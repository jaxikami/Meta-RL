"""The fermentation plant both environments share.

FermentationPlant is the part of FermentationEnv (metarl/env/safe.py) and
FermentationEnvBench (metarl/env/bench.py) that does not depend on how constraints
are priced: dimensions and hardware limits, feeds, the reactor reset, the
product ledger with per-batch lots, the normalized observation, the SERL stage
mask and the heat-load proxy.  Each environment adds its own step (dynamics
and reward) and constraint shaping.
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import (
    BASE_CREDITS, C_C5_BASE, C_LF_BASE, C_MAX_LIMITS, CHARGE_MASS_KG, DISCHARGE_FLOOR_KG,
    DRY_FLOOR_KG, FC5_NORM, FLF_NORM, HEAT_LOAD_SCALE, INIT_CELL_CONC, INITIAL_C5, INITIAL_LF,
    M_NORM, MAX_FLOW_OUT_REAC, MAX_HEAT_LOAD, MIN_HARVEST_TITER, MIN_PRODUCT_TITER,
    N_INNER_STEPS, S_INITIAL, S_MAX, S_MIN, S_OUT_RATE, SPECIES_IDX, SPECIES_NAMES,
    TIME_STEP_DURATION, UNLOAD_COMPLETE_KG
)
from metarl.env.disturbances import DisturbanceMixin
from metarl.env.kinetics import (_ETA, _PH_G_FACTOR, _PH_X_FACTOR, calculate_rates_numba,
                           integrate_rk4)

# Commanded discharge, as a fraction of the maximum outflow, below which the
# valve is shut, so a batch can be held in the unload stage without draining
# it (and without tripping the harvest titer floor).  Meta-RL's value.
DISCHARGE_CUTOFF = 0.02

# Constraint ids with a relative depth past the limit.
DEPTH_CONSTRAINTS = (1, 2, 3, 4, 7, 8, 9)

# A violation's spike grows from 1x at the limit to (1 + DEPTH_GAIN)x deep past
# it, so a shallower violation is always cheaper (a gradient on how to improve),
# yet a runaway state (overflow depths reach ~5) cannot swamp the return.
DEPTH_GAIN = 4.0


def saturate(depth):
    """Relative violation depth mapped to [0, 1): d / (1 + d), monotone in d."""
    d = np.maximum(depth, 0.0)
    return d / (1.0 + d)


def spike_scale(depth):
    """Spike multiplier of a violation ``depth`` past its limit, in [1, 1 + DEPTH_GAIN)."""
    return 1.0 + DEPTH_GAIN * saturate(depth)


# Plant fields of a state snapshot (_plant_state()).
_ARRAY_FIELDS = ("C", "M", "current_stage", "stage_credits", "prev_flow_action",
                 "lot_mass", "lot_eth", "lot_prod_reward")
_SCALAR_FIELDS = ("total_lf", "total_c5", "S_mass", "time_step_count",
                  "episode_mass_discharged", "episode_initial_total_mass", "product_mass",
                  "product_eth", "on_spec_eth", "lots_passed", "lots_failed")


def merge_events(into, counts, depth_sum, depth_max):
    """Add one step's event dicts to the running (counts, sum, max) dicts."""
    c, s, m = into
    for cid, k in counts.items():
        c[cid] = c.get(cid, 0) + k
    for cid, d in depth_sum.items():
        s[cid] = s.get(cid, 0.0) + d
    for cid, d in depth_max.items():
        m[cid] = max(m.get(cid, 0.0), d)


class StepEvents:
    """The violation events of one inner step.

    Attributes:
        counts: Constraint id -> events (one per instance past its limit).
        depth_sum, depth_max: Constraint id -> summed / worst relative depth
            past the limit (DEPTH_CONSTRAINTS only).
        ids: Every constraint id with an event this step.
        violated: Whether a constraint (anything but instability) was violated.
    """

    def __init__(self):
        self.counts, self.depth_sum, self.depth_max = {}, {}, {}
        self.ids = set()
        self.violated = False

    def record(self, cid, depths, violated=True):
        """Count one event of ``cid`` per entry of ``depths`` (relative depth past the limit)."""
        if violated:
            self.violated = True
        self.ids.add(cid)
        depths = np.atleast_1d(np.asarray(depths, dtype=np.float64))
        if depths.size == 0:
            return
        self.counts[cid] = self.counts.get(cid, 0) + int(depths.size)
        if cid in DEPTH_CONSTRAINTS:
            self.depth_sum[cid] = self.depth_sum.get(cid, 0.0) + float(depths.sum())
            self.depth_max[cid] = max(self.depth_max.get(cid, 0.0), float(depths.max()))

    @property
    def violation_id(self):
        """The highest id with an event (0: none)."""
        return max(self.ids) if self.ids else 0


class StepFlows:
    """What the hardware did in one inner step (FermentationPlant._execute_flows()).

    Attributes:
        F_in, F_out: Realized inflow and outflow per reactor (kg/s).
        C_in: Inlet concentrations per reactor [n, 15].
        titer_now: Ethanol titer at the start of the step (what the harvest draws).
        harvested_eth: Ethanol harvested per reactor (g).
        draining: Reactors discharging this step.
        total_outflow: Plant outflow (kg/s).
        total_harvested_eth: Plant harvest (g).
    """

    __slots__ = ("F_in", "F_out", "C_in", "titer_now", "harvested_eth", "draining",
                 "total_outflow", "total_harvested_eth")


# Production reward per gram of harvested ethanol (0.5 per 1e7 g).
PROD_REWARD_COEF = 0.5
PROD_REWARD_SCALE = 1e7
# Penalty per reactor-step spent unloading (discourages lazy harvesting).
UNLOAD_TRICKLE_PENALTY = 0.0005
# Raw-material penalty per fraction of the initial feedstock consumed.
RAW_MAT_PENALTY_COEF = 10.0


class FermentationPlant(DisturbanceMixin):
    """Reactors, storage tank, feeds and product ledger, without a reward.

    Attributes:
        control_action_interval: Number of inner 1-hour steps per outer action.
        n_reactors: Number of parallel reactors.
        stateDim: Per-reactor observation width (includes shared global fields).
        actionDim: Per-reactor action width.
        max_steps: Episode horizon in inner steps.
    """

    def _init_plant(self, n_reactors, Mmax, max_flow_LF, max_flow_C5, control_action_interval):
        """Set the plant's dimensions, limits, feeds and timing.

        Args:
            n_reactors: Number of reactors to simulate in parallel.
            Mmax: Optional per-reactor max mass values.
            max_flow_LF: Optional per-reactor LF max flow rates.
            max_flow_C5: Optional per-reactor C5 max flow rates.
            control_action_interval: Number of inner simulation steps to repeat
                each outer agent action.
        """
        # ---------------- Core dimensions ----------------
        self.control_action_interval = control_action_interval
        self.n_reactors = n_reactors
        self.n_stage = 5

        # RL Agent Input Dimensionality Layout:
        # [15 Chemical Channels + 1 Mass + 5 Stage One-Hot + 1 Time Credit + 3 Structural Limits] = 25 per reactor.
        # Plus 3 global variables: 1 Central Storage Mass, 1 LF supply, 1 C5 supply.
        self.stateDim = 28
        # One kinetic-parameter row per reactor, so a biochemical disturbance
        # can act on one vessel alone.
        self.params = np.tile(np.array(
            [_ETA, _PH_G_FACTOR, _PH_X_FACTOR], dtype=np.float64), (self.n_reactors, 1))

        # RL Agent Output Dimensionality:
        # 4 actions per reactor: [Time Multiplier, Feed LF, Feed C5, Outflow Unload]
        self.actionDim = 4
        self.max_flow_out_reac = MAX_FLOW_OUT_REAC

        # ---------------- Global Resource Pool ----------------
        self.initial_lf = INITIAL_LF
        self.initial_c5 = INITIAL_C5

        # Central Storage Hardware Specifications
        self.S_max = S_MAX
        self.S_min = S_MIN
        self.S_out_rate = S_OUT_RATE
        self.S_initial = S_INITIAL

        # ---------------- Reactor Limits ----------------
        self.Mmax = self._array_expand(Mmax, 220000.0, self.n_reactors)
        self.max_flow_LF = self._array_expand(
            max_flow_LF, (2487/3600), self.n_reactors)
        self.max_flow_C5 = self._array_expand(
            max_flow_C5, (628/3600), self.n_reactors)

        # ---------------- Tracked Species Definition ----------------
        self.species_names = SPECIES_NAMES
        self.idx = SPECIES_IDX

        # ---------------- Feed Flow Properties ----------------
        # Per reactor: a toxin_spike fault contaminates one reactor's feed line.
        self.C_C5 = np.tile(C_C5_BASE, (self.n_reactors, 1))
        self.C_LF = np.tile(C_LF_BASE, (self.n_reactors, 1))

        # ---------------- Time Credit and Flow Integration ----------------
        self.base_credits = BASE_CREDITS.copy()
        self.time_step_duration = TIME_STEP_DURATION

        # RK4 sub-steps per 1-hour control tick.
        # 30 sub-steps (dt=120 s) is numerically stable for the fermentation
        # kinetics (fastest rates ~0.07/h → Δc < 0.003 per sub-step at dt=120s)
        # and halves the Numba ODE integration cost vs the original 60 sub-steps.
        self.n_inner_steps = N_INNER_STEPS
        self.smoothing_coef = 0.005

        # Reactor dry floor (kg): a leak never drains a vessel below it, and
        # the underflow violation (id 2) is measured against it.
        self.DRY_FLOOR = DRY_FLOOR_KG

        # Product quality (the Meta-RL plant's concentration constraints) and
        # the shared cooling loop's capacity.
        self.min_harvest_titer = MIN_HARVEST_TITER
        self.min_product_titer = MIN_PRODUCT_TITER
        self.max_heat_load = MAX_HEAT_LOAD

        # Episode-end numerical stability guardrail via mass conservation residual.
        self.mass_balance_abs_tol = 5.0
        self.mass_balance_rel_tol = 1e-6

        self.max_steps = 600

    def _reset_plant(self):
        """Reset the plant (biology, resources, ledger) for a new episode."""
        self.time_step_count = 0

        # 0. Nominal plant, with every scheduled disturbance re-armed
        self._rearm_disturbances()

        # 1. Global Resource Initialization
        self.total_lf = self.initial_lf
        self.total_c5 = self.initial_c5
        self.S_mass = self.S_initial

        # 2. Reactor State Initialization
        self.M = np.full(self.n_reactors, CHARGE_MASS_KG)
        self.C = np.zeros((self.n_reactors, len(self.species_names)))

        self.current_stage = np.zeros(self.n_reactors, dtype=np.int32)

        # 3. Inoculation Setup (Initial Yeast Loading)
        self.C[:, self.idx['Cell']] = INIT_CELL_CONC

        # 4. Tracking and Housekeeping
        self.prev_flow_action = np.zeros(self.n_reactors * 3)
        self.stage_credits = np.zeros(self.n_reactors, dtype=np.float64)
        for i in range(self.n_reactors):
            self.stage_credits[i] = self.base_credits[self.current_stage[i]]

        # 5. Product ledger.  Ethanol is kg x titer (g, as eth_harvested), so
        # eth / mass is a mass-weighted mean titer in the units of C[Eth].
        self.product_mass = 0.0
        self.product_eth = 0.0
        self.lot_mass = np.zeros(self.n_reactors)
        self.lot_eth = np.zeros(self.n_reactors)
        self.lot_prod_reward = np.zeros(self.n_reactors)
        self.on_spec_eth = 0.0
        self.lots_passed = 0
        self.lots_failed = 0

        self.episode_mass_discharged = 0.0
        self.episode_initial_total_mass = self._compute_total_tracked_mass()

    def _plant_state(self):
        """Copies of the plant's dynamic fields (for snapshots and resumption)."""
        out = {k: getattr(self, k).copy() for k in _ARRAY_FIELDS}
        out.update({k: getattr(self, k) for k in _SCALAR_FIELDS})
        return out

    def _set_plant_state(self, state_dict):
        """Restore the plant's dynamic fields from _plant_state()."""
        for k in _ARRAY_FIELDS:
            setattr(self, k, state_dict[k].copy())
        for k in _SCALAR_FIELDS:
            setattr(self, k, state_dict[k])

    @staticmethod
    def _array_expand(arr, default_value, size):
        """Expand scalar/short arrays to match reactor count.

        Args:
            arr: Optional input array-like to expand.
            default_value: Fill value when arr is None or shorter than size.
            size: Target output length.

        Returns:
            NumPy array of length size with dtype float64.

        Raises:
            ValueError: If arr has more elements than size.
        """
        if arr is None:
            return np.full(size, default_value, dtype=np.float64)
        arr = np.array(arr, dtype=np.float64)
        if arr.size > size:
            raise ValueError(
                f"Array length {arr.size} exceeds number of reactors {size}.")
        if arr.size < size:
            padding = np.full(size - arr.size, default_value, dtype=np.float64)
            arr = np.concatenate([arr, padding])
        return arr


    @property
    def lot_mean_titer(self):
        """Mass-weighted mean titer of each reactor's lot in progress (0 if empty)."""
        return np.where(self.lot_mass > 0.0,
                        self.lot_eth / np.maximum(self.lot_mass, 1e-12), 0.0)


    def _close_lot(self, i):
        """Release or reject reactor ``i``'s lot in progress, then open an empty one.

        Returns:
            None when the lot is empty, else a dict with ``passed``,
            ``shortfall`` (relative, >= 0) and ``clawback`` (production
            reward to withdraw, >= 0).
        """
        if self.lot_mass[i] <= 0.0:
            return None
        spec = self.min_product_titer
        shortfall = max(0.0, (spec - self.lot_eth[i] / self.lot_mass[i]) / spec)
        passed = shortfall <= 0.0
        clawback = 0.0
        if passed:
            self.on_spec_eth += float(self.lot_eth[i])
            self.lots_passed += 1
        else:
            self.lots_failed += 1
            clawback = float(self.lot_prod_reward[i])
        self.lot_mass[i] = self.lot_eth[i] = self.lot_prod_reward[i] = 0.0
        return {"passed": passed, "shortfall": shortfall, "clawback": clawback}


    def heat_load(self):
        """Each reactor's CO2 evolution rate in kg/h, the heat-load proxy.

        Returns:
            NumPy array of shape [n_reactors]; id 9 caps its sum.
        """
        rates = calculate_rates_numba(self.C, self.params)
        return rates[:, self.idx['CO2']] * self.M * HEAT_LOAD_SCALE


    def _compute_total_tracked_mass(self):
        """Compute tracked in-system mass in kilograms.

        Includes reactor masses, central storage mass, and remaining feedstocks.

        Returns:
            Total tracked mass in kg.
        """
        return float(np.sum(self.M) + self.S_mass + self.total_lf + self.total_c5)


    def get_state_norm(self):
        """Build the normalized observation tensor for the RL policy.

        Builds the normalized flat array required by the RL algorithm tensor.
        Every dimension is strictly squashed to mathematically stable bounds (usually [0,1]).

        Returns:
            NumPy array of shape [n_reactors, 28].
        """
        # 1. Normalize Chemical Concentrations
        norm_C = self.C / C_MAX_LIMITS

        # 2. Normalize Reactor Masses
        norm_M = (self.M.reshape(-1, 1) / self.Mmax.reshape(-1, 1))

        # 3. One-Hot Encode Reactor Stages (5 stages: 0–4)
        # Each reactor gets a binary indicator vector of length n_stage.
        # e.g. stage 0 → [1,0,0,0,0], stage 2 → [0,0,1,0,0]
        stage_onehot = np.eye(self.n_stage, dtype=np.float32)[
            self.current_stage]

        # 4. Normalize Stage Credits
        # Calculates the percentage of credit remaining relative to the base credit for the CURRENT stage.
        current_base_credits = self.base_credits[self.current_stage].reshape(
            -1, 1)
        norm_credit_rem = (
            self.stage_credits.reshape(-1, 1) / current_base_credits)

        # 4.5 Normalize Structural Limits (Heterogeneous capabilities exposed to the network)
        norm_Mmax = (self.Mmax.reshape(-1, 1) / M_NORM)
        norm_Fmax_LF = (self.max_flow_LF.reshape(-1, 1) / FLF_NORM)
        norm_Fmax_C5 = (self.max_flow_C5.reshape(-1, 1) / FC5_NORM)

        # 5. Normalize Central Storage Mass
        norm_S = np.array([self.S_mass / self.S_max])

        # 6. Normalize Remaining Feedstock Stocks
        norm_LF_stock = np.array([self.total_lf / self.initial_lf])
        norm_C5_stock = np.array([self.total_c5 / self.initial_c5])

        # Form the shared global observation state
        global_state = np.concatenate(
            [norm_S, norm_LF_stock, norm_C5_stock]).astype(np.float32)
        global_state_tiled = np.tile(global_state, (self.n_reactors, 1))

        # --- Grouping Per Reactor ---
        reactor_states = np.hstack(
            [norm_C, norm_M, stage_onehot, norm_credit_rem, norm_Mmax, norm_Fmax_LF, norm_Fmax_C5]).astype(np.float32)

        # Batch-shaped state output: (n_reactors, 28)
        state = np.hstack([reactor_states, global_state_tiled])
        return state


    @staticmethod
    def get_action_mask(state_tensor):
        """Calculate stage-based structural action mask from normalized state.

        Calculates the SE-RL discrete logistic constraint mask directly from the normalized state tensor.
        A return value of 1.0 means the flow action is structurally allowed for that reactor and stage.
        A return value of 0.0 means the action is 'greyed out' and forced to a safe baseline.

        Args:
            state_tensor: Tensor with final dimension 28 (or broadcastable batch
                shape ending in 28).

        Returns:
            Torch tensor mask with final dimension 4 ordered as
            [time_multiplier, LF, C5, outflow].
        """
        import torch
        # state_tensor shape is [..., 28]
        # stage_onehot is at indices 16:21.
        stage_0 = state_tensor[..., 16]
        stage_1 = state_tensor[..., 17]
        stage_3 = state_tensor[..., 19]
        stage_4 = state_tensor[..., 20]

        # Time multiplier (index 0) is allowed except in Stage 4 (Clean up) where it is fixed to 1.0
        mask_time = 1.0 - stage_4

        # LF (index 1) is allowed in Stage 0 and 1
        mask_lf = stage_0 + stage_1

        # C5 (index 2) is allowed in Stage 1 only
        mask_c5 = stage_1

        # Outflow drain (index 3) is allowed in Stage 3
        mask_out = stage_3

        mask = torch.stack([mask_time, mask_lf, mask_c5, mask_out], dim=-1)
        return mask


    def _reset_single_reactor(self, i):
        """Reset one reactor after completing cleanup stage (4).

        Reinitializes a singular reactor upon completing Clean up Stage (4).
        Prepares the vessel for the next seamless batch integration.

        Mass-balance accounting:
          The cleanup physically empties the vessel (all M[i] kg exits the system)
          and then re-charges it with 10,000 kg of fresh inoculation from outside.
          Net effect on the system mass ledger = M[i] - 10,000 (signed).

          - M[i] > 10,000 → net positive discharge (more left than re-added)
          - M[i] < 10,000 → net negative discharge (inoculation adds mass into system)

          Using a signed accumulation keeps `episode_mass_discharged` in sync with
          the actual change in tracked in-system mass so the episode-end balance
          check (violation ID 6) does not fire spuriously.

                Args:
                        i: Reactor index to reset.
        """
        # Signed net discharge: positive = mass left system, negative = mass added from outside.
        net_discharge = float(self.M[i]) - CHARGE_MASS_KG
        self.episode_mass_discharged += net_discharge

        self.M[i] = CHARGE_MASS_KG
        self.C[i, :] = 0.0
        self.C[i, self.idx['Cell']] = INIT_CELL_CONC
        # A turnaround_delay fault stretches the charge, as the cleanup.
        self.stage_credits[i] = self.base_credits[0] * self._turnaround_mult

    # ─── Inner Step ──────────────────────────────────────────────────────────

    def _step_internal(self, action, is_first=True):
        """Execute one 1-hour inner step given the agent's flow request.

        The stages run in this order; only _enforce_stage_rules() and
        _price_constraints() differ between the environments.

        Args:
            action: Flattened action array in [-1, 1], shape [n_reactors * 4].
            is_first: True on the first inner step of an outer action (the
                smoothing penalty compares against the previous action then).

        Returns:
            A tuple of:
                - state: Next normalized state, shape [n_reactors, 28].
                - reward_arr: Per-reactor reward array.
                - done: Episode termination flag.
                - info: Dict of decomposed reward terms and constraint flags.
        """
        # Mid-episode disturbances due at this step (leak, contamination, ...)
        leaked_mass = self._apply_disturbances()
        a_clipped, a_reshaped, multipliers = self._decode_action(action)
        self._enforce_stage_rules(a_reshaped, multipliers)
        self._cut_unavailable_flows(a_reshaped)
        closed_lots = self._advance_stages(a_reshaped, multipliers)
        fl = self._execute_flows(a_reshaped)
        instability = self._integrate(fl)
        self._update_storage(fl)
        prod_reward = self._production_reward(fl)
        ev = StepEvents()
        constraint_penalty_arr, clawback, info = self._price_constraints(fl, closed_lots, ev)
        info["eth_harvested"] = fl.total_harvested_eth
        info["leaked_mass"] = leaked_mass
        return self._finish_inner_step(a_clipped, a_reshaped, is_first, prod_reward, clawback,
                                       constraint_penalty_arr, instability, ev, info)

    def _price_constraints(self, fl, closed_lots, ev):
        """The environment's constraint detection and pricing of one inner step.

        Args:
            fl: StepFlows of the step.
            closed_lots: _advance_stages() result.
            ev: StepEvents to record every violation in.

        Returns:
            A tuple of the constraint penalty per reactor [n], the production
            reward withdrawn from rejected lots, and the environment's info
            entries (the violation magnitudes, 'heat_load', ...).
        """
        raise NotImplementedError

    def _decode_action(self, action):
        """Map an action in [-1, 1] to commands in [0, 1] and time multipliers.

        Returns:
            A tuple of the clipped action [n * 4], the commands [n, 4] (a view
            the later stages edit in place) and the stage-credit multipliers [n].
        """
        a_clipped = np.clip(action, -1.0, 1.0)
        a = (a_clipped + 1.0) / 2.0
        a_reshaped = a.reshape(self.n_reactors, 4)
        multipliers = (a_reshaped[:, 0] * 1.5) + 0.5
        return a_clipped, a_reshaped, multipliers

    def _enforce_stage_rules(self, a_reshaped, multipliers):
        """Environment-side stage overrides (none: the agent's mask enforces them)."""

    def _cut_unavailable_flows(self, a_reshaped):
        """Empty feed tanks feed nothing; a discharge below the cutoff shuts the valve."""
        if self.total_lf <= 0:
            a_reshaped[:, 1] = 0.0
        if self.total_c5 <= 0:
            a_reshaped[:, 2] = 0.0
        a_reshaped[a_reshaped[:, 3] < DISCHARGE_CUTOFF, 3] = 0.0

    def _advance_stages(self, a_reshaped, multipliers):
        """Spend stage credits and move every reactor whose stage ran out.

        A batch completes when its cleanup ends: its lot is judged then and
        the reactor is recharged.  Unload holds until the reactor is nearly
        empty.  Then the batch-boundary interlock: the discharge valve opens
        only in unload, so a reactor that leaves stage 3 (or cleans up and
        recharges mid-hold) does not drain on.

        Returns:
            Per reactor, the _close_lot() result of a batch that completed
            this step, else None.
        """
        dt_hours = self.time_step_duration / 3600.0
        closed_lots = [None] * self.n_reactors
        self.stage_credits -= multipliers * dt_hours
        for i in np.where(self.stage_credits <= 0)[0]:
            if self.current_stage[i] == 4:
                self.current_stage[i] = 0
                closed_lots[i] = self._close_lot(i)
                self._reset_single_reactor(i)
            elif self.current_stage[i] == 3 and self.M[i] >= UNLOAD_COMPLETE_KG:
                self.stage_credits[i] = 0.0
            else:
                self.current_stage[i] += 1
                self.stage_credits[i] = self.base_credits[self.current_stage[i]]
                if self.current_stage[i] == 4:
                    # A turnaround_delay fault stretches the cleanup.
                    self.stage_credits[i] *= self._turnaround_mult
        a_reshaped[self.current_stage != 3, 3] = 0.0
        return closed_lots

    def _execute_flows(self, a_reshaped):
        """Feeds, discharges and the harvest ledger of one inner step.

        A feed_gain fault scales what the pumps deliver for a given command.
        A reactor drains at most down to 50 kg (CO2 loss during the step could
        otherwise push it below the dry floor).  The harvest is drawn at the
        titer standing at the start of the step, into the reactor's lot and
        the storage tank.  Feedstock is allocated reactor by reactor from the
        shared supply.

        Returns:
            StepFlows of the step.
        """
        fl = StepFlows()
        f_in_lf_req = a_reshaped[:, 1] * self.max_flow_LF * self.feed_gain
        f_in_c5_req = a_reshaped[:, 2] * self.max_flow_C5 * self.feed_gain
        max_possible_f = np.maximum(0.0, (self.M - DISCHARGE_FLOOR_KG) / self.time_step_duration)
        fl.F_out = np.minimum(a_reshaped[:, 3] * self.max_flow_out_reac, max_possible_f)
        fl.total_outflow = float(np.sum(fl.F_out))

        fl.titer_now = self.C[:, self.idx['Eth']].copy()
        harvested_mass = fl.F_out * self.time_step_duration
        fl.harvested_eth = harvested_mass * fl.titer_now
        fl.draining = fl.F_out > 0.0
        fl.total_harvested_eth = float(np.sum(fl.harvested_eth))
        self.lot_mass += harvested_mass
        self.lot_eth += fl.harvested_eth
        self.product_mass += float(np.sum(harvested_mass))
        self.product_eth += fl.total_harvested_eth

        fl.F_in = np.zeros(self.n_reactors)
        fl.C_in = np.zeros((self.n_reactors, 15))
        for i in range(self.n_reactors):
            act_lf_mass = min(f_in_lf_req[i] * self.time_step_duration, self.total_lf)
            act_c5_mass = min(f_in_c5_req[i] * self.time_step_duration, self.total_c5)
            self.total_lf -= act_lf_mass
            self.total_c5 -= act_c5_mass
            f_lf_act = act_lf_mass / self.time_step_duration
            f_c5_act = act_c5_mass / self.time_step_duration
            fl.F_in[i] = f_lf_act + f_c5_act
            fl.C_in[i] = (f_c5_act * self.C_C5[i] + f_lf_act * self.C_LF[i]) / \
                fl.F_in[i] if fl.F_in[i] > 0 else np.zeros(15)
        return fl

    def _integrate(self, fl):
        """Advance the reactors' kinetics over the step (RK4); True on instability."""
        c_new, m_new, instability = integrate_rk4(
            self.C.astype(np.float64), self.M.astype(np.float64), fl.F_in, fl.F_out, fl.C_in,
            self.params, self.time_step_duration / self.n_inner_steps, self.n_inner_steps)
        self.C = c_new
        self.M = m_new
        return instability

    def _update_storage(self, fl):
        """Reactor discharges in, the (possibly disturbed) downstream draw out."""
        prev_s_mass = self.S_mass
        self.S_mass += (fl.total_outflow * self.time_step_duration -
                        self.S_out_rate * self._downstream_mult * self.time_step_duration)
        self.S_mass = np.maximum(0.0, self.S_mass)
        # Track outlet discharge actually removed from the in-system inventory.
        self.episode_mass_discharged += max(0.0, prev_s_mass +
                                            fl.total_outflow * self.time_step_duration
                                            - self.S_mass)

    def _production_reward(self, fl):
        """The plant's production reward this step; each lot banks its own share.

        A lot's banked reward is withdrawn if the lot is rejected.
        """
        prod_reward = PROD_REWARD_COEF * (fl.total_harvested_eth / PROD_REWARD_SCALE)
        self.lot_prod_reward += PROD_REWARD_COEF * (fl.harvested_eth / PROD_REWARD_SCALE)
        return prod_reward

    def _operating_penalties(self, a_clipped, is_first):
        """Unload trickle, flow smoothing and raw-material penalties, per reactor.

        Smoothing charges the change of the flow commands on the first inner
        step of an outer action.  The raw-material penalty is the fraction of
        the initial feedstock consumed (read after this step's allocation).

        Returns:
            A tuple of (eff, smoothing, raw_mat) arrays [n].
        """
        eff_penalty_arr = np.where(self.current_stage == 3, UNLOAD_TRICKLE_PENALTY, 0.0)
        if is_first:
            current_flow_action = a_clipped.reshape(self.n_reactors, 4)[:, 1:4]
            prev_flow = self.prev_flow_action.reshape(self.n_reactors, 3)
            smoothing_penalty_arr = self.smoothing_coef * \
                np.mean(np.square(current_flow_action - prev_flow), axis=1)
            self.prev_flow_action = current_flow_action.flatten().copy()
        else:
            smoothing_penalty_arr = np.zeros(self.n_reactors)
        lf_ratio = self.total_lf / self.initial_lf
        c5_ratio = self.total_c5 / self.initial_c5
        lf_consumed_this_step = max(0.0, lf_ratio * self.initial_lf - self.total_lf)
        c5_consumed_this_step = max(0.0, c5_ratio * self.initial_c5 - self.total_c5)
        raw_mat_penalty_global = RAW_MAT_PENALTY_COEF * (
            lf_consumed_this_step / max(self.initial_lf, 1.0) +
            c5_consumed_this_step / max(self.initial_c5, 1.0)
        )
        raw_mat_penalty_arr = np.full(self.n_reactors, raw_mat_penalty_global / self.n_reactors)
        return eff_penalty_arr, smoothing_penalty_arr, raw_mat_penalty_arr

    def _finish_inner_step(self, a_clipped, a_reshaped, is_first, prod_reward, clawback,
                           constraint_penalty_arr, instability, ev, info):
        """Assemble the step's reward, advance the clock and complete ``info``.

        Args:
            a_clipped, a_reshaped: From _decode_action() (after the overrides).
            is_first: First inner step of an outer action.
            prod_reward: From _production_reward().
            clawback: Production reward withdrawn from rejected lots this step.
            constraint_penalty_arr: The environment's constraint pricing [n].
            instability: From _integrate().
            ev: StepEvents of the step (instability is recorded here).
            info: The environment's own info entries; completed in place.

        Returns:
            The (state, reward, done, info) of _step_internal().
        """
        eff_penalty_arr, smoothing_penalty_arr, raw_mat_penalty_arr = \
            self._operating_penalties(a_clipped, is_first)
        # Equal share of the plant's production reward (less any rejected
        # lot's), minus every penalty.
        reward_arr = ((prod_reward - clawback) / self.n_reactors) - (
            smoothing_penalty_arr + eff_penalty_arr + constraint_penalty_arr + raw_mat_penalty_arr)

        self.time_step_count += 1
        done = (self.time_step_count >= self.max_steps or instability)
        # Numerical instability in the ODE integrator ends the episode (id 5).
        if instability:
            ev.record(5, 1.0, violated=False)

        info.update({
            "prod": prod_reward,
            "constraint": np.sum(constraint_penalty_arr),
            "smoothing": np.sum(smoothing_penalty_arr),
            "eff": np.sum(eff_penalty_arr),
            "raw_mat_penalty": float(np.sum(raw_mat_penalty_arr)),
            "lf_remaining": self.total_lf,
            "c5_remaining": self.total_c5,
            "clawback": float(clawback),
            "hard_violation": instability or ev.violated,
            "constraint_violated": ev.violated,
            "violation_id": ev.violation_id,
            "violated_constraints": list(ev.ids),
            "violation_counts": ev.counts,
            "violation_depth_sum": ev.depth_sum,
            "violation_depth_max": ev.depth_max,
            "actual_a_reshaped": a_reshaped.copy(),
        })
        return self.get_state_norm(), reward_arr, done, info

    # ─── Outer Step (action repeat) ──────────────────────────────────────────

    def _new_step_extras(self):
        """Environment-specific accumulators of an outer step (see step())."""
        return {}

    def _accumulate_step_extras(self, extras, info):
        """Fold one inner step's info into the environment's accumulators."""

    def _step_extras_info(self, extras, steps_completed):
        """The environment's own outer-step info entries."""
        return {}

    def _on_lot_released(self, i):
        """Hook: reactor ``i``'s lot met the purity spec."""

    def _rejected_lot_penalty(self, i, shortfall):
        """Penalty of reactor ``i``'s rejected lot at episode end."""
        raise NotImplementedError

    def _empty_shipment_penalty(self):
        """Plant-wide penalty when nothing at all was shipped (a full purity miss)."""
        raise NotImplementedError

    def step(self, action):
        """Execute one outer environment step with action repeat.

        Repeats the agent's action for control_action_interval inner steps
        (fewer if the episode ends) and aggregates their diagnostics.  At the
        episode end every open lot is judged, the empty-shipment rule applies
        and the mass balance is checked; any of them counts as one more
        violation step.

        Args:
            action: Flattened action array in [-1, 1], shape [n_reactors * 4].

        Returns:
            A tuple of:
                - state: Next normalized state, shape [n_reactors, 28].
                - total_reward: Per-reactor accumulated reward over repeats.
                - done: Episode termination flag.
                - info_out: Aggregated diagnostics for the outer step.
        """
        total_prod = 0.0
        total_constraint = 0.0
        total_smoothing = 0.0
        total_eff = 0.0
        total_reward = np.zeros(self.n_reactors)
        total_harvested_eth_accum = 0.0
        raw_mat_penalty_accum = 0.0

        max_storage_overflow = 0.0
        max_storage_underflow = 0.0
        max_reactor_overflow = 0.0
        max_reactor_underflow = 0.0
        max_titer_violation = 0.0
        max_purity_violation = 0.0
        max_heat_violation = 0.0
        heat_load_peak = 0.0
        clawback_accum = 0.0
        # Violation events: id -> count, depth sum and depth max.
        events = ({}, {}, {})
        extras = self._new_step_extras()

        hard_violation_accum = False
        constraint_violated_accum = False
        violation_id_accum = 0
        violation_count_accum = 0
        violation_ids_accum = set()

        actual_a_reshaped_first = None
        lf_remaining_last = 0.0
        c5_remaining_last = 0.0

        # Track actual ticks executed (may be < interval if done fires early)
        steps_completed = 0
        for idx in range(self.control_action_interval):
            state, reward, done, info = self._step_internal(action, is_first=(idx == 0))
            steps_completed += 1

            total_prod += info.get("prod", 0.0)
            total_constraint += info.get("constraint", 0.0)
            total_smoothing += info.get("smoothing", 0.0)
            total_eff += info.get("eff", 0.0)
            total_reward += reward
            total_harvested_eth_accum += info.get("eth_harvested", 0.0)
            raw_mat_penalty_accum += info.get("raw_mat_penalty", 0.0)

            # Track worst-case violation magnitude seen across any sub-step
            max_storage_overflow = max(max_storage_overflow, info.get(
                "storage_overflow_violation", 0.0))
            max_storage_underflow = max(max_storage_underflow, info.get(
                "storage_underflow_violation", 0.0))
            max_reactor_overflow = max(max_reactor_overflow, info.get(
                "reactor_overflow_violation", 0.0))
            max_reactor_underflow = max(max_reactor_underflow, info.get(
                "reactor_underflow_violation", 0.0))
            max_titer_violation = max(max_titer_violation, info["harvest_titer_violation"])
            max_purity_violation = max(max_purity_violation, info["product_purity_violation"])
            max_heat_violation = max(max_heat_violation, info["heat_load_violation"])
            heat_load_peak = max(heat_load_peak, info["heat_load"])
            clawback_accum += info["clawback"]
            self._accumulate_step_extras(extras, info)
            merge_events(events, info["violation_counts"], info["violation_depth_sum"],
                         info["violation_depth_max"])

            hard_violation_accum = hard_violation_accum or info.get("hard_violation", False)
            constraint_violated_accum = constraint_violated_accum or info.get(
                "constraint_violated", False)
            if info.get("violation_id", 0) > 0:
                violation_id_accum = max(violation_id_accum, info["violation_id"])
            if "violated_constraints" in info:
                violation_ids_accum.update(info["violated_constraints"])
            if info.get("hard_violation", False) or info.get("constraint_violated", False):
                violation_count_accum += 1

            lf_remaining_last = info.get("lf_remaining", 0.0)
            c5_remaining_last = info.get("c5_remaining", 0.0)

            if idx == 0:
                actual_a_reshaped_first = info.get("actual_a_reshaped")

            if done:
                break

        info_out = {
            "prod": total_prod,
            "constraint": total_constraint,
            "smoothing": total_smoothing,
            "eff": total_eff,
            "eth_harvested": total_harvested_eth_accum,
            "raw_mat_penalty": raw_mat_penalty_accum,
            "lf_remaining": lf_remaining_last,
            "c5_remaining": c5_remaining_last,
            "storage_overflow_violation": float(max_storage_overflow),
            "storage_underflow_violation": float(max_storage_underflow),
            "reactor_overflow_violation": float(max_reactor_overflow),
            "reactor_underflow_violation": float(max_reactor_underflow),
            "harvest_titer_violation": float(max_titer_violation),
            "product_purity_violation": float(max_purity_violation),
            "heat_load_violation": float(max_heat_violation),
            "heat_load_peak": float(heat_load_peak),
            "clawback": float(clawback_accum),
            "hard_violation": hard_violation_accum,
            "constraint_violated": constraint_violated_accum,
            "violation_id": violation_id_accum,
            "violation_ids": list(violation_ids_accum),
            "violation_count": violation_count_accum,
            "actual_a_reshaped": actual_a_reshaped_first,
        }
        info_out.update(self._step_extras_info(extras, steps_completed))

        if done:
            self._episode_end_accounting(total_reward, events, info_out, max_purity_violation)

        counts, depth_sum, depth_max = events
        info_out["violation_counts"] = counts
        info_out["violation_depth_sum"] = depth_sum
        info_out["violation_depth_max"] = depth_max
        info_out["violation_events"] = int(sum(counts.values()))
        # Product ledger so far (released lots only count as on-spec).
        info_out["on_spec_eth"] = float(self.on_spec_eth)
        info_out["lots_passed"] = int(self.lots_passed)
        info_out["lots_failed"] = int(self.lots_failed)
        info_out["product_mass"] = float(self.product_mass)
        info_out["product_eth"] = float(self.product_eth)
        return state, total_reward, done, info_out

    def _episode_end_accounting(self, total_reward, events, info_out, max_purity_violation):
        """Judge every open lot, the empty-shipment rule and the mass balance.

        Updates ``total_reward``, ``events`` and ``info_out`` in place.
        """
        terminal_ids = set()

        # ── Constraint 8: every open lot is released or rejected ──────────
        purity_penalty = np.zeros(self.n_reactors)
        clawback = 0.0
        for i in range(self.n_reactors):
            lot = self._close_lot(i)
            if lot is None:
                continue
            if lot["passed"]:
                self._on_lot_released(i)
                continue
            purity_penalty[i] += self._rejected_lot_penalty(i, lot["shortfall"])
            clawback += lot["clawback"]
            merge_events(events, {8: 1}, {8: lot["shortfall"]}, {8: lot["shortfall"]})
            max_purity_violation = max(max_purity_violation, lot["shortfall"])
            terminal_ids.add(8)
        if self.product_mass <= 0.0:
            # Shipping nothing at all fails the spec, so doing nothing is not
            # "safe": one event for the plant, priced as a full miss.
            purity_penalty += self._empty_shipment_penalty() / self.n_reactors
            merge_events(events, {8: 1}, {8: 1.0}, {8: 1.0})
            max_purity_violation = 1.0
            terminal_ids.add(8)
        total_reward -= purity_penalty + clawback / self.n_reactors
        info_out["constraint"] += float(np.sum(purity_penalty))
        info_out["clawback"] += clawback
        info_out["product_purity_violation"] = float(max_purity_violation)

        # End-of-episode mass balance check to catch long-horizon numerical drift.
        current_total_mass = self._compute_total_tracked_mass()
        expected_total_mass = self.episode_initial_total_mass - self.episode_mass_discharged
        mass_balance_error = current_total_mass - expected_total_mass
        mass_balance_tol = max(
            self.mass_balance_abs_tol,
            self.mass_balance_rel_tol * max(self.episode_initial_total_mass, 1.0)
        )
        mass_balance_ok = abs(mass_balance_error) <= mass_balance_tol

        info_out["mass_balance_error"] = float(mass_balance_error)
        info_out["mass_balance_tol"] = float(mass_balance_tol)
        info_out["mass_balance_ok"] = bool(mass_balance_ok)

        if not mass_balance_ok:
            merge_events(events, {6: 1}, {}, {})
            terminal_ids.add(6)

        # The terminal accounting counts as one more violation step.
        if terminal_ids:
            info_out["hard_violation"] = True
            info_out["constraint_violated"] = True
            info_out["violation_id"] = max(int(info_out["violation_id"]), max(terminal_ids))
            info_out["violation_ids"] = list(set(info_out["violation_ids"]) | terminal_ids)
            info_out["violation_count"] = int(info_out["violation_count"]) + 1
