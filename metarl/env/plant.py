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
    BASE_CREDITS, C_C5_BASE, C_LF_BASE, C_MAX_LIMITS, HEAT_LOAD_SCALE, INITIAL_C5,
    INITIAL_LF, MAX_FLOW_OUT_REAC, MAX_HEAT_LOAD, MIN_HARVEST_TITER, MIN_PRODUCT_TITER,
    S_INITIAL, S_MAX, S_MIN, S_OUT_RATE, SPECIES_IDX, SPECIES_NAMES, TIME_STEP_DURATION
)
from metarl.env.disturbances import DisturbanceMixin
from metarl.env.kinetics import _ETA, _PH_G_FACTOR, _PH_X_FACTOR, calculate_rates_numba

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
        self.max_stage_time = 150.0
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
        self.n_inner_steps = 30
        self.smoothing_coef = 0.005

        # Reactor dry floor (kg): a leak never drains a vessel below it, and
        # the underflow violation (id 2) is measured against it.
        self.DRY_FLOOR = 10.0

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
        self.M = np.full(self.n_reactors, 10000.0)
        self.C = np.zeros((self.n_reactors, len(self.species_names)))

        self.current_stage = np.zeros(self.n_reactors, dtype=np.int32)
        self.stage_time_remaining = np.zeros(self.n_reactors, dtype=np.float64)

        # 3. Inoculation Setup (Initial Yeast Loading)
        init_cell_conc = (1000 * 147.0) / 10000.0
        self.C[:, self.idx['Cell']] = init_cell_conc

        # 4. Tracking and Housekeeping
        self.prev_flow_action = np.zeros(self.n_reactors * 3)
        self.stage_credits = np.zeros(self.n_reactors, dtype=np.float64)
        for i in range(self.n_reactors):
            self.stage_credits[i] = self.base_credits[self.current_stage[i]]
        # Dense reward: track per-reactor ethanol mass for Δ computation
        self.prev_eth_total = np.zeros(self.n_reactors)

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
        norm_Mmax = (self.Mmax.reshape(-1, 1) / 220000.0)
        norm_Fmax_LF = (self.max_flow_LF.reshape(-1, 1) / (2487/3600 * 4.0))
        norm_Fmax_C5 = (self.max_flow_C5.reshape(-1, 1) / (628/3600 * 4.0))

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
        net_discharge = float(self.M[i]) - 10000.0
        self.episode_mass_discharged += net_discharge

        self.M[i] = 10000.0
        self.C[i, :] = 0.0

        init_cell_conc = (1000 * 147.0) / 10000.0
        self.C[i, self.idx['Cell']] = init_cell_conc
        # A turnaround_delay fault stretches the charge, as the cleanup.
        self.stage_credits[i] = self.base_credits[0] * self._turnaround_mult
        # reset dense-reward baseline for new batch
        self.prev_eth_total[i] = 0.0
