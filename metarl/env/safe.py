"""Safe fermentation environment with Lagrangian constraint shaping.

This module defines FermentationEnv, the environment variant used with the
safe PPO agent. It includes biochemical simulation, staged process logic, and
constraint-aware reward shaping with both soft buffers and adaptive Lagrange
multipliers.

Violation ids (the reactor-local ones are counted per reactor):

    1  reactor overflow       M_i > Mmax                               (per reactor, per step)
    2  reactor underflow      M_i < DRY_FLOOR                          (per reactor, per step)
    3  storage overflow       S > usable storage volume                (plant, per step)
    4  storage underflow      S < S_MIN                                (plant, per step)
    5  numerical instability  ODE blow-up / NaN                        (plant)
    6  mass-balance drift     conservation residual beyond tolerance   (plant, episode end)
    7  harvest titer floor    reactor i drains below MIN_HARVEST_TITER (per reactor, per step)
    8  product purity spec    a lot's mean titer below MIN_PRODUCT_TITER
                              (per lot, and once if nothing was shipped)
    9  heat-removal capacity  plant CO2 evolution rate > MAX_HEAT_LOAD   (plant, per step)

Ids 7 and 8 are the Meta-RL plant's concentration constraints.  Each batch's
discharge is one lot, released or rejected when that reactor's cleanup ends
(and at episode end); a rejected lot's production reward is withdrawn.

Id 9 couples scheduling and control: the reactors share one cooling loop, so
how fast each one is fed (control) and when the batches' fed-batch phases
overlap (scheduling) together decide whether the plant stays under the cap.

An outer step's ``violation_counts`` splits its violation EVENTS by id: every
(inner step, constraint, instance) triple, the instance being the reactor
for ids 1, 2 and 7, the lot for 8 and the plant otherwise.
``violation_depth_max`` / ``violation_depth_sum`` hold the relative depth
past the limit of those events (ids 5 and 6 have no depth).
``violation_count`` is unchanged: the number of inner steps with any event.
``near_miss_counts`` splits the NEAR MISSES the same way: (inner step,
constraint, instance) triples inside that constraint's soft-wall buffer zone
but not past its limit (reactor mass above 90% of Mmax or below 100 kg, storage
above 90% of its usable volume or below 8 x S_MIN, a draining reactor's titer
or a lot's running mean titer below 1.15 x spec, heat load above 90% of the
cap).  ``near_miss_steps`` is the number of inner steps with any near miss.

Violations are priced by the Lagrange multipliers alone.  The fixed
per-violation penalties (a spike on every violation, the storage-underflow
spike and the fixed part of a rejected lot's penalty) apply only with
``fixed_violation_penalties``, the no-Lagrangian case study.  The soft walls
and barriers that act before a limit is reached are reward shaping and apply
in both.

Mid-episode disturbances (``schedule_disturbance``, see metarl/env/disturbances.py) are
the Meta-RL plant's fault kinds.  The plant is observed exactly.
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.kinetics import integrate_rk4
from metarl.env.plant import (DEPTH_CONSTRAINTS, DISCHARGE_CUTOFF, FermentationPlant, merge_events,
                         saturate, spike_scale)

# Constraint ids counted per reactor (the rest are plant-wide; 8 is counted per lot).
REACTOR_CONSTRAINTS = (1, 2, 7, 8)

_LAMBDA_SCALARS = ("storage_overflow_lambda", "storage_underflow_lambda", "heat_load_lambda")
_LAMBDA_ARRAYS = ("reactor_overflow_lambda", "reactor_underflow_lambda",
                  "harvest_titer_lambda", "product_purity_lambda")


class FermentationEnv(FermentationPlant):
    """Full RL environment wrapping biochemical physics and safe shaping.

    Adds to FermentationPlant the step (stage progression, kinetics, storage)
    and constraint-aware rewards with soft walls and adaptive Lagrange
    multipliers.
    """

    def __init__(self, n_reactors=3, Mmax=None, max_flow_LF=None, max_flow_C5=None, control_action_interval=5):
        """Initialise the safe fermentation environment.

        Args:
            n_reactors: Number of reactors to simulate in parallel.
            Mmax: Optional per-reactor max mass values.
            max_flow_LF: Optional per-reactor LF max flow rates.
            max_flow_C5: Optional per-reactor C5 max flow rates.
            control_action_interval: Number of inner simulation steps to repeat
                each outer agent action.
        """
        self._init_plant(n_reactors, Mmax, max_flow_LF, max_flow_C5, control_action_interval)

        # ── Lagrangian Multiplier Constraint Enforcement ─────────────────────
        # Each constraint has a dedicated Lagrangian multiplier λ that grows
        # when it is violated and decays when satisfied.  The penalty injected
        # into the reward = λ × violation_magnitude, giving the PPO policy
        # gradient a negative signal proportional to both the severity and
        # persistence of violations.
        #
        # Two gradient-signal mechanisms work in concert:
        #   1. λ-weighted hard penalty — fires when constraint is actively violated;
        #      λ accumulates across episodes so repeated violations amplify.
        #   2. Buffer-zone soft wall  — λ-independent quadratic ramp that activates
        #      BEFORE the hard constraint, providing smooth early-warning gradient
        #      so the policy steers away before hitting the cliff.
        # A violation itself is priced by its multiplier only.  The no-Lagrangian
        # case study instead prices it with fixed penalties (BASE_SPIKE and the
        # rest; fixed_violation_penalties) and keeps every multiplier at 0.
        self.lagrange_lr = 0.5             # Δλ = lr × violation per violated step
        self.lagrange_decay = 0.5 / 600.0  # Per-step exponential decay when satisfied
        self.lagrange_target_slack = 0.0   # Zero tolerance: any violation grows λ
        self.lagrange_max = 2.0            # Upper clamp: a saturated dual costs <= 2/step per constraint

        # Buffer-zone soft-wall coefficients.  A quadratic penalty ramps smoothly
        # from 0 at the buffer edge to BUFFER_COEF at the hard constraint boundary.
        # Gradient = 2·BUFFER_COEF·depth / zone_width — smooth, nonzero, monotonic.
        # This is λ-independent: fires every step regardless of violation history,
        # ensuring PPO gradient always has a nonzero constraint-aware signal.
        # (The underflow violation is normalized by DRY_FLOOR, so its signal
        # spans [0, 1] instead of being crushed by division by Mmax.)
        # Kept well below BASE_SPIKE (0.1): at 0.02 the walls plus purity
        # shaping cost about as much as production in violation-free episodes.
        self.BUFFER_COEF = 0.005
        self.OVERFLOW_BUFFER_FRAC = 0.10        # Activates at 90% Mmax
        self.UNDERFLOW_BUFFER_KG = 100.0        # Activates at M < 100 kg
        self.STORAGE_OVERFLOW_BUFFER_FRAC = 0.10  # Activates at 90% S_max
        # Activates at S < 8 × S_min (~10% of S_max)
        self.STORAGE_UNDERFLOW_BUFFER_MULT = 8.0

        # ── Product quality (the Meta-RL plant's concentration constraints) ─
        # Soft walls as for the inventory constraints: the titer floor's from
        # 15% above spec on draining reactors, the purity spec's on each lot's
        # running mean titer.  A rejected lot costs PURITY_PENALTY_COEF x its
        # relative shortfall plus its dual, and its production reward is
        # withdrawn.  Meta-RL prices it at 10x its violation spike (3.0 vs
        # 0.3); the same ratio here against BASE_SPIKE = 0.1 gives 1.0.
        self.TITER_BUFFER_FRAC = 0.15
        self.PURITY_BUFFER_FRAC = 0.15
        # Dense, on every lot below 1.15x spec (including lots still ramping
        # up), so it is kept small to not penalise ordinary production.
        self.PURITY_SHAPING_COEF = 0.002
        self.PURITY_PENALTY_COEF = 1.0

        # ── Plant heat removal (shared cooling loop) ──────────────────────
        # Soft wall from HEAT_BUFFER_FRAC below the cap.  The spike, soft wall
        # and dual are split over the reactors by their share of the load, so
        # the reactors driving it carry its cost.
        self.HEAT_BUFFER_FRAC = 0.10

        self._zero_lambdas()
        self.lagrange_updates_enabled = True
        self.reset_lambdas_per_episode = False
        # Fixed per-violation penalties, for the no-Lagrangian case study only.
        self.fixed_violation_penalties = False

        self._init_disturbances()
        self.reset()

    def _zero_lambdas(self):
        """Every Lagrange multiplier back to 0."""
        for k in _LAMBDA_SCALARS:
            setattr(self, k, 0.0)
        for k in _LAMBDA_ARRAYS:
            setattr(self, k, np.zeros(self.n_reactors))

    def reset(self):
        """Reset biological state, resources, and episode bookkeeping.

        Returns:
            Normalized observation array of shape [n_reactors, 28].
        """
        if self.reset_lambdas_per_episode:
            self._zero_lambdas()
        self._reset_plant()
        return self.get_state_norm()

    def _get_full_internal_state(self):
        """Capture full internal state (plant and multipliers) for diagnostics and resumption.

        Returns:
            Dict containing copies of dynamic environment state fields.
        """
        out = self._plant_state()
        out.update({k: getattr(self, k) for k in _LAMBDA_SCALARS})
        out.update({k: getattr(self, k).copy() for k in _LAMBDA_ARRAYS})
        return out

    def _set_full_internal_state(self, state_dict):
        """Restore full internal state from a _get_full_internal_state() snapshot."""
        self._set_plant_state(state_dict)
        for k in _LAMBDA_SCALARS:
            setattr(self, k, state_dict[k])
        for k in _LAMBDA_ARRAYS:
            setattr(self, k, state_dict[k].copy())

    # --- LAGRANGE MULTIPLIERS ---

    def _update_lambda(self, lam, violation):
        """Advance dual multipliers one step: grow while violated, decay otherwise."""
        lam = np.asarray(lam, dtype=np.float64)
        if not self.lagrange_updates_enabled:
            return lam.copy()
        out = np.where(violation > 0.0, lam + self.lagrange_lr * violation,
                       lam * (1.0 - self.lagrange_decay))
        return np.clip(out, 0.0, self.lagrange_max)

    def _lot_purity_penalty(self, i, shortfall, severity=None):
        """Purity cost of a rejected lot of reactor ``i`` (None: the plant).

        Args:
            i: Reactor whose dual prices it, or None for "the plant shipped
                nothing", priced on the mean of the reactors' duals.
            shortfall: Relative shortfall below MIN_PRODUCT_TITER.
            severity: Multiplier on PURITY_PENALTY_COEF; defaults to shortfall.

        Returns:
            The penalty, >= 0: the dual's price of the shortfall, plus the
            fixed PURITY_PENALTY_COEF x severity with fixed_violation_penalties.
        """
        sev = shortfall if severity is None else severity
        idx = np.arange(self.n_reactors) if i is None else np.array([i])
        lag = max(0.0, shortfall - self.lagrange_target_slack)
        fixed = self.PURITY_PENALTY_COEF * sev if self.fixed_violation_penalties else 0.0
        penalty = fixed + float(np.mean(self.product_purity_lambda[idx])) * lag
        self.product_purity_lambda[idx] = self._update_lambda(
            self.product_purity_lambda[idx], np.full(idx.size, max(lag, 1e-9)))
        return penalty

    def _decay_purity_lambda(self, i):
        """Relax reactor ``i``'s purity dual after a released lot."""
        self.product_purity_lambda[i] = self._update_lambda(
            self.product_purity_lambda[i:i + 1], np.zeros(1))[0]

    # --- INTERNAL STATE MANAGEMENT ---

    def _step_internal(self, action, is_first=True):
        """Execute one inner simulation step given the agent action request.

        Executes a 1-hour interval simulation given an Agent Flow Request.
        Integrates biology, time credits, penalty scaling, and Lagrangian constraints.

        Args:
            action: Flattened action array in [-1, 1], shape [n_reactors * 4].
            is_first: True when this is the first inner step of an outer action,
                used for smoothing penalty bookkeeping.

        Returns:
            A tuple of:
                - state: Next normalized state, shape [n_reactors, 28].
                - reward_arr: Per-reactor reward array.
                - done: Episode termination flag.
                - info: Dict of decomposed reward terms and constraint flags.
        """
        # Mid-episode disturbances due at this step (leak, contamination, ...)
        leaked_mass = self._apply_disturbances()

        # Lock action space bounds
        a_clipped = np.clip(action, -1.0, 1.0)
        a = (a_clipped + 1.0) / 2.0
        a_reshaped = a.reshape(self.n_reactors, 4)

        multipliers = (a_reshaped[:, 0] * 1.5) + 0.5

        # ==================== Phase 1: Native Stage Rules (SE-RL Failsafe DEMOUNTED) ====================
        # Enforced solely by PyTorch probability masking boundary constraints to permit true leaky-actuator simulations.

        # Master stock constraints remain local to the physical environment payload:
        if self.total_lf <= 0:
            a_reshaped[:, 1] = 0.0
        if self.total_c5 <= 0:
            a_reshaped[:, 2] = 0.0
        # A discharge commanded below the cutoff shuts the valve.
        a_reshaped[a_reshaped[:, 3] < DISCHARGE_CUTOFF, 3] = 0.0

        # Removed SE-RL explicit penalty mapping as constraints are strictly upheld by the agent mask

        total_reactor_outflow, total_harvested_eth = 0.0, 0.0
        dt_hours = self.time_step_duration / 3600.0

        # ==================== Phase 2: Time Credit Handling ====================
        # Vectorised decrement: update all reactors in one NumPy op, then
        # handle stage transitions only for the (rare) reactors that hit 0.
        # A batch completes when its cleanup ends: its lot is judged then.
        closed_lots = [None] * self.n_reactors
        self.stage_credits -= multipliers * dt_hours
        for i in np.where(self.stage_credits <= 0)[0]:
            if self.current_stage[i] == 4:
                self.current_stage[i] = 0
                closed_lots[i] = self._close_lot(i)
                self._reset_single_reactor(i)
            elif self.current_stage[i] == 3 and self.M[i] >= 100.0:
                self.stage_credits[i] = 0.0
            else:
                self.current_stage[i] += 1
                self.stage_credits[i] = self.base_credits[self.current_stage[i]]
                if self.current_stage[i] == 4:
                    # A turnaround_delay fault stretches the cleanup.
                    self.stage_credits[i] *= self._turnaround_mult
        # Batch-boundary interlock: the discharge valve opens only in unload.
        # An action decided in unload repeats for the whole outer step, so
        # without this a reactor that cleans up and recharges mid-step would
        # drain its fresh zero-titer charge into the lot and the tank.
        a_reshaped[self.current_stage != 3, 3] = 0.0

        # ==================== Phase 3: Hardware Execution ====================
        instability_detected = False

        # Use a constant number of RK4 inner steps regardless of stage.
        current_n_steps = self.n_inner_steps
        simulation_dt = self.time_step_duration / current_n_steps

        F_ins = np.zeros(self.n_reactors)
        Cins = np.zeros((self.n_reactors, 15))

        # Vectorised outflow / harvest computation.  A feed_gain fault scales
        # what the pumps deliver for a given command.
        f_in_lf_req = a_reshaped[:, 1] * self.max_flow_LF * self.feed_gain
        f_in_c5_req = a_reshaped[:, 2] * self.max_flow_C5 * self.feed_gain
        max_possible_f = np.maximum(
            0.0, (self.M - 50.0) / self.time_step_duration)
        F_outs = np.minimum(a_reshaped[:, 3] *
                            self.max_flow_out_reac, max_possible_f)
        total_reactor_outflow = float(np.sum(F_outs))

        # Harvest is drawn at the titer standing at the start of the step,
        # into each reactor's lot and into the storage tank.
        titer_now = self.C[:, self.idx['Eth']].copy()
        harvested_mass = F_outs * self.time_step_duration
        harvested_eth = harvested_mass * titer_now
        draining = F_outs > 0.0
        total_harvested_eth = float(np.sum(harvested_eth))
        self.lot_mass += harvested_mass
        self.lot_eth += harvested_eth
        self.product_mass += float(np.sum(harvested_mass))
        self.product_eth += total_harvested_eth

        # Feedstock allocation is inherently sequential (shared global supply).
        for i in range(self.n_reactors):
            act_lf_mass = min(
                f_in_lf_req[i] * self.time_step_duration, self.total_lf)
            act_c5_mass = min(
                f_in_c5_req[i] * self.time_step_duration, self.total_c5)
            self.total_lf -= act_lf_mass
            self.total_c5 -= act_c5_mass
            f_lf_act = act_lf_mass / self.time_step_duration
            f_c5_act = act_c5_mass / self.time_step_duration
            F_ins[i] = f_lf_act + f_c5_act
            Cins[i] = (f_c5_act * self.C_C5[i] + f_lf_act * self.C_LF[i]) / \
                F_ins[i] if F_ins[i] > 0 else np.zeros(15)

        c_new, m_new, instability_detected = integrate_rk4(
            self.C.astype(np.float64), self.M.astype(
                np.float64), F_ins, F_outs, Cins, self.params, simulation_dt, current_n_steps
        )
        self.C = c_new
        self.M = m_new

        # ==================== Phase 4: Tank Storage Update ====================
        # The downstream draw runs at its (possibly disturbed) rate.
        prev_s_mass = self.S_mass
        self.S_mass += (total_reactor_outflow * self.time_step_duration -
                        self.S_out_rate * self._downstream_mult * self.time_step_duration)
        self.S_mass = np.maximum(0.0, self.S_mass)
        # Track outlet discharge actually removed from the in-system inventory.
        self.episode_mass_discharged += max(0.0, prev_s_mass +
                                            total_reactor_outflow * self.time_step_duration - self.S_mass)

        # ==================== Phase 5: Reward Construction ====================
        reward_coef = 0.5
        prod_reward = reward_coef * (total_harvested_eth/1e7)
        # Each lot carries the production reward it earned, withdrawn if rejected.
        self.lot_prod_reward += reward_coef * (harvested_eth / 1e7)

        # Violation events of this step: id -> count, depth sum and depth max.
        counts, depth_sum, depth_max = {}, {}, {}

        def record(cid, depths):
            """Count one event per entry of ``depths`` (relative depth past the limit)."""
            depths = np.atleast_1d(np.asarray(depths, dtype=np.float64))
            if depths.size == 0:
                return
            counts[cid] = counts.get(cid, 0) + int(depths.size)
            if cid in DEPTH_CONSTRAINTS:
                depth_sum[cid] = depth_sum.get(cid, 0.0) + float(depths.sum())
                depth_max[cid] = max(depth_max.get(cid, 0.0), float(depths.max()))

        # Base spike coefficient: a fixed, λ-independent penalty that fires the
        # moment any hard constraint is violated, scaled with its severity
        # (spike_scale: 1x at the limit to 5x deep past it).  Only in the
        # no-Lagrangian case study (fixed_violation_penalties); with the
        # Lagrangian a violation costs its multiplier alone, so the first one
        # is free and the multiplier grows from there.
        BASE_SPIKE = 0.1 if self.fixed_violation_penalties else 0.0

        # Default state: no violation this step
        constraint_violated = False
        violation_id = 0
        violated_constraints = set()
        constraint_violated_local = np.zeros(self.n_reactors, dtype=bool)

        # Lagrangian State Update Blocks
        lagrange_penalty_local = np.zeros(self.n_reactors)
        storage_lagrange_penalty_global = 0.0

        # ── Buffer-zone soft walls (vectorised) ───────────────────────────
        # Quadratic penalty = BUFFER_COEF × depth² where depth ∈ [0, 1]
        # measures normalized penetration into the buffer zone.
        overflow_edge = 1.0 - self.OVERFLOW_BUFFER_FRAC
        m_frac_arr = self.M / self.Mmax
        overflow_depth = np.clip(
            (m_frac_arr - overflow_edge) / self.OVERFLOW_BUFFER_FRAC, 0.0, 1.0)
        lagrange_penalty_local += self.BUFFER_COEF * overflow_depth * overflow_depth
        zone_width_uf = max(
            float(self.UNDERFLOW_BUFFER_KG - self.DRY_FLOOR), 1.0)
        underflow_depth = np.clip(
            (self.UNDERFLOW_BUFFER_KG - self.M) / zone_width_uf, 0.0, 1.0)
        lagrange_penalty_local += self.BUFFER_COEF * underflow_depth * underflow_depth

        # Storage buffer zones (applied to the global penalty shared across reactors).
        # A storage_capacity fault moves the overflow wall (not the reading).
        s_cap = self.S_max * self._storage_cap_mult
        s_frac = self.S_mass / s_cap
        overflow_edge_s = 1.0 - self.STORAGE_OVERFLOW_BUFFER_FRAC
        if s_frac > overflow_edge_s:
            depth = min(1.0, (s_frac - overflow_edge_s) /
                        self.STORAGE_OVERFLOW_BUFFER_FRAC)
            storage_lagrange_penalty_global += self.BUFFER_COEF * depth * depth
        underflow_edge_s = self.STORAGE_UNDERFLOW_BUFFER_MULT * self.S_min
        if self.S_mass < underflow_edge_s:
            zone_width_s = underflow_edge_s - self.S_min
            depth = min(1.0, (underflow_edge_s - self.S_mass) /
                        max(zone_width_s, 1.0))
            storage_lagrange_penalty_global += self.BUFFER_COEF * depth * depth

        # ── Hard constraint detection + Lagrangian penalties (vectorised) ─
        r_overflow = np.maximum(0.0, self.M / self.Mmax - 1.0)
        overflow_viol_mask = r_overflow > 0
        if np.any(overflow_viol_mask):
            constraint_violated = True
            constraint_violated_local |= overflow_viol_mask
            violation_id = 1
            violated_constraints.add(1)
            record(1, r_overflow[overflow_viol_mask])
        lagrange_r_overflow = np.maximum(
            0.0, r_overflow - self.lagrange_target_slack)
        lagrange_penalty_local += self.reactor_overflow_lambda * saturate(lagrange_r_overflow)
        lagrange_penalty_local += np.where(
            overflow_viol_mask, BASE_SPIKE * spike_scale(r_overflow), 0.0)
        if self.lagrange_updates_enabled:
            self.reactor_overflow_lambda = np.clip(
                np.where(lagrange_r_overflow > 0,
                         self.reactor_overflow_lambda + self.lagrange_lr * lagrange_r_overflow,
                         self.reactor_overflow_lambda * (1.0 - self.lagrange_decay)),
                0.0, self.lagrange_max)

        r_underflow = np.maximum(
            0.0, (self.DRY_FLOOR - self.M) / self.DRY_FLOOR)
        underflow_viol_mask = r_underflow > 0
        if np.any(underflow_viol_mask):
            constraint_violated = True
            constraint_violated_local |= underflow_viol_mask
            violation_id = 2
            violated_constraints.add(2)
            record(2, r_underflow[underflow_viol_mask])
        lagrange_r_underflow = np.maximum(
            0.0, r_underflow - self.lagrange_target_slack)
        lagrange_penalty_local += self.reactor_underflow_lambda * saturate(lagrange_r_underflow)
        lagrange_penalty_local += np.where(
            underflow_viol_mask, BASE_SPIKE * spike_scale(r_underflow), 0.0)
        if self.lagrange_updates_enabled:
            self.reactor_underflow_lambda = np.clip(
                np.where(lagrange_r_underflow > 0,
                         self.reactor_underflow_lambda + self.lagrange_lr * lagrange_r_underflow,
                         self.reactor_underflow_lambda * (1.0 - self.lagrange_decay)),
                0.0, self.lagrange_max)

        # Central Hub Boundaries
        # Storage overflow: normalized by the usable volume → proportional
        # fractional excess (S_max unless a storage_capacity fault acts).
        storage_overflow_violation = max(
            0.0, (self.S_mass - s_cap) / s_cap)
        if storage_overflow_violation > 0:
            constraint_violated = True
            violation_id = 3
            violated_constraints.add(3)
            record(3, storage_overflow_violation)

        lagrange_s_overflow = max(
            0.0, storage_overflow_violation - self.lagrange_target_slack)
        storage_lagrange_penalty_global += self.storage_overflow_lambda * saturate(lagrange_s_overflow)
        if storage_overflow_violation > 0:
            storage_lagrange_penalty_global += BASE_SPIKE * spike_scale(storage_overflow_violation)

        if self.lagrange_updates_enabled:
            if lagrange_s_overflow > 0.0:
                new_u = self.storage_overflow_lambda + self.lagrange_lr * lagrange_s_overflow
            else:
                new_u = self.storage_overflow_lambda * (
                    1.0 - self.lagrange_decay)
            self.storage_overflow_lambda = min(
                self.lagrange_max, max(0.0, new_u))

        # Storage underflow: normalized by S_min (5,000 kg) so the gradient signal
        # spans [0, 1] instead of being suppressed to max 0.0125 by S_max (400,000).
        # At S=2500, violation = (5000-2500)/5000 = 0.5 — a strong, actionable signal.
        storage_underflow_violation = max(
            0.0, (self.S_min - self.S_mass) / self.S_min)
        if storage_underflow_violation > 0:
            constraint_violated = True
            violation_id = 4
            violated_constraints.add(4)
            record(4, storage_underflow_violation)

        lagrange_s_underflow = max(
            0.0, storage_underflow_violation - self.lagrange_target_slack)
        storage_lagrange_penalty_global += self.storage_underflow_lambda * saturate(lagrange_s_underflow)
        if storage_underflow_violation > 0:
            storage_lagrange_penalty_global += BASE_SPIKE * spike_scale(storage_underflow_violation)

        if self.lagrange_updates_enabled:
            if lagrange_s_underflow > 0.0:
                new_u = self.storage_underflow_lambda + self.lagrange_lr * lagrange_s_underflow
            else:
                new_u = self.storage_underflow_lambda * (
                    1.0 - self.lagrange_decay)
            self.storage_underflow_lambda = min(
                self.lagrange_max, max(0.0, new_u))

        # ── Constraint 9: plant heat-removal capacity ─────────────────────
        # The cooling loop caps the plant's total CO2 evolution rate; a
        # cooling_capacity fault moves the cap.  Each reactor pays its share
        # of the load's spike, soft wall and dual.
        heat_q = self.heat_load()
        heat_total = float(np.sum(heat_q))
        heat_cap = self.max_heat_load * self._cooling_cap_mult
        heat_violation = max(0.0, (heat_total - heat_cap) / heat_cap)
        heat_penalty = 0.0
        if heat_violation > 0:
            constraint_violated = True
            violated_constraints.add(9)
            record(9, heat_violation)
            heat_penalty += BASE_SPIKE * spike_scale(heat_violation)
        heat_edge = 1.0 - self.HEAT_BUFFER_FRAC
        if heat_total / heat_cap > heat_edge:
            depth = min(1.0, (heat_total / heat_cap - heat_edge) / self.HEAT_BUFFER_FRAC)
            heat_penalty += self.BUFFER_COEF * depth * depth
        lagrange_heat = max(0.0, heat_violation - self.lagrange_target_slack)
        heat_penalty += self.heat_load_lambda * saturate(lagrange_heat)
        self.heat_load_lambda = float(self._update_lambda(self.heat_load_lambda, lagrange_heat))
        heat_share = (heat_q / heat_total if heat_total > 0.0
                      else np.full(self.n_reactors, 1.0 / self.n_reactors))
        lagrange_penalty_local += heat_penalty * heat_share

        # ── Constraint 7: harvest titer floor (per draining reactor) ──────
        spec = self.min_harvest_titer
        titer_violation = np.where(
            draining, np.maximum(0.0, (spec - titer_now) / spec), 0.0)
        titer_viol_mask = titer_violation > 0
        if np.any(titer_viol_mask):
            constraint_violated = True
            constraint_violated_local |= titer_viol_mask
            violated_constraints.add(7)
            record(7, titer_violation[titer_viol_mask])
        lagrange_penalty_local += np.where(
            titer_viol_mask, BASE_SPIKE * spike_scale(titer_violation), 0.0)
        titer_edge = spec * (1.0 + self.TITER_BUFFER_FRAC)
        near = draining & ~titer_viol_mask & (titer_now < titer_edge)
        titer_depth = np.where(
            near, np.minimum(1.0, (titer_edge - titer_now) / (titer_edge - spec)), 0.0)
        lagrange_penalty_local += self.BUFFER_COEF * titer_depth * titer_depth
        lagrange_titer = np.maximum(0.0, titer_violation - self.lagrange_target_slack)
        lagrange_penalty_local += self.harvest_titer_lambda * saturate(lagrange_titer)
        # Draining reactors update their dual; the others decay.
        self.harvest_titer_lambda = self._update_lambda(
            self.harvest_titer_lambda, np.where(draining, lagrange_titer, 0.0))

        # ── Constraint 8: product purity, lot by lot ──────────────────────
        # A rejected lot's production reward is withdrawn from the shared pool,
        # as the reward was paid into it; its penalty is the reactor's own.
        clawback = 0.0
        purity_shortfall_closed = 0.0
        for i, lot in enumerate(closed_lots):
            if lot is None:
                continue
            if lot["passed"]:
                self._decay_purity_lambda(i)
            else:
                constraint_violated = True
                constraint_violated_local[i] = True
                violated_constraints.add(8)
                record(8, lot["shortfall"])
                lagrange_penalty_local[i] += self._lot_purity_penalty(i, lot["shortfall"])
                clawback += lot["clawback"]
                purity_shortfall_closed = max(purity_shortfall_closed, lot["shortfall"])
        # Dense shaping on each lot's running purity.
        lot_titer = self.lot_mean_titer
        spec_p = self.min_product_titer
        purity_edge = spec_p * (1.0 + self.PURITY_BUFFER_FRAC)
        shaped = (self.lot_mass > 0.0) & (lot_titer < purity_edge)
        purity_depth = np.where(
            shaped, np.minimum(1.0, (purity_edge - lot_titer) / (purity_edge - spec_p)), 0.0)
        lagrange_penalty_local += self.PURITY_SHAPING_COEF * purity_depth * purity_depth

        # ── Near misses: inside a buffer zone, not past the limit ────────
        near_miss = {
            1: int(np.sum((m_frac_arr > overflow_edge) & ~overflow_viol_mask)),
            2: int(np.sum((self.M < self.UNDERFLOW_BUFFER_KG) & ~underflow_viol_mask)),
            3: int(overflow_edge_s < s_frac <= 1.0),
            4: int(self.S_min <= self.S_mass < underflow_edge_s),
            7: int(np.sum(near)),
            8: int(np.sum(shaped & (lot_titer >= spec_p))),
            9: int(heat_edge < heat_total / heat_cap <= 1.0),
        }
        near_miss = {cid: k for cid, k in near_miss.items() if k}

        global_constraint_penalty = storage_lagrange_penalty_global / self.n_reactors
        constraint_penalty_arr = lagrange_penalty_local + global_constraint_penalty

        total_lagrange_penalty = np.sum(
            lagrange_penalty_local) + storage_lagrange_penalty_global

        # Feedstock low-supply barrier (independent of Lagrangian, pure soft penalty)
        lf_ratio = self.total_lf / self.initial_lf
        c5_ratio = self.total_c5 / self.initial_c5
        feedstock_barrier = 0.0
        _feedstock_barrier_coef = 0.005
        if 0.0 < lf_ratio < 0.02:
            feedstock_barrier += _feedstock_barrier_coef * \
                (-np.log(lf_ratio + 0.001))
        if 0.0 < c5_ratio < 0.02:
            feedstock_barrier += _feedstock_barrier_coef * \
                (-np.log(c5_ratio + 0.001))
        if feedstock_barrier > 0.0:
            constraint_penalty_arr += feedstock_barrier / self.n_reactors

        # ── Storage underflow spike + log-barrier (λ-independent) ─────────
        # Two complementary mechanisms prevent the central storage tank from
        # running dry, providing gradient signal well before the hard S_min
        # constraint fires and escalating sharply once it does:
        #
        #   Log-barrier: activates when S < 16·S_min (≈20% of S_max, ~80,000 kg).
        #     Raising the activation threshold from 2×S_min to 16×S_min gives the
        #     policy ~50 extra steps of warning before hitting the hard limit, since
        #     storage drains at 1,500 kg/hr and reactors can only replenish during
        #     stage 3.  The -log(ratio) term steepens as S→0 so the signal escalates
        #     naturally without a separate decay schedule.
        #
        #   Hard spike: fires when S < S_min.  A flat base plus a term
        #     proportional to the fractional shortfall ensures the penalty
        #     dominates the reward landscape once the hard limit is breached,
        #     making constraint violation unambiguously costly.  A fixed
        #     violation penalty: no-Lagrangian case study only.
        _storage_barrier_coef = 0.001
        _storage_spike_base = 0.005
        s_ratio = self.S_mass / self.S_max
        s_barrier_threshold = (16.0 * self.S_min) / \
            self.S_max  # normalized (~0.20)
        storage_underflow_extra = 0.0
        if 0.0 < s_ratio < s_barrier_threshold:
            # Log-barrier: -log(s_ratio) rises steeply as storage empties.
            # The +0.001 prevents log(0); at s_ratio ≈ s_barrier_threshold
            # the penalty is near zero so it blends smoothly with normal ops.
            storage_underflow_extra += _storage_barrier_coef * \
                (-np.log(s_ratio / s_barrier_threshold + 0.001))
        if self.S_mass < self.S_min and self.fixed_violation_penalties:
            # Hard spike: flat base + proportional depth below S_min.
            shortfall = (self.S_min - self.S_mass) / self.S_min
            storage_underflow_extra += _storage_spike_base * \
                (1.0 + shortfall)
        if storage_underflow_extra > 0.0:
            constraint_penalty_arr += storage_underflow_extra / self.n_reactors

        # trickle penalty during Stage 3 (Unload)
        eff_penalty_arr = np.where(self.current_stage == 3, 0.0005, 0.0)

        if is_first:
            current_flow_action = a_clipped.reshape(self.n_reactors, 4)[:, 1:4]
            prev_flow = self.prev_flow_action.reshape(self.n_reactors, 3)
            # localise smoothing
            smoothing_penalty_arr = self.smoothing_coef * \
                np.mean(np.square(current_flow_action - prev_flow), axis=1)
            self.prev_flow_action = current_flow_action.flatten().copy()
        else:
            smoothing_penalty_arr = np.zeros(self.n_reactors)

        # ==================== Raw Material Consumption Penalty ====================
        # Penalises wasteful feedstock use: deducted each step proportional to the
        # fraction of the total initial supply consumed this tick.  This discourages
        # the agent from flooding reactors with feed early and depleting the global
        # LF/C5 pool before downstream reactors reach their feed stages.
        #
        # The penalty is symmetric for LF and C5 so both resources are conserved
        # equally, and it is normalised per-reactor so the signal magnitude is
        # independent of n_reactors.
        lf_consumed_this_step = max(
            0.0, lf_ratio * self.initial_lf - self.total_lf)
        c5_consumed_this_step = max(
            0.0, c5_ratio * self.initial_c5 - self.total_c5)
        _raw_mat_penalty_coef = 10
        raw_mat_penalty_global = _raw_mat_penalty_coef * (
            lf_consumed_this_step / max(self.initial_lf, 1.0) +
            c5_consumed_this_step / max(self.initial_c5, 1.0)
        )
        raw_mat_penalty_arr = np.full(
            self.n_reactors, raw_mat_penalty_global / self.n_reactors)

        # Equal share of global product reward (less any rejected lot's) plus
        # shaping signals (no dense ethanol bonus)
        reward_arr = ((prod_reward - clawback) / self.n_reactors) - (
            smoothing_penalty_arr + eff_penalty_arr + constraint_penalty_arr + raw_mat_penalty_arr)

        self.time_step_count += 1

        # Episode completion logic
        done = (self.time_step_count >= self.max_steps or instability_detected)

        # Numerical instability in the ODE integrator terminates the episode (violation 5: instability)
        if instability_detected:
            violated_constraints.add(5)
            record(5, 1.0)
        if violated_constraints:
            violation_id = max(violated_constraints)

        hard_violation = instability_detected or constraint_violated

        info = {
            "prod": prod_reward,
            "constraint": np.sum(constraint_penalty_arr),
            "smoothing": np.sum(smoothing_penalty_arr),
            "eff": np.sum(eff_penalty_arr),
            "eth_harvested": total_harvested_eth,
            "raw_mat_penalty": float(np.sum(raw_mat_penalty_arr)),
            "lf_remaining": self.total_lf,
            "c5_remaining": self.total_c5,
            "storage_overflow_violation": float(storage_overflow_violation),
            "storage_underflow_violation": float(storage_underflow_violation),
            "reactor_overflow_violation": float(np.max(r_overflow)),
            "reactor_underflow_violation": float(np.max(r_underflow)),
            "harvest_titer_violation": float(np.max(titer_violation)),
            "product_purity_violation": float(purity_shortfall_closed),
            "heat_load": heat_total,
            "heat_load_violation": float(heat_violation),
            "clawback": float(clawback),
            "leaked_mass": leaked_mass,
            "total_lagrange_penalty": float(total_lagrange_penalty),
            "hard_violation": hard_violation,
            "constraint_violated": constraint_violated,
            "violation_id": violation_id,
            "violated_constraints": list(violated_constraints),
            "violation_counts": counts,
            "violation_depth_sum": depth_sum,
            "violation_depth_max": depth_max,
            "near_miss_counts": near_miss,
            "actual_a_reshaped": a_reshaped.copy(),
        }
        return self.get_state_norm(), reward_arr, done, info

    def step(self, action):
        """Execute one outer environment step with action repeat.

        Outer step function that repeats the agent's action for
        control_action_interval inner steps.

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
        near_miss_counts = {}
        near_miss_steps = 0

        lagrange_penalty_accum = 0.0
        raw_mat_penalty_accum = 0.0

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
            state, reward, done, info = self._step_internal(
                action, is_first=(idx == 0))
            steps_completed += 1

            total_prod += info.get("prod", 0.0)
            total_constraint += info.get("constraint", 0.0)
            total_smoothing += info.get("smoothing", 0.0)
            total_eff += info.get("eff", 0.0)
            total_reward += reward

            total_harvested_eth_accum += info.get("eth_harvested", 0.0)

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
            for cid, k in info["near_miss_counts"].items():
                near_miss_counts[cid] = near_miss_counts.get(cid, 0) + k
            near_miss_steps += int(bool(info["near_miss_counts"]))
            merge_events(events, info["violation_counts"], info["violation_depth_sum"],
                          info["violation_depth_max"])

            # Accumulate penalties — will be normalised below
            lagrange_penalty_accum += info.get("total_lagrange_penalty", 0.0)
            raw_mat_penalty_accum += info.get("raw_mat_penalty", 0.0)

            hard_violation_accum = hard_violation_accum or info.get(
                "hard_violation", False)
            constraint_violated_accum = constraint_violated_accum or info.get(
                "constraint_violated", False)
            if info.get("violation_id", 0) > 0:
                violation_id_accum = max(
                    violation_id_accum, info["violation_id"])
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

        norm = max(steps_completed, 1)
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
            # Divide by steps_completed so per-decision magnitude stays consistent
            # with the original 1-hour control regime regardless of interval length.
            "storage_lagrange_penalty": float(lagrange_penalty_accum / norm),
        }

        if done:
            terminal_ids = set()

            # ── Constraint 8: every open lot is released or rejected ──────
            purity_penalty = np.zeros(self.n_reactors)
            clawback = 0.0
            for i in range(self.n_reactors):
                lot = self._close_lot(i)
                if lot is None:
                    continue
                if lot["passed"]:
                    self._decay_purity_lambda(i)
                    continue
                purity_penalty[i] += self._lot_purity_penalty(i, lot["shortfall"])
                clawback += lot["clawback"]
                merge_events(events, {8: 1}, {8: lot["shortfall"]}, {8: lot["shortfall"]})
                max_purity_violation = max(max_purity_violation, lot["shortfall"])
                terminal_ids.add(8)
            if self.product_mass <= 0.0:
                # Shipping nothing at all fails the spec, so doing nothing is not
                # "safe": one event for the plant, priced as a full miss.
                # A full miss: the duals price it at shortfall 1 too.
                purity_penalty += self._lot_purity_penalty(None, 1.0) / self.n_reactors
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
                self.mass_balance_rel_tol *
                max(self.episode_initial_total_mass, 1.0)
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
                info_out["violation_id"] = max(
                    int(info_out["violation_id"]), max(terminal_ids))
                info_out["violation_ids"] = list(set(info_out["violation_ids"]) | terminal_ids)
                info_out["violation_count"] = int(info_out["violation_count"]) + 1

        counts, depth_sum, depth_max = events
        info_out["violation_counts"] = counts
        info_out["near_miss_counts"] = near_miss_counts
        info_out["near_miss_steps"] = near_miss_steps
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
