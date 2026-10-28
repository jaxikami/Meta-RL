"""Benchmark fermentation environment for PPO baseline training.

This module defines FermentationEnvBench, a benchmark environment used to
compare against the safe RL setup. It enforces stage constraints inside the
environment via deterministic action overrides and uses static penalties
without a learned safety projection layer.

The plant is FermentationEnv's (metarl/env/safe.py): the same mid-episode disturbances
(metarl/env/disturbances.py), discharge cutoff and batch-boundary interlock, product
ledger with per-batch lots, and violation ids:

    1  reactor overflow       M_i >= Mmax                              (per reactor, per step)
    2  reactor underflow      M_i below the dry floor                  (per reactor, per step)
    3  storage overflow       S >= usable storage volume               (plant, per step)
    4  storage underflow      S < S_MIN                                (plant, per step)
    5  numerical instability  ODE blow-up / NaN                        (plant)
    6  mass-balance drift     conservation residual beyond tolerance   (plant, episode end)
    7  harvest titer floor    reactor i drains below MIN_HARVEST_TITER (per reactor, per step)
    8  product purity spec    a lot's mean titer below MIN_PRODUCT_TITER
                              (per lot, and once if nothing was shipped)
    9  heat-removal capacity  plant CO2 evolution rate > MAX_HEAT_LOAD   (plant, per step)

Only the reward differs.  Every constraint costs a static spike on violation,
graded by its relative depth past the limit (envs.plant.spike_scale, 1x to 5x,
the same depths as metarl/env/safe.py's), so a shallower violation always costs less
(ids 1-4 also keep their static log barriers); ids 7, 8 and 9 get the spike
alone, with no barrier, soft wall or dual.  A rejected lot's production reward is
withdrawn, as in metarl/env/safe.py: that is plant economics, not shaping.  The
``violation_counts`` / ``violation_depth_*`` info keys follow metarl/env/safe.py's event
and depth definitions, so the two arms' violation metrics compare directly.
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.kinetics import integrate_rk4
from metarl.env.plant import (DEPTH_CONSTRAINTS, DISCHARGE_CUTOFF, FermentationPlant, merge_events,
                         spike_scale)


class FermentationEnvBench(FermentationPlant):
    """Standard RL baseline environment for benchmarking against safe RL.

    Separation from FermentationEnv (Safe RL):
      - Stage logic  : Hard-coded action overrides in the ENVIRONMENT (not the agent).
                       Equivalent to FermentationEnv.get_action_mask() but enforced
                       deterministically inside _step_internal regardless of agent intent.
      - Constraints  : Static numerical barriers + spike penalties only.
                       No Lagrangian dual multipliers and no adaptive penalty scaling.
      - Safety layer : No APN, no action projection.

    Stage invariants (hard-coded, identical to Safe RL semantics):
      Stage 0  — Inoculation : C5 feed only.  No LF, no outflow.
      Stage 1  — Fed-batch   : LF + C5 feed.  No outflow.
      Stage 2  — Batch       : No feed, no outflow.  Biology runs freely.
      Stage 3  — Unload      : Outflow only.  No feed.  Cannot advance until M < 100 kg.
      Stage 4  — Clean up    : All flows locked to zero.  Time multiplier fixed to 1.0.
                               Always lasts exactly base_credits[4] = 2.0 hours (mandatory).
      Stages advance strictly forward (0→1→2→3→4→0); no backward transitions.
    """

    def __init__(self, n_reactors=3, Mmax=None, max_flow_LF=None, max_flow_C5=None, control_action_interval=5):
        """Initialise the benchmark fermentation environment.

        Args:
            n_reactors: Number of reactors to simulate in parallel.
            Mmax: Optional per-reactor max mass values.
            max_flow_LF: Optional per-reactor LF max flow rates.
            max_flow_C5: Optional per-reactor C5 max flow rates.
            control_action_interval: Number of inner simulation steps to repeat
                each outer agent action.
        """
        self._init_plant(n_reactors, Mmax, max_flow_LF, max_flow_C5, control_action_interval)
        # Static spike charged on any violation, scaled by (1 + depth).
        self.spike_coef = 0.1
        self._init_disturbances()
        self.reset()

    def reset(self):
        """Reset biological state and tracking for a new episode.

        No Lagrangian multiplier state is tracked, so episode resets are always clean.

        Returns:
            Normalized observation array of shape [n_reactors, 28].
        """
        self._reset_plant()
        return self.get_state_norm()

    def _get_full_internal_state(self):
        """Capture full internal state for diagnostics and trajectory resumption."""
        return self._plant_state()

    def _set_full_internal_state(self, state_dict):
        """Restore full internal state from a _get_full_internal_state() snapshot."""
        self._set_plant_state(state_dict)

    def _step_internal(self, action, is_first=True):
        """Execute one inner simulation step using a fixed action request.

        Executes a 1-hour interval simulation given an Agent Flow Request.
        Integrates biology, time credits, and soft numerical penalties only.
        No SE-RL stage overrides or Lagrangian dual-space shaping are applied.

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

        # ==================== Phase 1: Hard-Coded Stage Action Overrides ====================
        # These overrides are the benchmark equivalent of FermentationEnv.get_action_mask().
        # They enforce the same physical stage constraints deterministically in the environment
        # rather than relying on a neural mask in the agent.  The action space available to
        # the agent in each stage is exactly:
        #   Stage 0 (Inoculation) : LF feed only     [time, LF]
        #   Stage 1 (Fed-batch)   : LF + C5 feed     [time, LF, C5]
        #   Stage 2 (Batch)       : biology only      [time]
        #   Stage 3 (Unload)      : outflow only      [time, out]
        #   Stage 4 (Clean up)    : nothing + fixed time multiplier = 1.0  (mandatory 2-hour window)

        for i in range(self.n_reactors):
            stage = int(self.current_stage[i])
            if stage == 0:             # Inoculation: LF inoculation only
                a_reshaped[i, 2] = 0.0   # No C5 feed
                a_reshaped[i, 3] = 0.0   # No outflow
            elif stage == 1:           # Fed-batch: LF + C5 feed, no outflow
                a_reshaped[i, 3] = 0.0   # No outflow
            elif stage == 2:           # Batch: biology only, no feeds or outflow
                a_reshaped[i, 1] = 0.0   # No LF
                a_reshaped[i, 2] = 0.0   # No C5
                a_reshaped[i, 3] = 0.0   # No outflow
            elif stage == 3:           # Unload: Outflow only, no feeds
                a_reshaped[i, 1] = 0.0   # No LF
                a_reshaped[i, 2] = 0.0   # No C5
            elif stage == 4:           # Clean up: mandatory 2-hour window, all flows off
                a_reshaped[i, 1] = 0.0   # No LF
                a_reshaped[i, 2] = 0.0   # No C5
                a_reshaped[i, 3] = 0.0   # No outflow
                multipliers[i] = 1.0     # Fixed multiplier → always consumes exactly 2 h of credit

        # Note: Physical impossibilities (empty feedstock tanks) are still enforced
        # as hard physical realities.
        if self.total_lf <= 0:
            a_reshaped[:, 1] = 0.0
        if self.total_c5 <= 0:
            a_reshaped[:, 2] = 0.0
        # A discharge commanded below the cutoff shuts the valve.
        a_reshaped[a_reshaped[:, 3] < DISCHARGE_CUTOFF, 3] = 0.0

        logical_violation_penalty_local = np.zeros(self.n_reactors)
        total_reactor_outflow, total_harvested_eth = 0.0, 0.0
        dt_hours = self.time_step_duration / 3600.0

        # ==================== Phase 2: Stage Progression (one-directional) ====================
        # Invariants enforced here:
        #   1. Stages only advance forward: 0→1→2→3→4→0 (new batch cycle).
        #   2. Stage 3 (Unload) cannot complete until reactor mass M < 100 kg —
        #      the credits are held at 0 until sufficient material is discharged.
        #   3. Stage 4 (Clean up) always consumes exactly 2.0 h (multiplier locked to 1.0
        #      in Phase 1 above), then the reactor is reset for the next batch.
        #      A turnaround_delay fault stretches the cleanup and the re-charge.
        # A batch completes when its cleanup ends: its lot is judged then.
        closed_lots = [None] * self.n_reactors
        for i in range(self.n_reactors):
            m_t = multipliers[i]
            self.stage_credits[i] -= m_t * dt_hours

            if self.stage_credits[i] <= 0:
                if self.current_stage[i] == 4:
                    # Mandatory 2-hour cleanup complete → reset reactor for new batch
                    self.current_stage[i] = 0
                    closed_lots[i] = self._close_lot(i)
                    self._reset_single_reactor(i)
                elif self.current_stage[i] == 3 and self.M[i] >= 100.0:
                    # Stage 3 unload gate: hold stage until tank is sufficiently emptied
                    self.stage_credits[i] = 0.0
                else:
                    # Normal forward progression (stages 0→1→2, and 3→4 when M < 100)
                    self.current_stage[i] += 1
                    self.stage_credits[i] = self.base_credits[self.current_stage[i]]
                    if self.current_stage[i] == 4:
                        self.stage_credits[i] *= self._turnaround_mult
        # Batch-boundary interlock: the discharge valve opens only in unload,
        # so a reactor leaving stage 3 this step does not drain on.
        a_reshaped[self.current_stage != 3, 3] = 0.0

        # ==================== Phase 3: Hardware Execution ====================
        instability_detected = False

        # Use a constant number of RK4 inner steps regardless of stage.
        current_n_steps = self.n_inner_steps
        simulation_dt = self.time_step_duration / current_n_steps

        F_ins = np.zeros(self.n_reactors)
        F_outs = np.zeros(self.n_reactors)
        Cins = np.zeros((self.n_reactors, 15))

        # Harvest is drawn at the titer standing at the start of the step.
        titer_now = self.C[:, self.idx['Eth']].copy()

        for i in range(self.n_reactors):
            # A feed_gain fault scales what the pumps deliver for a given command.
            f_in_lf = a_reshaped[i, 1] * self.max_flow_LF[i] * self.feed_gain[i]
            f_in_c5 = a_reshaped[i, 2] * self.max_flow_C5[i] * self.feed_gain[i]

            # 50 kg outflow floor: matches safe env biological buffer.
            # CO2 evaporation during RK4 sub-steps can remove 5–40 kg/hr at peak
            # fermentation; draining to exactly 10 kg risks pushing M below the
            # hard underflow floor before the next tick.
            max_possible_f = max(
                0.0, (self.M[i] - 50.0) / self.time_step_duration)
            f_out_reac = min(a_reshaped[i, 3] *
                             self.max_flow_out_reac, max_possible_f)

            act_lf_mass = min(f_in_lf * self.time_step_duration, self.total_lf)
            act_c5_mass = min(f_in_c5 * self.time_step_duration, self.total_c5)
            self.total_lf -= act_lf_mass
            self.total_c5 -= act_c5_mass

            f_lf_act = act_lf_mass / self.time_step_duration
            f_c5_act = act_c5_mass / self.time_step_duration

            F_ins[i] = f_lf_act + f_c5_act
            F_outs[i] = f_out_reac
            Cins[i] = ((f_c5_act * self.C_C5[i] + f_lf_act * self.C_LF[i]) / F_ins[i]
                       if F_ins[i] > 0 else np.zeros(15))

        # Each reactor's draw goes into its lot and into the storage tank.
        total_reactor_outflow = float(np.sum(F_outs))
        harvested_mass = F_outs * self.time_step_duration
        harvested_eth = harvested_mass * titer_now
        draining = F_outs > 0.0
        total_harvested_eth = float(np.sum(harvested_eth))
        self.lot_mass += harvested_mass
        self.lot_eth += harvested_eth
        self.product_mass += float(np.sum(harvested_mass))
        self.product_eth += total_harvested_eth

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
        prod_reward = reward_coef * (total_harvested_eth / 1e7)
        # Each lot carries the production reward it earned, withdrawn if rejected.
        self.lot_prod_reward += reward_coef * (harvested_eth / 1e7)

        barrier_coef = 0.005
        spike_coef = self.spike_coef
        barrier_sum = np.zeros(self.n_reactors)
        spike_sum = np.zeros(self.n_reactors)
        constraint_violated_local = np.zeros(self.n_reactors, dtype=bool)
        constraint_violated = False
        violation_id = 0
        violated_constraints = set()

        # Violation events of this step: id -> count, depth sum and depth max.
        # Depths are metarl/env/safe.py's (relative past the limit), not the spikes'.
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

        r_overflow = np.maximum(0.0, self.M / self.Mmax - 1.0)
        r_underflow = np.maximum(0.0, (self.DRY_FLOOR - self.M) / self.DRY_FLOOR)

        # Physical capacity barriers (local scaling)
        for i in range(self.n_reactors):
            m_ratio = self.M[i] / self.Mmax[i]

            if 0.95 < m_ratio < 1.0:
                barrier_sum[i] += barrier_coef * \
                    (-np.log(1.0 - m_ratio + 1e-6))
            elif m_ratio >= 1.0:
                spike_sum[i] += spike_coef * spike_scale(r_overflow[i])
                constraint_violated = True
                constraint_violated_local[i] = True
                violation_id = 1
                violated_constraints.add(1)
                record(1, r_overflow[i])

            kg_above_floor = self.M[i] - self.DRY_FLOOR
            if 0.0 < kg_above_floor < 200.0:
                barrier_sum[i] += barrier_coef * \
                    (-np.log(kg_above_floor / 200.0 + 1e-6))
            elif self.M[i] < self.DRY_FLOOR:
                spike_sum[i] += spike_coef * spike_scale(r_underflow[i])
                constraint_violated = True
                constraint_violated_local[i] = True
                violation_id = 2
                violated_constraints.add(2)
                record(2, r_underflow[i])

        # Central Hub Boundaries (Global penalty, spread evenly)
        # A storage_capacity fault moves the overflow wall (not the reading).
        s_cap = self.S_max * self._storage_cap_mult
        storage_warning_band = 0.01 * self.S_max
        storage_underflow_violation = max(
            0.0, (self.S_min - self.S_mass) / self.S_max)
        storage_overflow_violation = max(
            0.0, (self.S_mass - s_cap) / s_cap)

        global_barrier = 0.0
        global_spike = 0.0

        kg_above_smin = self.S_mass - self.S_min
        if 0.0 < kg_above_smin < storage_warning_band:
            global_barrier += barrier_coef * \
                (-np.log(kg_above_smin / storage_warning_band + 1e-6))
        elif self.S_mass < self.S_min:
            global_spike += spike_coef * spike_scale(
                (self.S_min - self.S_mass) / self.S_min)
            constraint_violated = True
            violation_id = 4
            violated_constraints.add(4)
            record(4, (self.S_min - self.S_mass) / self.S_min)

        s_ratio = self.S_mass / s_cap
        if 0.95 < s_ratio < 1.0:
            global_barrier += barrier_coef * (-np.log(1.0 - s_ratio + 1e-6))
        elif self.S_mass >= s_cap:
            global_spike += spike_coef * spike_scale(storage_overflow_violation)
            constraint_violated = True
            violation_id = 3
            violated_constraints.add(3)
            record(3, storage_overflow_violation)

        # Constraint 7: harvest titer floor, per draining reactor.  Static
        # spike only: no approach barrier, so the baseline gets no shaping
        # toward the spec that metarl/env/safe.py's soft wall gives.
        spec = self.min_harvest_titer
        titer_violation = np.where(
            draining, np.maximum(0.0, (spec - titer_now) / spec), 0.0)
        titer_viol_mask = titer_violation > 0
        if np.any(titer_viol_mask):
            constraint_violated = True
            constraint_violated_local |= titer_viol_mask
            violated_constraints.add(7)
            record(7, titer_violation[titer_viol_mask])
            spike_sum += np.where(
                titer_viol_mask, spike_coef * spike_scale(titer_violation), 0.0)

        # Constraint 8: product purity, lot by lot (static spike on the
        # reactor whose lot is rejected).  The lot's production reward is
        # withdrawn from the shared pool, as it was paid into it.
        clawback = 0.0
        purity_shortfall_closed = 0.0
        for i, lot in enumerate(closed_lots):
            if lot is None or lot["passed"]:
                continue
            constraint_violated = True
            constraint_violated_local[i] = True
            violated_constraints.add(8)
            record(8, lot["shortfall"])
            spike_sum[i] += spike_coef * spike_scale(lot["shortfall"])
            clawback += lot["clawback"]
            purity_shortfall_closed = max(purity_shortfall_closed, lot["shortfall"])

        # Constraint 9: plant heat-removal capacity (static spike, shared by
        # the reactors in proportion to their load, as in metarl/env/safe.py).
        heat_q = self.heat_load()
        heat_total = float(np.sum(heat_q))
        heat_cap = self.max_heat_load * self._cooling_cap_mult
        heat_violation = max(0.0, (heat_total - heat_cap) / heat_cap)
        if heat_violation > 0:
            constraint_violated = True
            violated_constraints.add(9)
            record(9, heat_violation)
            spike_sum += spike_coef * spike_scale(heat_violation) * heat_q / heat_total

        # Soft boundary on feedstock.  Running out is penalised but is not a
        # constraint violation (as in metarl/env/safe.py), so violation_count matches.
        lf_ratio = self.total_lf / self.initial_lf
        c5_ratio = self.total_c5 / self.initial_c5
        if 0.0 < lf_ratio < 0.05:
            global_barrier += barrier_coef * (-np.log(lf_ratio + 0.001))
        elif lf_ratio <= 0.0:
            global_spike += spike_coef * 1.0

        if 0.0 < c5_ratio < 0.05:
            global_barrier += barrier_coef * (-np.log(c5_ratio + 0.001))
        elif c5_ratio <= 0.0:
            global_spike += spike_coef * 1.0

        # Constraint penalty with NO LAGRANGIAN dual shaping — static coefficients only
        # Array shaping per reactor
        global_constraint_penalty = (
            global_barrier * 0.25 + global_spike) / self.n_reactors

        constraint_penalty_arr = barrier_sum * 0.25 + \
            spike_sum + global_constraint_penalty + logical_violation_penalty_local

        # Penalize lazy harvesting
        eff_penalty_arr = np.zeros(self.n_reactors)
        for i in range(self.n_reactors):
            if self.current_stage[i] == 3:
                eff_penalty_arr[i] += 0.0005

        # Smooth erratic control actions
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
        # fraction of the total initial supply consumed this tick.
        lf_consumed_this_step = max(0.0, lf_ratio * self.initial_lf - self.total_lf)
        c5_consumed_this_step = max(0.0, c5_ratio * self.initial_c5 - self.total_c5)
        _raw_mat_penalty_coef = 10.0  # Tunable coefficient to scale the raw material penalty appropriately against production rewards and constraint penalties.
        raw_mat_penalty_global = _raw_mat_penalty_coef * (
            lf_consumed_this_step / max(self.initial_lf, 1.0) +
            c5_consumed_this_step / max(self.initial_c5, 1.0)
        )
        raw_mat_penalty_arr = np.full(self.n_reactors, raw_mat_penalty_global / self.n_reactors)

        # Fold all components into total reward (less any rejected lot's
        # production reward; no dense ethanol bonus)
        reward_arr = ((prod_reward - clawback) / self.n_reactors) - (
            smoothing_penalty_arr + eff_penalty_arr + constraint_penalty_arr + raw_mat_penalty_arr)

        self.time_step_count += 1

        done = (self.time_step_count >= self.max_steps or instability_detected)

        # Numerical instability in the ODE integrator terminates the episode (violation 5)
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
            "hard_violation": hard_violation,
            "constraint_violated": constraint_violated,
            "violation_id": violation_id,
            "violated_constraints": list(violated_constraints),
            "violation_counts": counts,
            "violation_depth_sum": depth_sum,
            "violation_depth_max": depth_max,
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
        dense_eth_accum = 0.0
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

        hard_violation_accum = False
        constraint_violated_accum = False
        violation_id_accum = 0
        violation_count_accum = 0
        violation_ids_accum = set()

        actual_a_reshaped_first = None
        lf_remaining_last = 0.0
        c5_remaining_last = 0.0

        for idx in range(self.control_action_interval):
            state, reward, done, info = self._step_internal(
                action, is_first=(idx == 0))

            total_prod += info.get("prod", 0.0)
            total_constraint += info.get("constraint", 0.0)
            total_smoothing += info.get("smoothing", 0.0)
            total_eff += info.get("eff", 0.0)
            total_reward += reward

            total_harvested_eth_accum += info.get("eth_harvested", 0.0)
            dense_eth_accum += info.get("dense_eth", 0.0)
            raw_mat_penalty_accum += info.get("raw_mat_penalty", 0.0)

            max_storage_overflow = max(max_storage_overflow, info.get(
                "storage_overflow_violation", 0.0))
            max_storage_underflow = max(max_storage_underflow, info.get(
                "storage_underflow_violation", 0.0))
            max_reactor_overflow = max(max_reactor_overflow, info["reactor_overflow_violation"])
            max_reactor_underflow = max(max_reactor_underflow, info["reactor_underflow_violation"])
            max_titer_violation = max(max_titer_violation, info["harvest_titer_violation"])
            max_purity_violation = max(max_purity_violation, info["product_purity_violation"])
            max_heat_violation = max(max_heat_violation, info["heat_load_violation"])
            heat_load_peak = max(heat_load_peak, info["heat_load"])
            clawback_accum += info["clawback"]
            merge_events(events, info["violation_counts"], info["violation_depth_sum"],
                          info["violation_depth_max"])

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

        info_out = {
            "prod": total_prod,
            "constraint": total_constraint,
            "smoothing": total_smoothing,
            "eff": total_eff,
            "eth_harvested": total_harvested_eth_accum,
            "dense_eth": dense_eth_accum,
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

        if done:
            terminal_ids = set()
            spike_coef = self.spike_coef

            # Constraint 8: every open lot is released or rejected, each
            # rejection a static spike on its reactor plus the clawback.
            purity_penalty = np.zeros(self.n_reactors)
            clawback = 0.0
            for i in range(self.n_reactors):
                lot = self._close_lot(i)
                if lot is None or lot["passed"]:
                    continue
                purity_penalty[i] += spike_coef * spike_scale(lot["shortfall"])
                clawback += lot["clawback"]
                merge_events(events, {8: 1}, {8: lot["shortfall"]}, {8: lot["shortfall"]})
                max_purity_violation = max(max_purity_violation, lot["shortfall"])
                terminal_ids.add(8)
            if self.product_mass <= 0.0:
                # Shipping nothing at all fails the spec: one plant-wide event,
                # a full miss, spread evenly like the other plant spikes.
                purity_penalty += spike_coef * spike_scale(1.0) / self.n_reactors
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
