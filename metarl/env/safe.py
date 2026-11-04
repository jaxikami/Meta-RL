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

The inner step is FermentationPlant._step_internal(); this module adds its
constraint pricing (_price_constraints(), stage by stage below).
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
from types import SimpleNamespace

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.plant import FermentationPlant, saturate, spike_scale

# Constraint ids counted per reactor (the rest are plant-wide; 8 is counted per lot).
REACTOR_CONSTRAINTS = (1, 2, 7, 8)

_LAMBDA_SCALARS = ("storage_overflow_lambda", "storage_underflow_lambda", "heat_load_lambda")
_LAMBDA_ARRAYS = ("reactor_overflow_lambda", "reactor_underflow_lambda",
                  "harvest_titer_lambda", "product_purity_lambda")

# Fixed spike on every violation, scaled with its severity (spike_scale: 1x at
# the limit to 5x deep past it): the no-Lagrangian case study only.
BASE_SPIKE = 0.1
# Feedstock low-supply log barrier (below 2% of the initial supply).
FEEDSTOCK_BARRIER_COEF = 0.005
FEEDSTOCK_BARRIER_FRAC = 0.02
# Storage log barrier (below 16 x S_min) and the fixed underflow spike.
STORAGE_BARRIER_COEF = 0.001
STORAGE_BARRIER_MULT = 16.0
STORAGE_SPIKE_BASE = 0.005


class FermentationEnv(FermentationPlant):
    """Full RL environment wrapping biochemical physics and safe shaping.

    Adds to FermentationPlant the constraint-aware rewards with soft walls and
    adaptive Lagrange multipliers.
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

        The Lagrange multipliers carry over from one episode to the next.

        Returns:
            Normalized observation array of shape [n_reactors, 28].
        """
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

    # --- CONSTRAINT PRICING (one inner step) ---

    def _price_constraints(self, fl, closed_lots, ev):
        """Soft walls, hard constraints with their duals, barriers and near misses.

        The stages run in a fixed order, each adding to the reactors' penalty
        (``pen.local``) and the plant's (``pen.plant``, split evenly).
        """
        spike = BASE_SPIKE if self.fixed_violation_penalties else 0.0
        pen = SimpleNamespace(local=np.zeros(self.n_reactors), plant=0.0)
        ctx = SimpleNamespace()
        info = {}
        self._soft_walls(pen, ctx)
        self._reactor_mass_constraints(pen, ctx, ev, spike, info)
        self._storage_constraints(pen, ctx, ev, spike, info)
        self._heat_constraint(pen, ctx, ev, spike, info)
        self._titer_constraint(pen, ctx, fl, ev, spike, info)
        clawback = self._purity_constraint(pen, ctx, closed_lots, ev, info)
        info["near_miss_counts"] = self._near_misses(ctx)

        constraint_penalty_arr = pen.local + pen.plant / self.n_reactors
        info["total_lagrange_penalty"] = float(np.sum(pen.local) + pen.plant)
        self._supply_barriers(constraint_penalty_arr)
        return constraint_penalty_arr, clawback, info

    def _soft_walls(self, pen, ctx):
        """Quadratic BUFFER_COEF x depth^2 ramps inside the mass and storage buffer zones.

        A storage_capacity fault moves the overflow wall (not the reading).
        """
        ctx.overflow_edge = 1.0 - self.OVERFLOW_BUFFER_FRAC
        ctx.m_frac = self.M / self.Mmax
        overflow_depth = np.clip(
            (ctx.m_frac - ctx.overflow_edge) / self.OVERFLOW_BUFFER_FRAC, 0.0, 1.0)
        pen.local += self.BUFFER_COEF * overflow_depth * overflow_depth
        zone_width_uf = max(float(self.UNDERFLOW_BUFFER_KG - self.DRY_FLOOR), 1.0)
        underflow_depth = np.clip((self.UNDERFLOW_BUFFER_KG - self.M) / zone_width_uf, 0.0, 1.0)
        pen.local += self.BUFFER_COEF * underflow_depth * underflow_depth

        ctx.s_cap = self.S_max * self._storage_cap_mult
        ctx.s_frac = self.S_mass / ctx.s_cap
        ctx.overflow_edge_s = 1.0 - self.STORAGE_OVERFLOW_BUFFER_FRAC
        if ctx.s_frac > ctx.overflow_edge_s:
            depth = min(1.0, (ctx.s_frac - ctx.overflow_edge_s) /
                        self.STORAGE_OVERFLOW_BUFFER_FRAC)
            pen.plant += self.BUFFER_COEF * depth * depth
        ctx.underflow_edge_s = self.STORAGE_UNDERFLOW_BUFFER_MULT * self.S_min
        if self.S_mass < ctx.underflow_edge_s:
            zone_width_s = ctx.underflow_edge_s - self.S_min
            depth = min(1.0, (ctx.underflow_edge_s - self.S_mass) / max(zone_width_s, 1.0))
            pen.plant += self.BUFFER_COEF * depth * depth

    def _reactor_mass_constraints(self, pen, ctx, ev, spike, info):
        """Ids 1 and 2: reactor overflow and underflow, with their duals."""
        r_overflow = np.maximum(0.0, self.M / self.Mmax - 1.0)
        ctx.overflow_viol = r_overflow > 0
        if np.any(ctx.overflow_viol):
            ev.record(1, r_overflow[ctx.overflow_viol])
        lagrange_r_overflow = np.maximum(0.0, r_overflow - self.lagrange_target_slack)
        pen.local += self.reactor_overflow_lambda * saturate(lagrange_r_overflow)
        pen.local += np.where(ctx.overflow_viol, spike * spike_scale(r_overflow), 0.0)
        if self.lagrange_updates_enabled:
            self.reactor_overflow_lambda = np.clip(
                np.where(lagrange_r_overflow > 0,
                         self.reactor_overflow_lambda + self.lagrange_lr * lagrange_r_overflow,
                         self.reactor_overflow_lambda * (1.0 - self.lagrange_decay)),
                0.0, self.lagrange_max)

        r_underflow = np.maximum(0.0, (self.DRY_FLOOR - self.M) / self.DRY_FLOOR)
        ctx.underflow_viol = r_underflow > 0
        if np.any(ctx.underflow_viol):
            ev.record(2, r_underflow[ctx.underflow_viol])
        lagrange_r_underflow = np.maximum(0.0, r_underflow - self.lagrange_target_slack)
        pen.local += self.reactor_underflow_lambda * saturate(lagrange_r_underflow)
        pen.local += np.where(ctx.underflow_viol, spike * spike_scale(r_underflow), 0.0)
        if self.lagrange_updates_enabled:
            self.reactor_underflow_lambda = np.clip(
                np.where(lagrange_r_underflow > 0,
                         self.reactor_underflow_lambda + self.lagrange_lr * lagrange_r_underflow,
                         self.reactor_underflow_lambda * (1.0 - self.lagrange_decay)),
                0.0, self.lagrange_max)
        info["reactor_overflow_violation"] = float(np.max(r_overflow))
        info["reactor_underflow_violation"] = float(np.max(r_underflow))

    def _storage_constraints(self, pen, ctx, ev, spike, info):
        """Ids 3 and 4: storage overflow (of the usable volume) and underflow, with their duals.

        Overflow is relative to the usable volume, underflow to S_min (so its
        signal spans [0, 1] instead of being crushed by S_max).
        """
        storage_overflow_violation = max(0.0, (self.S_mass - ctx.s_cap) / ctx.s_cap)
        if storage_overflow_violation > 0:
            ev.record(3, storage_overflow_violation)
        lagrange_s_overflow = max(0.0, storage_overflow_violation - self.lagrange_target_slack)
        pen.plant += self.storage_overflow_lambda * saturate(lagrange_s_overflow)
        if storage_overflow_violation > 0:
            pen.plant += spike * spike_scale(storage_overflow_violation)
        if self.lagrange_updates_enabled:
            if lagrange_s_overflow > 0.0:
                new_u = self.storage_overflow_lambda + self.lagrange_lr * lagrange_s_overflow
            else:
                new_u = self.storage_overflow_lambda * (1.0 - self.lagrange_decay)
            self.storage_overflow_lambda = min(self.lagrange_max, max(0.0, new_u))

        storage_underflow_violation = max(0.0, (self.S_min - self.S_mass) / self.S_min)
        if storage_underflow_violation > 0:
            ev.record(4, storage_underflow_violation)
        lagrange_s_underflow = max(0.0, storage_underflow_violation - self.lagrange_target_slack)
        pen.plant += self.storage_underflow_lambda * saturate(lagrange_s_underflow)
        if storage_underflow_violation > 0:
            pen.plant += spike * spike_scale(storage_underflow_violation)
        if self.lagrange_updates_enabled:
            if lagrange_s_underflow > 0.0:
                new_u = self.storage_underflow_lambda + self.lagrange_lr * lagrange_s_underflow
            else:
                new_u = self.storage_underflow_lambda * (1.0 - self.lagrange_decay)
            self.storage_underflow_lambda = min(self.lagrange_max, max(0.0, new_u))
        info["storage_overflow_violation"] = float(storage_overflow_violation)
        info["storage_underflow_violation"] = float(storage_underflow_violation)

    def _heat_constraint(self, pen, ctx, ev, spike, info):
        """Id 9: the plant's heat-removal capacity, priced on each reactor's share of the load.

        A cooling_capacity fault moves the cap.
        """
        heat_q = self.heat_load()
        ctx.heat_total = float(np.sum(heat_q))
        ctx.heat_cap = self.max_heat_load * self._cooling_cap_mult
        heat_violation = max(0.0, (ctx.heat_total - ctx.heat_cap) / ctx.heat_cap)
        heat_penalty = 0.0
        if heat_violation > 0:
            ev.record(9, heat_violation)
            heat_penalty += spike * spike_scale(heat_violation)
        ctx.heat_edge = 1.0 - self.HEAT_BUFFER_FRAC
        if ctx.heat_total / ctx.heat_cap > ctx.heat_edge:
            depth = min(1.0, (ctx.heat_total / ctx.heat_cap - ctx.heat_edge) / self.HEAT_BUFFER_FRAC)
            heat_penalty += self.BUFFER_COEF * depth * depth
        lagrange_heat = max(0.0, heat_violation - self.lagrange_target_slack)
        heat_penalty += self.heat_load_lambda * saturate(lagrange_heat)
        self.heat_load_lambda = float(self._update_lambda(self.heat_load_lambda, lagrange_heat))
        heat_share = (heat_q / ctx.heat_total if ctx.heat_total > 0.0
                      else np.full(self.n_reactors, 1.0 / self.n_reactors))
        pen.local += heat_penalty * heat_share
        info["heat_load"] = ctx.heat_total
        info["heat_load_violation"] = float(heat_violation)

    def _titer_constraint(self, pen, ctx, fl, ev, spike, info):
        """Id 7: the harvest titer floor on every draining reactor, with its soft wall and dual."""
        spec = self.min_harvest_titer
        titer_violation = np.where(
            fl.draining, np.maximum(0.0, (spec - fl.titer_now) / spec), 0.0)
        titer_viol_mask = titer_violation > 0
        if np.any(titer_viol_mask):
            ev.record(7, titer_violation[titer_viol_mask])
        pen.local += np.where(titer_viol_mask, spike * spike_scale(titer_violation), 0.0)
        titer_edge = spec * (1.0 + self.TITER_BUFFER_FRAC)
        ctx.titer_near = fl.draining & ~titer_viol_mask & (fl.titer_now < titer_edge)
        titer_depth = np.where(
            ctx.titer_near, np.minimum(1.0, (titer_edge - fl.titer_now) / (titer_edge - spec)), 0.0)
        pen.local += self.BUFFER_COEF * titer_depth * titer_depth
        lagrange_titer = np.maximum(0.0, titer_violation - self.lagrange_target_slack)
        pen.local += self.harvest_titer_lambda * saturate(lagrange_titer)
        # Draining reactors update their dual; the others decay.
        self.harvest_titer_lambda = self._update_lambda(
            self.harvest_titer_lambda, np.where(fl.draining, lagrange_titer, 0.0))
        info["harvest_titer_violation"] = float(np.max(titer_violation))

    def _purity_constraint(self, pen, ctx, closed_lots, ev, info):
        """Id 8: every lot judged this step, then the dense shaping on each lot's running purity.

        A rejected lot's production reward is withdrawn from the shared pool,
        as the reward was paid into it; its penalty is the reactor's own.

        Returns:
            The production reward withdrawn.
        """
        clawback = 0.0
        purity_shortfall_closed = 0.0
        for i, lot in enumerate(closed_lots):
            if lot is None:
                continue
            if lot["passed"]:
                self._on_lot_released(i)
            else:
                ev.record(8, lot["shortfall"])
                pen.local[i] += self._rejected_lot_penalty(i, lot["shortfall"])
                clawback += lot["clawback"]
                purity_shortfall_closed = max(purity_shortfall_closed, lot["shortfall"])
        ctx.lot_titer = self.lot_mean_titer
        ctx.spec_p = self.min_product_titer
        purity_edge = ctx.spec_p * (1.0 + self.PURITY_BUFFER_FRAC)
        ctx.purity_shaped = (self.lot_mass > 0.0) & (ctx.lot_titer < purity_edge)
        purity_depth = np.where(
            ctx.purity_shaped,
            np.minimum(1.0, (purity_edge - ctx.lot_titer) / (purity_edge - ctx.spec_p)), 0.0)
        pen.local += self.PURITY_SHAPING_COEF * purity_depth * purity_depth
        info["product_purity_violation"] = float(purity_shortfall_closed)
        return clawback

    def _near_misses(self, ctx):
        """Instances inside a buffer zone but not past the limit, per constraint id."""
        near_miss = {
            1: int(np.sum((ctx.m_frac > ctx.overflow_edge) & ~ctx.overflow_viol)),
            2: int(np.sum((self.M < self.UNDERFLOW_BUFFER_KG) & ~ctx.underflow_viol)),
            3: int(ctx.overflow_edge_s < ctx.s_frac <= 1.0),
            4: int(self.S_min <= self.S_mass < ctx.underflow_edge_s),
            7: int(np.sum(ctx.titer_near)),
            8: int(np.sum(ctx.purity_shaped & (ctx.lot_titer >= ctx.spec_p))),
            9: int(ctx.heat_edge < ctx.heat_total / ctx.heat_cap <= 1.0),
        }
        return {cid: k for cid, k in near_miss.items() if k}

    def _supply_barriers(self, constraint_penalty_arr):
        """λ-independent barriers on running out of feedstock and of storage.

        Feedstock: a log barrier below FEEDSTOCK_BARRIER_FRAC of the initial
        supply.  Storage: a log barrier below STORAGE_BARRIER_MULT x S_min
        (about 20% of S_max, ~50 steps of warning at the downstream draw),
        steepening as the tank empties; and, in the no-Lagrangian case study
        only, a flat-plus-proportional spike below S_min.  Spread evenly over
        the reactors (in place).
        """
        lf_ratio = self.total_lf / self.initial_lf
        c5_ratio = self.total_c5 / self.initial_c5
        feedstock_barrier = 0.0
        if 0.0 < lf_ratio < FEEDSTOCK_BARRIER_FRAC:
            feedstock_barrier += FEEDSTOCK_BARRIER_COEF * (-np.log(lf_ratio + 0.001))
        if 0.0 < c5_ratio < FEEDSTOCK_BARRIER_FRAC:
            feedstock_barrier += FEEDSTOCK_BARRIER_COEF * (-np.log(c5_ratio + 0.001))
        if feedstock_barrier > 0.0:
            constraint_penalty_arr += feedstock_barrier / self.n_reactors

        s_ratio = self.S_mass / self.S_max
        s_barrier_threshold = (STORAGE_BARRIER_MULT * self.S_min) / self.S_max  # ~0.20
        storage_underflow_extra = 0.0
        if 0.0 < s_ratio < s_barrier_threshold:
            # The +0.001 prevents log(0); near the threshold the penalty is ~0.
            storage_underflow_extra += STORAGE_BARRIER_COEF * \
                (-np.log(s_ratio / s_barrier_threshold + 0.001))
        if self.S_mass < self.S_min and self.fixed_violation_penalties:
            shortfall = (self.S_min - self.S_mass) / self.S_min
            storage_underflow_extra += STORAGE_SPIKE_BASE * (1.0 + shortfall)
        if storage_underflow_extra > 0.0:
            constraint_penalty_arr += storage_underflow_extra / self.n_reactors

    # --- LOT HOOKS ---

    def _on_lot_released(self, i):
        self._decay_purity_lambda(i)

    def _rejected_lot_penalty(self, i, shortfall):
        return self._lot_purity_penalty(i, shortfall)

    def _empty_shipment_penalty(self):
        # A full miss: the duals price it at shortfall 1 too.
        return self._lot_purity_penalty(None, 1.0)

    # --- OUTER STEP EXTRAS ---

    def _new_step_extras(self):
        return {"near_miss_counts": {}, "near_miss_steps": 0, "lagrange_penalty": 0.0}

    def _accumulate_step_extras(self, extras, info):
        near = extras["near_miss_counts"]
        for cid, k in info["near_miss_counts"].items():
            near[cid] = near.get(cid, 0) + k
        extras["near_miss_steps"] += int(bool(info["near_miss_counts"]))
        extras["lagrange_penalty"] += info.get("total_lagrange_penalty", 0.0)

    def _step_extras_info(self, extras, steps_completed):
        return {
            # Per inner step, so the magnitude does not depend on the interval.
            "storage_lagrange_penalty": float(extras["lagrange_penalty"]
                                              / max(steps_completed, 1)),
            "near_miss_counts": extras["near_miss_counts"],
            "near_miss_steps": extras["near_miss_steps"],
        }
