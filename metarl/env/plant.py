"""Three-reactor fermentation plant with central storage and quality constraints.

Three heterogeneous fed-batch reactors share one feedstock pool and discharge
into one central storage tank, which feeds downstream at a fixed rate. Each
batch's discharge is one *lot*, released as on-spec product or rejected when
that reactor's batch completes; the purity constraint acts on the lot, and
each reactor's lot in progress is what the policy observes. Observations are
``(STATE_DIM,)`` = ``(78,)``, actions ``(ACTION_DIM,)`` = ``(12,)`` (four per
reactor), reward is a scalar. The policy observes the plant through the lean
study's instrument model (see ``config``); reward, constraints and metrics
are always computed from true state.

Violation ids (the lean study's 1-6, then the storage tank's):

    1  reactor overflow          M_i > Mmax                         (per reactor)
    2  reactor underflow         M_i < DRY_FLOOR                    (per reactor)
    3  harvest titer floor       reactor i drains below MIN_HARVEST_TITER
                                                                    (per reactor, per step)
    4  product purity spec       a lot's mean titer below spec      (per lot, and
                                 once if the plant shipped nothing)
    5  numerical instability     ODE blow-up / NaN                  (per reactor)
    6  mass-balance drift        conservation residual beyond tolerance (plant)
    7  storage overflow          S > S_MAX                          (plant, per step)
    8  storage underflow         S < S_MIN                          (plant, per step)

The headline violation count of an outer step is the number of violation
EVENTS in it: every (inner step, constraint, instance) triple, where the
instance is the reactor for ids 1-5 and the plant for 6-8 (a lot is an
instance of 4). ``violation_counts`` splits it by constraint and sums to it
exactly. Two vessels overflowing in the same hour are two events.

Mid-episode disturbances (``schedule_disturbance``): flow faults ``leak`` and
``feed_gain`` (one reactor or all), ``feed_conc`` (the shared feed tanks);
biochemical faults ``disturb_kinetics_decay``, ``toxin_spike``,
``cell_viability``, ``contamination`` (one reactor or all); and the storage
tank's ``downstream_rate`` (downstream draw x magnitude).

A discharge commanded below ``DISCHARGE_CUTOFF`` of the maximum outflow shuts
the valve, so a batch can be held in the unload stage without draining it.
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.core import integrate_rk4, co2_formation_rates, PARAM_DIM
from metarl.env.config import (N_REACTORS, INITIAL_LF, INITIAL_C5, S_MAX,
                               S_MIN, S_OUT_RATE, S_INITIAL, SPECIES_NAMES,
                               SPECIES_IDX, N_SPECIES, C_MAX_LIMITS, I_ETH,
                               BASE_CREDITS, TIME_STEP_DURATION,
                               MAX_FLOW_OUT_REAC, MIN_HARVEST_TITER,
                               MIN_PRODUCT_TITER, PRODUCT_MASS_SCALE,
                               TITER_SCALE, R_MASS, R_STAGE, R_CREDIT,
                               R_LOT_MASS, R_LOT_TITER, REACTOR_OBS_DIM,
                               get_reactor_config, nominal_task,
                               SENSOR_MODEL_ENABLED, ANALYZER_SAMPLE_PERIOD_H,
                               ANALYZER_REPORT_LAG_H, ANALYZER_NOISE_RSD,
                               ANALYZER_NOISE_FLOOR, ONLINE_NOISE_RSD,
                               SENSOR_SEED, MASK_INTERNAL_CONCENTRATIONS,
                               OBSERVE_CO2_RATE, CER_SCALE, CER_NOISE_RSD)
from metarl.env.layout import (GLOBAL_START, CER_START, STATE_DIM,
                               ACTION_DIM_PER_REACTOR, ACTION_DIM,
                               STORAGE_CHANNEL, LF_CHANNEL, C5_CHANNEL,
                               PRIVILEGED_SPECIES_CLIP, reactor_channel,
                               cer_channel)
from metarl.env.faults import FaultMixin, parse_active_disturbances
from metarl.env.instruments import InstrumentMixin
from metarl.env.ledger import LedgerMixin

# Commanded discharge, as a fraction of the maximum outflow, below which the
# valve is shut (see MetaRL_Lean/env_lean.DISCHARGE_CUTOFF).
DISCHARGE_CUTOFF = 0.02

# g of CO2 per g of ethanol a contaminant oxidises (C2H5OH + 3 O2 ->
# 2 CO2 + 3 H2O: 2 x 44 / 46).
CONTAM_CO2_PER_ETH = 88.0 / 46.0

# Constraint ids by scope: reactor-local ones are counted per reactor.
REACTOR_CONSTRAINTS = (1, 2, 3, 4, 5)
PLANT_CONSTRAINTS = (6, 7, 8)


class PlantFermentationEnv(InstrumentMixin, FaultMixin, LedgerMixin):
    """Three-reactor plant with central storage and Lagrangian constraint shaping.

    Manages the reactors' dynamics, the shared feedstock pool, the storage
    tank, normalized observation construction, staged batch progression, a
    product ledger with per-reactor lots, and constraint-aware rewards over
    inventory, product-quality and storage constraints.

    Attributes:
        n_reactors: Number of reactors (3).
        control_action_interval: Inner 1-hour steps per outer action.
        stateDim: Observation width (78).
        actionDim: Action width (12).
        max_steps: Episode horizon in inner steps.
    """

    def __init__(self, Mmax=None, max_flow_LF=None, max_flow_C5=None,
                 control_action_interval=5, max_steps=600,
                 active_disturbances=None, sensor_model=None,
                 mask_internal_concentrations=None, observe_co2_rate=None):
        """Initialise the plant.

        Args:
            Mmax: Per-reactor maximum mass (kg); defaults to
                ``config.get_reactor_config``.
            max_flow_LF: Per-reactor maximum LF feed flow (kg/s); default as
                above.
            max_flow_C5: Per-reactor maximum C5 feed flow (kg/s); default as
                above.
            control_action_interval: Inner 1-hour steps per outer agent action.
            max_steps: Episode horizon in inner steps.
            active_disturbances: Disturbance kinds random scheduling may draw
                from, in any form ``parse_active_disturbances`` accepts.
            sensor_model: Override ``SENSOR_MODEL_ENABLED``.
            mask_internal_concentrations: Override
                ``MASK_INTERNAL_CONCENTRATIONS``.
            observe_co2_rate: Override ``OBSERVE_CO2_RATE``.
        """
        cfg = get_reactor_config()
        n = N_REACTORS
        self.n_reactors = n
        self.control_action_interval = control_action_interval
        self.n_stage = 5
        self.stateDim = STATE_DIM
        self.actionDim = ACTION_DIM
        self.max_steps = max_steps
        self.active_disturbances = (
            parse_active_disturbances(active_disturbances)
            if active_disturbances is not None else None)
        self.mask_internal_concentrations = (
            MASK_INTERNAL_CONCENTRATIONS if mask_internal_concentrations is None
            else bool(mask_internal_concentrations))
        self.observe_co2_rate = (OBSERVE_CO2_RATE if observe_co2_rate is None
                                 else bool(observe_co2_rate))

        def _per_reactor(v, default):
            arr = np.asarray(default if v is None else v, dtype=np.float64)
            arr = np.broadcast_to(arr, (n,)).astype(np.float64).copy()
            return arr

        self.Mmax = _per_reactor(Mmax, cfg["Mmax"])
        self.max_flow_LF = _per_reactor(max_flow_LF, cfg["max_flow_LF"])
        self.max_flow_C5 = _per_reactor(max_flow_C5, cfg["max_flow_C5"])
        self.max_flow_out_reac = MAX_FLOW_OUT_REAC

        self.initial_lf = INITIAL_LF
        self.initial_c5 = INITIAL_C5
        self.S_max = S_MAX
        self.S_min = S_MIN
        self.S_out_rate = S_OUT_RATE
        self.S_initial = S_INITIAL

        self.species_names = SPECIES_NAMES
        self.idx = SPECIES_IDX
        self.base_credits = BASE_CREDITS.copy()
        self.time_step_duration = TIME_STEP_DURATION
        self.n_inner_steps = 30
        self.smoothing_coef = 0.005
        self.cer = np.zeros(n)

        # ── Uncertainty / task context ───────────────────────────────────────
        self.fixed_task = None
        self.current_task = None
        # List of {kind, onset, magnitude, reactor, channel, fired}.
        self._disturbances = []
        self.params = np.empty((n, PARAM_DIM), dtype=np.float64)
        self.feed_gain = np.ones(n)
        self.leak_frac = np.zeros(n)
        self._fault_seen = False
        self._fault_kind = 0
        self._apply_task(nominal_task())

        # ── Product quality specification ────────────────────────────────────
        self.min_harvest_titer = MIN_HARVEST_TITER
        self.min_product_titer = MIN_PRODUCT_TITER
        self.product_mass_scale = PRODUCT_MASS_SCALE
        self.titer_scale = TITER_SCALE

        # ── Lagrangian constraint enforcement (the lean study's) ─────────────
        self.lagrange_lr = 0.5
        self.lagrange_decay = 0.5 / 600.0
        self.lagrange_target_slack = 0.0
        self.lagrange_max = 10.0

        self.DRY_FLOOR = 10.0
        self.BUFFER_COEF = 0.02
        self.OVERFLOW_BUFFER_FRAC = 0.10        # activates at 90% Mmax
        self.UNDERFLOW_BUFFER_KG = 100.0        # activates at M < 100 kg
        self.TITER_BUFFER_FRAC = 0.15           # draining within 15% of spec
        self.PURITY_BUFFER_FRAC = 0.15
        self.PURITY_SHAPING_COEF = 0.02
        # Storage soft walls. Overflow as for a reactor (90% of S_MAX).
        # Underflow from 60,000 kg (15% of S_MAX): 37 h of downstream draw
        # above S_MIN, because the only fix is a reactor ready to unload, and
        # a vessel takes at least ~65 h from charge to unload.
        self.STORAGE_OVERFLOW_BUFFER_FRAC = 0.10
        self.STORAGE_UNDERFLOW_BUFFER_KG = 60000.0

        self.reactor_overflow_lambda = np.zeros(n)
        self.reactor_underflow_lambda = np.zeros(n)
        self.harvest_titer_lambda = np.zeros(n)
        self.product_purity_lambda = np.zeros(n)
        self.storage_overflow_lambda = 0.0
        self.storage_underflow_lambda = 0.0

        self.lagrange_updates_enabled = True
        self.reset_lambdas_per_episode = False
        self._use_lagrangian = True

        # ── Reward coefficients (the lean study's, per kg and per event) ─────
        self.PROD_REWARD_COEF = 10.0
        self.PROD_REWARD_SCALE = 1e7
        # Per pool fraction consumed. The lean study's 1.5 on pools of
        # 600,000 / 180,000 kg, rescaled to these 2,000,000 / 600,000 kg pools
        # so a kilogram of feed costs what it cost there.
        self.RAW_MAT_COEF = 5.0
        self.BASE_SPIKE = 0.3
        self.INSTABILITY_PENALTY_COEF = 5.0
        self.MASS_BALANCE_PENALTY_COEF = 1.0
        self.PURITY_PENALTY_COEF = 3.0
        self.OFF_SPEC_VALUE_FRAC = 0.0

        self.mass_balance_abs_tol = 5.0
        self.mass_balance_rel_tol = 1e-6

        # ── Instrumentation / sensor model ───────────────────────────────────
        self.sensor_model = bool(SENSOR_MODEL_ENABLED if sensor_model is None
                                 else sensor_model)
        # Per reactor: its 15 species, then its lot titer.
        per = [list(range(N_SPECIES)) + [R_LOT_TITER]] * n
        self.analyzer_channels = np.asarray(
            [reactor_channel(i, c) for i, chans in enumerate(per) for c in chans],
            dtype=np.intp)
        self._lot_assay_pos = np.asarray(
            [i * (N_SPECIES + 1) + N_SPECIES for i in range(n)], dtype=np.intp)
        self.online_channels = np.asarray(
            [reactor_channel(i, R_MASS) for i in range(n)]
            + [reactor_channel(i, R_LOT_MASS) for i in range(n)]
            + [STORAGE_CHANNEL, LF_CHANNEL, C5_CHANNEL], dtype=np.intp)
        self.cer_channels = np.asarray([cer_channel(i) for i in range(n)],
                                       dtype=np.intp)
        self.lot_titer_channels = np.asarray(
            [reactor_channel(i, R_LOT_TITER) for i in range(n)], dtype=np.intp)
        self.species_channels = np.asarray(
            [reactor_channel(i, c) for i in range(n) for c in range(N_SPECIES)],
            dtype=np.intp)
        self.analyzer_sample_period = int(ANALYZER_SAMPLE_PERIOD_H)
        self.analyzer_report_lag = int(ANALYZER_REPORT_LAG_H)
        self.analyzer_noise_rsd = float(ANALYZER_NOISE_RSD)
        self.analyzer_noise_floor = float(ANALYZER_NOISE_FLOOR)
        self.online_noise_rsd = float(ONLINE_NOISE_RSD)
        self.cer_noise_rsd = float(CER_NOISE_RSD)
        self.sensor_stream_key = None
        self._sensor_counter = 0
        self._sensor_rng = np.random.default_rng(SENSOR_SEED)
        self._analyzer_held = None
        self._analyzer_pending = []
        self._analyzer_last_draw = 0

        self.reset()

    # ------------------------------------------------------------------ reset

    def reset(self):
        """Reset biology, resources, storage, ledger and episode bookkeeping.

        Returns:
            Measured observation array of shape (STATE_DIM,).
        """
        n = self.n_reactors
        self.time_step_count = 0

        self._apply_task(self.fixed_task if self.fixed_task is not None
                         else nominal_task())
        for d in self._disturbances:
            d["fired"] = False
        self._fault_seen = False
        self._fault_kind = 0

        self.total_lf = self.initial_lf
        self.total_c5 = self.initial_c5
        self.S_mass = float(self.S_initial)

        if self.reset_lambdas_per_episode:
            self._zero_lambdas()

        self.M = np.full(n, 10000.0)
        self.C = np.zeros((n, N_SPECIES), dtype=np.float64)
        self.C[:, self.idx['Cell']] = (1000.0 * 147.0) / 10000.0
        self.current_stage = np.zeros(n, dtype=np.int64)
        self.stage_credits = np.full(n, float(self.base_credits[0]))

        # product_eth is kg x concentration, so product_eth / product_mass is
        # a mass-weighted mean titer in the units of C[Eth].
        self.product_mass = 0.0
        self.product_eth = 0.0
        self.reactor_product_mass = np.zeros(n)
        self.reactor_product_eth = np.zeros(n)
        # Each reactor's lot in progress (see _close_lot).
        self.lot_mass = np.zeros(n)
        self.lot_eth = np.zeros(n)
        self.lot_prod_reward = np.zeros(n)
        self.on_spec_eth = 0.0
        self.reactor_on_spec_eth = np.zeros(n)
        self.lots_passed = 0
        self.lots_failed = 0
        self.reactor_lots_passed = np.zeros(n, dtype=np.int64)
        self.reactor_lots_failed = np.zeros(n, dtype=np.int64)
        self.worst_lot_shortfall = 0.0

        self.prev_flow_action = np.zeros((n, 3))
        self.episode_mass_discharged = 0.0
        self.episode_initial_total_mass = self._compute_total_tracked_mass()
        self.n_batches_completed = 0
        self.reactor_batches_completed = np.zeros(n, dtype=np.int64)
        self.storage_min_seen = self.S_mass
        self.storage_max_seen = self.S_mass
        self.cer = self._co2_evolution_rate()

        self._reset_sensors()
        return self._sensor_observe(self.get_state_norm())

    def privileged_state(self):
        """The plant's true state and clock, for training-time critics only.

        The exact observation (no instrument noise), with every reactor's
        species channels filled in even where they are masked, then the
        elapsed share of the episode. Nothing that acts reads it.

        Returns:
            Float32 array of shape (PRIVILEGED_STATE_DIM,), finite.
        """
        s = self.get_state_norm().astype(np.float64)
        s[self.species_channels] = np.clip(
            (self.C / C_MAX_LIMITS).reshape(-1), 0.0, PRIVILEGED_SPECIES_CLIP)
        s = np.append(s, self.time_step_count / max(1, self.max_steps))
        return np.nan_to_num(s, nan=0.0, posinf=PRIVILEGED_SPECIES_CLIP,
                             neginf=0.0).astype(np.float32)

    def get_state_norm(self):
        """Build the exact (pre-instrument) normalized observation.

        Returns:
            Float32 NumPy array of shape (STATE_DIM,); layout in config.
        """
        n = self.n_reactors
        obs = np.zeros(STATE_DIM, dtype=np.float32)
        blocks = obs[:GLOBAL_START].reshape(n, REACTOR_OBS_DIM)
        if not self.mask_internal_concentrations:
            blocks[:, 0:N_SPECIES] = self.C / C_MAX_LIMITS
        blocks[:, R_MASS] = self.M / self.Mmax
        blocks[np.arange(n), R_STAGE + self.current_stage] = 1.0
        blocks[:, R_CREDIT] = self.stage_credits / self.base_credits[self.current_stage]
        blocks[:, R_LOT_MASS] = self.lot_mass / self.product_mass_scale
        blocks[:, R_LOT_TITER] = self.lot_mean_titer / self.titer_scale
        obs[STORAGE_CHANNEL] = self.S_mass / self.S_max
        obs[LF_CHANNEL] = self.total_lf / self.initial_lf
        obs[C5_CHANNEL] = self.total_c5 / self.initial_c5
        if self.observe_co2_rate:
            obs[CER_START:CER_START + n] = self.cer / CER_SCALE
        return obs

    def _co2_evolution_rate(self):
        """Each reactor's volumetric CO2 evolution rate right now, in g/L/h.

        Returns:
            Non-negative float array (n,); 0.0 where the state is not finite.
        """
        rate = co2_formation_rates(self.C, self.params) * 3600.0
        rate = rate + (self._current_contam_rate * CONTAM_CO2_PER_ETH
                       * self.C[:, I_ETH])
        return np.where(np.isfinite(rate) & (rate > 0.0), rate, 0.0)

    @staticmethod
    def get_action_mask(state_tensor):
        """Compute the stage-based structural action mask from normalized state.

        Per reactor: [time multiplier, LF, C5, outflow]; time multiplier free
        except during cleanup, LF in charge and co-feed, C5 in co-feed only,
        discharge in unload only.

        Args:
            state_tensor: Tensor whose final dimension holds at least the
                reactor blocks; any batch shape.

        Returns:
            Torch tensor mask with final dimension ACTION_DIM.
        """
        import torch
        cols = []
        for i in range(N_REACTORS):
            s = reactor_channel(i, R_STAGE)
            cols += [1.0 - state_tensor[..., s + 4],
                     state_tensor[..., s + 0] + state_tensor[..., s + 1],
                     state_tensor[..., s + 1],
                     state_tensor[..., s + 3]]
        return torch.stack(cols, dim=-1)

    def _reset_reactor_for_next_batch(self, i):
        """Empty reactor ``i`` and re-charge it with a fresh 10,000 kg inoculum.

        The ledger is credited ``M - 10,000`` (signed), which keeps the
        balance check exact.
        """
        self.episode_mass_discharged += float(self.M[i]) - 10000.0
        self.M[i] = 10000.0
        self.C[i, :] = 0.0
        self.C[i, self.idx['Cell']] = (1000.0 * 147.0) / 10000.0
        self.stage_credits[i] = float(self.base_credits[0])
        self.n_batches_completed += 1
        self.reactor_batches_completed[i] += 1

    # ------------------------------------------------------- inner step

    def _step_internal(self, action, is_first=True):
        """Execute one inner 1-hour step for the whole plant.

        Disturbances, stage credits, feed and discharge, RK4 integration, the
        storage tank, then reward and the per-step constraints.

        Args:
            action: Action array in [-1, 1], shape (ACTION_DIM,).
            is_first: True on the first inner step of an outer action, used
                for action-smoothing bookkeeping.

        Returns:
            A tuple (state, reward, done, info).
        """
        n = self.n_reactors
        leaked_mass = self._apply_disturbances()

        a_clipped = np.clip(np.asarray(action, dtype=np.float64).reshape(
            n, ACTION_DIM_PER_REACTOR), -1.0, 1.0)
        a = (a_clipped + 1.0) / 2.0
        multiplier = a[:, 0] * 1.5 + 0.5      # time multiplier in [0.5, 2.0]
        a_exec = a.copy()
        if self.total_lf <= 0:
            a_exec[:, 1] = 0.0
        if self.total_c5 <= 0:
            a_exec[:, 2] = 0.0
        a_exec[a_exec[:, 3] < DISCHARGE_CUTOFF, 3] = 0.0

        dt = self.time_step_duration
        dt_hours = dt / 3600.0

        # ==================== Phase 1: time credits ====================
        self.stage_credits = self.stage_credits - multiplier * dt_hours
        batch_completed = np.zeros(n, dtype=bool)
        closed_lots = [None] * n
        for i in np.flatnonzero(self.stage_credits <= 0):
            if self.current_stage[i] == 4:
                self.current_stage[i] = 0
                closed_lots[i] = self._close_lot(i)
                self._reset_reactor_for_next_batch(i)
                batch_completed[i] = True
                # The vessel was just recharged; a drain command carried over
                # from the unload must not drain the zero-titer charge.
                a_exec[i, 3] = 0.0
            elif self.current_stage[i] == 3 and self.M[i] >= 100.0:
                # Unload cannot end until the vessel is essentially empty.
                self.stage_credits[i] = 0.0
            else:
                self.current_stage[i] += 1
                self.stage_credits[i] = float(self.base_credits[self.current_stage[i]])

        # ==================== Phase 2: hardware execution ====================
        simulation_dt = dt / self.n_inner_steps
        f_lf_req = a_exec[:, 1] * self.max_flow_LF * self.feed_gain
        f_c5_req = a_exec[:, 2] * self.max_flow_C5 * self.feed_gain
        max_possible_f = np.maximum(0.0, (self.M - 50.0) / dt)
        F_out = np.minimum(a_exec[:, 3] * self.max_flow_out_reac, max_possible_f)

        # Harvest is drawn at the titer standing at the start of the step,
        # into each reactor's lot and into the storage tank.
        titer_now = self.C[:, I_ETH].copy()
        harvested_mass = F_out * dt
        harvested_eth = harvested_mass * titer_now
        draining = F_out > 0.0
        self.product_mass += float(harvested_mass.sum())
        self.product_eth += float(harvested_eth.sum())
        self.reactor_product_mass += harvested_mass
        self.reactor_product_eth += harvested_eth
        self.lot_mass += harvested_mass
        self.lot_eth += harvested_eth
        storage_in = float(harvested_mass.sum())
        self.S_mass += storage_in

        # The shared pool is drawn in reactor order.
        F_in = np.zeros(n)
        Cin = np.zeros((n, N_SPECIES))
        act_lf_mass = np.zeros(n)
        act_c5_mass = np.zeros(n)
        for i in range(n):
            lf_m = min(f_lf_req[i] * dt, self.total_lf)
            c5_m = min(f_c5_req[i] * dt, self.total_c5)
            self.total_lf -= lf_m
            self.total_c5 -= c5_m
            act_lf_mass[i], act_c5_mass[i] = lf_m, c5_m
            f_lf, f_c5 = lf_m / dt, c5_m / dt
            F_in[i] = f_lf + f_c5
            if F_in[i] > 0:
                Cin[i] = (f_c5 * self.C_C5[i] + f_lf * self.C_LF[i]) / F_in[i]

        c_new, m_new, instability_detected, severity = integrate_rk4(
            self.C.astype(np.float64), self.M.astype(np.float64),
            F_in, F_out, Cin, self.params, simulation_dt, self.n_inner_steps)
        self.C = c_new
        self.M = m_new
        self.cer = self._co2_evolution_rate()

        # The storage tank feeds downstream at its (possibly disturbed) rate,
        # never below empty; what it sends leaves the system.
        drain = min(self.S_out_rate * self._downstream_mult * dt,
                    max(self.S_mass, 0.0))
        self.S_mass -= drain
        self.episode_mass_discharged += drain
        self.storage_min_seen = min(self.storage_min_seen, self.S_mass)
        self.storage_max_seen = max(self.storage_max_seen, self.S_mass)

        # ==================== Phase 3: reward construction ====================
        prod_reward = self.PROD_REWARD_COEF * (harvested_eth / self.PROD_REWARD_SCALE)
        self.lot_prod_reward += prod_reward
        clawback = float(sum(c["clawback"] for c in closed_lots if c is not None))

        counts = {}            # constraint id -> violation events this step
        depth_sum, depth_max = {}, {}
        reactor_events = np.zeros(n, dtype=np.int64)
        penalty = 0.0

        def record(cid, depths, where=None):
            """Count events of ``cid`` with relative depths past the limit."""
            depths = np.atleast_1d(np.asarray(depths, dtype=float))
            if depths.size == 0:
                return
            counts[cid] = counts.get(cid, 0) + int(depths.size)
            depth_sum[cid] = depth_sum.get(cid, 0.0) + float(depths.sum())
            depth_max[cid] = max(depth_max.get(cid, 0.0), float(depths.max()))
            if where is not None:
                reactor_events[where] += 1

        # ── Reactor soft walls ───────────────────────────────────────────────
        m_frac = self.M / self.Mmax
        overflow_edge = 1.0 - self.OVERFLOW_BUFFER_FRAC
        over_depth = np.clip((m_frac - overflow_edge) / self.OVERFLOW_BUFFER_FRAC,
                             0.0, 1.0)
        penalty += self.BUFFER_COEF * float(np.sum(over_depth ** 2))
        zone_uf = max(self.UNDERFLOW_BUFFER_KG - self.DRY_FLOOR, 1.0)
        under_depth = np.clip((self.UNDERFLOW_BUFFER_KG - self.M) / zone_uf, 0.0, 1.0)
        penalty += self.BUFFER_COEF * float(np.sum(under_depth ** 2))

        # ── Constraint 1: reactor overflow ───────────────────────────────────
        r_overflow = np.maximum(0.0, m_frac - 1.0)
        v1 = r_overflow > 0
        if v1.any():
            record(1, r_overflow[v1], np.flatnonzero(v1))
            penalty += self.BASE_SPIKE * float(np.sum(1.0 + r_overflow[v1]))
        lag = np.maximum(0.0, r_overflow - self.lagrange_target_slack)
        penalty += float(np.sum(self.reactor_overflow_lambda * lag))
        self.reactor_overflow_lambda = self._update_lambda(
            self.reactor_overflow_lambda, lag)

        # ── Constraint 2: reactor underflow ──────────────────────────────────
        r_underflow = np.maximum(0.0, (self.DRY_FLOOR - self.M) / self.DRY_FLOOR)
        v2 = r_underflow > 0
        if v2.any():
            record(2, r_underflow[v2], np.flatnonzero(v2))
            penalty += self.BASE_SPIKE * float(np.sum(1.0 + r_underflow[v2]))
        lag = np.maximum(0.0, r_underflow - self.lagrange_target_slack)
        penalty += float(np.sum(self.reactor_underflow_lambda * lag))
        self.reactor_underflow_lambda = self._update_lambda(
            self.reactor_underflow_lambda, lag)

        # ── Constraint 3: harvest titer floor (per draining reactor) ─────────
        spec = self.min_harvest_titer
        titer_violation = np.where(
            draining, np.maximum(0.0, (spec - titer_now) / spec), 0.0)
        v3 = titer_violation > 0
        if v3.any():
            record(3, titer_violation[v3], np.flatnonzero(v3))
            penalty += self.BASE_SPIKE * float(np.sum(1.0 + titer_violation[v3]))
        edge = spec * (1.0 + self.TITER_BUFFER_FRAC)
        near = draining & ~v3 & (titer_now < edge)
        if near.any():
            depth = np.minimum(1.0, (edge - titer_now[near]) / max(edge - spec, 1e-9))
            penalty += self.BUFFER_COEF * float(np.sum(depth ** 2))
        lag = np.maximum(0.0, titer_violation - self.lagrange_target_slack)
        penalty += float(np.sum(self.harvest_titer_lambda * lag))
        # Draining reactors update their dual; the others decay.
        self.harvest_titer_lambda = self._update_lambda(
            self.harvest_titer_lambda, np.where(draining, lag, 0.0))

        # ── Constraint 4: product purity, lot by lot ─────────────────────────
        for i, lot in enumerate(closed_lots):
            if lot is None:
                continue
            if lot["passed"]:
                self.product_purity_lambda[i] = self._update_lambda(
                    self.product_purity_lambda[i:i + 1], np.zeros(1))[0]
            else:
                record(4, lot["shortfall"], i)
                penalty += self._lot_purity_penalty(i, lot["shortfall"])

        # Dense shaping on each lot's running purity.
        lot_titer = self.lot_mean_titer
        spec_p = self.min_product_titer
        edge_p = spec_p * (1.0 + self.PURITY_BUFFER_FRAC)
        has_lot = self.lot_mass > 0.0
        purity_shortfall = np.where(
            has_lot, np.maximum(0.0, (spec_p - lot_titer) / spec_p), 0.0)
        shaped = has_lot & (lot_titer < edge_p)
        if shaped.any():
            depth = np.minimum(1.0, (edge_p - lot_titer[shaped]) / max(edge_p - spec_p, 1e-9))
            penalty += self.PURITY_SHAPING_COEF * float(np.sum(depth ** 2))

        # ── Storage soft walls ───────────────────────────────────────────────
        s_frac = self.S_mass / self.S_max
        s_edge = 1.0 - self.STORAGE_OVERFLOW_BUFFER_FRAC
        if s_frac > s_edge:
            d_ = min(1.0, (s_frac - s_edge) / self.STORAGE_OVERFLOW_BUFFER_FRAC)
            penalty += self.BUFFER_COEF * d_ * d_
        u_edge = self.STORAGE_UNDERFLOW_BUFFER_KG
        if self.S_mass < u_edge:
            d_ = min(1.0, (u_edge - self.S_mass) / max(u_edge - self.S_min, 1.0))
            penalty += self.BUFFER_COEF * d_ * d_

        # ── Constraint 7: storage overflow ───────────────────────────────────
        s_over = max(0.0, (self.S_mass - self.S_max) / self.S_max)
        if s_over > 0:
            record(7, s_over)
            penalty += self.BASE_SPIKE * (1.0 + s_over)
        lag = max(0.0, s_over - self.lagrange_target_slack)
        penalty += self.storage_overflow_lambda * lag
        self.storage_overflow_lambda = float(self._update_lambda(
            np.array([self.storage_overflow_lambda]), np.array([lag]))[0])

        # ── Constraint 8: storage underflow ──────────────────────────────────
        # Normalised by S_MIN, so the signal spans [0, 1] instead of being
        # crushed by S_MAX.
        s_under = max(0.0, (self.S_min - self.S_mass) / self.S_min)
        if s_under > 0:
            record(8, s_under)
            penalty += self.BASE_SPIKE * (1.0 + s_under)
        lag = max(0.0, s_under - self.lagrange_target_slack)
        penalty += self.storage_underflow_lambda * lag
        self.storage_underflow_lambda = float(self._update_lambda(
            np.array([self.storage_underflow_lambda]), np.array([lag]))[0])

        # ── Constraint 5: instability ────────────────────────────────────────
        unstable = np.zeros(n, dtype=bool)
        if instability_detected:
            unstable = severity > 0.0
            if not unstable.any():
                unstable[:] = True
            record(5, np.ones(int(unstable.sum())), np.flatnonzero(unstable))
            penalty += self.INSTABILITY_PENALTY_COEF * float(
                np.sum(1.0 + severity[unstable]))
            # Instability has no comparable depth.
            depth_sum.pop(5, None)
            depth_max.pop(5, None)

        # ── Feedstock low-supply barrier ─────────────────────────────────────
        lf_ratio = self.total_lf / self.initial_lf
        c5_ratio = self.total_c5 / self.initial_c5
        _fb_coef = 0.005
        if 0.0 < lf_ratio < 0.02:
            penalty += _fb_coef * (-np.log(lf_ratio + 0.001))
        if 0.0 < c5_ratio < 0.02:
            penalty += _fb_coef * (-np.log(c5_ratio + 0.001))

        # ── Raw material, unload trickle, action smoothing ───────────────────
        raw_mat_penalty = self.RAW_MAT_COEF * (
            float(act_lf_mass.sum()) / max(self.initial_lf, 1.0)
            + float(act_c5_mass.sum()) / max(self.initial_c5, 1.0))
        eff_penalty = 0.0005 * float(np.sum(self.current_stage == 3))
        if is_first:
            current_flow = a_clipped[:, 1:4]
            smoothing_penalty = self.smoothing_coef * float(
                np.sum(np.mean((current_flow - self.prev_flow_action) ** 2, axis=1)))
            self.prev_flow_action = current_flow.copy()
        else:
            smoothing_penalty = 0.0

        reward = (float(prod_reward.sum()) - clawback
                  - (penalty + raw_mat_penalty + eff_penalty + smoothing_penalty))

        self.time_step_count += 1
        done = (self.time_step_count >= self.max_steps or instability_detected)

        info = {
            "prod": float(prod_reward.sum()),
            "clawback": clawback,
            "constraint": penalty,
            "smoothing": smoothing_penalty,
            "eff": eff_penalty,
            "raw_mat_penalty": raw_mat_penalty,
            "eth_harvested": float(harvested_eth.sum()),
            "mass_harvested": float(harvested_mass.sum()),
            "storage_in": storage_in,
            "storage_out": drain,
            "storage_mass": float(self.S_mass),
            "product_mass": self.product_mass,
            "product_eth": self.product_eth,
            "mean_titer": self.mean_product_titer,
            "lot_titer": lot_titer.copy(),
            "titer_now": titer_now,
            "draining": draining,
            "titer_violation": titer_violation,
            "purity_shortfall": purity_shortfall,
            "leaked_mass": leaked_mass,
            "lf_remaining": self.total_lf,
            "c5_remaining": self.total_c5,
            "instability_severity": float(np.max(severity)) if instability_detected else 0.0,
            "hard_violation": bool(counts),
            "constraint_violated": bool(counts),
            "violation_id": max(counts) if counts else 0,
            "violated_constraints": sorted(counts),
            "violation_count": int(sum(counts.values())),
            "violation_counts": counts,
            "violation_depth_sum": depth_sum,
            "violation_depth_max": depth_max,
            "reactor_violation_events": reactor_events,
            "actual_action": a_exec.reshape(-1).copy(),
            "batch_completed": bool(batch_completed.any()),
            "batch_completed_mask": batch_completed,
            "stage": self.current_stage.copy(),
        }
        return (self._sensor_observe(self.get_state_norm()), float(reward),
                done, info)

    # ------------------------------------------------------- outer step

    def step(self, action):
        """Execute one outer environment step with action repeat.

        Repeats the action for up to ``control_action_interval`` inner steps,
        stopping early when any reactor completes a batch (the policy then
        decides again for every reactor), and aggregates. On the terminating
        step, also releases or rejects every open lot and applies the
        mass-conservation guard.

        Args:
            action: Action array in [-1, 1], shape (ACTION_DIM,).

        Returns:
            A tuple (state, total_reward, done, info_out).
        """
        total = {k: 0.0 for k in ("reward", "prod", "clawback", "constraint",
                                  "smoothing", "eff", "raw_mat_penalty",
                                  "eth_harvested", "mass_harvested",
                                  "storage_in", "storage_out")}
        total_leaked = np.zeros(self.n_reactors)
        max_titer_violation = np.zeros(self.n_reactors)
        drain_steps = np.zeros(self.n_reactors, dtype=np.int64)
        drain_titers = []
        violation_count = 0
        violation_counts, depth_sum, depth_max = {}, {}, {}
        reactor_events = np.zeros(self.n_reactors, dtype=np.int64)
        storage_lo = storage_hi = float(self.S_mass)
        first_action = None
        state = self.get_state_norm()
        done = False
        info = {}

        for idx in range(self.control_action_interval):
            state, reward, done, info = self._step_internal(
                action, is_first=(idx == 0))
            total["reward"] += reward
            for k in ("prod", "clawback", "constraint", "smoothing", "eff",
                      "raw_mat_penalty", "eth_harvested", "mass_harvested",
                      "storage_in", "storage_out"):
                total[k] += info[k]
            total_leaked += info["leaked_mass"]
            max_titer_violation = np.maximum(max_titer_violation,
                                             info["titer_violation"])
            drain_steps += info["draining"].astype(np.int64)
            for i in np.flatnonzero(info["draining"]):
                drain_titers.append(float(info["titer_now"][i]))
            violation_count += info["violation_count"]
            for cid, k in info["violation_counts"].items():
                violation_counts[cid] = violation_counts.get(cid, 0) + k
            for cid, d in info["violation_depth_sum"].items():
                depth_sum[cid] = depth_sum.get(cid, 0.0) + d
            for cid, d in info["violation_depth_max"].items():
                depth_max[cid] = max(depth_max.get(cid, 0.0), d)
            reactor_events += info["reactor_violation_events"]
            storage_lo = min(storage_lo, info["storage_mass"])
            storage_hi = max(storage_hi, info["storage_mass"])
            if idx == 0:
                first_action = info["actual_action"]
            if done or info["batch_completed"]:
                break

        info_out = {
            "prod": total["prod"],
            "clawback": total["clawback"],
            "constraint": total["constraint"],
            "smoothing": total["smoothing"],
            "eff": total["eff"],
            "raw_mat_penalty": total["raw_mat_penalty"],
            "eth_harvested": total["eth_harvested"],
            "mass_harvested": total["mass_harvested"],
            "storage_in": total["storage_in"],
            "storage_out": total["storage_out"],
            "storage_mass": float(self.S_mass),
            "storage_min": storage_lo,
            "storage_max": storage_hi,
            "leaked_mass": total_leaked,
            "product_mass": self.product_mass,
            "product_eth": self.product_eth,
            "mean_titer": self.mean_product_titer,
            "lot_mass": self.lot_mass.copy(),
            "lot_titer": self.lot_mean_titer,
            "lots_passed": self.lots_passed,
            "lots_failed": self.lots_failed,
            "titer_violation": max_titer_violation,
            "drain_steps": drain_steps,
            "drain_titers": drain_titers,
            "lf_remaining": self.total_lf,
            "c5_remaining": self.total_c5,
            "hard_violation": violation_count > 0,
            "constraint_violated": violation_count > 0,
            "violation_id": max(violation_counts) if violation_counts else 0,
            "violation_ids": sorted(violation_counts),
            # Violation EVENTS (see the module docstring); violation_counts
            # splits it by constraint and sums to it exactly.
            "violation_count": int(violation_count),
            "violation_counts": dict(violation_counts),
            "violation_depth_sum": depth_sum,
            "violation_depth_max": depth_max,
            # Reactor-local events (ids 1-5) per reactor.
            "reactor_violation_events": reactor_events,
            "actual_action": first_action,
            "stage": self.current_stage.copy(),
            "n_batches_completed": self.n_batches_completed,
        }
        if done:
            total["reward"] = self._finalize_episode(total["reward"], info_out)
        return state, float(total["reward"]), done, info_out
