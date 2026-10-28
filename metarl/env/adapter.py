"""The meta-RL interface over the Fermentation-PPO environments.

``PlantFermentationEnv`` is Fermentation-PPO's ``FermentationEnv`` (Safe PPO's
environment: Lagrangian shaping, soft walls) and ``BenchPlantFermentationEnv``
its ``FermentationEnvBench`` (Standard PPO's: static spikes, stage logic in
the environment), each with ``MetaPlantMixin`` on top. The mixin changes
nothing about the plant, its reward or its constraints; it adds what the
meta-RL side reads:

  * a flat observation, ``get_state_norm()``'s ``(N_REACTORS, 28)`` rows in
    order, and the scalar reward, the sum of the per-reactor rewards;
  * tasks (``set_task``): a task's feed composition, pump gains and leaks are
    installed as the plant's nominal values before every reset, so a fault
    acts relative to them. The nominal task is the Fermentation-PPO plant
    exactly;
  * privileged views for training (``get_effective_context``,
    ``fault_kind``, ``privileged_state``) and the probes and per-reactor
    ledger the evaluation reports.

Violation ids, magnitudes and event counts are the environment's (see
``metarl.env.safe``): 1 / 2 reactor overflow / underflow, 3 / 4 storage
overflow / underflow, 5 numerical instability, 6 mass-balance drift, 7
harvest titer floor, 8 product purity (per lot), 9 plant heat-removal
capacity. ``info["violation_events"]`` is an outer step's event count.
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import (N_REACTORS, I_ETH, SPECIES_IDX,
                               REACTOR_OBS_DIM, PRODUCT_MASS_SCALE,
                               TITER_SCALE, TASK_CONTEXT_DIM,
                               get_main_reactor_config, nominal_task)
from metarl.env.layout import STATE_DIM, ACTION_DIM
from metarl.env.faults import (VALID_DISTURBANCES, ADDITIVE_DISTURBANCES,
                               DISTURBANCE_ALIASES, DISTURBANCE_KIND_INDEX,
                               parse_active_disturbances)
from metarl.env.plant import FermentationPlant
from metarl.env.safe import FermentationEnv
from metarl.env.bench import FermentationEnvBench

# Constraint ids attributed to a reactor in ``reactor_violation_events``.
REACTOR_CONSTRAINTS = (1, 2, 7, 8)


class MetaPlantMixin:
    """Tasks, fault tracking and flat views over a Fermentation-PPO environment.

    Attributes:
        stateDim: Flat observation width (STATE_DIM).
        actionDim: Flat action width (ACTION_DIM).
        fixed_task: Task installed at every reset (None: the nominal task).
        current_task: The task of the running episode.
    """

    def _init_meta(self, active_disturbances):
        """Meta-RL state that must exist before the environment's own __init__."""
        n = N_REACTORS
        self.fixed_task = None
        self.current_task = nominal_task()
        self.active_disturbances = (
            parse_active_disturbances(active_disturbances)
            if active_disturbances is not None else None)
        self._task_gain = np.ones(n)
        self._task_leak = np.zeros(n)
        self._feed_conc_mag = 1.0
        self._kinetic_activity = np.ones(n)
        self._fault_seen = False
        self._fault_kind = 0
        self._step_events = np.zeros(n, dtype=np.int64)
        self._pre_M = np.zeros(n)
        self._pre_titer = np.zeros(n)

    def _finish_meta_init(self, max_steps=None):
        """Flat dimensions; ``max_steps`` shortens the horizon (tests only)."""
        self.stateDim = STATE_DIM
        self.actionDim = ACTION_DIM
        if max_steps is not None:
            self.max_steps = int(max_steps)

    # ------------------------------------------------------------------ tasks

    def set_task(self, task):
        """Force a specific task on every subsequent reset() (None: nominal)."""
        self.fixed_task = task

    def set_active_disturbances(self, selection):
        """Restrict which disturbance kinds random scheduling may draw."""
        self.active_disturbances = (None if selection is None
                                    else parse_active_disturbances(selection))

    def get_active_disturbances(self):
        """Return the active disturbance kinds (all kinds when unrestricted)."""
        return (self.active_disturbances if self.active_disturbances is not None
                else VALID_DISTURBANCES)

    def _install_task(self, task):
        """Make a task's feeds, pump gains and leaks the plant's nominal values."""
        n = self.n_reactors
        self.current_task = task
        self._C_C5_0 = np.tile(np.asarray(task["C_C5"], dtype=np.float64), (n, 1))
        self._C_LF_0 = np.tile(np.asarray(task["C_LF"], dtype=np.float64), (n, 1))
        flow = task.get("flow", {})
        self._task_gain = np.broadcast_to(np.asarray(
            flow.get("feed_gain", 1.0), dtype=np.float64), (n,)).copy()
        self._task_leak = np.broadcast_to(np.asarray(
            flow.get("leak_frac", 0.0), dtype=np.float64), (n,)).copy()

    def _restore_nominal(self):
        """The environment's nominal values, then the task's pump gains."""
        super()._restore_nominal()
        self.feed_gain = self._task_gain.copy()
        self._feed_conc_mag = 1.0
        self._kinetic_activity = np.ones(self.n_reactors)
        self._fault_seen = False
        self._fault_kind = 0

    def reset(self):
        """Install the task, reset the plant; returns the flat observation."""
        self._install_task(self.fixed_task if self.fixed_task is not None
                           else nominal_task())
        super().reset()
        n = self.n_reactors
        self.n_batches_completed = 0
        self.reactor_batches_completed = np.zeros(n, dtype=np.int64)
        self.reactor_on_spec_eth = np.zeros(n)
        self.reactor_lots_passed = np.zeros(n, dtype=np.int64)
        self.reactor_lots_failed = np.zeros(n, dtype=np.int64)
        self.worst_lot_shortfall = 0.0
        self.storage_min_seen = float(self.S_mass)
        self.storage_max_seen = float(self.S_mass)
        return self.observe()

    def observe(self):
        """The flat observation of the current state, shape (STATE_DIM,)."""
        return self.get_state_norm().reshape(-1)

    # ------------------------------------------------------- disturbances

    def schedule_disturbance(self, kind, onset_step, magnitude, reactor=None,
                             end_step=None):
        """``schedule_disturbance`` of the environment, with kind aliases."""
        super().schedule_disturbance(DISTURBANCE_ALIASES.get(kind, kind),
                                     onset_step, magnitude, reactor=reactor,
                                     end_step=end_step)

    def _apply_disturbances(self):
        """The environment's faults, then what they made true for the probes.

        A ``feed_gain`` fault acts on the task's gain (realized = commanded x
        task gain x magnitude; the environment's own nominal gain is 1.0).
        """
        t = self.time_step_count
        firing = []
        for d in self._disturbances:
            if t < d["onset"] or (d["end"] is not None and t >= d["end"]):
                continue
            if d["kind"] in ADDITIVE_DISTURBANCES:
                if d["magnitude"] != 0.0:
                    self._mark_fault(d["kind"])
            elif not d["fired"]:
                firing.append(d)
                if d["magnitude"] != 1.0:
                    self._mark_fault(d["kind"])
        leaked = super()._apply_disturbances()
        for d in firing:
            kind, mag, tgt = d["kind"], d["magnitude"], self._targets(d)
            if kind == "feed_gain":
                self.feed_gain[tgt] = self._task_gain[tgt] * mag
            elif kind == "feed_conc":
                self._feed_conc_mag = float(mag)
            elif kind in ("disturb_kinetics_decay", "cell_viability"):
                self._kinetic_activity[tgt] = float(mag)
            elif kind == "toxin_spike":
                self._kinetic_activity[tgt] = 1.0 / max(1.0, float(mag))
        # The state the step's discharge and harvest titer are drawn from.
        self._pre_M = self.M.copy()
        self._pre_titer = self.C[:, I_ETH].copy()
        return leaked

    def _apply_leak(self, leak_frac_per_hour):
        """The faults' leak plus the task's own."""
        return super()._apply_leak(leak_frac_per_hour + self._task_leak)

    def _mark_fault(self, kind):
        """Record that a fault of ``kind`` acted on the plant this step."""
        if not self._fault_seen:
            self._fault_kind = DISTURBANCE_KIND_INDEX.get(kind, 0)
        self._fault_seen = True

    def get_effective_context(self):
        """Return the task descriptor as it stands now, disturbances included.

        Returns:
            Length-``TASK_CONTEXT_DIM`` float64 array in
            ``config.TASK_CONTEXT_NAMES`` order.
        """
        ctx0 = np.asarray(self.current_task["context"], dtype=np.float64)
        out = np.concatenate([
            [float(ctx0[0]) * self._feed_conc_mag, float(ctx0[1])],
            self.feed_gain, self._current_leak_rate + self._task_leak,
            self._kinetic_activity, self._current_contam_rate,
            [self._downstream_mult, self._turnaround_mult,
             self._storage_cap_mult, self._cooling_cap_mult]])
        assert out.size == TASK_CONTEXT_DIM
        return out

    def _off_task(self):
        """Whether the effective descriptor differs from the task's prior one."""
        ctx0 = np.asarray(self.current_task["context"], dtype=np.float64)
        return bool(np.any(np.abs(self.get_effective_context() - ctx0)
                           > 1e-12 * (1.0 + np.abs(ctx0))))

    def fault_active(self):
        """Whether a scheduled fault holds the plant off its task's values now.

        Privileged (training only): True from the first inner step in which a
        disturbance changed something, while the effective descriptor differs
        from the task's own. A step fault therefore latches; a spike, or each
        off-window of a recurrent fault, clears it (a kill keeps it: the
        descriptor records the knock-down).
        """
        return self._fault_seen and self._off_task()

    def fault_kind(self):
        """The ``DISTURBANCE_INDEX_MAP`` code of the first fault, while active; else 0."""
        return self._fault_kind if self.fault_active() else 0

    def disturbance_probe(self):
        """Current values of the quantities each disturbance kind acts on.

        Returns:
            Dict. Per-reactor arrays: ``mass`` (kg), ``feed_gain`` (x
            commanded), ``kinetic_activity`` (x nominal), broth ``inhibitors``
            (F + ACT + HMF), ``cells`` and ``ethanol`` (g/L). Plant scalars:
            ``feed_conc`` (x the task's feed), ``storage`` (kg),
            ``downstream_rate`` (x S_OUT_RATE), ``turnaround`` (x nominal),
            ``storage_capacity`` (usable volume, kg), ``heat_load`` (plant
            CO2 evolution, kg/h) and ``cooling_capacity`` (its cap, kg/h).
        """
        C = self.C
        return {
            "mass": self.M.astype(float).copy(),
            "feed_gain": self.feed_gain.astype(float).copy(),
            "kinetic_activity": self._kinetic_activity.astype(float).copy(),
            "inhibitors": (C[:, SPECIES_IDX['F']] + C[:, SPECIES_IDX['ACT']]
                           + C[:, SPECIES_IDX['HMF']]).astype(float),
            "cells": C[:, SPECIES_IDX['Cell']].astype(float).copy(),
            "ethanol": C[:, I_ETH].astype(float).copy(),
            "feed_conc": float(self._feed_conc_mag),
            "storage": float(self.S_mass),
            "downstream_rate": float(self._downstream_mult),
            "turnaround": float(self._turnaround_mult),
            "storage_capacity": float(self.S_max * self._storage_cap_mult),
            "heat_load": float(np.sum(self.heat_load())),
            "cooling_capacity": float(self.max_heat_load * self._cooling_cap_mult),
        }

    # -------------------------------------------------------- views

    @staticmethod
    def get_action_mask(state_tensor):
        """The environment's stage mask on flat observations.

        Args:
            state_tensor: Tensor [..., >= STATE_DIM] of flat observations.

        Returns:
            Torch tensor mask [..., ACTION_DIM].
        """
        st = state_tensor[..., :STATE_DIM]
        lead = st.shape[:-1]
        mask = FermentationPlant.get_action_mask(
            st.reshape(*lead, N_REACTORS, REACTOR_OBS_DIM))
        return mask.reshape(*lead, ACTION_DIM)

    def privileged_state(self):
        """The plant's hidden state and clock, for training-time critics only.

        The observation, then each reactor's lot in progress (mass and mean
        titer, which the observation does not carry), then the elapsed share
        of the episode. Nothing that acts reads it.

        Returns:
            Float32 array of shape (PRIVILEGED_STATE_DIM,), finite.
        """
        s = np.concatenate([self.observe().astype(np.float64),
                            self.lot_mass / PRODUCT_MASS_SCALE,
                            self.lot_mean_titer / TITER_SCALE,
                            [self.time_step_count / max(1, self.max_steps)]])
        return np.nan_to_num(s, nan=0.0, posinf=5.0, neginf=0.0).astype(np.float32)

    @property
    def mean_product_titer(self):
        """Mass-weighted mean ethanol titer of everything discharged so far."""
        if self.product_mass <= 0.0:
            return 0.0
        return float(self.product_eth / self.product_mass)

    # ---------------------------------------------- per-reactor ledger hooks

    def _reset_single_reactor(self, i):
        super()._reset_single_reactor(i)
        self.n_batches_completed += 1
        self.reactor_batches_completed[i] += 1

    def _close_lot(self, i):
        eth = float(self.lot_eth[i])
        lot = super()._close_lot(i)
        if lot is not None:
            if lot["passed"]:
                self.reactor_on_spec_eth[i] += eth
                self.reactor_lots_passed[i] += 1
            else:
                self.reactor_lots_failed[i] += 1
                self.worst_lot_shortfall = max(self.worst_lot_shortfall,
                                               lot["shortfall"])
                self._step_events[i] += 1
        return lot

    def _step_internal(self, action, is_first=True):
        """The environment's inner step, then its reactor-local events."""
        state, reward, done, info = super()._step_internal(action, is_first)
        a = info["actual_a_reshaped"]
        draining = (a[:, 3] > 0.0) & (self._pre_M > 50.0)
        self._step_events += (
            (self.M / self.Mmax - 1.0 > 0.0).astype(np.int64)
            + ((self.DRY_FLOOR - self.M) / self.DRY_FLOOR > 0.0).astype(np.int64)
            + (draining & (self._pre_titer < self.min_harvest_titer)).astype(np.int64))
        self.storage_min_seen = min(self.storage_min_seen, float(self.S_mass))
        self.storage_max_seen = max(self.storage_max_seen, float(self.S_mass))
        return state, reward, done, info

    def step(self, action):
        """One outer step of the environment on a flat action.

        Returns:
            Tuple (flat observation, scalar reward, done, info): the
            environment's info plus ``reactor_violation_events`` (ids 1, 2, 7
            and 8 by reactor), ``actual_action`` (the first inner step's
            executed action in [0, 1], flat), the per-reactor ledger, and at
            the episode's end ``purity_ok``.
        """
        self._step_events = np.zeros(self.n_reactors, dtype=np.int64)
        state, reward_arr, done, info = super().step(
            np.asarray(action, dtype=np.float64).reshape(-1))
        info["reward_per_reactor"] = np.asarray(reward_arr, dtype=float).copy()
        info["reactor_violation_events"] = self._step_events.copy()
        info["actual_action"] = np.asarray(info["actual_a_reshaped"]).reshape(-1)
        info["on_spec_ethanol"] = float(self.on_spec_eth)
        info["reactor_on_spec_ethanol"] = self.reactor_on_spec_eth.copy()
        info["reactor_lots_passed"] = self.reactor_lots_passed.copy()
        info["reactor_lots_failed"] = self.reactor_lots_failed.copy()
        info["storage_mass"] = float(self.S_mass)
        info["stage"] = self.current_stage.copy()
        info["n_batches_completed"] = int(self.n_batches_completed)
        if done:
            info["purity_ok"] = bool(self.product_mass > 0.0
                                     and self.lots_failed == 0)
            info["purity_shortfall"] = float(self.worst_lot_shortfall)
        return (np.asarray(state, dtype=np.float32).reshape(-1),
                float(np.sum(reward_arr)), done, info)


class PlantFermentationEnv(MetaPlantMixin, FermentationEnv):
    """Safe PPO's environment (``FermentationEnv``) with the meta-RL interface."""

    def __init__(self, control_action_interval=5, active_disturbances=None,
                 max_steps=None):
        self._init_meta(active_disturbances)
        hw = get_main_reactor_config()
        super().__init__(n_reactors=N_REACTORS, Mmax=hw['Mmax'],
                         max_flow_LF=hw['max_flow_LF'],
                         max_flow_C5=hw['max_flow_C5'],
                         control_action_interval=control_action_interval)
        self._finish_meta_init(max_steps)

    @property
    def use_lagrangian(self):
        """Whether the adaptive multipliers price violations.

        False is Fermentation-PPO's no-Lagrangian case study: every
        multiplier at zero and frozen, fixed per-violation penalties instead.
        """
        return bool(self.lagrange_updates_enabled
                    and not self.fixed_violation_penalties)

    @use_lagrangian.setter
    def use_lagrangian(self, value):
        self.lagrange_updates_enabled = bool(value)
        self.fixed_violation_penalties = not bool(value)
        if not value:
            self._zero_lambdas()


class BenchPlantFermentationEnv(MetaPlantMixin, FermentationEnvBench):
    """Standard PPO's environment (``FermentationEnvBench``) with the meta-RL interface."""

    use_lagrangian = False

    def __init__(self, control_action_interval=5, active_disturbances=None,
                 max_steps=None):
        self._init_meta(active_disturbances)
        hw = get_main_reactor_config()
        super().__init__(n_reactors=N_REACTORS, Mmax=hw['Mmax'],
                         max_flow_LF=hw['max_flow_LF'],
                         max_flow_C5=hw['max_flow_C5'],
                         control_action_interval=control_action_interval)
        self._finish_meta_init(max_steps)
