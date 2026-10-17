"""Mid-episode disturbances: the fault registry and ``FaultMixin``.

``FaultMixin`` holds the ``PlantFermentationEnv`` methods that install a task,
schedule faults and apply them at every inner step.
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import (SPECIES_IDX, C_MAX_LIMITS, I_ETH,
                               TASK_CONTEXT_DIM)

# ─── Disturbances ─────────────────────────────────────────────────────────────
VALID_DISTURBANCES = (
    "leak", "feed_gain", "feed_conc", "disturb_kinetics_decay",
    "toxin_spike", "cell_viability", "contamination", "downstream_rate",
)

DISTURBANCE_INDEX_MAP = {
    1: "leak",
    2: "feed_gain",
    3: "feed_conc",
    4: "disturb_kinetics_decay",
    5: "toxin_spike",
    6: "cell_viability",
    7: "contamination",
    8: "downstream_rate",
}

DISTURBANCE_ALIASES = {
    "kinetics_decay": "disturb_kinetics_decay",
    "disturb_toxin_spike": "toxin_spike",
    "disturb_cell_viability": "cell_viability",
    "disturb_contamination": "contamination",
    "disturb_downstream_rate": "downstream_rate",
}

# Each kind's code in DISTURBANCE_INDEX_MAP, as fault_kind() reports it (0 is
# "no fault has acted").
DISTURBANCE_KIND_INDEX = {k: i for i, k in DISTURBANCE_INDEX_MAP.items()}
N_DISTURBANCE_KINDS = len(DISTURBANCE_INDEX_MAP)

# Kinds that act on the plant as a whole and take no reactor: the shared feed
# tanks and the storage tank's downstream draw. Every other kind acts on one
# reactor, or on all of them when scheduled with ``reactor=None``.
PLANT_DISTURBANCES = ("feed_conc", "downstream_rate")

# Kinds that act continuously from onset and ADD across entries.
ADDITIVE_DISTURBANCES = ("leak", "contamination")


def parse_active_disturbances(selection):
    """Parse a disturbance selection into a tuple of disturbance kind strings.

    Accepted forms:
        - None or 'all': every kind in VALID_DISTURBANCES.
        - Integers, e.g. [1, 3] -> ("leak", "feed_conc"), 1-based per
          DISTURBANCE_INDEX_MAP; the whole list is read 0-based if it
          contains a 0.
        - Kind names or aliases.
        - Presets: 'flow', 'kinetics', 'biology' / 'biochemical', 'storage',
          'none'.

    Args:
        selection: Ints, strings, or preset name.

    Returns:
        Tuple of disturbance kind strings, in selection order, deduplicated.

    Raises:
        ValueError: On an unknown index or kind name.
        TypeError: On an element that is neither int nor str.
    """
    if selection is None or selection == "all":
        return VALID_DISTURBANCES
    if selection == "none":
        return ()
    if selection == "flow":
        return ("leak", "feed_gain", "feed_conc")
    if selection == "storage":
        return ("downstream_rate",)
    if selection in ("kinetics", "disturb_kinetics_decay", "kinetics_decay"):
        return ("disturb_kinetics_decay",)
    if selection in ("biology", "biochemical"):
        return ("disturb_kinetics_decay", "toxin_spike", "cell_viability",
                "contamination")

    if isinstance(selection, (int, str)):
        selection = [selection]

    use_zero_based = any(isinstance(x, int) and x == 0 for x in selection)

    active = []
    for item in selection:
        if isinstance(item, int):
            idx = item if not use_zero_based else item + 1
            if idx in DISTURBANCE_INDEX_MAP:
                k = DISTURBANCE_INDEX_MAP[idx]
                if k not in active:
                    active.append(k)
            else:
                raise ValueError(
                    f"invalid disturbance index {item}; expected 1..{len(DISTURBANCE_INDEX_MAP)}")
        elif isinstance(item, str):
            k = DISTURBANCE_ALIASES.get(item, item)
            if k in VALID_DISTURBANCES:
                if k not in active:
                    active.append(k)
            else:
                raise ValueError(
                    f"unknown disturbance {item!r}; expected one of {VALID_DISTURBANCES}")
        else:
            raise TypeError(f"unsupported disturbance selector type: {type(item)}")
    return tuple(active)


class FaultMixin:
    """Task installation and mid-episode faults of ``PlantFermentationEnv``."""

    def _apply_task(self, task):
        """Install a task into the physics, caching its undisturbed values.

        Args:
            task: Task dict from config.nominal_task() / sample_task().
        """
        n = self.n_reactors
        self.params = np.ascontiguousarray(np.tile(
            np.asarray(task["params"], dtype=np.float64).reshape(1, -1), (n, 1)))
        self.C_C5 = np.tile(np.asarray(task["C_C5"], dtype=np.float64), (n, 1))
        self.C_LF = np.tile(np.asarray(task["C_LF"], dtype=np.float64), (n, 1))
        flow = task.get("flow", {})
        self.feed_gain = np.broadcast_to(np.asarray(
            flow.get("feed_gain", 1.0), dtype=np.float64), (n,)).copy()
        self.leak_frac = np.broadcast_to(np.asarray(
            flow.get("leak_frac", 0.0), dtype=np.float64), (n,)).copy()
        self.current_task = task
        self._task_params0 = self.params.copy()
        self._task_C_C5_0 = self.C_C5.copy()
        self._task_C_LF_0 = self.C_LF.copy()
        self._task_feed_gain0 = self.feed_gain.copy()
        self._task_leak_frac0 = self.leak_frac.copy()
        self._reset_fault_state()

    def _reset_fault_state(self):
        """What fired disturbances have made true, back at the task's values."""
        n = self.n_reactors
        self._current_feed_conc_mag = 1.0
        self._current_leak_rate = self.leak_frac.copy()
        self._current_kinetic_activity = np.ones(n)
        self._current_contam_rate = np.zeros(n)
        self._downstream_mult = 1.0

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

    # ------------------------------------------------------- disturbances

    def schedule_disturbance(self, kind, onset_step, magnitude, reactor=None,
                             channel=None):
        """Register a mid-episode disturbance.

        ``leak`` and ``contamination`` act continuously from onset and ADD to
        each other (and a leak to the task's baseline). Every other kind fires
        once at onset, relative to the task's own values, so a later entry of
        the same kind on the same reactor overwrites an earlier one.

        Args:
            kind: One of VALID_DISTURBANCES (or an alias):
                'leak'      -- fraction of current mass lost per hour, never
                               below DRY_FLOOR; credited to the mass ledger.
                'feed_gain' -- realized feed = commanded x magnitude x task
                               gain on the reactor's pumps.
                'feed_conc' -- the shared feed tanks' composition x magnitude
                               at unchanged mass flow (plant-wide).
                'disturb_kinetics_decay' -- hydrolysis and/or uptake rate
                               modifiers x magnitude (see ``channel``).
                'toxin_spike' -- the reactor's feed-line inhibitors x
                               magnitude, an acute broth pulse when > 1, and
                               uptake slowed by 1 / magnitude for the rest of
                               the episode.
                'cell_viability' -- viable cell mass x magnitude.
                'contamination' -- bacteria oxidise this fraction of the
                               reactor's broth ethanol per hour, to the end of
                               the episode; their CO2 reaches its off-gas.
                'downstream_rate' -- the storage tank's downstream draw x
                               magnitude (plant-wide): < 1 is a distillation
                               slowdown that fills the tank, > 1 a surge that
                               empties it.
            onset_step: Inner-step index at which the disturbance begins.
            magnitude: Disturbance size (see per-kind semantics above).
            reactor: Reactor index the fault acts on, or None for every
                reactor. Must be None for PLANT_DISTURBANCES.
            channel: ``disturb_kinetics_decay`` only: 'eta' (hydrolysis),
                'uptake' / 'ph' (sugar uptake), or None for both.

        Raises:
            ValueError: On an unknown kind, a reactor index out of range, or a
                reactor given for a plant-wide kind.
        """
        kind = DISTURBANCE_ALIASES.get(kind, kind)
        if kind not in VALID_DISTURBANCES:
            raise ValueError(f"unknown disturbance kind {kind!r}; expected one "
                             f"of {VALID_DISTURBANCES}")
        if reactor is not None:
            if kind in PLANT_DISTURBANCES:
                raise ValueError(f"{kind} acts on the whole plant; schedule it "
                                 f"with reactor=None")
            reactor = int(reactor)
            if not 0 <= reactor < self.n_reactors:
                raise ValueError(f"reactor {reactor} out of range "
                                 f"0..{self.n_reactors - 1}")
        self._disturbances.append({
            "kind": kind, "onset": int(onset_step),
            "magnitude": float(magnitude), "reactor": reactor,
            "channel": channel, "fired": False,
        })

    def clear_disturbances(self):
        """Remove all scheduled disturbances and undo any that already fired."""
        self._disturbances = []
        self._fault_seen = False
        self._fault_kind = 0
        if hasattr(self, "_task_params0"):
            self.params = self._task_params0.copy()
            self.C_C5 = self._task_C_C5_0.copy()
            self.C_LF = self._task_C_LF_0.copy()
            self.feed_gain = self._task_feed_gain0.copy()
            self.leak_frac = self._task_leak_frac0.copy()
            self._reset_fault_state()

    def disturbance_probe(self):
        """Current values of the quantities each disturbance kind acts on.

        Returns:
            Dict. Per-reactor arrays (length ``n_reactors``): ``mass`` (kg),
            ``feed_gain`` (x commanded), ``kinetic_activity`` (x nominal),
            broth ``inhibitors`` (F + ACT + HMF), ``cells`` and ``ethanol``
            (g/L). Plant scalars: ``feed_conc`` (x the task's feed),
            ``storage`` (kg) and ``downstream_rate`` (x S_OUT_RATE).
        """
        C = self.C
        return {
            "mass": self.M.astype(float).copy(),
            "feed_gain": self.feed_gain.astype(float).copy(),
            "kinetic_activity": self._current_kinetic_activity.astype(float).copy(),
            "inhibitors": (C[:, SPECIES_IDX['F']] + C[:, SPECIES_IDX['ACT']]
                           + C[:, SPECIES_IDX['HMF']]).astype(float),
            "cells": C[:, SPECIES_IDX['Cell']].astype(float).copy(),
            "ethanol": C[:, I_ETH].astype(float).copy(),
            "feed_conc": float(self._current_feed_conc_mag),
            "storage": float(self.S_mass),
            "downstream_rate": float(self._downstream_mult),
        }

    def fault_active(self):
        """Whether a scheduled fault has acted on the plant this episode.

        Privileged (training only), like ``get_effective_context``: True from
        the first inner step in which a disturbance changed something, for the
        rest of the episode.
        """
        return self._fault_seen

    def fault_kind(self):
        """Which kind of fault acted first this episode, as a code.

        0 until ``fault_active`` turns on, then the ``DISTURBANCE_INDEX_MAP``
        code of the kind that turned it on. Privileged like ``fault_active``.

        Returns:
            int in [0, N_DISTURBANCE_KINDS].
        """
        return self._fault_kind

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
            [float(ctx0[0]) * self._current_feed_conc_mag, float(ctx0[1])],
            self.feed_gain, self._current_leak_rate,
            self._current_kinetic_activity, self._current_contam_rate,
            [self._downstream_mult]])
        assert out.size == TASK_CONTEXT_DIM
        return out

    def _targets(self, d):
        """Reactor indices a disturbance entry acts on."""
        if d["reactor"] is None:
            return np.arange(self.n_reactors)
        return np.array([d["reactor"]], dtype=np.intp)

    def _apply_disturbances(self):
        """Apply any due mid-episode disturbances at the current inner step.

        Returns:
            Mass removed by leakage this step per reactor, kg, shape (n,).
        """
        leak_rate = self.leak_frac.copy()
        contam_rate = np.zeros(self.n_reactors)
        t = self.time_step_count
        for d in self._disturbances:
            if t < d["onset"]:
                continue
            kind, mag = d["kind"], d["magnitude"]
            tgt = self._targets(d)
            if kind == "leak":
                leak_rate[tgt] += mag
                if mag != 0.0:
                    self._mark_fault(kind)
                continue
            if kind == "contamination":
                contam_rate[tgt] += mag
                if mag != 0.0:
                    self._mark_fault(kind)
                continue
            if d["fired"]:
                continue
            if mag != 1.0:
                self._mark_fault(kind)
            if kind == "feed_conc":
                self.C_C5 = self._task_C_C5_0 * mag
                self.C_LF = self._task_C_LF_0 * mag
                self._current_feed_conc_mag = mag
            elif kind == "downstream_rate":
                self._downstream_mult = float(mag)
            elif kind == "feed_gain":
                self.feed_gain[tgt] = self._task_feed_gain0[tgt] * mag
            elif kind == "disturb_kinetics_decay":
                ch = d.get("channel")
                if ch in (None, "eta"):
                    self.params[tgt, 0] = self._task_params0[tgt, 0] * mag
                if ch in (None, "uptake", "ph"):
                    self.params[tgt, 1:3] = self._task_params0[tgt, 1:3] * mag
                self._current_kinetic_activity[tgt] = float(mag)
            elif kind == "toxin_spike":
                inh = [SPECIES_IDX['F'], SPECIES_IDX['ACT'], SPECIES_IDX['HMF']]
                for i in tgt:
                    self.C_C5[i, inh] = self._task_C_C5_0[i, inh] * mag
                    self.C_LF[i, inh] = self._task_C_LF_0[i, inh] * mag
                    if mag > 1.0:
                        # Acute broth pulse at onset, capped below C_MAX_LIMITS.
                        i_F, i_ACT, i_HMF = inh
                        self.C[i, i_F] = min(float(C_MAX_LIMITS[i_F]),
                                             self.C[i, i_F] + min(4.5, (mag - 1.0) * 1.5))
                        self.C[i, i_HMF] = min(float(C_MAX_LIMITS[i_HMF]),
                                               self.C[i, i_HMF] + min(4.5, (mag - 1.0) * 1.5))
                        self.C[i, i_ACT] = min(float(C_MAX_LIMITS[i_ACT]),
                                               self.C[i, i_ACT] + min(0.85, (mag - 1.0) * 0.35))
                # The lasting harm is the uptake cut (the lean study's
                # 1 / magnitude since 2026-09-26).
                act = float(1.0 / max(1.0, mag))
                self._current_kinetic_activity[tgt] = act
                self.params[tgt, 1:3] = self._task_params0[tgt, 1:3] * act
            elif kind == "cell_viability":
                i_Cell = SPECIES_IDX['Cell']
                self.C[tgt, i_Cell] = np.maximum(0.1, self.C[tgt, i_Cell] * mag)
                self._current_kinetic_activity[tgt] = float(mag)
            d["fired"] = True

        self._current_leak_rate = leak_rate
        self._current_contam_rate = contam_rate
        self._apply_contamination(contam_rate)
        return self._apply_leak(leak_rate)

    def _apply_contamination(self, frac_per_hour):
        """Remove the broth ethanol a bacterial contaminant oxidises this step.

        Ethanol already discharged is safe; reactor mass is left alone (env.core
        does not debit the yeast's own CO2 either), so the mass balance holds.

        Args:
            frac_per_hour: Per-reactor fraction of broth ethanol lost per hour.
        """
        if not np.any(frac_per_hour > 0.0):
            return
        dt_hours = self.time_step_duration / 3600.0
        self.C[:, I_ETH] *= np.maximum(0.0, 1.0 - frac_per_hour * dt_hours)

    def _apply_leak(self, leak_frac_per_hour):
        """Remove leaked mass, never below DRY_FLOOR, crediting exactly that.

        Args:
            leak_frac_per_hour: Per-reactor fraction of mass lost per hour.

        Returns:
            Mass removed this step per reactor, kg.
        """
        if not np.any(leak_frac_per_hour > 0.0):
            return np.zeros(self.n_reactors)
        dt_hours = self.time_step_duration / 3600.0
        want = self.M * np.maximum(leak_frac_per_hour, 0.0) * dt_hours
        avail = np.maximum(0.0, self.M - self.DRY_FLOOR)
        delta = np.minimum(want, avail)
        self.M = self.M - delta
        self.episode_mass_discharged += float(delta.sum())
        return delta
