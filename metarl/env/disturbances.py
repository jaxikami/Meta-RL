"""Mid-episode plant disturbances, ported from the Meta-RL plant.

The fault kinds, temporal patterns and evaluation scenarios of ../Meta-RL
(metarl/env/faults.py, metarl/training/tasks.py and metarl/training/config.py),
adapted to FermentationEnv.  Unlike Meta-RL, the plant is observed exactly:
there is no instrument model and no masked concentrations, so this is not a
POMDP in Meta-RL's sense.  A fault acts on the plant unannounced and shows
only through the state it changes; its parameters are not in the observation.

Kinds (``reactor`` = one reactor, or all of them when scheduled with None):
    leak                    reactor  extra fraction of mass lost per hour (additive)
    feed_gain               reactor  realized feed = commanded x magnitude
    feed_conc               plant    the shared feed tanks' composition x magnitude
    disturb_kinetics_decay  reactor  hydrolysis and uptake rate modifiers x magnitude
    toxin_spike             reactor  feed-line inhibitors x magnitude, an acute broth
                                     pulse, uptake x 1 / magnitude
    cell_viability          reactor  viable cells x magnitude (once)
    contamination           reactor  fraction of broth ethanol oxidised per hour
                                     (additive)
    downstream_rate         plant    the storage tank's downstream draw x magnitude
    turnaround_delay        plant    every cleanup and re-charge that starts after
                                     onset x magnitude long
    storage_capacity        plant    usable storage volume x magnitude: the overflow
                                     wall moves, the level reading does not
    cooling_capacity        plant    heat-removal capacity x magnitude (a warmer
                                     cooling-water supply): the heat-load cap moves,
                                     nothing in the observation does

Patterns (``pattern_schedule``): ``step`` (full magnitude from onset to the
end), ``spike`` (cleared after a while), ``gradual`` (ramped up in treads) and
``recurrent`` (on and off in windows).
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import C_MAX_LIMITS, SPECIES_IDX

# ─── Registry ─────────────────────────────────────────────────────────────────
VALID_DISTURBANCES = (
    "leak", "feed_gain", "feed_conc", "disturb_kinetics_decay",
    "toxin_spike", "cell_viability", "contamination", "downstream_rate",
    "turnaround_delay", "storage_capacity", "cooling_capacity",
)

# Kinds that act on the plant as a whole and take no reactor: the shared feed
# tanks, the storage tank's downstream draw and volume, the shared turnaround
# equipment and the shared cooling loop.
PLANT_DISTURBANCES = ("feed_conc", "downstream_rate", "turnaround_delay",
                      "storage_capacity", "cooling_capacity")

# Kinds that act continuously from onset (until their end step) and ADD across
# entries.  Every other kind fires once at onset, relative to the nominal
# value, so an entry of magnitude 1.0 restores it.
ADDITIVE_DISTURBANCES = ("leak", "contamination")

FAULT_PATTERNS = ("step", "spike", "gradual", "recurrent")

# A gradual fault reaches full magnitude in this many treads.
DRIFT_TREADS = 6

# ─── Evaluation Scenarios ─────────────────────────────────────────────────────
# Meta-RL's evaluation magnitudes (calibrated in its lean study), plus
# contamination, which Meta-RL defines but does not evaluate: at 0.2% of the
# broth ethanol per hour it takes ~20% off a 120 h ferment's titer, so it
# tests the titer and purity constraints directly; and cooling_capacity, this
# plant's own, which moves the heat-load cap the way storage_capacity moves the
# storage wall.  A per-reactor fault
# ROTATES with the episode index (episode k faults reactor k % n_reactors), so
# every reactor is faulted on an equal share of the paired episodes.
EVAL_DISTURBANCES = {
    "leak": {"kind": "leak", "magnitude": 0.010, "reactor": "rotate"},
    "feed_gain": {"kind": "feed_gain", "magnitude": 1.30, "reactor": "rotate"},
    # Both feed tanks' composition at 55%: a dilute feedstock lot.
    "dilution": {"kind": "feed_conc", "magnitude": 0.55, "reactor": None},
    "kinetics_decay": {"kind": "disturb_kinetics_decay", "magnitude": 0.20,
                       "reactor": "rotate"},
    "toxin": {"kind": "toxin_spike", "magnitude": 6.50, "reactor": "rotate"},
    "cell_viability": {"kind": "cell_viability", "magnitude": 0.50,
                       "reactor": "rotate"},
    "contamination": {"kind": "contamination", "magnitude": 0.002,
                      "reactor": "rotate"},
    # Downstream distillation at half rate: the tank fills 0.75 t/h faster.
    "downstream_rate": {"kind": "downstream_rate", "magnitude": 0.50,
                        "reactor": None},
    # Every turnaround (cleanup + charge, ~12 h) 4x as long.
    "turnaround_delay": {"kind": "turnaround_delay", "magnitude": 4.0,
                         "reactor": None},
    # The tank holds 90% of S_MAX (360 t).
    "storage_capacity": {"kind": "storage_capacity", "magnitude": 0.90,
                         "reactor": None},
    # The cooling loop removes 80% of its rated heat load (432 kg CO2/h).
    "cooling_capacity": {"kind": "cooling_capacity", "magnitude": 0.80,
                         "reactor": None},
}

# Fixed evaluation timings, in inner steps (hours of the 600 h episode): step,
# spike and gradual start at mid-episode; the spike clears after EVAL_SPIKE_H,
# the ramp reaches full strength after EVAL_GRADUAL_H; the recurrent fault is
# on in each window and off between.
EVAL_ONSET = 300
EVAL_SPIKE_H = 24
EVAL_GRADUAL_H = 180
EVAL_RECURRENT_WINDOWS = ((150, 230), (310, 390), (470, 550))


def pattern_schedule(kind, magnitude, pattern, onset=EVAL_ONSET, reactor=None,
                     horizon=600, spike_h=EVAL_SPIKE_H, gradual_h=EVAL_GRADUAL_H,
                     treads=DRIFT_TREADS, windows=EVAL_RECURRENT_WINDOWS):
    """One fault kind under one temporal pattern, as a list of fault entries.

    A kind that fires once is cleared by an entry of magnitude 1.0
    (cell_viability's kill is not undone: the cells regrow), an additive kind
    by its ``end``.  A ramp of an additive kind adds ``magnitude / treads`` per
    tread, of cell_viability multiplies by ``magnitude ** (1 / treads)``, of
    any other kind steps linearly from 1.0.

    Args:
        kind: One of VALID_DISTURBANCES.
        magnitude: Full-strength magnitude, as schedule_disturbance() reads it.
        pattern: One of FAULT_PATTERNS.
        onset: First inner step (step / spike / gradual).
        reactor: Reactor index, or None for every reactor / a plant-wide kind.
        horizon: Episode length in inner steps; entries past it are dropped.
        spike_h: Length of a spike, in inner steps.
        gradual_h: Length of a ramp, in inner steps.
        treads: Number of treads of a ramp.
        windows: (start, end) inner steps of each recurrent on-window.

    Returns:
        List of dicts with keys kind / onset / magnitude / reactor / end.

    Raises:
        ValueError: On an unknown pattern.
    """
    additive = kind in ADDITIVE_DISTURBANCES
    n = max(1, int(treads))

    def entry(t, mag, end=None):
        end = None if end is None or end >= horizon else int(end)
        return {"kind": kind, "onset": int(t), "magnitude": float(mag),
                "reactor": reactor, "end": end}

    def pulse(t0, t1):
        return ([entry(t0, magnitude, t1)] if additive
                else [entry(t0, magnitude), entry(t1, 1.0)])

    if pattern == "step":
        out = [entry(onset, magnitude)]
    elif pattern == "spike":
        out = pulse(onset, onset + int(spike_h))
    elif pattern == "gradual":
        times = [onset + int(round(k * gradual_h / max(n - 1, 1))) for k in range(n)]
        if additive:
            mags = [magnitude / n] * n
        elif kind == "cell_viability":
            mags = [magnitude ** (1.0 / n)] * n
        else:
            mags = [1.0 + (magnitude - 1.0) * (k + 1) / n for k in range(n)]
        out = [entry(t, m) for t, m in zip(times, mags)]
    elif pattern == "recurrent":
        out = [e for t0, t1 in windows for e in pulse(t0, t1)]
    else:
        raise ValueError(f"unknown fault pattern {pattern!r}; expected one of {FAULT_PATTERNS}")
    return [e for e in out if 1 <= e["onset"] <= horizon - 1]


def eval_schedule(label, pattern, index=0, n_reactors=3, horizon=600):
    """The fault entries of evaluation scenario ``label`` under ``pattern``.

    Args:
        label: Key of EVAL_DISTURBANCES.
        pattern: One of FAULT_PATTERNS.
        index: Episode index within the scenario (picks the rotated reactor).
        n_reactors: Number of reactors to rotate over.
        horizon: Episode length in inner steps.

    Returns:
        List of fault entries for apply_schedule().
    """
    spec = EVAL_DISTURBANCES[label]
    reactor = spec["reactor"]
    if spec["kind"] in PLANT_DISTURBANCES or reactor is None:
        reactor = None
    elif reactor == "rotate":
        reactor = int(index) % n_reactors
    return pattern_schedule(spec["kind"], spec["magnitude"], pattern,
                            reactor=reactor, horizon=horizon)


def apply_schedule(env, schedule):
    """Clear ``env``'s faults, then schedule every entry of ``schedule``."""
    env.clear_disturbances()
    for e in schedule:
        env.schedule_disturbance(e["kind"], e["onset"], e["magnitude"],
                                 reactor=e["reactor"], end_step=e["end"])


def _toxin_pulse(mag, cap, per_unit):
    """Acute broth pulse (g/L) of a toxin step from clean feed to ``mag``."""
    return min(cap, max(0.0, mag - 1.0) * per_unit)


class DisturbanceMixin:
    """Mid-episode faults of FermentationEnv.

    The host provides n_reactors, params [n, 3], C_C5 / C_LF [n, 15], C, M,
    DRY_FLOOR, time_step_count, time_step_duration and
    episode_mass_discharged, calls _init_disturbances() once its nominal
    values are set, _rearm_disturbances() in reset(), and
    _apply_disturbances() at the start of every inner step.  A schedule
    survives reset(): it replays identically every episode until cleared.
    """

    def _init_disturbances(self):
        """Capture the nominal values faults act on, with no fault scheduled."""
        self._params0 = np.ascontiguousarray(self.params, dtype=np.float64).copy()
        self._C_C5_0 = self.C_C5.copy()
        self._C_LF_0 = self.C_LF.copy()
        self._disturbances = []
        self._restore_nominal()

    def _restore_nominal(self):
        """Put every quantity a fault acts on back at its nominal value."""
        n = self.n_reactors
        self.params = self._params0.copy()
        self.C_C5 = self._C_C5_0.copy()
        self.C_LF = self._C_LF_0.copy()
        self.feed_gain = np.ones(n)
        self._current_toxin_mag = np.ones(n)
        self._current_leak_rate = np.zeros(n)
        self._current_contam_rate = np.zeros(n)
        self._downstream_mult = 1.0
        self._turnaround_mult = 1.0
        self._storage_cap_mult = 1.0
        self._cooling_cap_mult = 1.0

    def _rearm_disturbances(self):
        """Restore nominal values and re-arm every scheduled fault (reset())."""
        self._restore_nominal()
        for d in self._disturbances:
            d["fired"] = False

    def schedule_disturbance(self, kind, onset_step, magnitude, reactor=None,
                             end_step=None):
        """Register a mid-episode disturbance.

        Args:
            kind: One of VALID_DISTURBANCES (see the module docstring).
            onset_step: Inner-step index at which the disturbance begins.
            magnitude: Disturbance size.
            reactor: Reactor index the fault acts on, or None for every
                reactor.  Must be None for PLANT_DISTURBANCES.
            end_step: ADDITIVE_DISTURBANCES only: first inner step at which
                the entry no longer acts (None: to the end of the episode).

        Raises:
            ValueError: On an unknown kind, a reactor index out of range, a
                reactor given for a plant-wide kind, or an ``end_step`` on a
                kind that fires once or not after its onset.
        """
        if kind not in VALID_DISTURBANCES:
            raise ValueError(f"unknown disturbance kind {kind!r}; expected one of "
                             f"{VALID_DISTURBANCES}")
        if reactor is not None:
            if kind in PLANT_DISTURBANCES:
                raise ValueError(f"{kind} acts on the whole plant; schedule it with reactor=None")
            reactor = int(reactor)
            if not 0 <= reactor < self.n_reactors:
                raise ValueError(f"reactor {reactor} out of range 0..{self.n_reactors - 1}")
        if end_step is not None:
            if kind not in ADDITIVE_DISTURBANCES:
                raise ValueError(f"{kind} fires once: end it with a later entry of "
                                 f"magnitude 1.0, not end_step")
            if int(end_step) <= int(onset_step):
                raise ValueError(f"end_step {end_step} is not after onset {onset_step}")
            end_step = int(end_step)
        self._disturbances.append({
            "kind": kind, "onset": int(onset_step), "magnitude": float(magnitude),
            "reactor": reactor, "end": end_step, "fired": False,
        })

    def clear_disturbances(self):
        """Remove all scheduled disturbances and undo any that already fired."""
        self._disturbances = []
        self._restore_nominal()

    def _targets(self, d):
        """Reactor indices a disturbance entry acts on."""
        if d["reactor"] is None:
            return np.arange(self.n_reactors)
        return np.array([d["reactor"]], dtype=np.intp)

    def _apply_disturbances(self):
        """Apply any due disturbances at the current inner step.

        Returns:
            Mass removed by leakage this step per reactor, kg, shape [n].
        """
        leak_rate = np.zeros(self.n_reactors)
        contam_rate = np.zeros(self.n_reactors)
        t = self.time_step_count
        for d in self._disturbances:
            if t < d["onset"] or (d["end"] is not None and t >= d["end"]):
                continue
            kind, mag = d["kind"], d["magnitude"]
            tgt = self._targets(d)
            if kind == "leak":
                leak_rate[tgt] += mag
                continue
            if kind == "contamination":
                contam_rate[tgt] += mag
                continue
            if d["fired"]:
                continue
            if kind == "feed_conc":
                self.C_C5 = self._C_C5_0 * mag
                self.C_LF = self._C_LF_0 * mag
            elif kind == "downstream_rate":
                self._downstream_mult = float(mag)
            elif kind == "turnaround_delay":
                self._turnaround_mult = float(mag)
            elif kind == "storage_capacity":
                self._storage_cap_mult = float(mag)
            elif kind == "cooling_capacity":
                self._cooling_cap_mult = float(mag)
            elif kind == "feed_gain":
                self.feed_gain[tgt] = mag
            elif kind == "disturb_kinetics_decay":
                self.params[tgt, :] = self._params0[tgt, :] * mag
            elif kind == "toxin_spike":
                inh = [SPECIES_IDX['F'], SPECIES_IDX['ACT'], SPECIES_IDX['HMF']]
                i_F, i_ACT, i_HMF = inh
                for i in tgt:
                    self.C_C5[i, inh] = self._C_C5_0[i, inh] * mag
                    self.C_LF[i, inh] = self._C_LF_0[i, inh] * mag
                    # Acute broth pulse, capped below C_MAX_LIMITS, sized by
                    # the RISE in feed toxin: a ramp's pulses add up to one
                    # step's, and each recurrence (after a 1.0 entry restores
                    # the feed) pulses anew.
                    prev = float(self._current_toxin_mag[i])
                    dFH = _toxin_pulse(mag, 4.5, 1.5) - _toxin_pulse(prev, 4.5, 1.5)
                    dA = _toxin_pulse(mag, 0.85, 0.35) - _toxin_pulse(prev, 0.85, 0.35)
                    if dFH > 0.0:
                        self.C[i, i_F] = min(float(C_MAX_LIMITS[i_F]), self.C[i, i_F] + dFH)
                        self.C[i, i_HMF] = min(float(C_MAX_LIMITS[i_HMF]), self.C[i, i_HMF] + dFH)
                    if dA > 0.0:
                        self.C[i, i_ACT] = min(float(C_MAX_LIMITS[i_ACT]), self.C[i, i_ACT] + dA)
                    self._current_toxin_mag[i] = mag
                # The lasting harm is the uptake cut.
                act = 1.0 / max(1.0, mag)
                self.params[tgt, 1:3] = self._params0[tgt, 1:3] * act
            elif kind == "cell_viability":
                i_Cell = SPECIES_IDX['Cell']
                self.C[tgt, i_Cell] = np.maximum(0.1, self.C[tgt, i_Cell] * mag)
            d["fired"] = True

        self._current_leak_rate = leak_rate
        self._current_contam_rate = contam_rate
        self._apply_contamination(contam_rate)
        return self._apply_leak(leak_rate)

    def _apply_contamination(self, frac_per_hour):
        """Remove the broth ethanol a bacterial contaminant oxidises this step.

        Reactor mass is left alone (metarl/env/kinetics does not debit the yeast's own
        CO2 either), so the mass balance holds.
        """
        if not np.any(frac_per_hour > 0.0):
            return
        dt_hours = self.time_step_duration / 3600.0
        self.C[:, SPECIES_IDX['Eth']] *= np.maximum(0.0, 1.0 - frac_per_hour * dt_hours)

    def _apply_leak(self, leak_frac_per_hour):
        """Remove leaked mass, never below DRY_FLOOR, crediting exactly that.

        Returns:
            Mass removed this step per reactor, kg.
        """
        if not np.any(leak_frac_per_hour > 0.0):
            return np.zeros(self.n_reactors)
        dt_hours = self.time_step_duration / 3600.0
        want = self.M * np.maximum(leak_frac_per_hour, 0.0) * dt_hours
        delta = np.minimum(want, np.maximum(0.0, self.M - self.DRY_FLOOR))
        self.M = self.M - delta
        self.episode_mass_discharged += float(delta.sum())
        return delta
