"""Environments, task pools and fault schedules."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import random

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import (sample_task, N_REACTORS, CTX_SOLIDS, CTX_GX,
                               CTX_GAIN)
from metarl.env.faults import (PLANT_DISTURBANCES, DISTURBANCE_ALIASES,
                               ADDITIVE_DISTURBANCES,
                               parse_active_disturbances)
from metarl.env.plant import PlantFermentationEnv
from metarl.training.config import (HELDOUT_CORNER_TASKS, HELDOUT_CORNER_SEED,
                                    HELDOUT_CORNER_CANDIDATES,
                                    RESET_LAMBDAS_PER_EPISODE,
                                    META_DISTURB_PROB, META_DISTURB_KINDS,
                                    META_PLANTWIDE_FRAC,
                                    META_DISTURB_ONSET_FRAC,
                                    META_DISTURB_MAG_RANGES,
                                    META_DISTURB_STRATA,
                                    META_DISTURB_TAIL_WEIGHT,
                                    META_DRIFT_TREADS, META_COMPOUND_PAIRS,
                                    META_COMPOUND_PAIR_FRAC, FAULT_PATTERNS,
                                    META_SPIKE_H, META_GRADUAL_FRAC,
                                    META_RECURRENT_ON_H, META_RECURRENT_OFF_H,
                                    META_RECURRENT_MAX)

ACTIVE_DISTURBANCES = None


def set_active_disturbances(selection):
    """Restrict which kinds meta-training disturbance draws may use."""
    global ACTIVE_DISTURBANCES
    ACTIVE_DISTURBANCES = (None if selection is None
                           else parse_active_disturbances(selection))
    return ACTIVE_DISTURBANCES


def seed_everything(seed):
    """Seed Python, NumPy and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(lagrangian=True, sensor_model=None):
    """Construct the plant with the standard configuration.

    Args:
        lagrangian: Enable the adaptive dual multipliers.
        sensor_model: Instrument noise; None means SENSOR_MODEL_ENABLED (what
            evaluation uses). Training passes the arm's train_sensor_noise.

    Returns:
        A configured ``PlantFermentationEnv``.
    """
    env = PlantFermentationEnv(sensor_model=sensor_model)
    env.reset_lambdas_per_episode = RESET_LAMBDAS_PER_EPISODE
    env.use_lagrangian = lagrangian
    return env


def make_task_pool(n, seed=0, ood=False):
    """Draw a reproducible pool of tasks."""
    rng = np.random.default_rng(seed)
    return [sample_task(rng, ood=ood) for _ in range(n)]


def corner_risk(task):
    """Overflow-proneness of a task, as a scalar (higher is more prone).

    The lean study's ``feed_solids * feed_gain / feed_gx``, at the most
    over-delivering pump of the three. A heuristic: revisit it if a run
    overflows on a task this ranks mid-pool.
    """
    ctx = np.asarray(task["context"], dtype=float)
    gains = ctx[CTX_GAIN:CTX_GAIN + N_REACTORS]
    return float(ctx[CTX_SOLIDS] * np.max(gains) / max(ctx[CTX_GX], 1e-9))


def make_corner_tasks(n=None, seed=None, candidates=None):
    """Draw the most overflow-prone tasks the prior produces (rejection sampling)."""
    n = HELDOUT_CORNER_TASKS if n is None else int(n)
    if n <= 0:
        return []
    seed = HELDOUT_CORNER_SEED if seed is None else seed
    candidates = (HELDOUT_CORNER_CANDIDATES if candidates is None
                  else int(candidates))
    rng = np.random.default_rng(seed)
    pool = [sample_task(rng) for _ in range(max(candidates, n))]
    pool.sort(key=corner_risk, reverse=True)
    return pool[:n]


def resolve_reactor(kind, reactor, index=0):
    """The reactor a fault spec acts on, for the ``index``-th episode.

    Args:
        kind: Disturbance kind (aliases accepted).
        reactor: None (every reactor), an int, or "rotate" (episode
            ``index`` faults reactor ``index % N_REACTORS``).
        index: Episode index within its scenario.

    Returns:
        None for plant-wide kinds and plant-wide specs, else an int.
    """
    kind = DISTURBANCE_ALIASES.get(kind, kind)
    if kind in PLANT_DISTURBANCES or reactor is None:
        return None
    if reactor == "rotate":
        return int(index) % N_REACTORS
    return int(reactor)


def as_fault_schedule(disturbance, index=0):
    """Normalise any accepted disturbance spec into a list of fault dicts.

    Accepted forms: None; one dict with kind / onset / magnitude [/ reactor
    / end]; a list of (kind, onset, magnitude[, reactor[, end]]) tuples or
    such dicts. "rotate" reactors are resolved for episode ``index``.

    Returns:
        List of dicts with keys kind / onset / magnitude / reactor / end.
    """
    if disturbance is None:
        return []
    items = [disturbance] if isinstance(disturbance, dict) else list(disturbance)
    out = []
    for d in items:
        if isinstance(d, dict):
            kind, onset, mag = d["kind"], d["onset"], d["magnitude"]
            reactor, end = d.get("reactor"), d.get("end")
        else:
            kind, onset, mag = d[0], d[1], d[2]
            reactor = d[3] if len(d) > 3 else None
            end = d[4] if len(d) > 4 else None
        out.append({"kind": str(kind), "onset": int(onset),
                    "magnitude": float(mag),
                    "reactor": resolve_reactor(kind, reactor, index),
                    "end": None if end is None else int(end)})
    return out


def apply_fault_schedule(env, disturbance, index=0):
    """Clear ``env``'s faults and schedule ``disturbance`` (any accepted form)."""
    env.clear_disturbances()
    for d in as_fault_schedule(disturbance, index):
        env.schedule_disturbance(d["kind"], d["onset"], d["magnitude"],
                                 reactor=d["reactor"], end_step=d["end"])


def pattern_schedule(kind, magnitude, pattern, onset, reactor=None,
                     horizon=600, spike_h=24, gradual_h=180, treads=None,
                     windows=None):
    """One fault kind under one temporal pattern, as a list of fault dicts.

    ``step``: ``magnitude`` from ``onset`` to the end. ``spike``: the same,
    cleared after ``spike_h``. ``gradual``: ``treads`` steps over
    ``gradual_h`` up to ``magnitude``. ``recurrent``: on in each (start, end)
    of ``windows``, off between. A kind that fires once is cleared by an
    entry of magnitude 1.0 (cell_viability's kill is not undone: the cells
    regrow), an additive kind by its ``end``; a ramp of an additive kind adds
    ``magnitude / treads`` per tread, of cell_viability multiplies by
    ``magnitude ** (1 / treads)``, of any other kind steps linearly from 1.0.

    Args:
        kind: Disturbance kind (aliases accepted).
        magnitude: Full-strength magnitude, as ``schedule_disturbance`` reads it.
        pattern: One of FAULT_PATTERNS.
        onset: First inner step (step / spike / gradual).
        reactor: As in ``as_fault_schedule`` ("rotate" is kept).
        horizon: Episode length in inner steps; entries past it are dropped.

    Returns:
        List of dicts with keys kind / onset / magnitude / reactor / end.

    Raises:
        ValueError: On an unknown pattern.
    """
    kind = DISTURBANCE_ALIASES.get(kind, kind)
    additive = kind in ADDITIVE_DISTURBANCES
    n = max(1, int(META_DRIFT_TREADS if treads is None else treads))
    last = int(horizon) - 1

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
        times = [onset + int(round(k * gradual_h / max(n - 1, 1)))
                 for k in range(n)]
        if additive:
            mags = [magnitude / n] * n
        elif kind == "cell_viability":
            mags = [magnitude ** (1.0 / n)] * n
        else:
            mags = [1.0 + (magnitude - 1.0) * (k + 1) / n for k in range(n)]
        out = [entry(t, m) for t, m in zip(times, mags)]
    elif pattern == "recurrent":
        out = [e for t0, t1 in (windows or ()) for e in pulse(t0, t1)]
    else:
        raise ValueError(f"unknown fault pattern {pattern!r}; expected one "
                         f"of {FAULT_PATTERNS}")
    return [e for e in out if 1 <= e["onset"] <= last]


class MagnitudeStratifier:
    """Per-(task, kind) stratified draws over a disturbance magnitude range."""

    def __init__(self, n_strata=None, tail_weight=None):
        self.n = max(1, int(META_DISTURB_STRATA if n_strata is None
                            else n_strata))
        self.tail_weight = float(META_DISTURB_TAIL_WEIGHT
                                 if tail_weight is None else tail_weight)
        self._cycles = {}

    def _fresh_cycle(self, rng):
        reps = max(1, int(round(self.tail_weight)))
        idx = []
        for s in range(self.n):
            edge = (s == 0 or s == self.n - 1) and self.n > 1
            idx.extend([s] * (reps if edge else 1))
        return [idx[i] for i in rng.permutation(len(idx))]

    def draw(self, rng, key, lo, hi):
        """Draw one magnitude for ``key``, advancing that key's cycle."""
        cycle = self._cycles.get(key)
        if not cycle:
            cycle = self._fresh_cycle(rng)
            self._cycles[key] = cycle
        s = cycle.pop()
        width = (hi - lo) / self.n
        return float(min(hi, lo + width * (s + rng.random())))


def maybe_schedule_disturbance(env, prob=None, rng=None, task_id=None,
                               strat=None, patterns=None,
                               active_disturbances=None):
    """Clear disturbances, then schedule a fault at random with probability ``prob``.

    One kind under one of ``patterns`` (FAULT_PATTERNS, timings drawn from
    META_SPIKE_H / META_GRADUAL_FRAC / META_RECURRENT_*), or "compound": two
    kinds stepping at one onset. A reactor is drawn for every per-reactor
    fault (one uniformly, or all with META_PLANTWIDE_FRAC); every entry of
    one pattern keeps its reactor.

    Returns:
        Dict describing the first fault (plus ``pattern`` and the full
        ``schedule`` of (kind, onset, magnitude, reactor, end)), or None.
    """
    env.clear_disturbances()
    p = META_DISTURB_PROB if prob is None else prob
    rng = np.random.default_rng() if rng is None else rng
    if rng.random() >= p:
        return None

    active = active_disturbances
    if active is None:
        active = getattr(env, "active_disturbances", None)
    if active is None:
        active = ACTIVE_DISTURBANCES
    if active is None:
        active = META_DISTURB_KINDS
    candidate_kinds = parse_active_disturbances(active)
    if not candidate_kinds:
        return None
    horizon = int(env.max_steps)

    def draw_mag(kind):
        lo, hi = META_DISTURB_MAG_RANGES[kind]
        if strat is None:
            return float(rng.uniform(lo, hi))
        return strat.draw(rng, (task_id, kind), lo, hi)

    def draw_onset():
        f_lo, f_hi = META_DISTURB_ONSET_FRAC
        o = int(rng.uniform(f_lo, f_hi) * horizon)
        return max(1, min(horizon - 1, o))

    def draw_reactor(kind):
        if kind in PLANT_DISTURBANCES or rng.random() < META_PLANTWIDE_FRAC:
            return None
        return int(rng.integers(N_REACTORS))

    def draw_windows(t):
        out = []
        while t < horizon - 1 and len(out) < META_RECURRENT_MAX:
            on = int(rng.integers(META_RECURRENT_ON_H[0],
                                  META_RECURRENT_ON_H[1] + 1))
            out.append((t, t + on))
            t += on + int(rng.integers(META_RECURRENT_OFF_H[0],
                                       META_RECURRENT_OFF_H[1] + 1))
        return out

    kind = str(rng.choice(candidate_kinds))
    mag = draw_mag(kind)
    onset = draw_onset()
    reactor = draw_reactor(kind)

    if not patterns:
        pattern = "step"
    else:
        names = list(patterns)
        weights = np.asarray([patterns[k] for k in names], dtype=float)
        pattern = names[int(rng.choice(len(names), p=weights / weights.sum()))]
    if pattern == "compound":
        pairs = [p_ for p_ in META_COMPOUND_PAIRS
                 if p_[0] in candidate_kinds and p_[1] in candidate_kinds]
        if pairs and rng.random() < META_COMPOUND_PAIR_FRAC:
            k1, k2 = pairs[int(rng.integers(len(pairs)))]
            parts = [(k1, draw_mag(k1), draw_reactor(k1)),
                     (k2, draw_mag(k2), draw_reactor(k2))]
        else:
            others = [k for k in candidate_kinds if k != kind]
            kind2 = str(rng.choice(others)) if others else kind
            parts = [(kind, mag, reactor),
                     (kind2, draw_mag(kind2), draw_reactor(kind2))]
        entries = [e for k, m, r in parts
                   for e in pattern_schedule(k, m, "step", onset, r, horizon)]
    else:
        entries = pattern_schedule(
            kind, mag, pattern, onset, reactor, horizon,
            spike_h=int(rng.integers(META_SPIKE_H[0], META_SPIKE_H[1] + 1)),
            gradual_h=int(META_GRADUAL_FRAC * horizon),
            treads=META_DRIFT_TREADS, windows=draw_windows(onset))

    schedule = [(e["kind"], e["onset"], e["magnitude"],
                 None if e["kind"] in PLANT_DISTURBANCES else e["reactor"],
                 e["end"]) for e in entries]
    for k, o, m, r, end in schedule:
        env.schedule_disturbance(k, o, m, reactor=r, end_step=end)
    first = schedule[0]
    return {"kind": first[0], "onset": first[1], "magnitude": first[2],
            "reactor": first[3], "pattern": pattern, "schedule": schedule}
