"""Scripted-recipe calibration of the five-reactor plant and its known change.

Two questions, answered before any training:

1. Does the five-vessel plant couple like the three-reactor one? The
   smoke test's scripted recipes on the nominal plant, no change: doing
   nothing runs the tank dry, unloading in lockstep overflows it, a
   staggered recipe runs nearly clean.
2. Is the change worth adapting to? For every magnitude m, the staggered
   recipe's post-change settings (clock pace, mass target, drain level) are
   retuned on m and compared with the settings tuned on the training mean.
   The gap is the headroom an agent that infers m can win over one that
   acts on the mean; where it is near zero, no inference can show.

Run from the repository root: ``python -m five_reactor calibrate``. Writes
``results/log/calibration.json`` under the sub-study folder.
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
import five_reactor  # noqa: F401  (selects the layout before metarl loads)
from metarl.env.config import (FIVE_REACTOR_MMAX, N_REACTORS, R_MASS, R_STAGE,
                               REACTOR_FLOW_SCALE, nominal_task, sample_task)
from metarl.env.layout import (STORAGE_CHANNEL, reactor_channel,
                               apply_stage_mask_np)
from metarl.training.config import DEPTH_CONSTRAINTS
from metarl.training.metrics import net_production, violation_burden
from metarl.training.tasks import apply_fault_schedule, make_env
from five_reactor.change import change_onset, change_schedule
from five_reactor.config import CHANGE_MEAN, LOG_DIR

T1 = -0.34657359          # clock action of exactly 1.0x
# Staggered clocks, 1.0x down to 0.6x, as the three-reactor recipe's
# 1.0 / 0.8 / 0.6.
BASE_CLOCKS = tuple(np.linspace(1.0, 0.6, N_REACTORS))
# The recipe's settings: (pace on the clocks, mass target / Mmax, drain
# below this storage fraction). The pre-change settings are tuned on the
# unchanged plant, the post-change ones retuned per magnitude, both over GRID.
GRID = tuple(itertools.product((0.6, 0.75, 0.9, 1.0, 1.15, 1.3, 1.5),
                               (0.80, 0.90, 0.95),
                               (0.55, 0.70, 0.85, 0.95)))
# The three-reactor recipe feeds each vessel at most a 1x pump's flow into
# 220 t; this one keeps that flow per tonne of vessel.
FEED_CAP = FIVE_REACTOR_MMAX / 220000.0
# At most this many vessels charge or co-feed at once; the rest wait at the
# slowest clock with their feeds shut. The cooling cap sits just below three
# vessels feeding at half rate in phase: with all five free (they all start
# at hour 0) the recipe ran 65-70 h over it on every task.
MAX_FEEDING = 3
MAGNITUDES = (0.6, 0.8, 1.0, 1.15, 1.3, 1.45, 1.6, 1.8, 2.0)
# Prior tasks the headroom is averaged over (nominal is the first).
N_TASKS = 4


def _clock(x):
    """Clock multiplier (0.5-2x) -> action value."""
    return float(np.clip(2.0 * (x - 0.5) / 1.5 - 1.0, -1.0, 1.0))


def _stage(obs, i):
    s = reactor_channel(i, R_STAGE)
    return int(np.argmax(obs[s:s + 5]))


def recipe(obs, pace, target, s_cap, max_feeding=MAX_FEEDING):
    """The smoke test's staggered closed-loop recipe, on five vessels.

    Clocks BASE_CLOCKS x ``pace`` through charge and co-feed, feed toward
    ``target`` x Mmax at most FEED_CAP of a 1x pump's flow, drain while the
    tank is below ``s_cap``. Only the first ``max_feeding`` vessels (in
    reactor order) in charge or co-feed run; the others hold.
    """
    a = []
    S = obs[STORAGE_CHANNEL]
    feeding = 0
    for i in range(N_REACTORS):
        m = obs[reactor_channel(i, R_MASS)]
        st = _stage(obs, i)
        run = st in (0, 1) and feeding < max_feeding
        feeding += int(run)
        hr = np.clip((target - m) / 0.15, 0.0, 1.0)
        frac = hr * min(1.0, FEED_CAP / REACTOR_FLOW_SCALE[i])
        f = 2.0 * frac - 1.0 if (run or st not in (0, 1)) else -1.0
        out = 1.0 if (st == 3 and S < s_cap) else -1.0
        if st not in (0, 1):
            c = T1
        else:
            c = _clock(BASE_CLOCKS[i] * pace) if run else -1.0
        a += [c, f, f, out]
    return np.array(a)


def _lockstep(obs):
    """Unit clocks for every vessel, feed toward 80%, drain whenever unloading."""
    a = recipe(obs, 1.0, 0.80, 2.0, max_feeding=N_REACTORS)
    a[0::4] = np.where(a[0::4] == T1, T1, _clock(1.0))
    return a


def _idle(obs):
    return np.tile([T1, -1.0, -1.0, -1.0], N_REACTORS)


def run_episode(policy, task=None, schedule=None, after=None):
    """Roll a scripted policy out; ``after`` replaces it from the change on.

    Returns:
        Dict: violation events by constraint, burden, on-spec ethanol, lots,
        heat-load peak, storage range and the net production Pi.
    """
    env = make_env()
    apply_fault_schedule(env, schedule or [])
    env.set_task(task if task is not None else nominal_task())
    obs = env.reset()
    onset = change_onset(env.max_steps)
    counts = {}
    depth = {c: 0.0 for c in DEPTH_CONSTRAINTS}
    smoothing, heat_peak, done, info = 0.0, 0.0, False, {}
    s_lo, s_hi = float(env.S_mass), float(env.S_mass)
    while not done:
        pol = after if (after is not None and env.time_step_count >= onset) else policy
        obs, _, done, info = env.step(apply_stage_mask_np(obs, pol(obs)))
        smoothing += float(info.get("smoothing", 0.0))
        heat_peak = max(heat_peak, float(info.get("heat_load_peak", 0.0)))
        s_lo, s_hi = min(s_lo, float(env.S_mass)), max(s_hi, float(env.S_mass))
        for k, v in info["violation_counts"].items():
            counts[int(k)] = counts.get(int(k), 0) + int(v)
        for k, d in info.get("violation_depth_sum", {}).items():
            if int(k) in depth:
                depth[int(k)] += float(d)
    burden = violation_burden(depth, counts)
    b = float(sum(burden.values()))
    on_spec = float(info.get("on_spec_ethanol", 0.0))
    return {"violations": counts, "events": int(sum(counts.values())),
            "burden": b, "on_spec_ethanol": on_spec,
            "lots_passed": int(info.get("lots_passed", 0)),
            "lots_failed": int(info.get("lots_failed", 0)),
            "heat_peak": heat_peak, "storage_range_t": (s_lo / 1e3, s_hi / 1e3),
            "pi": net_production(on_spec, b, smoothing)}


def _job(args):
    """One episode: ``before`` until the change, ``after`` from it (None: ``after`` throughout)."""
    task, m, before, after = args
    before = after if before is None else before
    return run_episode(lambda o: recipe(o, *before), task,
                       change_schedule(m) if m != 1.0 else None,
                       after=lambda o: recipe(o, *after))


def _grid_runs(ex, tasks, m, before):
    """(Pi, events) of every GRID post-change setting at ``m``, task-averaged."""
    jobs = [(t, m, before, g) for g in GRID for t in tasks]
    res = list(ex.map(_job, jobs, chunksize=4))
    pi = np.array([r["pi"] for r in res]).reshape(len(GRID), len(tasks)).mean(1)
    ev = np.array([r["events"] for r in res]).reshape(len(GRID), len(tasks)).mean(1)
    return pi, ev


def coupling():
    """The nominal plant under the scripted recipes, no change."""
    out = {"idle": run_episode(_idle),
           "lockstep": run_episode(_lockstep),
           "staggered, all five feeding":
               run_episode(lambda o: recipe(o, 1.0, 0.80, 0.85, N_REACTORS)),
           f"staggered, {MAX_FEEDING} feeding at once":
               run_episode(lambda o: recipe(o, 1.0, 0.80, 0.85))}
    for name, r in out.items():
        print(f"  {name:34s} events {r['events']:4d} {r['violations']}  lots "
              f"{r['lots_passed']}/{r['lots_passed'] + r['lots_failed']}  "
              f"on-spec {r['on_spec_ethanol']:.3e}  heat peak "
              f"{r['heat_peak']:.0f} kg/h  tank {r['storage_range_t'][0]:.0f}-"
              f"{r['storage_range_t'][1]:.0f} t  Pi {r['pi']:.3e}")
    return out


def headroom(workers=None):
    """Pi of the retuned and the mean-tuned post-change recipe, per magnitude.

    The pre-change settings are the best on the unchanged plant; the
    mean-tuned post-change settings the best at CHANGE_MEAN.

    Returns:
        Dict with both tuned settings and, per magnitude, the retuned
        settings, the three Pi (retuned, mean-tuned, not retuned; averaged
        over N_TASKS tasks) and the retuned-vs-mean-tuned gap.
    """
    rng = np.random.default_rng(7)
    tasks = [nominal_task()] + [sample_task(rng) for _ in range(N_TASKS - 1)]
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as ex:
        pi0, ev0 = _grid_runs(ex, tasks, 1.0, None)
        before = GRID[int(np.argmax(pi0))]
        print(f"  pre-change settings (pace, target, drain level) {before}: "
              f"Pi {pi0.max():.3e}, {ev0[int(np.argmax(pi0))]:.1f} events "
              f"on the unchanged plant")
        runs = {m: _grid_runs(ex, tasks, m, before) for m in MAGNITUDES}
    k_mean = int(np.argmax(runs[CHANGE_MEAN][0]))
    k_keep = GRID.index(before)
    print(f"  post-change settings tuned at m = {CHANGE_MEAN}: {GRID[k_mean]}")
    rows = {}
    for m, (pi, ev) in runs.items():
        k = int(np.argmax(pi))
        r = rows[str(m)] = {
            "retuned": GRID[k], "pi_retuned": float(pi[k]),
            "pi_mean_tuned": float(pi[k_mean]), "pi_kept": float(pi[k_keep]),
            "events_retuned": float(ev[k]),
            "events_mean_tuned": float(ev[k_mean]),
            "events_kept": float(ev[k_keep]),
            "gap": float(pi[k] - pi[k_mean])}
        print(f"  m={m:4.2f}  retuned {GRID[k]} Pi {r['pi_retuned']:.3e} "
              f"({r['events_retuned']:.1f} ev) | mean-tuned "
              f"{r['pi_mean_tuned']:.3e} ({r['events_mean_tuned']:.1f} ev) | "
              f"kept {r['pi_kept']:.3e} ({r['events_kept']:.1f} ev) | gap "
              f"{r['gap']:+.3e}")
    return {"before": before, "mean_tuned": GRID[k_mean], "per_magnitude": rows}


def calibrate():
    """Both checks; writes calibration.json."""
    print("[calibrate] the nominal five-reactor plant, no change")
    out = {"coupling": coupling()}
    print(f"[calibrate] headroom of retuning the recipe on m "
          f"({len(GRID)} settings x {len(MAGNITUDES)} magnitudes x "
          f"{N_TASKS} tasks)")
    out["headroom"] = headroom()
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, "calibration.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[calibrate] wrote {path}")
    return out
