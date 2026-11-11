"""Evaluating meta5 under the known change.

Every evaluation task (a prior task, EVAL_ID_SEED) runs once to the onset;
each branch continues from that snapshot, so all of a task's branches share
its first 300 h exactly. Per task:

    unchanged          no change (m = 1.0, ~4 training SDs below the mean)
    train              m from the training distribution
    wide               m from the evaluation distribution
    sweep@x            m = x for every x in EVAL_SWEEP

Every changed branch also runs FROZEN: the same policy with its latent held
at the onset's posterior mean, so it never infers the change (its recurrent
history still sees the plant). ``INFER = adaptive - frozen`` is paired per
task and branch: what online inference of m is worth, with the same weights.

Reported for each: net production Pi and the violation burden B
(docs/evaluation_metrics.md), violation events by constraint, on-spec
ethanol, the flow-change wear (the reward's smoothing term) and the aux
head's decoded draw against the true m, 25 / 50 / 100 h after the onset and
at the end. The wide rows are also split by whether m lies in the training
support (|m - mean| <= SUPPORT_SD training SDs).
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import copy
import json
import os
import time

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
import five_reactor  # noqa: F401  (selects the layout before metarl loads)
from metarl.env.layout import ACTION_DIM_PER_REACTOR
from metarl.training.arms import _build_agent, _load_preferring_best, arm_paths
from metarl.training.config import EVAL_WORKERS, SEED
from metarl.training.logs import _json_default
from metarl.training.rollouts import (_advance, _episode_metrics,
                                      _start_episode)
from metarl.training.tasks import (make_arm_env, make_task_pool,
                                   seed_everything)
from metarl.training.workers import RolloutPool, _worker_agent
from five_reactor.arms import SPEC
from five_reactor.change import (change_onset, change_schedule, in_support,
                                 sample_magnitude)
from five_reactor.config import (ARM, EVAL_CHANGE_SEED, EVAL_EPISODES,
                                 EVAL_ID_SEED, EVAL_SWEEP, LOG_DIR)
from five_reactor.train import decode_magnitude

# Hours after the onset at which the decoded draw is read (and at the end).
DECODE_AFTER_H = (25, 50, 100)
SCENARIOS = ("train", "wide")


def eval_plan(n):
    """The evaluation tasks and, per task, its (train, wide) magnitudes."""
    tasks = make_task_pool(n, seed=EVAL_ID_SEED)
    rng = np.random.default_rng(EVAL_CHANGE_SEED)
    draws = [(sample_magnitude(rng, "train"), sample_magnitude(rng, "eval"))
             for _ in range(n)]
    return tasks, draws


def task_branches(draw):
    """One task's branches: (name, m, frozen), the unchanged one first."""
    out = [("unchanged", 1.0, False)]
    for name, m in zip(SCENARIOS, draw):
        out += [(name, m, False), (name, m, True)]
    for x in EVAL_SWEEP:
        out += [(f"sweep@{x:g}", float(x), False),
                (f"sweep@{x:g}", float(x), True)]
    return out


def _compact(env, agent, row, m, frozen, onset_step, decoded):
    """The fields the summary and figures read, from one episode's metrics."""
    act = np.asarray(row["act_traj"], dtype=float)
    post = act[onset_step:].reshape(-1, env.n_reactors, ACTION_DIM_PER_REACTOR)
    return {
        "m": float(m), "frozen": bool(frozen), "in_support": in_support(m),
        "score": row["score"], "burden": row["burden"],
        "violations": row["violations"],
        "viol_by_constraint": row["viol_by_constraint"],
        "burden_by_constraint": row["burden_by_constraint"],
        "on_spec_ethanol": row["on_spec_ethanol"],
        "smoothing": row["smoothing"], "reward": row["reward"],
        "lots_passed": row["lots_passed"], "lots_failed": row["lots_failed"],
        "storage_min": row["storage_min"], "storage_max": row["storage_max"],
        "storage_traj": np.asarray(row["storage_traj"]).tolist(),
        # Mean executed clock, LF feed and discharge after the onset, over
        # the reactors (actions in [0, 1]).
        "post_clock": float(post[..., 0].mean()) if post.size else float("nan"),
        "post_feed": float(post[..., 1].mean()) if post.size else float("nan"),
        "post_discharge": float(post[..., 3].mean()) if post.size else float("nan"),
        "decoded": decoded + [decode_magnitude(agent, row["z_mu"])],
    }


def task_rollouts(env, agent, task, draw, index=0):
    """Every branch of one task from one pre-change prefix.

    Returns:
        List of compact rows, in ``task_branches`` order.
    """
    onset = change_onset(env.max_steps)
    onset_step = onset // env.control_action_interval
    prev = agent.z_mode
    try:
        agent.z_mode = "mean"
        ep = _start_episode(env, agent, task, None, index)
        _advance(env, agent, ep, stop_at=onset)
        snap = (copy.deepcopy(env), agent.posterior_state(), copy.deepcopy(ep),
                np.asarray(agent.proj_counts()))
        z_onset = agent.z_mu.detach().clone()
        rows = []
        for _, m, frozen in task_branches(draw):
            b_env, b_ep = copy.deepcopy(snap[0]), copy.deepcopy(snap[2])
            agent.set_posterior_state(snap[1])
            b_ep["proj0"] = b_ep["proj0"] + np.asarray(agent.proj_counts()) - snap[3]
            if frozen:
                agent.set_fixed_z(z_onset)
            agent.z_mode = "fixed" if frozen else "mean"
            agent._resample_z()
            if m != 1.0:
                for kind, o, mag, reactor, end in change_schedule(m, env.max_steps):
                    b_env.schedule_disturbance(kind, o, mag, reactor=reactor,
                                               end_step=end)
            decoded = []
            for h in DECODE_AFTER_H:
                _advance(b_env, agent, b_ep, stop_at=onset + h)
                decoded.append(decode_magnitude(agent, agent.z_mu))
            _advance(b_env, agent, b_ep)
            rows.append(_compact(b_env, agent, _episode_metrics(b_env, agent, b_ep),
                                 m, frozen, onset_step, decoded))
            agent.z_mode = "mean"
        return rows
    finally:
        agent.z_mode = prev


def _task_job(job):
    """``task_rollouts`` inside a worker process."""
    arm, weights, task, draw, index = job
    env, agent = _worker_agent(arm, weights)
    return task_rollouts(env, agent, task, draw, index)


def _all_rows(env, agent, tasks, draws):
    """Per task, its branch rows: in EVAL_WORKERS processes, else in-process."""
    if EVAL_WORKERS > 1:
        pool = None
        try:
            pool = RolloutPool(SPEC, EVAL_WORKERS)
            weights = agent.rollout_weights()
            futures = [pool._ex.submit(_task_job, (ARM, weights, t, d, i))
                       for i, (t, d) in enumerate(zip(tasks, draws))]
            return [f.result() for f in futures]
        except Exception as exc:                          # pragma: no cover
            print(f"[eval] worker pool failed ({exc!r}); running in-process")
        finally:
            if pool is not None:
                pool.close()
    return [task_rollouts(env, agent, t, d, i)
            for i, (t, d) in enumerate(zip(tasks, draws))]


def _mean_ci(x):
    """Mean and 95% half-width (normal approximation) of a sample."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "ci95": float("nan"), "n": 0}
    hw = 1.96 * x.std(ddof=1) / np.sqrt(x.size) if x.size > 1 else float("nan")
    return {"mean": float(x.mean()), "ci95": float(hw), "n": int(x.size)}


def _block(adaptive, frozen=None):
    """Summary of paired adaptive (and frozen) rows."""
    out = {k: _mean_ci([r[k] for r in adaptive])
           for k in ("score", "burden", "violations", "on_spec_ethanol",
                     "smoothing", "post_feed", "post_discharge", "post_clock")}
    out["decode_mae"] = [_mean_ci([abs(r["decoded"][j] - r["m"]) for r in adaptive])
                         for j in range(len(DECODE_AFTER_H) + 1)]
    cons = sorted({int(c) for r in adaptive for c in r["viol_by_constraint"]})
    out["violations_by_constraint"] = {
        c: float(np.mean([r["viol_by_constraint"].get(c, 0) for r in adaptive]))
        for c in cons}
    if frozen is not None:
        out["frozen"] = {k: _mean_ci([r[k] for r in frozen])
                         for k in ("score", "burden", "violations",
                                   "on_spec_ethanol", "smoothing")}
        out["infer"] = {k: _mean_ci([a[k] - f[k] for a, f in zip(adaptive, frozen)])
                        for k in ("score", "burden", "violations",
                                  "on_spec_ethanol", "smoothing")}
    return out


def summarize(per_task, draws):
    """The summary of every task's rows (also what ``plots`` reads)."""
    names = [b[0] for b in task_branches(draws[0])]
    frozen = [b[2] for b in task_branches(draws[0])]

    def col(name, is_frozen):
        k = next(j for j, (n, f) in enumerate(zip(names, frozen))
                 if n == name and f == is_frozen)
        return [rows[k] for rows in per_task]

    out = {"unchanged": _block(col("unchanged", False))}
    for sc in SCENARIOS:
        a, f = col(sc, False), col(sc, True)
        out[sc] = _block(a, f)
        if sc == "wide":
            for label, keep in (("in_support", True), ("outside_support", False)):
                idx = [i for i, r in enumerate(a) if r["in_support"] == keep]
                if idx:
                    out[f"wide_{label}"] = _block([a[i] for i in idx],
                                                  [f[i] for i in idx])
    out["sweep"] = {f"{x:g}": _block(col(f"sweep@{x:g}", False),
                                     col(f"sweep@{x:g}", True))
                    for x in EVAL_SWEEP}
    out["decode_after_h"] = list(DECODE_AFTER_H) + ["end"]
    return out


def _fmt(b, k, scale=1.0):
    return f"{b[k]['mean'] / scale:+.3f}±{b[k]['ci95'] / scale:.3f}"


def print_summary(summary):
    print(f"\n{'=' * 78}\nMETA5 under the known change "
          f"(Pi, burden B, events; INFER = adaptive - frozen z, paired)\n{'=' * 78}")
    for key in ("unchanged", "train", "wide", "wide_in_support",
                "wide_outside_support"):
        if key not in summary:
            continue
        b = summary[key]
        line = (f"  {key:22s} n={b['score']['n']:3d}  Pi {_fmt(b, 'score', 1e7)}e7"
                f"  B {b['burden']['mean']:.2f}  events {b['violations']['mean']:.1f}"
                f"  decode MAE (25h/end) {b['decode_mae'][0]['mean']:.3f}/"
                f"{b['decode_mae'][-1]['mean']:.3f}")
        if "infer" in b:
            line += (f"\n  {'':22s} INFER: Pi {_fmt(b['infer'], 'score', 1e7)}e7  "
                     f"B {_fmt(b['infer'], 'burden')}  events "
                     f"{_fmt(b['infer'], 'violations')}")
        print(line)
    print("  sweep (m: Pi adaptive / frozen x1e7, INFER Pi, decoded m at end)")
    for x, b in summary["sweep"].items():
        dec = np.nan if not b["decode_mae"] else b["decode_mae"][-1]["mean"]
        print(f"    m={float(x):4.2f}  {b['score']['mean'] / 1e7:+.3f} / "
              f"{b['frozen']['score']['mean'] / 1e7:+.3f}  INFER "
              f"{_fmt(b['infer'], 'score', 1e7)}  decode MAE {dec:.3f}")


def evaluate(n=None):
    """Evaluate meta5's selected checkpoint; writes eval.json and eval_summary.json."""
    n = EVAL_EPISODES if n is None else int(n)
    seed_everything(SEED)
    env = make_arm_env(SPEC)
    agent = _build_agent(env, SPEC)
    final_path, best_path = arm_paths(SPEC)
    _load_preferring_best(agent, best_path, final_path, ARM, strict=True)
    tasks, draws = eval_plan(n)
    t0 = time.time()
    per_task = _all_rows(env, agent, tasks, draws)
    print(f"[eval] {n} tasks x {len(task_branches(draws[0]))} branches in "
          f"{time.time() - t0:.0f}s")
    summary = summarize(per_task, draws)
    print_summary(summary)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "eval.json"), "w") as f:
        json.dump({"branches": [list(b) for b in task_branches(draws[0])],
                   "draws": draws, "rows": per_task}, f,
                  default=_json_default)
    with open(os.path.join(LOG_DIR, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"[eval] wrote {os.path.join(LOG_DIR, 'eval.json')} and eval_summary.json")
    from five_reactor.plots import plot_eval
    plot_eval()
    return summary
