"""Evaluation: both arms on every scenario, on both metric axes."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import json
import os
import time

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import N_REACTORS
from metarl.env.layout import ACTION_DIM_PER_REACTOR
from metarl.training.config import (SEED, EVAL_EPISODES, EVAL_Z_MODE,
                                    EVAL_WORKERS,
                                    EVAL_DISTURBANCES, FAULT_PATTERNS,
                                    EVAL_SPIKE_H, EVAL_GRADUAL_H,
                                    EVAL_RECURRENT_WINDOWS, META_DRIFT_TREADS,
                                    CONSTRAINT_NAMES, DEPTH_CONSTRAINTS,
                                    Z_INFORM_MIN_SNR, LOG_DIR, PLOT_DIR,
                                    DISTURB_PROBES)
from metarl.training.arms import (ARMS, ARM_ORDER, CONTRASTS, arm_paths,
                                  _build_agent,
                                  _load_preferring_best)
from metarl.training.tasks import (seed_everything, make_arm_env, make_task_pool,
                                   as_fault_schedule, pattern_schedule)
from metarl.training.rollouts import (rollout_deterministic,
                                      scenario_rollouts)
from metarl.training.workers import RolloutPool
from metarl.training.metrics import (latent_separability, gate_opening,
                                     gate_opening_in_stage, _finite_mean,
                                     _paired_delta, _verdict_sign, _fmt_delta,
                                     _fmt_delta_sci)
from metarl.training.logs import _json_default
from metarl.training.plots import (_plot_comparison, plot_training_curves,
                                   scenario_probe)


def _recovery_metrics(reward_traj, viol_traj, onset_outer, win=10):
    """Summarise behaviour either side of a disturbance onset."""
    n = len(reward_traj)
    if n < 2:
        return {}
    o = int(min(max(onset_outer, 1), n - 1))
    pre = reward_traj[max(0, o - win):o]
    post = reward_traj[o:]
    pre_r = float(np.nanmean(pre)) if len(pre) else float("nan")
    k = max(1, min(5, len(post)))
    post_sm = np.convolve(np.nan_to_num(post), np.ones(k) / k, mode="same")
    tol = 0.05 * abs(pre_r) + 1e-6
    recovery = None
    for i, v in enumerate(post_sm):
        if v >= pre_r - tol:
            recovery = i
            break
    return {
        "pre_reward": pre_r,
        "post_mean_reward": float(np.nanmean(post)) if len(post) else float("nan"),
        "post_min_reward": float(np.nanmin(post)) if len(post) else float("nan"),
        "recovery_steps": recovery,
        "pre_viol_rate": (float(np.nanmean(viol_traj[max(0, o - win):o]))
                          if o > 0 else float("nan")),
        "post_viol_rate": float(np.nanmean(viol_traj[o:])),
    }


def _eval_agent_on_tasks(env, agent, tasks, disturbance=None, z_mode=None,
                         trace_probe=None):
    """Roll out one arm over a task list and aggregate both metric axes.

    A "rotate" fault resolves per episode index, so every arm faults the same
    reactor on the same task.

    Returns:
        Dict of aggregated metrics plus per-episode arrays for paired tests.
    """
    mode = z_mode if z_mode is not None else EVAL_Z_MODE
    rows = [rollout_deterministic(env, agent, t, z_mode=mode,
                                  disturbance=disturbance, index=i)
            for i, t in enumerate(tasks)]
    return _aggregate_rows(rows, disturbance, env.control_action_interval,
                           mode if agent.use_context else "n/a (z=0)",
                           trace_probe)


def _aggregate_rows(rows, disturbance, interval, z_label, trace_probe=None):
    """Both metric axes of one arm on one scenario, from its episode rows.

    Args:
        rows: One ``_episode_metrics`` dict per task, in task order.
        disturbance: The scenario's fault schedule (None: undisturbed).
        interval: The environment's control interval, in inner steps.
        z_label: What the report records as the z mode.
        trace_probe: ``DISTURB_PROBES`` key to trace, or None.
    """
    calls, _, _, moved = np.sum([r["proj"] for r in rows], axis=0)
    shield = {"proj_calls": int(calls),
              "proj_moved_frac": moved / max(calls, 1)}
    targets = []
    for i in range(len(rows)):
        sched = as_fault_schedule(disturbance, i)
        targets.append(sched[0]["reactor"] if sched else None)

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    z_mus = np.stack([r["z_mu"] for r in rows])
    sep = latent_separability(z_mus, [r["z_sigma_sq"] for r in rows])
    acts = np.stack([r["act_mean"] for r in rows])            # (E, N, 3)

    L = max(len(r["reward_traj"]) for r in rows)

    def pad_mean(seqs):
        m = np.full((len(seqs), L), np.nan)
        for i, s in enumerate(seqs):
            m[i, :len(s)] = s
        return np.nanmean(m, axis=0)

    mean_r = pad_mean([r["reward_traj"] for r in rows])
    mean_v = pad_mean([r["viol_traj"] for r in rows])
    by_c = {cid: [float(r["viol_by_constraint"].get(cid, 0)) for r in rows]
            for cid in CONSTRAINT_NAMES}

    out = {
        "z_mode": z_label,
        "reward": float(np.mean(col("reward"))),
        "reward_std": float(np.std(col("reward"))),
        "violations": float(np.mean(col("violations"))),
        "violations_std": float(np.std(col("violations"))),
        "violations_median": float(np.median(col("violations"))),
        "violations_p90": float(np.percentile(col("violations"), 90)),
        "violations_worst": float(np.max(col("violations"))),
        "episodes_violated": int(np.sum(col("violations") > 0)),
        "episodes": len(rows),
        "feasible_rate": float(np.mean(col("feasible"))),
        "viol_by_constraint": {CONSTRAINT_NAMES[cid]: float(np.mean(v))
                               for cid, v in by_c.items()},
        "viol_by_constraint_per_ep": {CONSTRAINT_NAMES[cid]: v
                                      for cid, v in by_c.items()},
        # Reactor-local violation events (ids 1-5) per reactor, mean per ep.
        "viol_by_reactor": np.mean(np.asarray(
            [r["viol_by_reactor"] for r in rows], dtype=float), axis=0).tolist(),
        "viol_depth_sum_per_ep": {
            CONSTRAINT_NAMES[cid]: [float(r["viol_depth_sum"].get(cid, 0.0))
                                    for r in rows]
            for cid in DEPTH_CONSTRAINTS},
        "viol_depth_max_per_ep": {
            CONSTRAINT_NAMES[cid]: [float(r["viol_depth_max"].get(cid, 0.0))
                                    for r in rows]
            for cid in DEPTH_CONSTRAINTS},
        "ethanol": float(np.mean(col("ethanol"))),
        "ethanol_std": float(np.std(col("ethanol"))),
        "on_spec_ethanol": float(np.mean(col("on_spec_ethanol"))),
        "on_spec_ethanol_std": float(np.std(col("on_spec_ethanol"))),
        "reactor_on_spec_ethanol": np.mean(np.asarray(
            [r["reactor_on_spec_ethanol"] for r in rows]), axis=0).tolist(),
        "product_mass": float(np.mean(col("product_mass"))),
        "mean_titer": float(np.mean(col("mean_titer"))),
        "mean_titer_std": float(np.std(col("mean_titer"))),
        "purity_rate": float(np.mean(col("purity_ok"))),
        "lot_pass_rate": float(col("lots_passed").sum()
                               / max(col("lots_passed").sum()
                                     + col("lots_failed").sum(), 1.0)),
        "batches": float(np.mean(col("batches"))),
        "reactor_batches": np.mean(np.asarray(
            [r["reactor_batches"] for r in rows], dtype=float), axis=0).tolist(),
        "storage_min": float(np.mean(col("storage_min"))),
        "storage_max": float(np.mean(col("storage_max"))),
        "feed_used": float(np.mean(col("feed_used"))),
        "yield_per_feed": float(np.mean(col("ethanol")) /
                                max(np.mean(col("feed_used")), 1e-9)),
        "act_lf": float(np.nanmean(acts[:, :, 0])),
        "act_c5": float(np.nanmean(acts[:, :, 1])),
        "act_out": float(np.nanmean(acts[:, :, 2])),
        # Per reactor: mean executed (LF, C5, discharge).
        "act_by_reactor": np.nanmean(acts, axis=0).tolist(),
        "z_spread": sep["spread"],
        "z_post_std": sep["post_std"],
        "z_snr": sep["snr"],
        "z_snr_mean_ratio": sep["mean_ratio"],
        "z_abs_mean": float(np.mean(np.abs(z_mus))),
        "proj_calls": int(shield["proj_calls"]),
        "proj_moved_frac": (float(shield["proj_moved_frac"])
                            if shield["proj_calls"] else None),
        "fault_reactor_per_ep": [(-1 if t is None else int(t)) for t in targets],
        "viol_per_ep": col("violations").tolist(),
        "reward_per_ep": col("reward").tolist(),
        "ethanol_per_ep": col("ethanol").tolist(),
        "on_spec_ethanol_per_ep": col("on_spec_ethanol").tolist(),
        "feed_used_per_ep": col("feed_used").tolist(),
        "titer_per_ep": col("mean_titer").tolist(),
        "z_mu_per_ep": z_mus.tolist(),
        "reward_traj": mean_r.tolist(),
        "viol_traj": mean_v.tolist(),
    }
    if trace_probe is not None:
        a_idx = DISTURB_PROBES[trace_probe][2]
        key = DISTURB_PROBES[trace_probe][0]

        def actuator(r, t):
            act = r["act_traj"].reshape(-1, N_REACTORS, ACTION_DIM_PER_REACTOR)
            return act[:, :, a_idx].mean(axis=1) if t is None else act[:, t, a_idx]

        def quantity(r, t):
            q = r["probe_traj"][key]
            if q.ndim == 1:
                return q
            return q.mean(axis=1) if t is None else q[:, t]

        out["traces"] = {
            "probe": key,
            "reactor": [(-1 if t is None else int(t)) for t in targets],
            "act": [np.round(actuator(r, t), 4).tolist()
                    for r, t in zip(rows, targets)],
            "quantity": [[float(f"{v:.5g}") for v in quantity(r, t)]
                         for r, t in zip(rows, targets)],
            "storage": [np.round(r["storage_traj"], 1).tolist() for r in rows],
        }
    sched0 = as_fault_schedule(disturbance, 0)
    split = (int(min(d["onset"] for d in sched0) // interval)
             if sched0 else L // 2)
    for k, lo, hi in (("gate_pre", 0, split + 1), ("gate_post", split + 1, None)):
        v = _finite_mean([gate_opening(r["gate_traj"], lo, hi) for r in rows])
        out[k] = v if np.isfinite(v) else None
    for k, stage in (("gate_post_ferment", 2), ("gate_post_unload", 3)):
        v = _finite_mean([gate_opening_in_stage(r["gate_traj"], r["stage_traj"],
                                                stage, t, split + 1)
                          for r, t in zip(rows, targets)])
        out[k] = v if np.isfinite(v) else None
    if sched0:
        out["onset_outer"] = split
        out["recovery"] = _recovery_metrics(mean_r, mean_v, split)
    return out


def eval_fault_scenarios(onset, horizon=None):
    """The disturbed evaluation scenarios, exactly as ``evaluate`` runs them.

    Every EVAL_DISTURBANCES kind under every FAULT_PATTERNS entry, named
    "<label>_<pattern>"; step, spike and gradual start at ``onset``.

    Args:
        onset: First inner step of the step / spike / gradual patterns.
        horizon: Episode length in inner steps (default ``2 * onset``).

    Returns:
        Dict scenario name -> list of fault dicts ("rotate" reactors resolve
        per episode).
    """
    horizon = 2 * int(onset) if horizon is None else int(horizon)
    out = {}
    for label, d in EVAL_DISTURBANCES.items():
        for pattern in FAULT_PATTERNS:
            out[f"{label}_{pattern}"] = pattern_schedule(
                d["kind"], d["magnitude"], pattern, onset, d["reactor"],
                horizon, spike_h=EVAL_SPIKE_H, gradual_h=EVAL_GRADUAL_H,
                treads=META_DRIFT_TREADS, windows=EVAL_RECURRENT_WINDOWS)
    return out


def evaluate():
    """Compare both arms on every scenario, on both metric axes.

    Every arm sees the same tasks in the same order (and, for rotating
    faults, the same faulted reactor per task), so results are paired. Each
    arm gets its own environment so one arm's rollouts cannot move another's
    duals. One job per (arm, task) runs every scenario of that task from one
    undisturbed prefix (``scenario_rollouts``), in ``EVAL_WORKERS`` processes
    (in-process when that is 0 or 1, or the pool fails).

    Returns:
        The full results dict, also written to ``eval.json``.
    """
    seed_everything(SEED)
    arms = ARM_ORDER
    envs, agents = {}, {}
    for name in arms:
        spec = ARMS[name]
        envs[name] = make_arm_env(spec)
        agents[name] = _build_agent(envs[name], spec)
        final_path, best_path = arm_paths(spec)
        _load_preferring_best(agents[name], best_path, final_path, name)
    env = envs[arms[-1]]

    id_tasks = make_task_pool(EVAL_EPISODES, seed=12345, ood=False)
    ood_tasks = make_task_pool(EVAL_EPISODES, seed=99999, ood=True)
    onset = env.max_steps // 2
    scenarios = {
        "in_distribution": {"tasks": id_tasks, "dist": None},
        "out_of_distribution": {"tasks": ood_tasks, "dist": None},
    }
    for name, dist in eval_fault_scenarios(onset, env.max_steps).items():
        scenarios[name] = {"tasks": id_tasks, "dist": dist}

    # Scenarios grouped by task pool: each group is one job per task.
    groups = [(id_tasks, [n for n, sc in scenarios.items()
                          if sc["tasks"] is id_tasks]),
              (ood_tasks, ["out_of_distribution"])]
    n_eps = sum(len(t) * len(names) for t, names in groups) * len(arms)
    rows = _scenario_rows(envs, agents, arms, groups, scenarios, n_eps)

    results = {}
    for name, sc in scenarios.items():
        results[name] = {
            arm: _aggregate_rows(
                rows[arm][name], sc["dist"],
                envs[arm].control_action_interval,
                EVAL_Z_MODE if agents[arm].use_context else "n/a (z=0)",
                trace_probe=scenario_probe(name))
            for arm in arms
        }

    summary = _eval_summary(results, arms)
    _print_eval(results, summary)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "eval.json"), "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    with open(os.path.join(LOG_DIR, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\nwrote {os.path.join(LOG_DIR, 'eval.json')} and eval_summary.json")
    _plot_comparison(results)
    plot_training_curves()
    return results


def _scenario_rows(envs, agents, arms, groups, scenarios, n_eps):
    """Every arm's episode rows on every scenario: {arm: {scenario: rows}}.

    Args:
        groups: (tasks, scenario names) pairs; every scenario of a group runs
            on its tasks.
        n_eps: Total episodes, for the progress line.
    """
    plan = []                                   # (arm, tasks, names, traced)
    for arm in arms:
        for tasks, names in groups:
            plan.append((arm, tasks, names,
                         [scenario_probe(n) is not None for n in names]))

    def schedules(names):
        return [scenarios[n]["dist"] for n in names]

    t0 = time.time()
    per_task = None
    if EVAL_WORKERS > 1:
        pool = None
        try:
            pool = RolloutPool(None, EVAL_WORKERS)
            print(f"[eval] {n_eps} episodes in {EVAL_WORKERS} worker "
                  f"processes ...")
            futures = [pool.scenarios_async(agents[arm], tasks,
                                            schedules(names), EVAL_Z_MODE,
                                            traced, arm=arm)
                       for arm, tasks, names, traced in plan]
            per_task = [[f.result() for f in fs] for fs in futures]
        except Exception as exc:                          # pragma: no cover
            print(f"[eval] worker pool failed ({exc!r}); running in-process")
            per_task = None
        finally:
            if pool is not None:
                pool.close()
    if per_task is None:
        print(f"[eval] {n_eps} episodes in-process ...")
        per_task = [[scenario_rollouts(envs[arm], agents[arm], t,
                                       schedules(names), EVAL_Z_MODE, index=i)
                     for i, t in enumerate(tasks)]
                    for arm, tasks, names, _ in plan]
    print(f"[eval] rollouts done in {time.time() - t0:.0f}s")

    out = {arm: {} for arm in arms}
    for (arm, _, names, _), task_rows in zip(plan, per_task):
        for k, name in enumerate(names):
            out[arm][name] = [r[k] for r in task_rows]
    return out


def _eval_verdict(res, arm_a, arm_b):
    """One contrast's paired verdict on one scenario, on both axes."""
    a, b = res[arm_a], res[arm_b]
    d_viol = _paired_delta(b.get("viol_per_ep", []), a.get("viol_per_ep", []))
    d_eth = _paired_delta(a.get("on_spec_ethanol_per_ep", []),
                          b.get("on_spec_ethanol_per_ep", []))
    d_titer = a["mean_titer"] - b["mean_titer"]
    sv, sp = _verdict_sign(d_viol), _verdict_sign(d_eth)
    if sv > 0 and sp > 0:
        verdict = f"{arm_a} better (both axes)"
    elif sv > 0 and sp == 0:
        verdict = f"{arm_a} safer, output tied"
    elif sv == 0 and sp > 0:
        verdict = f"{arm_a} richer, safety tied"
    elif sv > 0 and sp < 0:
        rel = (a["on_spec_ethanol"] - b["on_spec_ethanol"]) / max(
            abs(b["on_spec_ethanol"]), 1e-12)
        verdict = f"safer but produces {rel * 100:+.0f}%"
    elif sv < 0 and sp > 0:
        verdict = "produces more but less safe"
    elif sv == 0 and sp == 0:
        verdict = "tied (no significant gap)"
    else:
        verdict = f"{arm_b} better" + (
            " (both axes)" if sv < 0 and sp < 0
            else " (safety)" if sv < 0 else " (output)")
    if sv > 0 and b["on_spec_ethanol"] > 0 and \
            a["on_spec_ethanol"] < 0.5 * b["on_spec_ethanol"]:
        verdict += " [!] <50% output"
    return {"d_viol": d_viol, "d_eth": d_eth, "d_titer": d_titer,
            "verdict": verdict}


def _eval_summary(results, arms):
    """Derived, per-scenario verdicts for every reportable contrast."""
    usable = [c for c in CONTRASTS if c[1] in arms and c[2] in arms]
    contrasts = {
        label: {name: _eval_verdict(res, arm_a, arm_b)
                for name, res in results.items()}
        for label, arm_a, arm_b, _ in usable
    }
    out = {"contrasts": contrasts}
    if "meta" in arms:
        snr = float(np.mean([res["meta"]["z_snr"] for res in results.values()]))
        out.update(z_snr_mean=snr, z_snr_pass=bool(snr >= Z_INFORM_MIN_SNR))
    return out


def _print_eval(results, summary):
    """Print the arms that ran and, per scenario, each contrast's verdict."""
    arms = tuple(a for a in ARM_ORDER if a in next(iter(results.values())))
    W = 100
    print("\n" + "=" * W)
    print("ARMS".center(W))
    print("=" * W)
    for a in arms:
        print(f"  {a:<10}{ARMS[a].summary():<58}{ARMS[a].blurb}")
    if "z_snr_mean" in summary:
        snr, ok = summary["z_snr_mean"], summary["z_snr_pass"]
        print(f"\nlatent separability: mean SNR {snr:.2f} "
              f"({'>= ' if ok else '< '}{Z_INFORM_MIN_SNR})")
    print(f"\n{'scenario':<26}" + "".join(
        f"{a + ' viol':>14}{a + ' on-spec':>16}" for a in arms))
    for name, res in results.items():
        print(f"{name:<26}" + "".join(
            f"{res[a]['violations']:>14.2f}{res[a]['on_spec_ethanol']:>16.3e}"
            for a in arms))
    for label, per_scenario in summary["contrasts"].items():
        print(f"\n{label}:")
        for name, v in per_scenario.items():
            print(f"  {name:<26}{v['verdict']:<30}"
                  f"viol {_fmt_delta(v['d_viol']):<22}"
                  f"eth {_fmt_delta_sci(v['d_eth'])}")
    print(f"\nfull per-scenario tables: eval.json  |  figures: {PLOT_DIR}  |  "
          f"redraw figures only: python -m metarl plot_eval")
