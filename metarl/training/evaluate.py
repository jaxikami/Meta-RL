"""Evaluation: both arms on every scenario, on both metric axes."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import json
import os
import time

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import N_REACTORS
from metarl.env.faults import ADDITIVE_DISTURBANCES
from metarl.env.layout import ACTION_DIM_PER_REACTOR
from metarl.training.config import (SEED, EVAL_EPISODES, EVAL_Z_MODE,
                                    EVAL_WORKERS,
                                    EVAL_DISTURBANCES, FAULT_PATTERNS,
                                    EVAL_SPIKE_H, EVAL_GRADUAL_H,
                                    EVAL_RECURRENT_WINDOWS, META_DRIFT_TREADS,
                                    CONSTRAINT_NAMES, DEPTH_CONSTRAINTS,
                                    Z_INFORM_MIN_SNR, LOG_DIR, PLOT_DIR,
                                    DISTURB_PROBES, META_DISTURB_MAG_RANGES,
                                    EVAL_EXTRAPOLATION,
                                    EVAL_COMBINATIONS, EVAL_TIERS,
                                    EVAL_MISMATCH)
from metarl.training.arms import (ARMS, ARM_ORDER, CONTRASTS, arm_paths,
                                  _build_agent,
                                  _load_preferring_best)
from metarl.training.tasks import (seed_everything, make_arm_env, make_task_pool,
                                   as_fault_schedule, pattern_schedule)
from metarl.training.rollouts import scenario_rollouts
from metarl.training.workers import RolloutPool
from metarl.training.metrics import (latent_separability, gate_opening,
                                     gate_opening_in_stage, _finite_mean,
                                     _paired_delta, _verdict_sign, _fmt_delta,
                                     _fmt_delta_sci)
from metarl.training.logs import _json_default
from metarl.training.plots import (_plot_comparison, _plot_tiers,
                                   plot_training_curves, scenario_probe)

# The evaluation's task pools (``eval_task_pools``).
EVAL_ID_SEED = 12345
EVAL_OOD_SEED = 99999


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


def _z_label(agent):
    """What a report records as an agent's z mode."""
    return EVAL_Z_MODE if agent.use_context else "n/a (z=0)"


def _fault_targets(disturbance, n):
    """Per episode, the reactor its first fault hits (None: none or plant-wide)."""
    targets = []
    for i in range(n):
        sched = as_fault_schedule(disturbance, i)
        targets.append(sched[0]["reactor"] if sched else None)
    return targets


def _pad_mean(seqs, length):
    """Per-step mean over episodes of series up to ``length`` long (NaN-padded)."""
    m = np.full((len(seqs), length), np.nan)
    for i, s in enumerate(seqs):
        m[i, :len(s)] = s
    return np.nanmean(m, axis=0)


def _traces(rows, targets, trace_probe):
    """Each episode's probed actuator, disturbed quantity and storage."""
    key, _, a_idx = DISTURB_PROBES[trace_probe][:3]

    def actuator(r, t):
        act = r["act_traj"].reshape(-1, N_REACTORS, ACTION_DIM_PER_REACTOR)
        return act[:, :, a_idx].mean(axis=1) if t is None else act[:, t, a_idx]

    def quantity(r, t):
        q = r["probe_traj"][key]
        if q.ndim == 1:
            return q
        return q.mean(axis=1) if t is None else q[:, t]

    return {
        "probe": key,
        "reactor": [(-1 if t is None else int(t)) for t in targets],
        "act": [np.round(actuator(r, t), 4).tolist()
                for r, t in zip(rows, targets)],
        "quantity": [[float(f"{v:.5g}") for v in quantity(r, t)]
                     for r, t in zip(rows, targets)],
        "storage": [np.round(r["storage_traj"], 1).tolist() for r in rows],
    }


def _gate_openings(rows, targets, split):
    """Mean gate opening before and after the outer step ``split`` (None: no gate)."""
    out = {}
    for k, lo, hi in (("gate_pre", 0, split + 1), ("gate_post", split + 1, None)):
        v = _finite_mean([gate_opening(r["gate_traj"], lo, hi) for r in rows])
        out[k] = v if np.isfinite(v) else None
    for k, stage in (("gate_post_ferment", 2), ("gate_post_unload", 3)):
        v = _finite_mean([gate_opening_in_stage(r["gate_traj"], r["stage_traj"],
                                                stage, t, split + 1)
                          for r, t in zip(rows, targets)])
        out[k] = v if np.isfinite(v) else None
    return out


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
    targets = _fault_targets(disturbance, len(rows))

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    z_mus = np.stack([r["z_mu"] for r in rows])
    sep = latent_separability(z_mus, [r["z_sigma_sq"] for r in rows])
    acts = np.stack([r["act_mean"] for r in rows])            # (E, N, 3)
    L = max(len(r["reward_traj"]) for r in rows)
    mean_r = _pad_mean([r["reward_traj"] for r in rows], L)
    mean_v = _pad_mean([r["viol_traj"] for r in rows], L)
    by_c = {cid: [float(r["viol_by_constraint"].get(cid, 0)) for r in rows]
            for cid in CONSTRAINT_NAMES}
    burden_c = {cid: [float(r["burden_by_constraint"].get(cid, 0.0))
                      for r in rows]
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
        # Violation burden (metrics.violation_burden) and net production
        # (metrics.net_production): the two numbers a report leads with.
        "burden": float(np.mean(col("burden"))),
        "burden_std": float(np.std(col("burden"))),
        "burden_p90": float(np.percentile(col("burden"), 90)),
        "burden_by_constraint": {CONSTRAINT_NAMES[cid]: float(np.mean(v))
                                 for cid, v in burden_c.items()},
        "score": float(np.mean(col("score"))),
        "score_std": float(np.std(col("score"))),
        # The reward's flow-change penalty over an episode (reward units).
        "smoothing": float(np.mean(col("smoothing"))),
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
        "proj_calls": int(calls),
        "proj_moved_frac": float(moved / max(calls, 1)) if calls else None,
        "fault_reactor_per_ep": [(-1 if t is None else int(t)) for t in targets],
        "viol_per_ep": col("violations").tolist(),
        "burden_per_ep": col("burden").tolist(),
        "score_per_ep": col("score").tolist(),
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
        out["traces"] = _traces(rows, targets, trace_probe)
    sched0 = as_fault_schedule(disturbance, 0)
    split = (int(min(d["onset"] for d in sched0) // interval)
             if sched0 else L // 2)
    out.update(_gate_openings(rows, targets, split))
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


def extreme_magnitude(label):
    """Tier 1's magnitude for EVAL_DISTURBANCES ``label``, past training.

    The training range's edge on the evaluation magnitude's side of nominal
    (1.0), EVAL_EXTRAPOLATION times as far from nominal in log space; an
    additive kind (nominal 0) takes EVAL_EXTRAPOLATION x its upper edge.
    """
    d = EVAL_DISTURBANCES[label]
    lo, hi = META_DISTURB_MAG_RANGES[d["kind"]]
    if d["kind"] in ADDITIVE_DISTURBANCES:
        return float(EVAL_EXTRAPOLATION * hi)
    edge = lo if d["magnitude"] < 1.0 else hi
    return float(edge ** EVAL_EXTRAPOLATION)


def eval_tier_scenarios(onset, horizon):
    """Tier 1's extrapolated magnitudes and tier 2's combinations.

    Returns:
        Dict scenario name -> (tier, fault schedule): "<label>_extreme" (T1,
        a step at ``onset``) and "combo_<name>" (T2, EVAL_COMBINATIONS).
    """
    out = {}
    for label, d in EVAL_DISTURBANCES.items():
        out[f"{label}_extreme"] = ("T1", pattern_schedule(
            d["kind"], extreme_magnitude(label), "step", onset, d["reactor"],
            horizon))
    for name, parts in EVAL_COMBINATIONS.items():
        sched = []
        for label, pattern, t0, shift in parts:
            d = EVAL_DISTURBANCES[label]
            reactor = (f"rotate+{shift}" if d["reactor"] == "rotate" and shift
                       else d["reactor"])
            sched += pattern_schedule(
                d["kind"], d["magnitude"], pattern,
                onset if t0 is None else t0, reactor, horizon,
                spike_h=EVAL_SPIKE_H, gradual_h=EVAL_GRADUAL_H,
                treads=META_DRIFT_TREADS, windows=EVAL_RECURRENT_WINDOWS)
        out[f"combo_{name}"] = ("T2", sched)
    return out


def with_mismatch(tasks, spec):
    """Copies of ``tasks`` whose plant runs the mismatch ``spec``."""
    return [{**t, "mismatch": dict(spec)} for t in tasks]


def eval_mismatch_scenarios(tasks):
    """Tier 3: each EVAL_MISMATCH case on copies of ``tasks``, undisturbed.

    Returns:
        Dict "mismatch_<name>" -> {"tasks", "dist" (None), "tier" ("T3")}.
        Each case is its own task pool, so it runs its own episodes from t = 0.
    """
    return {f"mismatch_{name}": {"tasks": with_mismatch(tasks, spec),
                                 "dist": None, "tier": "T3"}
            for name, spec in EVAL_MISMATCH.items()}


def eval_task_pools(n):
    """``evaluate``'s first ``n`` in-distribution and out-of-distribution tasks."""
    return (make_task_pool(n, seed=EVAL_ID_SEED, ood=False),
            make_task_pool(n, seed=EVAL_OOD_SEED, ood=True))


def eval_scenarios(id_tasks, ood_tasks, onset, horizon):
    """Every scenario ``evaluate`` runs, in report order: the three tiers.

    The fault grid (``eval_fault_scenarios``) is meta's training
    distribution, so it is no fair test against saferl and is not run here.

    Args:
        id_tasks, ood_tasks: The task pools (``eval_task_pools``).
        onset: First inner step of the stepped faults.
        horizon: Episode length in inner steps.

    Returns:
        Dict name -> {"tasks", "dist" (fault schedule or None), "tier"}.
    """
    scenarios = {"out_of_distribution": {"tasks": ood_tasks, "dist": None,
                                         "tier": "T1"}}
    for name, (tier, dist) in eval_tier_scenarios(onset, horizon).items():
        scenarios[name] = {"tasks": id_tasks, "dist": dist, "tier": tier}
    scenarios.update(eval_mismatch_scenarios(id_tasks))
    return scenarios


def ablation_scenarios(id_tasks, ood_tasks, onset, horizon):
    """The component ablation's scenarios: both undisturbed pools and the grid.

    Every ablation variant trains on the grid's distribution, so it compares
    them fairly. Untiered.
    """
    scenarios = {
        "in_distribution": {"tasks": id_tasks, "dist": None, "tier": None},
        "out_of_distribution": {"tasks": ood_tasks, "dist": None,
                                "tier": None},
    }
    for name, dist in eval_fault_scenarios(onset, horizon).items():
        scenarios[name] = {"tasks": id_tasks, "dist": dist, "tier": None}
    return scenarios


def scenario_groups(scenarios):
    """(tasks, scenario names) per task pool, in first-appearance order.

    Each task runs every scenario of its group from one undisturbed prefix
    (``scenario_rollouts``).
    """
    groups = {}
    for name, sc in scenarios.items():
        groups.setdefault(id(sc["tasks"]), (sc["tasks"], []))[1].append(name)
    return list(groups.values())


def _by_scenario(names, task_rows):
    """Per-task rows over ``names`` regrouped as {name: rows in task order}."""
    return {name: [r[k] for r in task_rows] for k, name in enumerate(names)}


def scenario_rows_in_process(env, agent, groups, scenarios):
    """One arm's episode rows on every scenario, in this process: {name: rows}."""
    out = {}
    for tasks, names in groups:
        dists = [scenarios[n]["dist"] for n in names]
        out.update(_by_scenario(names, [
            scenario_rollouts(env, agent, t, dists, EVAL_Z_MODE, index=i)
            for i, t in enumerate(tasks)]))
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

    id_tasks, ood_tasks = eval_task_pools(EVAL_EPISODES)
    scenarios = eval_scenarios(id_tasks, ood_tasks, env.max_steps // 2,
                               env.max_steps)
    rows = _scenario_rows(envs, agents, arms, scenario_groups(scenarios),
                          scenarios)

    results = {}
    for name, sc in scenarios.items():
        results[name] = {
            arm: _aggregate_rows(
                rows[arm][name], sc["dist"],
                envs[arm].control_action_interval, _z_label(agents[arm]),
                trace_probe=scenario_probe(name))
            for arm in arms
        }

    summary = _eval_summary(results, arms)
    summary["tiers"] = _tier_summary(
        results, summary, {n: sc["tier"] for n, sc in scenarios.items()})
    _print_eval(results, summary)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "eval.json"), "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    with open(os.path.join(LOG_DIR, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\nwrote {os.path.join(LOG_DIR, 'eval.json')} and eval_summary.json")
    _plot_comparison(results)
    _plot_tiers(summary["tiers"])
    plot_training_curves()
    return results


def _scenario_rows(envs, agents, arms, groups, scenarios):
    """Every arm's episode rows on every scenario: {arm: {scenario: rows}}.

    Args:
        groups: (tasks, scenario names) pairs (``scenario_groups``).
    """
    n_eps = sum(len(t) * len(names) for t, names in groups) * len(arms)
    t0 = time.time()
    out = None
    if EVAL_WORKERS > 1:
        out = _scenario_rows_pooled(agents, arms, groups, scenarios, n_eps)
    if out is None:
        print(f"[eval] {n_eps} episodes in-process ...")
        out = {arm: scenario_rows_in_process(envs[arm], agents[arm], groups,
                                             scenarios)
               for arm in arms}
    print(f"[eval] rollouts done in {time.time() - t0:.0f}s")
    return out


def _scenario_rows_pooled(agents, arms, groups, scenarios, n_eps):
    """``_scenario_rows`` in ``EVAL_WORKERS`` processes; None if the pool fails."""
    plan = [(arm, tasks, names) for arm in arms for tasks, names in groups]
    pool = None
    try:
        pool = RolloutPool(None, EVAL_WORKERS)
        print(f"[eval] {n_eps} episodes in {EVAL_WORKERS} worker "
              f"processes ...")
        futures = [pool.scenarios_async(
            agents[arm], tasks, [scenarios[n]["dist"] for n in names],
            EVAL_Z_MODE, [scenario_probe(n) is not None for n in names],
            arm=arm) for arm, tasks, names in plan]
        per_task = [[f.result() for f in fs] for fs in futures]
    except Exception as exc:                              # pragma: no cover
        print(f"[eval] worker pool failed ({exc!r}); running in-process")
        return None
    finally:
        if pool is not None:
            pool.close()
    out = {arm: {} for arm in arms}
    for (arm, _, names), task_rows in zip(plan, per_task):
        out[arm].update(_by_scenario(names, task_rows))
    return out

def _score_delta(a, b):
    """Paired net-production gap a - b, relative to b's on-spec ethanol.

    Net production can sit near zero, so b's gross output is the scale the
    relative gap and the negligibility rule read.
    """
    return _paired_delta(a.get("score_per_ep", []), b.get("score_per_ep", []),
                         ref=b.get("on_spec_ethanol"))


def _eval_verdict(res, arm_a, arm_b):
    """One contrast's paired verdict on one scenario, on both axes.

    Safety is the violation burden, output the on-spec ethanol; ``d_score``
    is the net-production gap that folds both (and actuator wear) into one.
    """
    a, b = res[arm_a], res[arm_b]
    d_viol = _paired_delta(b.get("viol_per_ep", []), a.get("viol_per_ep", []))
    d_burden = _paired_delta(b.get("burden_per_ep", []),
                             a.get("burden_per_ep", []))
    d_eth = _paired_delta(a.get("on_spec_ethanol_per_ep", []),
                          b.get("on_spec_ethanol_per_ep", []))
    d_score = _score_delta(a, b)
    d_titer = a["mean_titer"] - b["mean_titer"]
    sv, sp = _verdict_sign(d_burden), _verdict_sign(d_eth)
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
    return {"d_viol": d_viol, "d_burden": d_burden, "d_eth": d_eth,
            "d_score": d_score, "d_titer": d_titer, "verdict": verdict}


def _eval_summary(results, arms):
    """Derived, per-scenario verdicts for every reportable contrast."""
    usable = [c for c in CONTRASTS if c[1] in arms and c[2] in arms]
    contrasts = {
        label: {name: _eval_verdict(res, arm_a, arm_b)
                for name, res in results.items()}
        for label, arm_a, arm_b, _ in usable
    }
    out = {"contrasts": contrasts, "headline": _headline(results, arms)}
    if "meta" in arms:
        snr = float(np.mean([res["meta"]["z_snr"] for res in results.values()]))
        out.update(z_snr_mean=snr, z_snr_pass=bool(snr >= Z_INFORM_MIN_SNR))
    return out


def _headline(results, arms):
    """Per arm, burden, on-spec ethanol and net production over scenarios.

    Each scenario weighs equally. ``scenarios_score_better`` counts the
    scenarios on which the arm's net production is significantly above every
    other arm's, paired over the same episodes.
    """
    out = {arm: {k: float(np.mean([res[arm][k] for res in results.values()]))
                 for k in ("burden", "on_spec_ethanol", "score")}
           for arm in arms}
    for arm in arms:
        out[arm]["scenarios_score_better"] = int(sum(
            all(_verdict_sign(_score_delta(res[arm], res[o])) > 0
                for o in arms if o != arm)
            for res in results.values()))
    out["scenarios"] = len(results)
    return out


def _tier_summary(results, summary, tiers, arm_a="meta", arm_b="saferl"):
    """ADAPT per tier: significant wins per side, and the pooled gaps.

    ``burden_reduction`` is 1 - (meta's summed burden / saferl's) over the
    tier's scenarios; ``score_gain`` the summed net-production gap over
    saferl's summed on-spec ethanol (``_score_delta``'s scale). Wins are the
    per-scenario paired verdicts (``_verdict_sign``).

    Returns:
        Dict tier -> its numbers and scenario names; {} without both arms.
    """
    contrast = next((summary["contrasts"][label] for label, a, b, _
                     in CONTRASTS if (a, b) == (arm_a, arm_b)
                     and label in summary["contrasts"]), None)
    if contrast is None:
        return {}
    out = {}
    for tier, label in EVAL_TIERS.items():
        names = [n for n in results if tiers.get(n) == tier]
        if not names:
            continue

        def total(arm, k):
            return float(sum(results[n][arm][k] for n in names))

        def wins(key):
            s = [_verdict_sign(contrast[n][key]) for n in names]
            return {arm_a: s.count(1), arm_b: s.count(-1), "tied": s.count(0)}

        b_ref, e_ref = total(arm_b, "burden"), total(arm_b, "on_spec_ethanol")
        out[tier] = {
            "label": label, "n": len(names), "scenarios": names,
            "burden_wins": wins("d_burden"), "score_wins": wins("d_score"),
            "burden_reduction": (1.0 - total(arm_a, "burden") / b_ref
                                 if b_ref > 0 else float("nan")),
            "score_gain": ((total(arm_a, "score") - total(arm_b, "score"))
                           / e_ref if e_ref > 0 else float("nan")),
        }
    return out


def _print_tiers(tiers):
    """One line per tier: meta's pooled gaps and its significant wins."""
    if not tiers:
        return
    print("\nADAPT BY TIER (meta vs saferl, paired; wins = significant "
          "scenarios, meta / saferl / tied)")
    print(f"  {'tier':<5}{'':<60}{'n':>4}{'burden cut':>12}{'B wins':>12}"
          f"{'Pi gain':>10}{'Pi wins':>12}")
    for tier, t in tiers.items():
        bw, sw = t["burden_wins"], t["score_wins"]
        print(f"  {tier:<5}{t['label']:<60}{t['n']:>4}"
              f"{t['burden_reduction'] * 100:>+11.1f}%"
              f"{bw['meta']:>5}/{bw['saferl']}/{bw['tied']:<4}"
              f"{t['score_gain'] * 100:>+9.1f}%"
              f"{sw['meta']:>5}/{sw['saferl']}/{sw['tied']:<4}")


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
    print("\nB: violation burden, exceedance area past the limits in "
          "limit-hours (lower is better).  Pi: net production, on-spec ethanol less "
          "violations and wear at the reward's rates (higher is better).")
    print(f"\n{'scenario':<26}" + "".join(
        f"{a + ' B':>12}{a + ' on-spec':>16}{a + ' Pi':>14}" for a in arms))
    for name, res in results.items():
        print(f"{name:<26}" + "".join(
            f"{res[a]['burden']:>12.2f}{res[a]['on_spec_ethanol']:>16.3e}"
            f"{res[a]['score']:>14.3e}" for a in arms))
    head = summary["headline"]
    print(f"{'mean over scenarios':<26}" + "".join(
        f"{head[a]['burden']:>12.2f}{head[a]['on_spec_ethanol']:>16.3e}"
        f"{head[a]['score']:>14.3e}" for a in arms))
    print("Pi significantly higher on " + ", ".join(
        f"{a}: {head[a]['scenarios_score_better']}" for a in arms)
        + f" of {head['scenarios']} scenarios")
    for label, per_scenario in summary["contrasts"].items():
        print(f"\n{label}:")
        for name, v in per_scenario.items():
            print(f"  {name:<26}{v['verdict']:<30}"
                  f"B {_fmt_delta(v['d_burden']):<22}"
                  f"eth {_fmt_delta_sci(v['d_eth']):<26}"
                  f"Pi {_fmt_delta_sci(v['d_score'])}")
    _print_tiers(summary.get("tiers", {}))
    print(f"\nfull per-scenario tables: eval.json  |  figures: {PLOT_DIR}  |  "
          f"redraw figures only: python -m metarl plot_eval")
