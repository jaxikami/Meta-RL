"""Ablation study of the meta arm's components.

Trains and evaluates variants of the ``meta`` arm in which individual
components are switched off, to measure what each one contributes to
production and constraint satisfaction. It is the Safe PPO study's design
(``Fermentation-PPO/ablation.py``) on this study's arms: every variant is an
``ArmSpec`` that differs from ``ARMS["meta"]`` only in the switches below, and
is trained, selected and evaluated by the study's own code.

Components (all on in ``full``, which is ``ARMS["meta"]``):
    z           Latent task inference: the GRU posterior, FiLM conditioning,
                the aux head, the privileged critics and the adaptation gate.
                Ablated by ``use_context=False``: a context-free SAC agent on
                the same disturbed task stream, warm-started from saferl and
                anchored to it, acting without a gate.
    multitask   The disturbed training distribution: a fresh prior task every
                episode and mid-episode faults. Ablated by
                ``multitask=False``: meta's machinery trained on the nominal
                task only. Its gate's detector never sees a fault, and its
                held-out selection runs the undisturbed checks only, as
                saferl's does.
    lagrangian  Adaptive dual multipliers. Ablated by ``lagrangian=False``:
                the duals stay at 0. The static shaping (violation spike, soft
                walls, purity and lot terms) is kept, as for baseline.
    mask        Structural stage mask. Ablated by ``action_mask=False``. The
                plant has no interlock (baseline runs the same way), so the
                unmasked action reaches the shield and the plant: the variant
                can open a discharge valve mid-ferment. The shield was trained
                on stage-masked flows and meets off-distribution actions here.
                (The PPO study kept a plant-side interlock; this plant has none
                to keep.)
    apn         The hard inventory shield. Ablated by ``apn=False``: no
                projection and no mapping penalty. No-regression selection
                still compares against saferl as it runs, shield included.
    warmstart   Starting from saferl's trained policy. Ablated by
                ``warm_start=None``: trained from scratch, which also removes
                everything the warm start carries: the anchor, the weight decay
                toward saferl, the actor warm-up, the adaptation gate (saferl is
                its base) and no-regression selection.

What the switches cannot separate:
    The adaptation gate is on exactly when z and the warm start both are
    (``arm_gated``), so it has no switch of its own; ``no_z`` and
    ``no_warmstart`` both remove it. ``ADAPT = meta - saferl`` bundles the
    disturbed distribution and inference; ``--design factorial --components
    z multitask`` splits it: full, no_z, no_multitask and no_z_multitask
    (saferl fine-tuned on the nominal task under meta's anchor), each read
    against the saferl reference row.

Protocol:
    Seed k runs under the run tag ``<tag>/seed<k>`` (``PLANT_SEED``,
    ``PLANT_RUN_TAG``): checkpoints in ``policy/<tag>/seed<k>/``, logs in
    ``results/log/<tag>/seed<k>/``. The seed's saferl, the warm-start source
    and the reference row, trains there first with the study's protocol
    unless it is already there (copy ``policy/saferl.pth`` and
    ``saferl_best.pth`` in to reuse one). Each variant then trains with
    ``metarl.training.train._train`` unchanged: the same collection, updates,
    held-out checks, feasibility gate, no-regression selection (warm-started
    variants) and early stopping as meta, so variants can stop after
    different numbers of iterations; summary.csv reports them. ``--budget N``
    instead runs every variant for exactly N iterations with early stopping
    off (selection unchanged), comparing them at equal sample counts as the
    PPO study does; saferl keeps the study's protocol either way.

    Every variant and the seed's saferl are evaluated on ``evaluate``'s
    scenarios and task pools (their first ``--eval-episodes`` tasks), with
    the instrument noise, so within a seed every result is paired per task.
    Violations are events (see the README); severity is the worst exceedance
    of each depth constraint over every scenario and episode, as a percentage
    of its limit. Each delta against full is paired within a seed; a seed
    counts as "worse" or "better" only when its paired 95% interval excludes
    zero and the gap is not negligible (the evaluation's SIG rule).

    Every (config, seed) run is its own process, because the study reads its
    seed and output folders at import. Its held-out workers re-run this module
    when they spawn, which registers the variants in ``ARMS`` before they
    build an agent; if that ever fails, train.py runs the checks in-process.

Cost:
    Every variant is a meta-sized training run: leave-one-out over three
    seeds is 21 of them plus three saferl runs. ``--workers`` runs several
    at once and splits the held-out workers and CPU threads between them.

Outputs:
    results/log/<tag>/seed<k>/ablation_<config>.json   switches, training
                                                       stats and evaluation
    results/log/<tag>/seed<k>/abl_<config>_log.json    training log (the
                                                       study's format)
    policy/<tag>/seed<k>/abl_<config>[_best].pth       checkpoints
    results/log/<tag>/summary.csv          per (config, scenario): mean and
                                           std over seeds, paired deltas vs full
    results/plot/<tag>/ablation_curves.png     held-out learning curves vs full
    results/plot/<tag>/ablation_final.png      violations and on-spec ethanol
                                               per scenario, against full
    results/plot/<tag>/ablation_severity.png   worst exceedance per constraint

A run whose record holds an evaluation is skipped, so re-running the same
command resumes the study. A trained run without one (or evaluated with a
different ``--eval-episodes``) is evaluated from its checkpoint without
retraining; a run interrupted in training restarts from scratch.

Usage (from the repository root):
    python -m metarl.ablation                          # leave-one-out, seeds 0-2
    python -m metarl.ablation --configs full no_z --seeds 0 1 2 3 4
    python -m metarl.ablation --design factorial --components z multitask
    python -m metarl.ablation --budget 1500            # equal iterations
    python -m metarl.ablation --workers 2              # two runs at a time
    python -m metarl.ablation --dry-run                # list the runs only
    python -m metarl.ablation --summarize-only         # re-aggregate, re-plot
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import argparse
import csv
import glob
import itertools
import json
import math
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.training.config import (SEED, LOG_DIR, EVAL_EPISODES,
                                    CONSTRAINT_NAMES, DEPTH_CONSTRAINTS,
                                    META_ACTOR_WARMUP_ITERS)
from metarl.training.arms import (ARMS, ArmSpec, get_arm, arm_paths,
                                  _build_agent, _load_preferring_best)
from metarl.training.logs import _json_default
from metarl.training.metrics import _paired_delta, _verdict_sign

# ─── Study Defaults ───────────────────────────────────────────────────────────
DEFAULT_TAG = "ablation"
DEFAULT_SEEDS = [0, 1, 2]
# The reference row: the seed's saferl, also every warm-started variant's
# source.
REFERENCE = "saferl"
# ``evaluate``'s task pools.
EVAL_ID_SEED = 12345
EVAL_OOD_SEED = 99999
DEPTH_NAMES = tuple(CONSTRAINT_NAMES[c] for c in DEPTH_CONSTRAINTS)


# ─── Ablation Configurations ──────────────────────────────────────────────────

COMPONENTS = ("z", "multitask", "lagrangian", "mask", "apn", "warmstart")
COMPONENT_LABELS = {"z": "z", "multitask": "multi-task",
                    "lagrangian": "Lagrangian", "mask": "mask", "apn": "APN",
                    "warmstart": "warm start"}


@dataclass(frozen=True)
class AblationConfig:
    """Which of meta's components are active in one variant.

    Attributes:
        z: Latent task inference (``use_context``).
        multitask: Disturbed multi-task training (``multitask``).
        lagrangian: Adaptive dual multipliers (``lagrangian``).
        mask: Structural stage mask (``action_mask``).
        apn: Hard inventory shield (``apn``).
        warmstart: Warm start from saferl (``warm_start``).
    """
    z: bool = True
    multitask: bool = True
    lagrangian: bool = True
    mask: bool = True
    apn: bool = True
    warmstart: bool = True

    @property
    def removed(self):
        """Components switched off in this variant, in COMPONENTS order."""
        return [c for c in COMPONENTS if not getattr(self, c)]

    @property
    def name(self):
        """Identifier used on the command line and in output paths."""
        if not self.removed:
            return "full"
        if len(self.removed) == len(COMPONENTS):
            return "none"
        return "no_" + "_".join(self.removed)

    @property
    def label(self):
        """Human-readable name for tables and plots."""
        if not self.removed:
            return "Full"
        if len(self.removed) == len(COMPONENTS):
            return "w/o all"
        return "w/o " + " + ".join(COMPONENT_LABELS[c] for c in self.removed)

    @property
    def arm(self):
        """Key of the variant's ``ArmSpec`` in ``ARMS`` (and its file stem)."""
        return f"abl_{self.name}"

    def spec(self):
        """The variant's ``ArmSpec``: ``ARMS["meta"]`` less the removed parts."""
        meta = ARMS["meta"]
        return ArmSpec(
            self.arm, self.arm.upper(),
            multitask=meta.multitask and self.multitask,
            use_context=meta.use_context and self.z,
            lagrangian=meta.lagrangian and self.lagrangian,
            action_mask=meta.action_mask and self.mask,
            apn=meta.apn and self.apn,
            apn_soft=meta.apn_soft and self.apn,
            warm_start=meta.warm_start if self.warmstart else None,
            train_sensor_noise=meta.train_sensor_noise,
            blurb=f"ablation of meta: {self.label}")


# Every combination, ordered full -> single ablations -> pairs -> ... -> none.
ALL_CONFIGS = {
    cfg.name: cfg for cfg in sorted(
        (AblationConfig(*flags)
         for flags in itertools.product((True, False), repeat=len(COMPONENTS))),
        key=lambda c: (len(c.removed), [COMPONENTS.index(r) for r in c.removed]))
}
LOO_CONFIGS = ["full"] + [f"no_{c}" for c in COMPONENTS]

# Registered at import, here and in every held-out worker (which re-runs this
# module on spawn), so ``get_arm`` and ``ARMS`` resolve every variant.
for _cfg in ALL_CONFIGS.values():
    ARMS.setdefault(_cfg.arm, _cfg.spec())


def factorial_configs(components):
    """Every config whose removed components all lie in ``components``."""
    keep = set(components)
    return [n for n, c in ALL_CONFIGS.items() if set(c.removed) <= keep]


def config_label(name):
    """Display label of a config or of the reference row."""
    if name == REFERENCE:
        return "saferl (reference)"
    return ALL_CONFIGS[name].label


def _spec_of(name):
    """The ``ArmSpec`` a config (or the reference) trains."""
    return get_arm(REFERENCE if name == REFERENCE else ALL_CONFIGS[name].arm)


# ─── Paths and Records ────────────────────────────────────────────────────────

def _run_tag(tag, seed):
    """``PLANT_RUN_TAG`` of one seed's runs."""
    return f"{tag}/seed{seed}"


def _seed_log_dir(tag, seed):
    """``LOG_DIR`` of one seed's runs, as its processes compute it."""
    return os.path.join("results", "log", _run_tag(tag, seed))


def _record_path(log_dir, name):
    """One run's record: switches, training stats and evaluation."""
    return os.path.join(log_dir, f"ablation_{name}.json")


def _read_json(path):
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def _write_json(path, obj):
    """Write a JSON file atomically, so a reader never sees half of one."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


# ─── One Run (inside its own process) ─────────────────────────────────────────

# Per-scenario results kept in a record: the aggregates, plus the per-episode
# arrays the paired deltas need. Trajectories and traces are dropped.
EVAL_KEYS = ("violations", "violations_std", "violations_median",
             "violations_worst", "episodes_violated", "episodes",
             "feasible_rate", "viol_by_constraint", "viol_by_reactor",
             "ethanol", "on_spec_ethanol", "on_spec_ethanol_std",
             "reactor_on_spec_ethanol", "mean_titer", "purity_rate",
             "lot_pass_rate", "batches", "storage_min", "storage_max",
             "yield_per_feed", "z_snr", "proj_calls", "proj_moved_frac",
             "gate_pre", "gate_post", "fault_reactor_per_ep", "viol_per_ep",
             "on_spec_ethanol_per_ep")


def _compact(res):
    """Keep EVAL_KEYS of one scenario's results and add the severities."""
    out = {k: res[k] for k in EVAL_KEYS if k in res}
    peaks = {c: 100.0 * max(v, default=0.0)
             for c, v in res["viol_depth_max_per_ep"].items()}
    out["peak_pct"] = peaks
    out["peak_violation_pct"] = max(peaks.values(), default=0.0)
    out["episodes_violated_pct"] = (100.0 * res["episodes_violated"]
                                    / max(res["episodes"], 1))
    return out


def evaluate_run(spec, n_episodes):
    """Evaluate one trained arm on every scenario ``evaluate`` runs.

    Args:
        spec: The arm's ``ArmSpec``; its checkpoint is read from POLICY_DIR.
        n_episodes: Tasks per scenario (the first of evaluate's pools).

    Returns:
        Dict scenario name -> compact results.
    """
    from metarl.training.tasks import seed_everything, make_env, make_task_pool
    from metarl.training.evaluate import (_eval_agent_on_tasks,
                                          eval_fault_scenarios)
    seed_everything(SEED)
    env = make_env(lagrangian=spec.lagrangian)
    agent = _build_agent(env, spec)
    final_path, best_path = arm_paths(spec)
    _load_preferring_best(agent, best_path, final_path, spec.name, strict=True)
    id_tasks = make_task_pool(n_episodes, seed=EVAL_ID_SEED, ood=False)
    ood_tasks = make_task_pool(n_episodes, seed=EVAL_OOD_SEED, ood=True)
    scenarios = {"in_distribution": (id_tasks, None),
                 "out_of_distribution": (ood_tasks, None)}
    for name, dist in eval_fault_scenarios(env.max_steps // 2).items():
        scenarios[name] = (id_tasks, dist)
    out = {}
    for name, (tasks, dist) in scenarios.items():
        print(f"[eval] {spec.name}: {name} ...", flush=True)
        out[name] = _compact(_eval_agent_on_tasks(env, agent, tasks, dist))
    return out


def _new_record(name, spec, budget):
    """A run record without training stats or evaluation."""
    from metarl.agent.common import device
    cfg = ALL_CONFIGS.get(name)
    return {"config": name,
            "label": config_label(name),
            "seed": SEED,
            "arm": spec.name,
            "arm_summary": spec.summary(),
            "components": ({c: getattr(cfg, c) for c in COMPONENTS}
                           if cfg is not None else None),
            "settings": {"budget": None if name == REFERENCE else budget,
                         "device": str(device)}}


def _run(name, eval_episodes, budget, eval_only):
    """Train (unless eval-only) and evaluate one config for this process's seed.

    Args:
        name: Config name, or REFERENCE.
        eval_episodes: Tasks per evaluation scenario.
        budget: Fixed training iterations with early stopping off, or None
            for the study's protocol. Never applied to the reference.
        eval_only: Reload the trained checkpoint and replace the evaluation.
    """
    spec = _spec_of(name)
    path = _record_path(LOG_DIR, name)
    log_path = os.path.join(LOG_DIR, f"{spec.label.lower()}_log.json")
    if eval_only and os.path.exists(path):
        record = _read_json(path)
    elif eval_only:
        # A reference checkpoint copied into the seed's policy folder.
        record = _new_record(name, spec, budget)
        record["train"] = {"hours": None, "iterations": None,
                           "heldout_checks": None, "copied": True,
                           "log": log_path if os.path.exists(log_path)
                           else None}
    else:
        from metarl.training import train as train_module
        if budget and name != REFERENCE:
            train_module.META_ITERATIONS = int(budget)
            train_module.EARLY_STOP_ENABLED = False
        record = _new_record(name, spec, budget)
        # No "train" section yet: a run interrupted from here retrains.
        _write_json(path, record)
        t0 = time.time()
        train_module._train(spec)
        log = _read_json(log_path)
        record["train"] = {
            "hours": (time.time() - t0) / 3600.0,
            "iterations": int(log["iter"][-1]) if log["iter"] else 0,
            "heldout_checks": len(log["heldout_iter"]),
            "log": log_path}
        _write_json(path, record)
    t0 = time.time()
    record["eval"] = evaluate_run(spec, eval_episodes)
    record["settings"]["eval_episodes"] = int(eval_episodes)
    record["settings"]["eval_hours"] = (time.time() - t0) / 3600.0
    # Written last: an evaluation in the record marks the run complete.
    _write_json(path, record)
    print(f"[{name} seed {SEED}] "
          + ("re-evaluated" if eval_only
             else f"trained in {record['train']['hours']:.2f} h "
                  f"({record['train']['iterations']} iterations)") + ":")
    for sc, e in record["eval"].items():
        print(f"  {sc:<26}{e['violations']:>8.2f} viol/ep  "
              f"{e['episodes_violated_pct']:>6.1f}% eps violated  "
              f"{e['on_spec_ethanol']:>11.3e} on-spec eth")


# ─── Scheduling (driver) ──────────────────────────────────────────────────────

def _status(tag, name, seed, settings, force):
    """What a run needs: "train", "eval" or "done".

    A reference with no record but a checkpoint in the seed's policy folder
    was copied there, and is evaluated as it is.
    """
    path = _record_path(_seed_log_dir(tag, seed), name)
    if force:
        return "train"
    if not os.path.exists(path):
        policy_dir = os.path.join("policy", _run_tag(tag, seed))
        copied = name == REFERENCE and any(
            os.path.exists(os.path.join(policy_dir, f))
            for f in ("saferl_best.pth", "saferl.pth"))
        return "eval" if copied else "train"
    rec = _read_json(path)
    if "train" not in rec:
        return "train"
    if (name != REFERENCE
            and rec["settings"].get("budget") != settings["budget"]):
        print(f"  [note] {name} seed {seed} was trained with budget "
              f"{rec['settings'].get('budget')}, not {settings['budget']}; "
              f"kept (--force retrains it)")
    if ("eval" not in rec
            or rec["settings"].get("eval_episodes") != settings["eval_episodes"]):
        return "eval"
    return "done"


def _launch(job, settings):
    """Run one job in a fresh process for its seed; returns the exit code."""
    name, seed, action = job
    tag = settings["tag"]
    env = dict(os.environ, PLANT_SEED=str(seed),
               PLANT_RUN_TAG=_run_tag(tag, seed))
    workers = settings["workers"]
    if workers > 1:
        # Split the held-out workers and the CPU threads between the runs.
        cpus = os.cpu_count() or 2
        env.setdefault("PLANT_HELDOUT_WORKERS",
                       str(max(1, (cpus // 2) // workers)))
        env.setdefault("OMP_NUM_THREADS", str(max(1, cpus // workers)))
    cmd = [sys.executable, "-m", "metarl.ablation", "--_run", name,
           "--eval-episodes", str(settings["eval_episodes"])]
    if settings["budget"]:
        cmd += ["--budget", str(settings["budget"])]
    if action == "eval":
        cmd.append("--_eval-only")
    if workers <= 1:
        return subprocess.run(cmd, env=env).returncode
    log_dir = _seed_log_dir(tag, seed)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"ablation_{name}.out"), "w") as out:
        return subprocess.run(cmd, env=env, stdout=out,
                              stderr=subprocess.STDOUT).returncode


def run_jobs(jobs, settings):
    """Execute jobs, reporting each failure without stopping the others.

    Returns:
        Set of (name, seed) that failed.
    """
    failed = set()

    def report(job, rc):
        name, seed, action = job
        if rc == 0:
            print(f"[{name} seed {seed}] {action} done", flush=True)
            return
        failed.add((name, seed))
        where = ""
        if settings["workers"] > 1:
            out = os.path.join(_seed_log_dir(settings["tag"], seed),
                               f"ablation_{name}.out")
            where = f"; output in {out}"
        print(f"[{name} seed {seed}] FAILED (exit {rc}){where}", flush=True)

    if settings["workers"] <= 1:
        for job in jobs:
            try:
                report(job, _launch(job, settings))
            except Exception:
                failed.add(job[:2])
                print(f"[{job[0]} seed {job[1]}] FAILED\n"
                      f"{traceback.format_exc()}", flush=True)
        return failed
    with ThreadPoolExecutor(max_workers=settings["workers"]) as pool:
        futures = {pool.submit(_launch, job, settings): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                report(job, future.result())
            except Exception:
                failed.add(job[:2])
                print(f"[{job[0]} seed {job[1]}] FAILED\n"
                      f"{traceback.format_exc()}", flush=True)
    return failed


# ─── Aggregation ──────────────────────────────────────────────────────────────

SCENARIO_METRICS = ("violations", "episodes_violated_pct", "on_spec_ethanol",
                    "mean_titer", "lot_pass_rate", "peak_violation_pct")
TRAIN_METRICS = (("train_hours", "hours"), ("train_iterations", "iterations"))


def load_results(tag):
    """Every evaluated run under the tag, by config in canonical order.

    Returns:
        Dict config name -> list of records sorted by seed (the reference
        last).
    """
    grouped = {}
    pattern = os.path.join("results", "log", tag, "seed*", "ablation_*.json")
    for path in glob.glob(pattern):
        rec = _read_json(path)
        if "eval" in rec:
            grouped.setdefault(rec["config"], []).append(rec)
    order = [n for n in ALL_CONFIGS if n in grouped]
    order += [REFERENCE] if REFERENCE in grouped else []
    return {n: sorted(grouped[n], key=lambda r: r["seed"]) for n in order}


def _aggregate(vals):
    """(mean, std over seeds) of the available values; None when missing."""
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return (float(np.mean(vals)) if vals else None,
            float(np.std(vals, ddof=1)) if len(vals) > 1 else None)


def _paired_vs_full(runs, full_by_seed, scenario):
    """Per-seed paired deltas of a config against full on one scenario.

    Returns:
        Dict: seeds paired, mean (over seeds) violation delta and relative
        on-spec ethanol delta (variant - full), and how many seeds were
        significantly worse or better on each axis.
    """
    d_viol, d_eth = [], []
    counts = {"viol_worse": 0, "viol_better": 0, "eth_worse": 0,
              "eth_better": 0}
    for run in runs:
        ref = full_by_seed.get(run["seed"])
        if ref is None or scenario not in ref["eval"]:
            continue
        a, b = run["eval"][scenario], ref["eval"][scenario]
        dv = _paired_delta(a["viol_per_ep"], b["viol_per_ep"])
        de = _paired_delta(a["on_spec_ethanol_per_ep"],
                           b["on_spec_ethanol_per_ep"])
        if not (dv["paired"] and de["paired"]):
            continue
        d_viol.append(dv["mean"])
        d_eth.append(100.0 * de["rel"] if np.isfinite(de["rel"]) else None)
        sv, se = _verdict_sign(dv), _verdict_sign(de)
        counts["viol_worse"] += sv > 0
        counts["viol_better"] += sv < 0
        counts["eth_worse"] += se < 0
        counts["eth_better"] += se > 0
    return {"paired_seeds": len(d_viol),
            "d_viol_vs_full": _aggregate(d_viol)[0],
            "d_eth_vs_full_pct": _aggregate(d_eth)[0],
            **{f"seeds_{k}": v for k, v in counts.items()}}


def summary_rows(results):
    """One row per (config, scenario): means, stds and deltas against full."""
    full_by_seed = {r["seed"]: r for r in results.get("full", [])}
    rows = []
    for name, runs in results.items():
        for sc in runs[0]["eval"]:
            evals = [r["eval"][sc] for r in runs if sc in r["eval"]]
            row = {"config": name, "label": config_label(name),
                   "scenario": sc, "seeds": len(evals)}
            for key in SCENARIO_METRICS:
                row[f"{key}_mean"], row[f"{key}_std"] = _aggregate(
                    [e.get(key) for e in evals])
            for c in DEPTH_NAMES:
                row[f"{c}_peak_pct_mean"], row[f"{c}_peak_pct_std"] = (
                    _aggregate([e["peak_pct"].get(c) for e in evals]))
            for key, field in TRAIN_METRICS:
                row[f"{key}_mean"], row[f"{key}_std"] = _aggregate(
                    [r["train"].get(field) for r in runs])
            if name == "full":
                row.update(paired_seeds=0, d_viol_vs_full=None,
                           d_eth_vs_full_pct=None)
            else:
                row.update(_paired_vs_full(runs, full_by_seed, sc))
            rows.append(row)
    full = {r["scenario"]: r for r in rows if r["config"] == "full"}
    for row in rows:
        ref = full.get(row["scenario"])
        row["eth_vs_full_pct"] = (
            100.0 * (row["on_spec_ethanol_mean"] / ref["on_spec_ethanol_mean"]
                     - 1.0)
            if ref and ref["on_spec_ethanol_mean"]
            and row["on_spec_ethanol_mean"] is not None else None)
    return rows


def severity_by_config(results):
    """Worst exceedance per depth constraint over every scenario and episode.

    Returns:
        Dict config -> {constraint: (mean, std) over seeds}, in percent.
    """
    out = {}
    for name, runs in results.items():
        out[name] = {c: _aggregate([max(e["peak_pct"].get(c, 0.0)
                                        for e in r["eval"].values())
                                    for r in runs])
                     for c in DEPTH_NAMES}
    return out


def _fmt(mean, std, spec):
    """Format 'mean +- std' for the console; '-' when unavailable."""
    if mean is None:
        return "-"
    if std is None:
        return format(mean, spec)
    return f"{format(mean, spec)} +- {format(std, spec)}"


def _fmt_seeds(row, axis):
    """'worse/better/paired' seed counts on one axis, '-' for full."""
    n = row.get("paired_seeds") or 0
    if not n:
        return "-"
    return f"{row[f'seeds_{axis}_worse']}/{row[f'seeds_{axis}_better']}/{n}"


def summarize(tag):
    """Aggregate every evaluated run under the tag; write the table and plots."""
    results = load_results(tag)
    if not results:
        print(f"No evaluated runs under "
              f"{os.path.join('results', 'log', tag, 'seed*')}.")
        return
    rows = summary_rows(results)
    log_dir = os.path.join("results", "log", tag)
    plot_dir = os.path.join("results", "plot", tag)
    os.makedirs(log_dir, exist_ok=True)
    fields = (["config", "label", "scenario", "seeds", "eth_vs_full_pct",
               "paired_seeds", "d_viol_vs_full", "seeds_viol_worse",
               "seeds_viol_better", "d_eth_vs_full_pct", "seeds_eth_worse",
               "seeds_eth_better"]
              + [f"{key}_{stat}" for key in SCENARIO_METRICS
                 + tuple(f"{c}_peak_pct" for c in DEPTH_NAMES)
                 + tuple(k for k, _ in TRAIN_METRICS)
                 for stat in ("mean", "std")])
    csv_path = os.path.join(log_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    W = 146
    print(f"\n{'=' * W}\n  ABLATION SUMMARY: mean +- std over seeds; deltas "
          f"vs Full are paired per task within a seed (seeds worse/better/"
          f"paired, by the evaluation's SIG rule)\n{'=' * W}")
    for sc in dict.fromkeys(r["scenario"] for r in rows):
        print(f"\n  {sc}")
        print(f"  {'Config':<34}{'Seeds':>5}{'Viol/ep':>16}{'Viol eps %':>14}"
              f"{'On-spec eth (kg)':>24}{'vs Full':>9}{'dViol':>9}"
              f"{'w/b/n':>8}{'dEth %':>9}{'w/b/n':>8}{'Iters':>12}")
        for row in (r for r in rows if r["scenario"] == sc):
            d_v, d_e = row["d_viol_vs_full"], row["d_eth_vs_full_pct"]
            vs = row["eth_vs_full_pct"]
            print(f"  {row['label']:<34}{row['seeds']:>5}"
                  f"{_fmt(row['violations_mean'], row['violations_std'], '.2f'):>16}"
                  f"{_fmt(row['episodes_violated_pct_mean'], row['episodes_violated_pct_std'], '.1f'):>14}"
                  f"{_fmt(row['on_spec_ethanol_mean'], row['on_spec_ethanol_std'], '.3e'):>24}"
                  f"{'-' if vs is None else f'{vs:+.1f}%':>9}"
                  f"{'-' if d_v is None else f'{d_v:+.2f}':>9}"
                  f"{_fmt_seeds(row, 'viol'):>8}"
                  f"{'-' if d_e is None else f'{d_e:+.1f}':>9}"
                  f"{_fmt_seeds(row, 'eth'):>8}"
                  f"{_fmt(row['train_iterations_mean'], row['train_iterations_std'], '.0f'):>12}")
    print(f"{'=' * W}\n")

    plot_curves(_load_curves(results), os.path.join(plot_dir,
                                                    "ablation_curves.png"))
    plot_final(rows, os.path.join(plot_dir, "ablation_final.png"))
    plot_severity(severity_by_config(results),
                  os.path.join(plot_dir, "ablation_severity.png"))
    print(f"wrote {csv_path} and ablation_curves.png, ablation_final.png, "
          f"ablation_severity.png in {plot_dir}")


# ─── Plotting ─────────────────────────────────────────────────────────────────

# Light-surface chart tokens: one accent series against a gray reference.
_SURFACE = "#fcfcfb"
_REF = "#898781"
_ACCENT = "#2a78d6"
# Diverging pair, better than full -> no difference -> worse than full.
_BETTER, _NEUTRAL, _WORSE = "#2a78d6", "#f0efec", "#e34948"
# Sequential blue ramp (steps 100 -> 700) for magnitudes.
_SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
               "#184f95", "#0d366b")
CURVE_KEYS = (("heldout_undisturbed_violations", "violation events / episode",
               1.0),
              ("heldout_undisturbed_on_spec_ethanol", "on-spec ethanol (1e7 kg)",
               1e-7))


def _pyplot():
    """matplotlib.pyplot on the Agg backend, or None if unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:                             # pragma: no cover
        print(f"[plot] skipped: {exc}")
        return None


def _scenario_label(name):
    """Scenario tick label, as the evaluation figures write it."""
    return name.replace("disturb_", "D ").replace("_", " ")


def _load_curves(results):
    """Held-out checks of each config's runs, from their training logs.

    Returns:
        Dict config -> list (one per seed) of dicts with 'iter' and the
        CURVE_KEYS arrays.
    """
    curves = {}
    for name, runs in results.items():
        if name == REFERENCE:
            continue
        for run in runs:
            path = run["train"].get("log")
            if not path or not os.path.exists(path):
                continue
            lg = _read_json(path)
            curves.setdefault(name, []).append(
                {"iter": np.asarray(lg["heldout_iter"], dtype=float),
                 **{k: np.asarray(lg[k], dtype=float) for k, _, _ in CURVE_KEYS}})
    return curves


def _grid_stats(runs, key):
    """Mean and std over seeds on the union of the runs' check iterations.

    A seed contributes only within its own range (early stopping ends seeds
    at different iterations); the std is 0 where one seed remains.
    """
    xs = np.unique(np.concatenate([r["iter"] for r in runs]))
    m = np.full((len(runs), xs.size), np.nan)
    for i, r in enumerate(runs):
        x, y = r["iter"], r[key]
        ok = np.isfinite(y)
        if ok.sum() < 1:
            continue
        inside = (xs >= x[ok][0]) & (xs <= x[ok][-1])
        m[i, inside] = np.interp(xs[inside], x[ok], y[ok])
    n = np.sum(np.isfinite(m), axis=0)
    keep = n > 0
    xs, m, n = xs[keep], m[:, keep], n[keep]
    mean = np.nansum(m, axis=0) / n
    var = np.nansum((m - mean) ** 2, axis=0) / np.maximum(n - 1, 1)
    return xs, mean, np.where(n > 1, np.sqrt(var), 0.0)


def _band(ax, runs, key, scale, color, label):
    """Plot the seed mean +- 1 std of one curve."""
    xs, mean, std = _grid_stats(runs, key)
    if not xs.size:
        return
    ax.fill_between(xs, (mean - std) * scale, (mean + std) * scale,
                    color=color, alpha=0.18, lw=0)
    ax.plot(xs, mean * scale, color=color, lw=2.0, label=label)


def plot_curves(curves, out_path):
    """Each ablated variant's held-out learning curves against full's.

    The undisturbed held-out checks, which every variant runs (no_multitask
    has no disturbed panel), as small multiples: one column per variant,
    violations above and on-spec ethanol below.
    """
    plt = _pyplot()
    if plt is None:
        return
    from metarl.training.plots import PLOT_INK, PLOT_INK2, PLOT_GRID, _style_ax
    variants = [n for n in curves if n != "full"]
    if "full" not in curves or not variants:
        print("[plot] ablation_curves.png skipped: needs full and at least one "
              "ablated variant")
        return
    cols = min(len(variants), 4)
    groups = math.ceil(len(variants) / cols)
    fig, axs = plt.subplots(2 * groups, cols, figsize=(4.4 * cols,
                                                       6.4 * groups),
                            squeeze=False)
    fig.patch.set_facecolor(_SURFACE)
    seeds = sorted({len(v) for v in curves.values()})
    for i, name in enumerate(variants):
        g, c = divmod(i, cols)
        for p, (key, ylab, scale) in enumerate(CURVE_KEYS):
            ax = axs[2 * g + p, c]
            _style_ax(ax, PLOT_INK2, PLOT_GRID)
            _band(ax, curves["full"], key, scale, _REF, "Full")
            _band(ax, curves[name], key, scale, _ACCENT, config_label(name))
            if p == 0:
                ax.set_title(config_label(name), fontsize=10.5, color=PLOT_INK,
                             fontweight="bold", pad=8)
                ax.legend(frameon=False, fontsize=8.5, labelcolor=PLOT_INK2)
            else:
                ax.set_xlabel("training iteration", fontsize=9, color=PLOT_INK2)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=9, color=PLOT_INK2)
    for i in range(len(variants), groups * cols):
        g, c = divmod(i, cols)
        axs[2 * g, c].set_visible(False)
        axs[2 * g + 1, c].set_visible(False)
    for p in range(2):
        row_axes = [axs[2 * g + p, c] for g in range(groups)
                    for c in range(cols) if axs[2 * g + p, c].get_visible()]
        lo = min(a.get_ylim()[0] for a in row_axes)
        hi = max(a.get_ylim()[1] for a in row_axes)
        for a in row_axes:
            a.set_ylim(max(lo, 0.0), hi)
    seeds_note = "/".join(str(s) for s in seeds)
    fig.suptitle(f"Ablation learning curves: undisturbed held-out checks "
                 f"(mean +- 1 std over {seeds_note} seeds)",
                 fontweight="bold", fontsize=12, color=PLOT_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _luminance(rgba):
    """WCAG relative luminance of an RGB(A) tuple in [0, 1]."""
    r, g, b = (((c / 12.92) if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
               for c in rgba[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _text_color(rgba):
    """Ink or surface, whichever contrasts more with the cell."""
    from matplotlib.colors import to_rgba
    from metarl.training.plots import PLOT_INK
    cell = _luminance(rgba)
    ink = (cell + 0.05) / (_luminance(to_rgba(PLOT_INK)) + 0.05)
    surface = (_luminance(to_rgba(_SURFACE)) + 0.05) / (cell + 0.05)
    return PLOT_INK if ink >= surface else _SURFACE


def _heatmap(ax, values, texts, rows, cols, cmap, norm):
    """Cells with a 2 px surface gap, a text per cell, labels on both axes.

    NaN cells are left as the surface.
    """
    from matplotlib.colors import to_rgba
    from metarl.training.plots import PLOT_INK2
    mesh = ax.pcolormesh(np.ma.masked_invalid(values), cmap=cmap, norm=norm,
                         edgecolors=_SURFACE, linewidth=2.0)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = values[i, j]
            rgba = cmap(norm(v)) if np.isfinite(v) else to_rgba(_SURFACE)
            ax.text(j + 0.5, i + 0.5, texts[i][j], ha="center", va="center",
                    fontsize=8, color=_text_color(rgba), linespacing=1.15)
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(cols, rotation=25, ha="right", fontsize=9,
                       color=PLOT_INK2)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels(rows, fontsize=9, color=PLOT_INK2)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(rows), 0)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    return mesh


def _consistent(row, axis):
    """Whether every paired seed was significant in the same direction."""
    n = row.get("paired_seeds") or 0
    return bool(n) and (row[f"seeds_{axis}_worse"] == n
                        or row[f"seeds_{axis}_better"] == n)


def plot_final(rows, out_path):
    """Violations and on-spec ethanol per config and scenario, against full.

    Each cell prints the mean over seeds and the paired delta against full
    (``*``: significant in the same direction in every seed); its color is
    that delta, blue where the config does better than full and red where it
    does worse, on both panels.
    """
    plt = _pyplot()
    if plt is None:
        return
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from metarl.training.plots import PLOT_INK, PLOT_INK2
    configs = list(dict.fromkeys(r["config"] for r in rows))
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    get = {(r["config"], r["scenario"]): r for r in rows}
    worse_up = LinearSegmentedColormap.from_list(
        "abl_viol", (_BETTER, _NEUTRAL, _WORSE))
    worse_down = LinearSegmentedColormap.from_list(
        "abl_eth", (_WORSE, _NEUTRAL, _BETTER))
    panels = (
        ("violations", "d_viol_vs_full", "viol", 1.0, "{:.2f}", "{:+.2f}",
         worse_up, 0.5, "VIOLATION EVENTS PER EPISODE (lower is better)",
         "paired change vs Full (events / episode)"),
        ("on_spec_ethanol", "d_eth_vs_full_pct", "eth", 1e-7, "{:.2f}",
         "{:+.1f}%", worse_down, 2.0,
         "ON-SPEC ETHANOL, 1e7 kg (higher is better)",
         "paired change vs Full (%)"),
    )
    fig, axs = plt.subplots(2, 1, figsize=(max(9.0, 1.15 * len(scenarios) + 3.5),
                                           2 * (1.6 + 0.5 * len(configs))),
                            squeeze=False)
    fig.patch.set_facecolor(_SURFACE)
    for ax, (key, dkey, axis, scale, vfmt, dfmt, cmap, floor, title,
             cbar_label) in zip(axs[:, 0], panels):
        colors_of = np.full((len(configs), len(scenarios)), np.nan)
        texts = [["-"] * len(scenarios) for _ in configs]
        for i, c in enumerate(configs):
            for j, s in enumerate(scenarios):
                row = get.get((c, s))
                if row is None or row[f"{key}_mean"] is None:
                    continue
                d = 0.0 if c == "full" else row[dkey]
                colors_of[i, j] = np.nan if d is None else d
                delta = ("ref" if c == "full" else "-" if d is None
                         else dfmt.format(d) + ("*" if _consistent(row, axis)
                                                else ""))
                texts[i][j] = (vfmt.format(row[f"{key}_mean"] * scale)
                               + "\n" + delta)
        finite = np.abs(colors_of[np.isfinite(colors_of)])
        lim = max(floor, float(finite.max()) if finite.size else floor)
        norm = Normalize(-lim, lim)
        mesh = _heatmap(ax, colors_of, texts,
                        [config_label(c) for c in configs],
                        [_scenario_label(s) for s in scenarios], cmap, norm)
        ax.set_title(title, fontsize=11, color=PLOT_INK, fontweight="bold",
                     pad=10, loc="left")
        cb = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.015)
        cb.set_label(cbar_label + "\nblue: better than Full   red: worse",
                     fontsize=8.5, color=PLOT_INK2)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, colors=PLOT_INK2, length=0)
    seeds = sorted({r["seeds"] for r in rows})
    fig.suptitle("Ablation of the meta arm: mean over "
                 + "/".join(str(s) for s in seeds) + " seeds per cell, paired "
                 "delta vs Full beneath (* = significant in every seed)",
                 fontweight="bold", fontsize=12, color=PLOT_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_severity(severity, out_path):
    """Worst exceedance of each depth constraint per config (% of its limit)."""
    plt = _pyplot()
    if plt is None:
        return
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm
    from metarl.training.plots import PLOT_INK, PLOT_INK2
    configs = list(severity)
    values = np.array([[severity[c][k][0] if severity[c][k][0] is not None
                        else np.nan for k in DEPTH_NAMES] for c in configs])
    texts = [["-" if not np.isfinite(v) else f"{v:.1f}" for v in row]
             for row in values]
    finite = values[np.isfinite(values)]
    vmax = float(finite.max()) if finite.size and finite.max() > 0 else 1.0
    cmap = LinearSegmentedColormap.from_list("abl_sev", _SEQUENTIAL)
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(max(8.0, 1.3 * len(DEPTH_NAMES) + 4.0),
                                    1.8 + 0.5 * len(configs)))
    fig.patch.set_facecolor(_SURFACE)
    mesh = _heatmap(ax, values, texts, [config_label(c) for c in configs],
                    [k.replace("_", " ") for k in DEPTH_NAMES], cmap, norm)
    cb = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.015)
    cb.set_label("worst exceedance (% of the limit, square-root scale)",
                 fontsize=8.5, color=PLOT_INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=PLOT_INK2, length=0)
    ax.set_title("WORST VIOLATION SEVERITY over every scenario and episode "
                 "(mean over seeds)", fontsize=11, color=PLOT_INK,
                 fontweight="bold", pad=10, loc="left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── Entry Point ──────────────────────────────────────────────────────────────

def _parse_args(argv):
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        prog="python -m metarl.ablation",
        description="Ablation study of the meta arm (latent z, multi-task "
                    "training, Lagrangian, stage mask, APN, warm start).",
        epilog="configs: full, no_<component>[_<component>...] (components "
               "in the order " + ", ".join(COMPONENTS) + "), none; "
               "leave-one-out runs " + ", ".join(LOO_CONFIGS),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--design", choices=("loo", "factorial"), default="loo",
                        help="loo: full plus one run per removed component; "
                             "factorial: every combination of --components")
    parser.add_argument("--components", nargs="+", choices=COMPONENTS,
                        default=list(COMPONENTS),
                        help="the factorial design's axes (the rest stay on)")
    parser.add_argument("--configs", nargs="+", choices=list(ALL_CONFIGS),
                        metavar="NAME",
                        help="run only these configs instead of the --design set")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="each config is trained once per seed")
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES,
                        help="evaluation tasks per scenario")
    parser.add_argument("--budget", type=int, default=None,
                        help="train every variant for exactly this many "
                             "iterations with early stopping off; omitted, "
                             "the study's protocol")
    parser.add_argument("--tag", default=DEFAULT_TAG,
                        help="run-tag prefix: seed k runs under <tag>/seed<k>")
    parser.add_argument("--workers", type=int, default=1,
                        help="runs at a time (each in its own process)")
    parser.add_argument("--force", action="store_true",
                        help="retrain runs that already have results")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would run, then stop")
    parser.add_argument("--summarize-only", action="store_true",
                        help="skip training; aggregate existing runs and re-plot")
    parser.add_argument("--_run", help=argparse.SUPPRESS)
    parser.add_argument("--_eval-only", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.budget is not None and args.budget <= META_ACTOR_WARMUP_ITERS:
        parser.error(f"--budget must exceed META_ACTOR_WARMUP_ITERS "
                     f"({META_ACTOR_WARMUP_ITERS}): a warm-started actor is "
                     f"held that long before it trains")
    return args


def main(argv=None):
    """Run the requested ablation runs, then summarize everything on disk."""
    args = _parse_args(argv)
    if args._run:
        _run(args._run, args.eval_episodes, args.budget, args._eval_only)
        return 0
    if not args.summarize_only:
        names = args.configs or (LOO_CONFIGS if args.design == "loo"
                                 else factorial_configs(args.components))
        settings = {"tag": args.tag, "eval_episodes": args.eval_episodes,
                    "budget": args.budget, "workers": max(1, args.workers)}
        status = {(n, s): _status(args.tag, n, s, settings, args.force)
                  for n in [REFERENCE] + names for s in args.seeds}
        todo = {k: v for k, v in status.items() if v != "done"}
        n_train = sum(v == "train" for v in todo.values())
        print(f"--- Ablation: {len(names)} configs x {len(args.seeds)} seeds "
              f"(+ {REFERENCE} per seed) = {len(status)} runs, {n_train} to "
              f"train, {len(todo) - n_train} to evaluate "
              f"({len(status) - len(todo)} already complete) ---")
        print(f"  configs: {', '.join(names)}")
        print(f"  training: " + (f"{args.budget} iterations per variant, early "
                                 f"stopping off" if args.budget else
                                 "the study's protocol (early stopping on)")
              + f"; evaluation: {args.eval_episodes} tasks per scenario")
        print(f"  outputs: policy/{args.tag}/seed<k>/, "
              f"results/log/{args.tag}/, results/plot/{args.tag}/  |  "
              f"runs at a time: {settings['workers']}")
        for (n, s), v in todo.items():
            print(f"    {v:<6}{n} seed {s}")
        if args.dry_run:
            return 0
        failed = run_jobs([(n, s, v) for (n, s), v in todo.items()
                           if n == REFERENCE], settings)
        jobs = []
        for (n, s), v in todo.items():
            if n == REFERENCE:
                continue
            if (REFERENCE, s) in failed and ALL_CONFIGS[n].warmstart:
                print(f"[{n} seed {s}] skipped: {REFERENCE} seed {s} failed "
                      f"and this variant warm-starts from it")
                continue
            jobs.append((n, s, v))
        run_jobs(jobs, settings)
    summarize(args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
