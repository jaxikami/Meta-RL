"""Multi-seed meta-vs-baseline comparison with error bars.

This is the sibling driver for `train_pearl_single_run.py`. That script runs
ONE seed through train_meta -> train_baseline -> evaluate and is fine for
iterating on a fix, but this project's training pipeline is high-variance
run-to-run (observed: baseline eval violations 0.25 on one run, 46.1 on the
next, identical code). A single run therefore cannot tell you whether meta
beats baseline — the between-seed noise is larger than the meta-vs-baseline
gap. THIS script runs BOTH agents under several shared seeds (by importing
train_pearl_single_run and calling its train_meta/train_baseline/evaluate
repeatedly) and compares the DISTRIBUTIONS instead of one noisy point.

Because each seed trains meta + baseline with the SAME seed, the meta-baseline
difference is PAIRED per seed (same network init / sampling draws), which is
the right way to compare here: report the mean paired delta and how many seeds
meta actually won, not just marginal means.

Usage:
    python train_pearl_seed_sweep.py                # seeds 0,1,2
    python train_pearl_seed_sweep.py 0 1 2 3 4       # explicit seed list

RUNTIME WARNING: each seed is a full train_meta (~27 min) + train_baseline
(~8 min) + evaluate. 3 seeds ~= 1.75 h, 5 seeds ~= 3 h. Start with 3 for a
first read of the variance, extend to 5 for tighter bars.
"""
import json
import os
import shutil
import sys

import numpy as np

import train_pearl_single_run as tp

# Headline metric per scenario is violations (lower is better); reward is
# tracked too. Feasibility is ~always 100% here so it is omitted.
METRIC = "violations"


def _run_one_seed(seed):
    """Train meta + baseline under one seed and return evaluate()'s results."""
    tp.SEED = seed
    print(f"\n{'#'*70}\n# SEED {seed}\n{'#'*70}")
    tp.train_meta()
    tp.train_baseline()
    results = tp.evaluate()
    # Preserve this seed's best checkpoints so a later seed can't clobber a
    # run you might want to inspect.
    for src in (tp.BEST_META_PATH, tp.BEST_BASE_PATH):
        if os.path.exists(src):
            root, ext = os.path.splitext(src)
            shutil.copyfile(src, f"{root}_seed{seed}{ext}")
    return results


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
    print(f"Running seeds: {seeds}")

    # per_seed[scenario][agent] -> list of metric values across seeds
    scenarios, per_seed_viol, per_seed_rew = None, {}, {}
    paired_delta = {}   # scenario -> list of (baseline_viol - meta_viol) per seed

    for seed in seeds:
        try:
            res = _run_one_seed(seed)
        except Exception as exc:  # keep the sweep alive if one seed dies
            print(f"[seed {seed}] FAILED: {exc}")
            continue
        if scenarios is None:
            scenarios = list(res.keys())
            for sc in scenarios:
                per_seed_viol[sc] = {"meta": [], "baseline": []}
                per_seed_rew[sc] = {"meta": [], "baseline": []}
                paired_delta[sc] = []
        for sc in scenarios:
            for who in ("meta", "baseline"):
                per_seed_viol[sc][who].append(res[sc][who]["violations"])
                per_seed_rew[sc][who].append(res[sc][who]["reward"])
            paired_delta[sc].append(
                res[sc]["baseline"]["violations"] - res[sc]["meta"]["violations"])

    if not scenarios:
        print("No seeds completed successfully.")
        return

    n = len(paired_delta[scenarios[0]])
    print(f"\n{'='*88}\n  MULTI-SEED SUMMARY  (n={n} seeds: {seeds})\n{'='*88}")
    print(f"  {'Scenario':<22}{'meta viol':>18}{'baseline viol':>18}"
          f"{'paired Δ(base−meta)':>22}{'meta wins':>10}")
    print("  " + "-" * 86)
    for sc in scenarios:
        mv = np.array(per_seed_viol[sc]["meta"])
        bv = np.array(per_seed_viol[sc]["baseline"])
        dd = np.array(paired_delta[sc])
        wins = int(np.sum(dd > 0))   # meta fewer violations than baseline
        print(f"  {sc:<22}"
              f"{mv.mean():>10.2f}±{mv.std():<6.2f}"
              f"{bv.mean():>10.2f}±{bv.std():<6.2f}"
              f"{dd.mean():>14.2f}±{dd.std():<6.2f}"
              f"{wins:>7}/{n}")
    print("  " + "-" * 86)
    print("  Δ>0 means meta had FEWER violations (meta better). A verdict needs\n"
          "  the paired Δ mean to be positive AND large relative to its std;\n"
          "  'meta wins k/n' shows how consistent that is across seeds.")

    out = {
        "seeds": seeds,
        "violations": per_seed_viol,
        "reward": per_seed_rew,
        "paired_delta_viol": paired_delta,
    }
    with open(os.path.join(tp.MODEL_DIR, "pearl_seed_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  saved -> {os.path.join(tp.MODEL_DIR, 'pearl_seed_sweep.json')}")


if __name__ == "__main__":
    main()
