"""``python -m five_reactor <mode>``: the five-reactor sub-study.

Modes:
    calibrate   scripted-recipe checks of the plant and the change's headroom
    train       train meta5 (from scratch)
    evaluate    evaluate meta5 on the training and the wide change distributions
    plot        redraw the figures from the logs
    all         train -> evaluate

Environment variables are metarl's (``PLANT_SEED``, ``PLANT_RUN_TAG``,
``PLANT_HELDOUT_WORKERS``, ``PLANT_EVAL_EPISODES``, ``PLANT_META_ITERS``,
...). Outputs go under this folder (``policy/``, ``results/``).
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import sys

# ─── Local ────────────────────────────────────────────────────────────────────
import five_reactor  # noqa: F401  (selects the layout before metarl loads)
import five_reactor.arms  # noqa: F401  (registers meta5, here and in workers)

MODES = ("calibrate", "train", "evaluate", "plot", "all")


def main(argv):
    """Dispatch to the requested mode; returns the process exit code."""
    mode = argv[1] if len(argv) > 1 else "all"
    if mode not in MODES:
        print(f"unknown mode {mode!r}; expected one of {', '.join(MODES)}")
        return 2
    if mode == "calibrate":
        from five_reactor.calibrate import calibrate
        calibrate()
        return 0
    from five_reactor.evaluate import evaluate
    from five_reactor.plots import plot_all
    from five_reactor.train import train
    if mode in ("train", "all"):
        train()
    if mode in ("evaluate", "all"):
        evaluate()
    if mode == "plot":
        plot_all()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
