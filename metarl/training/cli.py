"""Train and evaluate three control arms on the three-reactor plant.

    arm       multi-task + latent z   Lagrangian   stage mask   APN
    -----------------------------------------------------------------------
    baseline        no                    no           no       no
    saferl          no                   yes          yes       hard: reactor overflow /
                                                                underflow, storage overflow
    meta           yes                   yes          yes       hard: the same shield

    SAFETY = saferl - baseline    what the safe-RL stack buys, on one regime
    ADAPT  = meta   - saferl      what meta-RL adds on top of safe RL

The lean study's protocol (``MetaRL_Lean/train_lean.py``) on the plant of
three heterogeneous reactors and a central storage tank. ``baseline`` and
``saferl`` train on the single undisturbed nominal task and meet uncertainty
first at evaluation; ``meta`` trains on a fresh prior task every episode, with
mid-episode faults on one reactor (or all, or the plant's feed tanks or its
storage tank). Both shielded arms run the same APN, which with concentrations
masked guards only the inventory constraints one control interval can keep;
harvest titer, purity and storage underflow have no shield, so there meta
must adapt. ``meta`` starts from saferl's trained policy, acts through it
until its adaptation gate opens, and its checkpoint must be no worse than
saferl on any held-out branch (``NO_REGRESS_*``).

Every table reports violations AND on-spec ethanol, and verdicts require both
axes.

Run ``python -m metarl <mode>`` from the repository root. Modes (default
``MODE``):

    train_baseline     train the unconstrained baseline on the nominal task
    train_saferl       train the safe-RL arm on the nominal task
    train_meta         meta-train PEARL over the disturbed task pool
    train_apn          train the shared action projection network
    evaluate           every arm on every scenario, both metric axes
    z_informativeness  gate: does the meta arm's z carry task information
                       the actor acts on?
    plot_curves        redraw the training curves from the logs
    plot_eval          redraw the evaluation figures from eval.json
    all                train every arm -> evaluate -> z_informativeness

Environment variables: ``PLANT_SEED`` (default 0), ``PLANT_RUN_TAG``
(subfolder for checkpoints, logs and plots), ``PLANT_APN_PATH`` (APN
checkpoint), ``PLANT_HELDOUT_WORKERS``, ``PLANT_DEVICE``.
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import json
import os

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import (MIN_HARVEST_TITER, MIN_PRODUCT_TITER,
                               N_REACTORS, S_MAX, S_MIN)
from metarl.apn.network import APN_PATH
from metarl.agent.common import LATENT_DIM
from metarl.training.config import (MODE, SEED, INCLUDE_BASELINE, USE_FILM,
                                    AUX_MODE, POLICY_DIR, LOG_DIR)
from metarl.training.arms import ARMS, ARM_ORDER, active_arms
from metarl.training.train import _train
from metarl.training.evaluate import evaluate
from metarl.training.plots import _plot_comparison, plot_training_curves
from metarl.training.z_informativeness import z_informativeness


def main(argv):
    """Dispatch to the requested mode; returns the process exit code."""
    mode = argv[1] if len(argv) > 1 else MODE
    print(f"plant 3-arm study ({N_REACTORS} reactors + central storage)   "
          f"mode={mode}   seed={SEED}   policy_dir={POLICY_DIR}   "
          f"log_dir={LOG_DIR}")
    print(f"latent_dim={LATENT_DIM}  film={USE_FILM}  aux={AUX_MODE}  "
          f"specs: harvest {MIN_HARVEST_TITER:.0f} / purity "
          f"{MIN_PRODUCT_TITER:.0f} g/L; storage {S_MIN:.0f}..{S_MAX:.0f} kg")
    print("arms: " + " | ".join(f"{a} ({ARMS[a].summary()})"
                                for a in active_arms())
          + ("" if INCLUDE_BASELINE
             else "   [baseline excluded: INCLUDE_BASELINE = False]"))
    print(f"projection network: {APN_PATH}")

    known = {f"train_{a}": a for a in ARM_ORDER}
    if mode in known:
        if known[mode] not in active_arms():
            print(f"{mode}: the {known[mode]} arm is excluded "
                  f"(INCLUDE_BASELINE = False); set it True to train it")
            return 2
        _train(known[mode])
    elif mode == "train_apn":
        from metarl.apn.train import train
        train()
    elif mode == "evaluate":
        evaluate()
    elif mode == "z_informativeness":
        z_informativeness()
    elif mode == "plot_curves":
        plot_training_curves()
    elif mode == "plot_eval":
        with open(os.path.join(LOG_DIR, "eval.json")) as f:
            _plot_comparison(json.load(f))
    elif mode == "all":
        for a in active_arms():
            _train(a)
        evaluate()
        z_informativeness()
    else:
        print(f"unknown mode {mode!r}; expected one of "
              f"{', '.join(sorted(known))}, train_apn, evaluate, "
              f"z_informativeness, plot_curves, plot_eval, all")
        return 2
    return 0
