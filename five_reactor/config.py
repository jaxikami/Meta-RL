"""Settings of the five-reactor sub-study (everything else is metarl's)."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
import five_reactor  # noqa: F401  (selects the layout before metarl loads)
from metarl.training.config import EVAL_EPISODES, LOG_DIR, PLOT_DIR  # noqa: F401

ARM = "meta5"

# ─── The known change ─────────────────────────────────────────────────────────
# The tank's downstream draw (1.5 t/h at m = 1) steps to m x nominal at the
# episode's midpoint. Kind and onset are known; m is not. The draw is the
# change the observation shows soonest: the storage level is a channel and the
# agent knows its own discharges, so m is in the level's slope within a few
# steps. The right answer to it is a schedule (start batches sooner and feed
# harder for a larger draw, hold finished batches for a smaller one), which
# the APN's storage heads cannot supply: they only clip discharge.
CHANGE_KIND = "downstream_rate"
CHANGE_ONSET_FRAC = 0.5
CHANGE_MEAN = 1.30
# Training: N(1.30, 0.08) cut at 3 SD. Evaluation: N(1.30, 0.25) cut to
# [0.5, 2.1]; ~11% of its draws fall below 1.0, where the draw drops instead.
CHANGE_TRAIN_SD = 0.08
CHANGE_EVAL_SD = 0.25
CHANGE_TRAIN_CLIP = (CHANGE_MEAN - 3 * CHANGE_TRAIN_SD,
                     CHANGE_MEAN + 3 * CHANGE_TRAIN_SD)
CHANGE_EVAL_CLIP = (0.5, 2.1)
# Evaluation rows inside |m - mean| <= SUPPORT_SD * CHANGE_TRAIN_SD are "in
# support", the rest "outside".
SUPPORT_SD = 2.0

# ─── Training and selection ───────────────────────────────────────────────────
# Held-out magnitudes, inside the training distribution (mean, +-1.9 SD), so
# selection never sees the evaluation's tails.
HELDOUT_MAGNITUDES = (1.15, 1.30, 1.45)
# The three-reactor gate (3 mean / 12 worst-episode events) scaled by 5/3.
SELECT_MAX_VIOL = 5.0
SELECT_MAX_WORST_VIOL = 20.0
EARLY_STOP_PATIENCE = int(os.environ.get("PLANT_EARLY_STOP_PATIENCE", 50))

# ─── Evaluation ───────────────────────────────────────────────────────────────
EVAL_ID_SEED = 12345
# Per-episode magnitude draws, independent of the task draws.
EVAL_CHANGE_SEED = 4242
# A sweep over m on every evaluation task, from the same pre-change prefix.
EVAL_SWEEP = tuple(np.round(np.arange(0.6, 2.01, 0.1), 2))
