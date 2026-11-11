"""The known change: its magnitude distributions and its fault schedule."""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from five_reactor.config import (CHANGE_KIND, CHANGE_ONSET_FRAC, CHANGE_MEAN,
                                 CHANGE_TRAIN_SD, CHANGE_EVAL_SD,
                                 CHANGE_TRAIN_CLIP, CHANGE_EVAL_CLIP,
                                 SUPPORT_SD)


def change_onset(horizon=600):
    """Inner step at which the change acts (hour 300 of 600)."""
    return int(round(CHANGE_ONSET_FRAC * horizon))


def sample_magnitude(rng, split="train"):
    """One magnitude m ~ N(CHANGE_MEAN, sd), clipped, for ``split`` train or eval."""
    sd, (lo, hi) = ((CHANGE_TRAIN_SD, CHANGE_TRAIN_CLIP) if split == "train"
                    else (CHANGE_EVAL_SD, CHANGE_EVAL_CLIP))
    return float(np.clip(rng.normal(CHANGE_MEAN, sd), lo, hi))


def change_schedule(magnitude, horizon=600):
    """The change as a fault schedule: a step of ``magnitude`` at the onset."""
    return [(CHANGE_KIND, change_onset(horizon), float(magnitude), None, None)]


def in_support(magnitude):
    """Whether ``magnitude`` lies within SUPPORT_SD training SDs of the mean."""
    return abs(float(magnitude) - CHANGE_MEAN) <= SUPPORT_SD * CHANGE_TRAIN_SD + 1e-12
