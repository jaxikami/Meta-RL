"""Observation and action layout of the plant, and the structural stage mask.

Channel indices of the ``(STATE_DIM,)`` observation (each reactor's block is
laid out in ``metarl.env.config``) and of the ``(ACTION_DIM,)`` action, four
actuators per reactor.
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import (N_REACTORS, I_ETH, R_STAGE, REACTOR_OBS_DIM,
                               G_STORAGE, G_LF, G_C5, GLOBAL_OBS_DIM)

# ─── Observation layout (see config) ──────────────────────────────────────────
GLOBAL_START = N_REACTORS * REACTOR_OBS_DIM            # 72
CER_START = GLOBAL_START + GLOBAL_OBS_DIM              # 75
STATE_DIM = CER_START + N_REACTORS                     # 78
ACTION_DIM_PER_REACTOR = 4
ACTION_DIM = N_REACTORS * ACTION_DIM_PER_REACTOR       # 12
# Bumped when a channel's meaning changes at the same width, so checkpoints
# trained on the old meaning are refused.
OBS_LAYOUT_VERSION = 1

STORAGE_CHANNEL = GLOBAL_START + G_STORAGE
LF_CHANNEL = GLOBAL_START + G_LF
C5_CHANNEL = GLOBAL_START + G_C5

# The action projection network reads channels [0, SHIELD_OBS_DIM): every
# reactor block and the plant-wide channels, not the CO2 rates.
SHIELD_OBS_DIM = CER_START

# Width of PlantFermentationEnv.privileged_state(): the exact observation with
# the species channels unmasked, then the elapsed share of the episode.
PRIVILEGED_STATE_DIM = STATE_DIM + 1
PRIVILEGED_SPECIES_CLIP = 5.0

# Action dimensions the projection never moves: each reactor's stage clock.
TIME_ACTION_DIMS = tuple(i * ACTION_DIM_PER_REACTOR for i in range(N_REACTORS))


def reactor_channel(i, offset):
    """Observation index of a per-reactor channel (``config.R_*``)."""
    return i * REACTOR_OBS_DIM + offset


def cer_channel(i):
    """Observation index of reactor ``i``'s off-gas CO2 rate."""
    return CER_START + i


def reactor_action(i, k):
    """Action index of reactor ``i``'s k-th actuator (0 clock, 1 LF, 2 C5, 3 out)."""
    return i * ACTION_DIM_PER_REACTOR + k


# Per reactor, the observation channels of its stage one-hot, shape (N, 5).
STAGE_CHANNELS = np.array([[reactor_channel(i, R_STAGE + s) for s in range(5)]
                           for i in range(N_REACTORS)], dtype=np.intp)
# Each reactor's broth ethanol channel (meaningful in privileged_state, and in
# the observation only when concentrations are not masked).
ETHANOL_CHANNELS = tuple(reactor_channel(i, I_ETH) for i in range(N_REACTORS))

# Action values for structurally locked dimensions, matching the agent's
# ``_default_a``: tanh(-0.34657359) maps to a time multiplier of exactly 1.0,
# and -1 maps to zero flow. One row of four per reactor.
DEFAULT_MASKED_ACTION = np.tile(
    np.array([np.tanh(-0.34657359), -1.0, -1.0, -1.0]), N_REACTORS)


def stage_mask_np(obs):
    """NumPy stage mask, for scripted policies and diagnostics.

    Per reactor: time multiplier free except during cleanup; LF in charge and
    co-feed; C5 in co-feed only; discharge in unload only. Must agree with
    ``PlantFermentationEnv.get_action_mask``; ``smoke_test`` asserts it.

    Args:
        obs: Observation array of shape (STATE_DIM,).

    Returns:
        Float array of shape (ACTION_DIM,) with 1.0 for legal action dimensions.
    """
    obs = np.asarray(obs)
    out = np.empty(ACTION_DIM, dtype=np.float64)
    for i in range(N_REACTORS):
        s = reactor_channel(i, R_STAGE)
        out[4 * i:4 * i + 4] = (1.0 - obs[s + 4], obs[s + 0] + obs[s + 1],
                                obs[s + 1], obs[s + 3])
    return out


def apply_stage_mask_np(obs, action):
    """Blend a raw action with the masked defaults, exactly as the agent does.

    Args:
        obs: Observation array of shape (STATE_DIM,).
        action: Raw action in [-1, 1], shape (ACTION_DIM,).

    Returns:
        Executed action of shape (ACTION_DIM,).
    """
    m = stage_mask_np(obs)
    return (np.asarray(action, dtype=np.float64) * m
            + DEFAULT_MASKED_ACTION * (1.0 - m))
