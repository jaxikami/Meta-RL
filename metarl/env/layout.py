"""Observation and action layout of the plant, and the structural stage mask.

The agent reads the environment's observation flattened: ``get_state_norm()``
is ``(N_REACTORS, REACTOR_OBS_DIM)``, one row per reactor (laid out in
``metarl.env.config``), and the flat ``(STATE_DIM,)`` vector is the rows in
order. The action is ``(ACTION_DIM,)``, four actuators per reactor, the
environment's flattened action.
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import (N_REACTORS, N_SPECIES, I_ETH, R_STAGE,
                               R_STORAGE, R_LF, R_C5, REACTOR_OBS_DIM)

# ─── Observation layout (see config) ──────────────────────────────────────────
STATE_DIM = N_REACTORS * REACTOR_OBS_DIM               # 84
ACTION_DIM_PER_REACTOR = 4
ACTION_DIM = N_REACTORS * ACTION_DIM_PER_REACTOR       # 12
# Bumped when a channel's meaning changes at the same width, so checkpoints
# trained on the old meaning are refused.
# v2: the Fermentation-PPO observation (rows of 28, species observed, no lot
# or off-gas channels).
OBS_LAYOUT_VERSION = 2


def reactor_channel(i, offset):
    """Observation index of a per-reactor channel (``config.R_*``)."""
    return i * REACTOR_OBS_DIM + offset


# The plant-wide channels are repeated in every row; reactor 0's copy is read.
STORAGE_CHANNEL = reactor_channel(0, R_STORAGE)
LF_CHANNEL = reactor_channel(0, R_LF)
C5_CHANNEL = reactor_channel(0, R_C5)

# The action projection network reads the whole observation but for the
# species channels: none of its margins depends on them.
SHIELD_OBS_DIM = STATE_DIM
SHIELD_IGNORED_CHANNELS = tuple(reactor_channel(i, c) for i in range(N_REACTORS)
                                for c in range(N_SPECIES))

# Width of PlantFermentationEnv.privileged_state(): the observation, then
# each reactor's lot in progress (mass, mean titer: the observation has no
# lot channel), then the elapsed share of the episode.
PRIVILEGED_STATE_DIM = STATE_DIM + 2 * N_REACTORS + 1
PRIV_LOT_MASS = STATE_DIM
PRIV_LOT_TITER = STATE_DIM + N_REACTORS

# Action dimensions the projection never moves: each reactor's stage clock.
TIME_ACTION_DIMS = tuple(i * ACTION_DIM_PER_REACTOR for i in range(N_REACTORS))


def reactor_action(i, k):
    """Action index of reactor ``i``'s k-th actuator (0 clock, 1 LF, 2 C5, 3 out)."""
    return i * ACTION_DIM_PER_REACTOR + k


# Per reactor, the observation channels of its stage one-hot, shape (N, 5).
STAGE_CHANNELS = np.array([[reactor_channel(i, R_STAGE + s) for s in range(5)]
                           for i in range(N_REACTORS)], dtype=np.intp)
# Each reactor's broth ethanol channel.
ETHANOL_CHANNELS = tuple(reactor_channel(i, I_ETH) for i in range(N_REACTORS))
# Each reactor's lot mean titer in the privileged state.
LOT_TITER_PRIV_CHANNELS = tuple(PRIV_LOT_TITER + i for i in range(N_REACTORS))

# Action values for structurally locked dimensions, matching the agent's
# ``_default_a`` and Fermentation-PPO's default intent: tanh(-0.34657359)
# maps to a time multiplier of exactly 1.0, and -1 maps to zero flow. One row
# of four per reactor.
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
