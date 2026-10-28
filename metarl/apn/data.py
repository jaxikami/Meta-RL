"""Safety-margin labels and synthetic training data for the plant's APN.

Each constraint becomes a signed, normalized distance (positive = feasible),
combined by a smooth minimum ``-tau * logsumexp(-m / tau)``:

    reactor i overflow     M_i <= Mmax                     (5% back-off)
    reactor i underflow    M_i >= DRY_FLOOR                (5% back-off)
    storage overflow       S <= S_MAX                      (5% back-off)

These are the inventory constraints one control interval can keep. The others
are left to the policy and the reward:

  * storage UNDERFLOW: within one control interval the only way to raise the
    tank is to drain a reactor that is already in unload; when none is, every
    action is equally unsafe, and the smooth minimum would then be dominated
    by a constraint no action can move -- the projection would shut every feed
    to "repair" it, costing production for nothing. Keeping the tank supplied
    is a scheduling decision made tens of hours earlier;
  * harvest titer and lot purity: the observation carries no lot (its mass
    and mean titer), so the purity margin cannot be computed from it;
  * the plant heat load: it needs the kinetics (Fermentation-PPO learns it).

The label rolls the action forward over one control interval, reproducing the
environment's discharge cap, the shared feedstock pool drawn in reactor order,
and the storage tank's inflow and downstream draw, holding every stage fixed
across the interval.

It is a NOMINAL-plant label: pump gains, leaks and the downstream rate are
latent and do not enter it, so coping with a disturbed plant stays the
policy's job.
"""

import numpy as np
import torch

from metarl.env.config import (INITIAL_C5, INITIAL_LF, MAX_FLOW_OUT_REAC,
                               TIME_STEP_DURATION, get_reactor_config,
                               N_REACTORS, S_MAX, S_OUT_RATE, R_MASS, R_STAGE,
                               R_CREDIT, R_MMAX, R_FMAX_LF, R_FMAX_C5,
                               R_STORAGE, R_LF, R_C5, N_SPECIES)
from metarl.env.layout import (SHIELD_OBS_DIM, ACTION_DIM,
                               ACTION_DIM_PER_REACTOR, STORAGE_CHANNEL,
                               LF_CHANNEL, C5_CHANNEL, TIME_ACTION_DIMS,
                               reactor_channel)
from metarl.env.adapter import PlantFermentationEnv

# ─── Constraint back-off ──────────────────────────────────────────────────────
BACKOFF_MASS = 0.05
SAFE_BUFFER = 1.0 - BACKOFF_MASS

# Mirrors of PlantFermentationEnv constants (checked by
# assert_plant_constants_match).
DRY_FLOOR = 10.0
UNDERFLOW_SCALE = 100.0          # env.UNDERFLOW_BUFFER_KG
GUARD_MASS = 50.0                # the discharge cap's floor

SMOOTH_TAU = 0.05
CONTROL_ACTION_INTERVAL = 5

_MASS_CH = [reactor_channel(i, R_MASS) for i in range(N_REACTORS)]


def plant_constants():
    """Return the fixed plant constants the margin label depends on.

    Returns:
        Dict of per-reactor limits (lists), flow maxima and storage specs.
    """
    cfg = get_reactor_config()
    return {
        "Mmax": torch.as_tensor(cfg["Mmax"], dtype=torch.float32),
        "max_flow_LF": torch.as_tensor(cfg["max_flow_LF"], dtype=torch.float32),
        "max_flow_C5": torch.as_tensor(cfg["max_flow_C5"], dtype=torch.float32),
        "max_flow_out": float(MAX_FLOW_OUT_REAC),
        "S_max": float(S_MAX),
        "S_out_rate": float(S_OUT_RATE),
        "dt": float(TIME_STEP_DURATION),
        "n_inner": int(CONTROL_ACTION_INTERVAL),
    }


def assert_plant_constants_match(env):
    """Check the label's mirrored constants against a live environment.

    Args:
        env: A ``PlantFermentationEnv``.

    Raises:
        AssertionError: On any mismatch.
    """
    c = plant_constants()
    assert env.DRY_FLOOR == DRY_FLOOR, (env.DRY_FLOOR, DRY_FLOOR)
    assert env.UNDERFLOW_BUFFER_KG == UNDERFLOW_SCALE
    assert env.control_action_interval == CONTROL_ACTION_INTERVAL
    assert env.time_step_duration == TIME_STEP_DURATION
    assert env.max_flow_out_reac == MAX_FLOW_OUT_REAC
    assert env.S_max == S_MAX and env.S_out_rate == S_OUT_RATE
    assert np.allclose(env.Mmax, c["Mmax"].numpy())
    assert np.allclose(env.max_flow_LF, c["max_flow_LF"].numpy())
    assert np.allclose(env.max_flow_C5, c["max_flow_C5"].numpy())


# =============================================================================
# MARGIN LABEL
# =============================================================================

def safety_margins(obs, act, const=None):
    """Compute per-constraint margins for a batch of state-actions.

    Args:
        obs: Normalized observations, shape ``(B, >= SHIELD_OBS_DIM)``.
        act: Actions in [-1, 1], shape ``(B, ACTION_DIM)``.
        const: Optional dict from ``plant_constants``.

    Returns:
        Tensor ``(B, 2N + 1)``: [overflow_0..N-1, underflow_0..N-1, storage
        overflow]. Positive means satisfied with room to spare.
    """
    c = plant_constants() if const is None else const
    dt, n_inner = c["dt"], c["n_inner"]
    Mmax = c["Mmax"].to(obs.device)
    fl_lf = c["max_flow_LF"].to(obs.device)
    fl_c5 = c["max_flow_C5"].to(obs.device)
    n = N_REACTORS
    obs = obs.to(torch.float32)
    act = act.to(torch.float32)
    B = obs.shape[0]

    mask = PlantFermentationEnv.get_action_mask(obs).reshape(
        B, n, ACTION_DIM_PER_REACTOR)
    u = ((torch.clamp(act, -1.0, 1.0) + 1.0) / 2.0).reshape(
        B, n, ACTION_DIM_PER_REACTOR)
    u_lf = u[..., 1] * mask[..., 1]
    u_c5 = u[..., 2] * mask[..., 2]
    u_out = u[..., 3] * mask[..., 3]

    M = obs[:, _MASS_CH] * Mmax                              # (B, n)
    S = obs[:, STORAGE_CHANNEL] * c["S_max"]                 # (B,)
    lf_stock = obs[:, LF_CHANNEL] * INITIAL_LF
    c5_stock = obs[:, C5_CHANNEL] * INITIAL_C5

    f_lf_cmd = u_lf * fl_lf
    f_c5_cmd = u_c5 * fl_c5
    f_out_cmd = u_out * c["max_flow_out"]
    drain_cap = c["S_out_rate"] * dt

    m_hi, m_lo, s_hi = M, M, S
    for _ in range(n_inner):
        f_out = torch.minimum(f_out_cmd, torch.clamp(M - GUARD_MASS, min=0.0) / dt)
        # The shared pool is drawn in reactor order, as in the environment.
        f_in = []
        for i in range(n):
            lf_m = torch.minimum(f_lf_cmd[:, i] * dt, torch.clamp(lf_stock, min=0.0))
            c5_m = torch.minimum(f_c5_cmd[:, i] * dt, torch.clamp(c5_stock, min=0.0))
            lf_stock = lf_stock - lf_m
            c5_stock = c5_stock - c5_m
            f_in.append((lf_m + c5_m) / dt)
        F_in = torch.stack(f_in, dim=-1)
        M = M + (F_in - f_out) * dt
        S = S + f_out.sum(dim=-1) * dt
        S = S - torch.minimum(torch.full_like(S, drain_cap), torch.clamp(S, min=0.0))
        m_hi = torch.maximum(m_hi, M)
        m_lo = torch.minimum(m_lo, M)
        s_hi = torch.maximum(s_hi, S)

    m_over = (Mmax * SAFE_BUFFER - m_hi) / Mmax
    # Underflow is normalized by the buffer zone, not by Mmax.
    m_under = (m_lo - DRY_FLOOR * (1.0 + BACKOFF_MASS)) / UNDERFLOW_SCALE
    s_over = (c["S_max"] * SAFE_BUFFER - s_hi) / c["S_max"]
    return torch.cat([m_over, m_under, s_over.unsqueeze(-1)], dim=-1)


def smooth_min(margins, tau=SMOOTH_TAU):
    """Differentiable minimum over the constraint axis."""
    return -tau * torch.logsumexp(-margins / tau, dim=-1)


def safety_margin(obs, act, const=None, tau=SMOOTH_TAU):
    """Scalar safety margin: the smooth minimum over shielded constraints.

    Returns:
        Tensor of shape ``(B,)``; positive means safe.
    """
    return smooth_min(safety_margins(obs, act, const=const), tau=tau)


def shut_action(n, device=None):
    """Every flow shut, every clock at mid-range: the shield's retreat point.

    Args:
        n: Batch size.
        device: Optional torch device.

    Returns:
        Tensor ``(n, ACTION_DIM)``.
    """
    a = torch.full((n, ACTION_DIM), -1.0, device=device)
    a[:, list(TIME_ACTION_DIMS)] = 0.0
    return a


# =============================================================================
# SYNTHETIC SAMPLING
# =============================================================================

def _mix(rng, n, parts):
    """Draw from a mixture of uniform ranges given as (weight, low, high)."""
    w = np.array([p[0] for p in parts], dtype=float)
    w = w / w.sum()
    which = rng.choice(len(parts), size=n, p=w)
    lo = np.array([p[1] for p in parts])[which]
    hi = np.array([p[2] for p in parts])[which]
    return rng.uniform(lo, hi)


def sample_states(n, rng):
    """Draw synthetic normalized observations in the environment's layout.

    Synthetic rather than collected: a policy's own states are where it
    already behaves. Biased toward every shielded boundary: each reactor's
    mass walls, a full or empty storage tank, exhausted feedstock. Stages are
    independent per reactor, so about two thirds of the draws have at least
    one vessel unloading into the tank. Species are drawn at random (the
    label does not read them, so the shield learns not to); each reactor's
    structural channels are its own, and the plant-wide channels are the
    same in every row, as in the environment.

    Args:
        n: Batch size.
        rng: NumPy generator.

    Returns:
        Float32 array of shape ``(n, SHIELD_OBS_DIM)``.
    """
    cfg = get_reactor_config()
    obs = np.zeros((n, SHIELD_OBS_DIM), dtype=np.float32)
    rows = np.arange(n)
    storage = _mix(rng, n, [(0.40, 0.0, 1.0), (0.45, 0.85, 1.02),
                            (0.15, 0.0, 0.05)])
    lf = _mix(rng, n, [(0.70, 0.0, 1.0), (0.30, 0.0, 0.03)])
    c5 = _mix(rng, n, [(0.70, 0.0, 1.0), (0.30, 0.0, 0.03)])
    for i in range(N_REACTORS):
        base = reactor_channel(i, 0)
        obs[:, base:base + N_SPECIES] = rng.uniform(0.0, 1.0, size=(n, N_SPECIES))
        obs[:, base + R_MASS] = _mix(rng, n, [(0.40, 0.0, 1.0),
                                              (0.40, 0.80, 1.02),
                                              (0.20, 0.0, 0.02)])
        stages = rng.choice(5, size=n, p=[0.25, 0.30, 0.10, 0.30, 0.05])
        obs[rows, base + R_STAGE + stages] = 1.0
        obs[:, base + R_CREDIT] = rng.uniform(0.0, 1.0, size=n)
        obs[:, base + R_MMAX] = cfg["Mmax"][i] / 220000.0
        obs[:, base + R_FMAX_LF] = cfg["max_flow_LF"][i] / (2487 / 3600 * 4.0)
        obs[:, base + R_FMAX_C5] = cfg["max_flow_C5"][i] / (628 / 3600 * 4.0)
        obs[:, base + R_STORAGE] = storage
        obs[:, base + R_LF] = lf
        obs[:, base + R_C5] = c5
    return obs


def sample_actions(n, rng):
    """Draw synthetic actions in [-1, 1] with saturated corners over-represented.

    Returns:
        Float32 array of shape ``(n, ACTION_DIM)``.
    """
    act = rng.uniform(-1.0, 1.0, size=(n, ACTION_DIM)).astype(np.float32)
    sat = rng.random(size=(n, ACTION_DIM)) < 0.25
    act[sat] = np.sign(rng.uniform(-1.0, 1.0, size=int(sat.sum()))).astype(np.float32)
    return act


def sample_labeled_batch(n, rng, const=None, oversample=4,
                         frac_unsafe=0.55, frac_boundary=0.25,
                         boundary_width=0.05):
    """Draw a stratified batch of (state, action, margin) triples.

    Candidates are over-drawn and re-mixed to a fixed unsafe / boundary / safe
    split, so the boundary -- where the operator runs -- is not undertrained.

    Returns:
        Tuple of float32 tensors (obs ``(n, SHIELD_OBS_DIM)``, act
        ``(n, ACTION_DIM)``, margin ``(n,)``).
    """
    c = plant_constants() if const is None else const
    m_cand = max(n * oversample, n)
    obs = torch.from_numpy(sample_states(m_cand, rng))
    act = torch.from_numpy(sample_actions(m_cand, rng))
    with torch.no_grad():
        marg = safety_margin(obs, act, const=c)

    m = marg.numpy()
    idx_bound = np.flatnonzero(np.abs(m) < boundary_width)
    idx_unsafe = np.flatnonzero(m < -boundary_width)
    idx_safe = np.flatnonzero(m >= boundary_width)

    def take(pool, k):
        if k <= 0 or pool.size == 0:
            return np.empty(0, dtype=int)
        return rng.choice(pool, size=k, replace=pool.size < k)

    n_b = int(round(n * frac_boundary))
    n_u = int(round(n * frac_unsafe))
    pick = np.concatenate([take(idx_bound, n_b), take(idx_unsafe, n_u)])
    pick = np.concatenate([pick, take(idx_safe, n - pick.size)])
    if pick.size < n:
        pick = np.concatenate([pick, take(np.arange(m.size), n - pick.size)])
    pick = pick[:n]
    rng.shuffle(pick)
    sel = torch.from_numpy(pick.astype(np.int64))
    return obs[sel], act[sel], marg[sel]


__all__ = ["safety_margins", "safety_margin", "smooth_min", "sample_states",
           "sample_actions", "sample_labeled_batch", "plant_constants",
           "assert_plant_constants_match", "shut_action", "BACKOFF_MASS",
           "SAFE_BUFFER", "SMOOTH_TAU"]
