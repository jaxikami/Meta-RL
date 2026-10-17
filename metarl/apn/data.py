"""Safety-margin labels and synthetic training data for the plant's APN.

Each constraint becomes a signed, normalized distance (positive = feasible),
combined by a smooth minimum ``-tau * logsumexp(-m / tau)``:

    reactor i overflow     M_i <= Mmax                     (5% back-off)
    reactor i underflow    M_i >= DRY_FLOOR                (5% back-off)
    storage overflow       S <= S_MAX                      (5% back-off)
    reactor i harvest titer, reactor i lot purity          (observable
                                                            concentrations only)

Storage UNDERFLOW is deliberately not shielded. Within one control interval
the only way to raise the tank is to drain a reactor that is already in
unload; when none is, every action is equally unsafe, and the smooth minimum
would then be dominated by a constraint no action can move -- the projection
would shut every feed to "repair" it, costing production for nothing. Keeping
the tank supplied is a scheduling decision made tens of hours earlier (when
each batch is started and how fast it runs), so it is left to the policy and
the reward, like harvest titer and purity under masked concentrations.

The label rolls the action forward over one control interval, reproducing the
environment's discharge cap, the shared feedstock pool drawn in reactor order,
and the storage tank's inflow and downstream draw. It models mass balance and
ethanol dilution exactly but drops fermentation, which under-predicts titer
(an error in the safe direction), and holds every stage fixed across the
interval.

It is a NOMINAL-plant label: pump gains, leaks and the downstream rate are
latent and do not enter it, so coping with a disturbed plant stays the
policy's job.
"""

import numpy as np
import torch

from metarl.env.config import (C_MAX_LIMITS, INITIAL_C5, INITIAL_LF, I_ETH,
                               MAX_FLOW_OUT_REAC, MIN_HARVEST_TITER,
                               MIN_PRODUCT_TITER, PRODUCT_MASS_SCALE,
                               TIME_STEP_DURATION, TITER_SCALE,
                               get_reactor_config,
                               MASK_INTERNAL_CONCENTRATIONS, N_REACTORS, S_MAX,
                               S_OUT_RATE, R_MASS, R_STAGE, R_CREDIT,
                               R_LOT_MASS, R_LOT_TITER, N_SPECIES)
from metarl.env.layout import (SHIELD_OBS_DIM, ACTION_DIM,
                               ACTION_DIM_PER_REACTOR, STORAGE_CHANNEL,
                               LF_CHANNEL, C5_CHANNEL, TIME_ACTION_DIMS,
                               reactor_channel)
from metarl.env.plant import PlantFermentationEnv

# ─── Constraint back-off ──────────────────────────────────────────────────────
BACKOFF_MASS = 0.05
BACKOFF_QUALITY = 0.0
SAFE_BUFFER = 1.0 - BACKOFF_MASS

# Mirrors of PlantFermentationEnv constants (checked by
# assert_plant_constants_match).
DRY_FLOOR = 10.0
UNDERFLOW_SCALE = 100.0          # env.UNDERFLOW_BUFFER_KG
GUARD_MASS = 50.0                # the discharge cap's floor

INACTIVE_SLACK = 1.5
PURITY_ACTIVATION_KG = 10000.0
DRAIN_EPS = 0.02
SMOOTH_TAU = 0.05
CONTROL_ACTION_INTERVAL = 5

_MASS_CH = [reactor_channel(i, R_MASS) for i in range(N_REACTORS)]
_ETH_CH = [reactor_channel(i, I_ETH) for i in range(N_REACTORS)]
_LOT_MASS_CH = [reactor_channel(i, R_LOT_MASS) for i in range(N_REACTORS)]
_LOT_TITER_CH = [reactor_channel(i, R_LOT_TITER) for i in range(N_REACTORS)]


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
        "mask_internal_concentrations": bool(MASK_INTERNAL_CONCENTRATIONS),
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
    assert env.min_harvest_titer == MIN_HARVEST_TITER
    assert env.min_product_titer == MIN_PRODUCT_TITER
    assert env.max_flow_out_reac == MAX_FLOW_OUT_REAC
    assert env.mask_internal_concentrations == MASK_INTERNAL_CONCENTRATIONS
    assert env.S_max == S_MAX and env.S_out_rate == S_OUT_RATE
    assert np.allclose(env.Mmax, c["Mmax"].numpy())
    assert np.allclose(env.max_flow_LF, c["max_flow_LF"].numpy())
    assert np.allclose(env.max_flow_C5, c["max_flow_C5"].numpy())


# =============================================================================
# MARGIN LABEL
# =============================================================================

def safety_margins(obs, act, const=None, mask_concentrations=None):
    """Compute per-constraint margins for a batch of state-actions.

    Args:
        obs: Normalized observations, shape ``(B, >= SHIELD_OBS_DIM)``.
        act: Actions in [-1, 1], shape ``(B, ACTION_DIM)``.
        const: Optional dict from ``plant_constants``.
        mask_concentrations: Whether in-reactor concentrations are masked;
            defaults to the constants' setting.

    Returns:
        Tensor ``(B, 2N + 1)`` [overflow_0..N-1, underflow_0..N-1, storage
        overflow] when masked, or ``(B, 4N + 1)`` with each reactor's harvest
        titer and lot purity appended when observable. Positive means
        satisfied with room to spare.
    """
    c = plant_constants() if const is None else const
    dt, n_inner = c["dt"], c["n_inner"]
    Mmax = c["Mmax"].to(obs.device)
    fl_lf = c["max_flow_LF"].to(obs.device)
    fl_c5 = c["max_flow_C5"].to(obs.device)
    if mask_concentrations is None:
        mask_concentrations = bool(c.get("mask_internal_concentrations",
                                         MASK_INTERNAL_CONCENTRATIONS))
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

    quality = not mask_concentrations
    if quality:
        eth = obs[:, _ETH_CH] * float(C_MAX_LIMITS[I_ETH])
        lot_mass = obs[:, _LOT_MASS_CH] * PRODUCT_MASS_SCALE
        lot_titer = obs[:, _LOT_TITER_CH] * TITER_SCALE
        harvest_mass = torch.zeros_like(M)
        harvest_eth = torch.zeros_like(M)
        titer_min = eth

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
        if quality:
            # Drawn at the titer at the START of the step; dilution only.
            harvest_mass = harvest_mass + f_out * dt
            harvest_eth = harvest_eth + f_out * dt * eth
            eth = eth * torch.exp(-F_in * dt / torch.clamp(M, min=1.0))
            titer_min = torch.minimum(titer_min, eth)
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
    parts = [m_over, m_under, s_over.unsqueeze(-1)]

    if quality:
        drain_gate = u_out / (u_out + DRAIN_EPS)
        rel_titer = ((titer_min - MIN_HARVEST_TITER * (1.0 + BACKOFF_QUALITY))
                     / MIN_HARVEST_TITER)
        m_titer = drain_gate * rel_titer + (1.0 - drain_gate) * INACTIVE_SLACK
        pm_next = lot_mass + harvest_mass
        eth_next = lot_mass * lot_titer + harvest_eth
        mean_next = eth_next / torch.clamp(pm_next, min=1e-6)
        rel_pur = ((mean_next - MIN_PRODUCT_TITER * (1.0 + BACKOFF_QUALITY))
                   / MIN_PRODUCT_TITER)
        w = drain_gate * (pm_next / (pm_next + PURITY_ACTIVATION_KG))
        m_pur = w * rel_pur + (1.0 - w) * INACTIVE_SLACK
        parts += [m_titer, m_pur]
    return torch.cat(parts, dim=-1)


def smooth_min(margins, tau=SMOOTH_TAU):
    """Differentiable minimum over the constraint axis."""
    return -tau * torch.logsumexp(-margins / tau, dim=-1)


def safety_margin(obs, act, const=None, tau=SMOOTH_TAU, mask_concentrations=None):
    """Scalar safety margin: the smooth minimum over shielded constraints.

    Returns:
        Tensor of shape ``(B,)``; positive means safe.
    """
    return smooth_min(safety_margins(obs, act, const=const,
                                     mask_concentrations=mask_concentrations),
                      tau=tau)


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


def sample_states(n, rng, mask_concentrations=None):
    """Draw synthetic normalized observations in the shield's channel layout.

    Synthetic rather than collected: a policy's own states are where it
    already behaves. Biased toward every shielded boundary: each reactor's
    mass walls, a full or empty storage tank, exhausted feedstock. Stages are
    independent per reactor, so about two thirds of the draws have at least
    one vessel unloading into the tank.

    Args:
        n: Batch size.
        rng: NumPy generator.
        mask_concentrations: Whether species channels are masked to 0.

    Returns:
        Float32 array of shape ``(n, SHIELD_OBS_DIM)``.
    """
    if mask_concentrations is None:
        mask_concentrations = MASK_INTERNAL_CONCENTRATIONS
    obs = np.zeros((n, SHIELD_OBS_DIM), dtype=np.float32)
    rows = np.arange(n)
    for i in range(N_REACTORS):
        base = reactor_channel(i, 0)
        if not mask_concentrations:
            obs[:, base:base + N_SPECIES] = rng.uniform(0.0, 1.0, size=(n, N_SPECIES))
            obs[:, base + I_ETH] = _mix(rng, n, [(0.35, 0.0, 1.0),
                                                 (0.45, 0.25, 0.60),
                                                 (0.20, 0.0, 0.10)])
        obs[:, base + R_MASS] = _mix(rng, n, [(0.40, 0.0, 1.0),
                                              (0.40, 0.80, 1.02),
                                              (0.20, 0.0, 0.02)])
        stages = rng.choice(5, size=n, p=[0.25, 0.30, 0.10, 0.30, 0.05])
        obs[rows, base + R_STAGE + stages] = 1.0
        obs[:, base + R_CREDIT] = rng.uniform(0.0, 1.0, size=n)
        obs[:, base + R_LOT_MASS] = _mix(rng, n, [(0.30, 0.0, 0.005),
                                                  (0.45, 0.005, 0.25),
                                                  (0.25, 0.25, 1.0)])
        if not mask_concentrations:
            around = np.clip(obs[:, base + I_ETH] * rng.uniform(0.7, 1.3, size=n)
                             + rng.normal(0.0, 0.05, size=n), 0.0, 1.0)
            indep = rng.uniform(0.0, 1.0, size=n)
            obs[:, base + R_LOT_TITER] = np.where(rng.random(size=n) < 0.15,
                                                  indep, around)
        else:
            obs[:, base + R_LOT_TITER] = _mix(rng, n, [(0.60, 0.35, 0.65),
                                                       (0.40, 0.0, 1.0)])
        empty = obs[:, base + R_LOT_MASS] <= 1e-6
        obs[empty, base + R_LOT_TITER] = 0.0

    # Storage: uniform, plus dense samples against the overflow wall and near
    # empty (where the draw-down below zero must be modelled right).
    obs[:, STORAGE_CHANNEL] = _mix(rng, n, [(0.40, 0.0, 1.0),
                                            (0.45, 0.85, 1.02),
                                            (0.15, 0.0, 0.05)])
    obs[:, LF_CHANNEL] = _mix(rng, n, [(0.70, 0.0, 1.0), (0.30, 0.0, 0.03)])
    obs[:, C5_CHANNEL] = _mix(rng, n, [(0.70, 0.0, 1.0), (0.30, 0.0, 0.03)])
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
    mask_conc = bool(c.get("mask_internal_concentrations", MASK_INTERNAL_CONCENTRATIONS))
    m_cand = max(n * oversample, n)
    obs = torch.from_numpy(sample_states(m_cand, rng, mask_concentrations=mask_conc))
    act = torch.from_numpy(sample_actions(m_cand, rng))
    with torch.no_grad():
        marg = safety_margin(obs, act, const=c, mask_concentrations=mask_conc)

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
           "BACKOFF_QUALITY", "SAFE_BUFFER", "SMOOTH_TAU"]
