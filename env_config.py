"""Static configuration for multi-reactor fermentation environments.

This module centralizes structural limits, feed compositions, species metadata,
and timing constants shared by benchmark and safe environment variants.
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from env_core import (
    kinetics_from_pH, PARAM_DIM,
    NOMINAL_MAX_ADS, NOMINAL_K_ADS,
    _PH as NOMINAL_PH,
)


# ─── Reactor Configuration ────────────────────────────────────────────────────
def get_main_reactor_config():
    """Define structural capacities for the heterogeneous 3-reactor setup.

    Returns:
        Dictionary containing arrays for:
            - Mmax: Maximum reactor mass limits.
            - max_flow_LF: Maximum LF feed flow rates.
            - max_flow_C5: Maximum C5 feed flow rates.
    """
    return {
        'Mmax': np.array([220000.0, 220000.0, 220000.0], dtype=np.float64),
        'max_flow_LF': np.array(
            [(2487/3600), (2487/3600) * 4, (2487/3600) / 2], dtype=np.float64),
        'max_flow_C5': np.array(
            [(628/3600), (628/3600) * 4, (628/3600) / 2], dtype=np.float64),
    }

# ─── Global Resource Pool ────────────────────────────────────────────────────
INITIAL_LF = 2000000.0
INITIAL_C5 = 600000.0

# ─── Central Storage Hardware Specifications ─────────────────────────────────
S_MAX = 400000.0
S_MIN = 5000.0
S_OUT_RATE = 1500.0 / 3600.0
S_INITIAL = 350000.0

# ─── Tracked Species Definition ───────────────────────────────────────────────
SPECIES_NAMES = ['CS', 'XS', 'LS', 'C', 'G', 'X', 'F', 'E', 'AC', 'Cell', 'Eth', 'CO2', 'ACT', 'HMF', 'Base']
SPECIES_IDX = {name: i for i, name in enumerate(SPECIES_NAMES)}

# Defines the physical maximum concentration (kg/L) realistically achievable for each specific species.
# Used to statically normalize signals perfectly into the [0, 1] neural network range without crushing trace elements.
C_MAX_LIMITS = np.array([
    100.0,  # CS
    5.0,    # XS
    100.0,  # LS
    200.0,  # C
    150.0,  # G
    100.0,  # X
    5.0,    # F
    10.0,   # E
    30.0,   # AC
    50.0,   # Cell
    150.0,  # Eth
    20.0,   # CO2
    1.0,    # ACT
    5.0,    # HMF
    20.0    # Base
], dtype=np.float32)

# ─── Feed Flow Properties ─────────────────────────────────────────────────────
def get_C_C5():
    """Build baseline C5 feed composition vector across tracked species.

    Returns:
        NumPy array of length 15 containing species concentrations for C5 feed.
    """
    arr = np.zeros(15)
    vals = {'CS': 1.2, 'XS': 0.5, 'LS': 0.7, 'G': 10.0, 'X': 29.7, 'F': 0.5, 'AC': 4.1, 'ACT': 0.1, 'HMF': 0.3}
    for k, v in vals.items():
        arr[SPECIES_IDX[k]] = v
    return arr


def get_C_LF():
    """Build baseline LF feed composition vector across tracked species.

    Returns:
        NumPy array of length 15 containing species concentrations for LF feed.
    """
    arr = np.zeros(15)
    vals = {'CS': 50.0, 'XS': 1.0, 'LS': 78.0, 'G': 98.0, 'X': 59.0, 'F': 0.2, 'E': 4.9, 'AC': 16.0, 'ACT': 0.1, 'HMF': 0.1, 'Base': 8.6}
    for k, v in vals.items():
        arr[SPECIES_IDX[k]] = v
    return arr

C_C5_BASE = get_C_C5()
C_LF_BASE = get_C_LF()

# ─── Time Credit and Flow Integration ────────────────────────────────────────
BASE_CREDITS = np.array([10.0, 60.0, 120.0, 70.0, 2.0])
TIME_STEP_DURATION = 3600.0
MAX_FLOW_OUT_REAC = 5000.0 / 3600.0


# =============================================================================
# UNCERTAINTY / DOMAIN-RANDOMIZATION TASK DISTRIBUTION  (proposal §2.3)
# =============================================================================
# A "task" T fixes one realization of the latent process parameters theta ~ p(theta).
# theta is sampled ONCE per episode and held constant for the whole episode, so the
# within-episode transition history is informative about a single underlying regime
# (proposal §2.3, guideline iii).  The agent never observes theta directly; a PEARL
# context encoder must infer it from (o, a, r) transitions.
#
# Priors are bounded and physically motivated (guideline i) and correlated within
# each group where a single upstream cause co-varies several coefficients
# (guideline ii).  All ranges below are intentionally centralized here so they can
# be tuned without touching the physics or the agent.
#
# Four randomization groups:
#   1. Kinetic modifiers (eta, gamma_G, gamma_X)  — driven by operating pH + a
#      shared enzyme-cocktail hydrolysis factor.  A single pH draw shifts all three
#      channels coherently via the Gaussian bump functions in env_core.
#   2. Toxin sensitivity   — per-toxin tolerance multipliers sharing a common
#      strain-tolerance base so glucose/xylose channels stay biologically linked.
#   3. Enzyme adsorption   — a shared lot-efficacy scale on capacities AND affinities
#      (drawn jointly, not independently).
#   4. Feedstock composition — correlated scaling of inlet solid fractions plus a
#      glucose/xylose split shift, applied to both LF and C5 feeds.

TASK_PRIORS = {
    # Group 1 — kinetics
    "pH_mean": NOMINAL_PH,
    "pH_std": 0.02,                 # ~correlated shift; ph_x is very pH-sensitive
    "pH_clip": (5.33, 5.45),
    "eta_cocktail": (0.85, 1.15),   # extra hydrolysis (enzyme-cocktail) drift on eta
    # Group 2 — toxin tolerance
    "tox_base": (0.7, 1.6),         # shared strain tolerance base
    "tox_jitter": (0.85, 1.15),     # per-toxin multiplicative jitter
    "tox_clip": (0.5, 2.5),
    # Group 3 — enzyme adsorption
    "enz_lot_scale": (0.7, 1.3),    # shared lot efficacy (capacities + affinities)
    "enz_comp_jitter": (0.9, 1.1),  # small per-component jitter
    "enz_aff_couple": 0.5,          # fraction of lot scale transferred to affinities
    # Group 4 — feedstock
    "feed_solids_scale": (0.8, 1.2),   # correlated CS/XS/LS scaling in both feeds
    "feed_gx_ratio": (0.8, 1.2),       # glucose up / xylose down (and vice-versa)
}

# Human-readable names for the true latent descriptors, for diagnostics/plots.
# NOTE: these are exposed only for analysis — the agent never sees them.
TASK_CONTEXT_NAMES = [
    "pH", "eta", "ph_g", "ph_x",
    "tox_F", "tox_ACT", "tox_HMF",
    "enz_scale", "feed_solids", "feed_gx",
]


def _u(rng, lohi):
    """Sample a scalar uniformly from a (lo, hi) tuple."""
    return float(rng.uniform(lohi[0], lohi[1]))


def build_param_vector(eta, ph_g, ph_x, max_ads, k_ads, tox):
    """Assemble the length-``PARAM_DIM`` kinetic vector for calculate_rates_numba.

    Args:
        eta: Overall hydrolysis modifier.
        ph_g: Glucose-uptake pH modifier.
        ph_x: Xylose-uptake pH modifier.
        max_ads: Length-3 enzyme adsorption capacities.
        k_ads: Length-3 enzyme adsorption affinities.
        tox: Length-3 toxin tolerance multipliers [F, ACT, HMF].

    Returns:
        NumPy float64 array of length PARAM_DIM.
    """
    p = np.empty(PARAM_DIM, dtype=np.float64)
    p[0], p[1], p[2] = eta, ph_g, ph_x
    p[3:6] = np.asarray(max_ads, dtype=np.float64)
    p[6:9] = np.asarray(k_ads, dtype=np.float64)
    p[9:12] = np.asarray(tox, dtype=np.float64)
    return p


def nominal_task():
    """Return the nominal task theta_0 (no randomization).

    Used both as the reference operating point and as the sole training
    distribution for the non-adaptive Safe-RL baseline (proposal §3.1).

    Returns:
        Task dict with keys: params, C_C5, C_LF, context, meta.
    """
    eta, ph_g, ph_x = kinetics_from_pH(NOMINAL_PH)
    tox = np.array([1.0, 1.0, 1.0])
    params = build_param_vector(
        eta, ph_g, ph_x, NOMINAL_MAX_ADS, NOMINAL_K_ADS, tox)
    context = np.array(
        [NOMINAL_PH, eta, ph_g, ph_x, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    return {
        "params": params,
        "C_C5": C_C5_BASE.copy(),
        "C_LF": C_LF_BASE.copy(),
        "context": context,
        "meta": {"nominal": True},
    }


def _widen(lohi, factor):
    """Expand a (lo, hi) range symmetrically about its centre by ``factor``.

    Args:
        lohi: A (lo, hi) tuple.
        factor: Multiplier on the half-width (1.0 = unchanged).

    Returns:
        A widened (lo, hi) tuple.
    """
    lo, hi = lohi
    c, h = 0.5 * (lo + hi), 0.5 * (hi - lo) * factor
    return (c - h, c + h)


def sample_task(rng=None, ood=False):
    """Sample one task theta ~ p(theta) from the bounded, correlated priors.

    Args:
        rng: Optional ``numpy.random.Generator``.  A fresh default generator is
            created when None.
        ood: If True, draw from *widened* priors that extend beyond the training
            ranges, producing genuinely out-of-distribution tasks for evaluation
            (proposal §2.3 guideline iv).  The correlation structure within each
            group is preserved.

    Returns:
        Task dict with keys:
            - params: length-PARAM_DIM kinetic vector for calculate_rates_numba.
            - C_C5, C_LF: randomized feed composition vectors (length 15).
            - context: the true latent descriptor vector (diagnostics only).
            - meta: human-readable dict of the raw draws.
    """
    if rng is None:
        rng = np.random.default_rng()

    # OOD widening factor applied to every group's sampling range.
    f = 1.8 if ood else 1.0
    W = (lambda k: _widen(TASK_PRIORS[k], f)) if ood else (lambda k: TASK_PRIORS[k])

    # ── Group 1: kinetics via pH (correlated) + enzyme-cocktail hydrolysis drift
    pH_std = TASK_PRIORS["pH_std"] * (2.0 if ood else 1.0)
    pH_clip = _widen(TASK_PRIORS["pH_clip"], f)
    pH = float(np.clip(
        rng.normal(TASK_PRIORS["pH_mean"], pH_std), pH_clip[0], pH_clip[1]))
    eta, ph_g, ph_x = kinetics_from_pH(pH)
    eta *= _u(rng, W("eta_cocktail"))

    # ── Group 2: toxin tolerance (shared base + per-toxin jitter, linked G/X)
    tox_base = _u(rng, W("tox_base"))
    tox = np.array([
        np.clip(tox_base * _u(rng, W("tox_jitter")), *TASK_PRIORS["tox_clip"])
        for _ in range(3)
    ])

    # ── Group 3: enzyme adsorption (shared lot scale on capacities + affinities)
    lot = _u(rng, W("enz_lot_scale"))
    comp_jit = rng.uniform(*W("enz_comp_jitter"), size=3)
    max_ads = NOMINAL_MAX_ADS * lot * comp_jit
    # Affinities co-vary with the same lot cause but only partially (coupling).
    aff_scale = 1.0 + TASK_PRIORS["enz_aff_couple"] * (lot - 1.0)
    k_ads = NOMINAL_K_ADS * aff_scale

    params = build_param_vector(eta, ph_g, ph_x, max_ads, k_ads, tox)

    # ── Group 4: feedstock composition (correlated solids + G/X split shift)
    solids = _u(rng, W("feed_solids_scale"))
    gx = _u(rng, W("feed_gx_ratio"))
    C_C5 = C_C5_BASE.copy()
    C_LF = C_LF_BASE.copy()
    for feed in (C_C5, C_LF):
        for sp in ("CS", "XS", "LS"):
            feed[SPECIES_IDX[sp]] *= solids
        feed[SPECIES_IDX["G"]] *= gx
        feed[SPECIES_IDX["X"]] /= gx

    context = np.array([pH, eta, ph_g, ph_x, tox[0], tox[1], tox[2],
                        lot, solids, gx])
    meta = {
        "pH": pH, "eta": eta, "ph_g": ph_g, "ph_x": ph_x,
        "tox": tox.tolist(), "enz_lot": lot,
        "feed_solids": solids, "feed_gx": gx, "nominal": False, "ood": ood,
    }
    return {"params": params, "C_C5": C_C5, "C_LF": C_LF,
            "context": context, "meta": meta}
