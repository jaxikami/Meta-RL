"""Static configuration of the plant, and the meta-RL task distribution.

The first half is Fermentation-PPO's ``envs/config.py`` verbatim: three
heterogeneous fed-batch reactors (feed pumps 1x, 4x and 0.5x) sharing one LF /
C5 feedstock pool and discharging into one central storage tank, the species,
feeds, stage credits, the two titer specs and the plant heat-removal cap. The
environment (``metarl.env.safe`` / ``metarl.env.bench``) is that project's,
so Safe PPO and the arms here are benchmarked on one plant;
``tests/smoke_test.py`` steps both side by side.

The second half is this project's: the observation's per-reactor layout
offsets and the meta-RL task distribution. Only FLOW quantities are
randomized across tasks (feed concentration, each reactor's feed-pump gain and
vessel leak); the chemistry is pinned at the nominal values on every task and
in every reactor. The nominal task is the Fermentation-PPO plant exactly.
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np


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

# ─── Product Quality Specification ───────────────────────────────────────────
# The Meta-RL plant's concentration constraints (same chemistry and units as
# C[Eth]).  The harvest titer floor acts on the outlet stream of every
# draining reactor at every step; the purity spec acts on each batch's lot,
# the mass-weighted mean titer of everything that batch discharged, judged
# when the batch completes (and at episode end).
MIN_HARVEST_TITER = 65.0    # g/L in the outlet stream while draining
MIN_PRODUCT_TITER = 75.0    # g/L, mass-weighted mean over a lot

# ─── Plant Heat Removal ───────────────────────────────────────────────────────
# One cooling loop removes every reactor's fermentation heat and one scrubber
# its CO2, so their capacity caps the plant's total CO2 evolution rate (CER),
# the heat-load proxy.  A reactor's CER is its broth's CO2 production rate
# times its mass, R[CO2] x M (g/s with concentrations in g/kg), times
# HEAT_LOAD_SCALE for kg/h.  The cap sits just below what the three reactors
# release when all feed at half rate in phase (~580 kg/h; ~1,100 kg/h at full
# rate), so they share the budget: feeding fast in one reactor leaves less for
# the others, and a high-titer batch must be fed slowly over a longer fed-batch.
HEAT_LOAD_SCALE = 3.6
MAX_HEAT_LOAD = 540.0       # kg CO2/h, plant-wide


# =============================================================================
# META-RL: OBSERVATION LAYOUT AND TASK DISTRIBUTION (not in Fermentation-PPO)
# =============================================================================
N_REACTORS = 3
N_SPECIES = len(SPECIES_NAMES)
I_ETH = SPECIES_IDX['Eth']


def get_reactor_config():
    """The plant's reactor capacities (``get_main_reactor_config``)."""
    return get_main_reactor_config()


# Each reactor's feed pumps as a multiple of the 1x reactor's (2487 kg/h LF,
# 628 kg/h C5), as in get_main_reactor_config.
REACTOR_FLOW_SCALE = (1.0, 4.0, 0.5)


# One observation row per reactor, the environment's get_state_norm() row
# (REACTOR_OBS_DIM = 28; the flat observation is the three rows in order):
#     0-14  species concentrations / C_MAX_LIMITS
#     15    M / Mmax
#     16-20 stage one-hot (5)
#     21    stage credit remaining / base_credits[stage]
#     22    Mmax / 220,000 kg
#     23    LF pump capacity / (4x the 1x reactor's)
#     24    C5 pump capacity / (4x the 1x reactor's)
#     25    central storage mass / S_MAX         (plant-wide, in every row)
#     26    LF feedstock remaining / INITIAL_LF  (plant-wide, in every row)
#     27    C5 feedstock remaining / INITIAL_C5  (plant-wide, in every row)
R_MASS = 15
R_STAGE = 16
R_CREDIT = 21
R_MMAX = 22
R_FMAX_LF = 23
R_FMAX_C5 = 24
R_STORAGE = 25
R_LF = 26
R_C5 = 27
REACTOR_OBS_DIM = 28

# The environment's reward exchange rates, locals of its step (safe.py /
# bench.py), so the selection can price a violation event in ethanol:
# the production reward per unit of harvested ethanol (kg x g/L), and the
# fixed spike of one violation at the limit (FermentationEnvBench's
# spike_coef, and BASE_SPIKE in FermentationEnv's no-Lagrangian case study).
# smoke_test checks both against the environment.
PROD_REWARD_PER_ETH = 0.5 / 1e7
VIOLATION_SPIKE = 0.1

# Normalise each reactor's lot in the privileged state (the observation has
# no lot channel). One batch's lot is at most a vessel (~0.55).
PRODUCT_MASS_SCALE = 400000.0
# The ethanol C_MAX, so the lot titer and the broth ethanol share one scale.
TITER_SCALE = float(C_MAX_LIMITS[I_ETH])


# ─── Uncertainty / domain-randomization task distribution ─────────────────────
# A task fixes one realization of the latent theta, sampled once per episode
# and never observed by the agent. NOT randomized: the chemistry, held at the
# nominal values on every task and in every reactor.
#
# Randomized: flow quantities only, each the persistent counterpart of a
# mid-episode disturbance kind:
#   A. feed concentration -- correlated solids scaling plus a glucose/xylose
#      split shift on both feeds. The feed tanks are shared, so this is one
#      draw for the whole plant (counterpart of `feed_conc`).
#   B. each reactor's feed-pump gain and vessel leak, drawn independently per
#      reactor (counterparts of `feed_gain` and `leak`): a plant has a pump
#      and a vessel per reactor, so the encoder must also infer WHICH vessel
#      is off.
TASK_PRIORS = {
    "feed_solids_scale": (0.8, 1.2),   # correlated CS/XS/LS scaling, both feeds
    "feed_gx_ratio": (0.8, 1.2),       # glucose up / xylose down (and vice-versa)
    "feed_gain": (0.85, 1.15),         # realized / commanded feed, per reactor
    "leak_frac": (0.0, 0.0015),        # fraction of mass lost per hour, per reactor
    "leak_clip": (0.0, 0.006),         # OOD widening must not produce negative leak
}

OOD_WIDEN_FACTOR = 1.8


def _per_reactor(name):
    """Descriptor names for one quantity, one per reactor."""
    return [f"{name}_r{i}" for i in range(N_REACTORS)]


# Latent descriptor layout, used for diagnostics, the auxiliary "task" head and
# the critics' task conditioning. kinetic_activity is 1.0 and contam_rate 0 in
# the prior; they move only when a biochemical fault or a contamination fires
# in that reactor. downstream_rate is the storage tank's outflow as a multiple
# of S_OUT_RATE, turnaround_mult the turnaround time, storage_capacity the
# tank's usable volume and cooling_capacity the heat-load cap as multiples of
# nominal, all 1.0 in the prior; the fault of the same name moves each.
TASK_CONTEXT_NAMES = (["feed_solids", "feed_gx"]
                      + _per_reactor("feed_gain")
                      + _per_reactor("leak_frac")
                      + _per_reactor("kinetic_activity")
                      + _per_reactor("contam_rate")
                      + ["downstream_rate", "turnaround_mult",
                         "storage_capacity", "cooling_capacity"])
TASK_CONTEXT_DIM = len(TASK_CONTEXT_NAMES)
# Offsets of each group in the descriptor.
CTX_SOLIDS, CTX_GX = 0, 1
CTX_GAIN = 2
CTX_LEAK = CTX_GAIN + N_REACTORS
CTX_KIN = CTX_LEAK + N_REACTORS
CTX_CONTAM = CTX_KIN + N_REACTORS
CTX_DOWNSTREAM = CTX_CONTAM + N_REACTORS
CTX_TURNAROUND = CTX_DOWNSTREAM + 1
CTX_STORAGE_CAP = CTX_TURNAROUND + 1
CTX_COOLING_CAP = CTX_STORAGE_CAP + 1

# Standardization for the auxiliary regression target (per-quantity centers
# and scales, repeated per reactor).
TASK_CONTEXT_CENTER = np.array(
    [1.00, 1.00] + [1.00] * N_REACTORS + [0.00075] * N_REACTORS
    + [1.00] * N_REACTORS + [0.0] * N_REACTORS + [1.00, 1.00, 1.00, 1.00],
    dtype=np.float64)
TASK_CONTEXT_SCALE = np.array(
    [0.12, 0.12] + [0.09] * N_REACTORS + [0.00045] * N_REACTORS
    + [0.20] * N_REACTORS + [0.001] * N_REACTORS + [0.20, 0.50, 0.10, 0.10],
    dtype=np.float64)


def standardize_context(context):
    """Standardize a raw task-context vector for use as a regression target.

    Args:
        context: Raw length-``TASK_CONTEXT_DIM`` descriptor vector.

    Returns:
        Float32 array of the same length, roughly zero-mean unit-scale.
    """
    c = np.asarray(context, dtype=np.float64)
    return ((c - TASK_CONTEXT_CENTER) / TASK_CONTEXT_SCALE).astype(np.float32)


def _u(rng, lohi, size=None):
    """Sample uniformly from a (lo, hi) tuple."""
    if size is None:
        return float(rng.uniform(lohi[0], lohi[1]))
    return rng.uniform(lohi[0], lohi[1], size=size).astype(np.float64)


def _context(solids, gx, gains, leaks):
    """A prior task's descriptor: no fault has acted yet."""
    return np.concatenate([[solids, gx], gains, leaks,
                           np.ones(N_REACTORS), np.zeros(N_REACTORS),
                           [1.0, 1.0, 1.0, 1.0]])


def nominal_task():
    """Return the nominal task theta_0: the Fermentation-PPO plant exactly.

    The sole training task for the single-regime arms (``baseline``,
    ``saferl``). Its flow is ideal -- every pump gain 1.0, no leak, unscaled
    feed -- so those arms never see a miscalibrated pump, a leaking vessel or
    off-spec feedstock until evaluation.

    Returns:
        Task dict with keys: C_C5, C_LF, flow, context, meta.
    """
    gains = np.ones(N_REACTORS)
    leaks = np.zeros(N_REACTORS)
    return {
        "C_C5": C_C5_BASE.copy(),
        "C_LF": C_LF_BASE.copy(),
        "flow": {"feed_gain": gains.copy(), "leak_frac": leaks.copy()},
        "context": _context(1.0, 1.0, gains, leaks),
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
    """Sample one task theta ~ p(theta).

    Only flow quantities vary. The feed concentration is one draw for the
    plant; each reactor's pump gain and leak are independent draws.

    Args:
        rng: Optional ``numpy.random.Generator``; a fresh one when None.
        ood: Draw from priors widened by ``OOD_WIDEN_FACTOR``.

    Returns:
        Task dict with keys:
            - C_C5, C_LF: randomized feed composition vectors (length 15).
            - flow: dict with per-reactor ``feed_gain`` and ``leak_frac``.
            - context: the true latent descriptor (TASK_CONTEXT_NAMES).
            - meta: human-readable dict of the raw draws.
    """
    if rng is None:
        rng = np.random.default_rng()

    f = OOD_WIDEN_FACTOR if ood else 1.0
    W = (lambda k: _widen(TASK_PRIORS[k], f)) if ood else (lambda k: TASK_PRIORS[k])

    solids = _u(rng, W("feed_solids_scale"))
    gx = _u(rng, W("feed_gx_ratio"))
    C_C5 = C_C5_BASE.copy()
    C_LF = C_LF_BASE.copy()
    for feed in (C_C5, C_LF):
        for sp in ("CS", "XS", "LS"):
            feed[SPECIES_IDX[sp]] *= solids
        feed[SPECIES_IDX["G"]] *= gx
        feed[SPECIES_IDX["X"]] /= gx

    gains = _u(rng, W("feed_gain"), size=N_REACTORS)
    leaks = np.clip(_u(rng, W("leak_frac"), size=N_REACTORS),
                    *TASK_PRIORS["leak_clip"])

    meta = {"feed_solids": solids, "feed_gx": gx,
            "feed_gain": gains.tolist(), "leak_frac": leaks.tolist(),
            "nominal": False, "ood": ood}
    return {"C_C5": C_C5, "C_LF": C_LF,
            "flow": {"feed_gain": gains, "leak_frac": leaks},
            "context": _context(solids, gx, gains, leaks), "meta": meta}
