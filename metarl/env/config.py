"""Static configuration for the three-reactor plant with central storage.

The plant is the multi-reactor study's: three heterogeneous fed-batch reactors
(feed pumps sized 1x, 4x and 0.5x) sharing one LF / C5 feedstock pool and
discharging into one central storage tank that feeds downstream at a fixed
rate. The constraints, quality specs, instrument model and flow-only task
distribution are the lean single-reactor study's (``MetaRL_Lean``), extended
to three vessels; the storage tank adds two inventory constraints of its own.

Only FLOW quantities are randomized across tasks (feed concentration, each
reactor's feed-pump gain and vessel leak). The chemistry -- pH, kinetic
modifiers, enzyme adsorption, toxin tolerance -- is pinned at the
Fermentation-PPO project's values on every task and in every reactor.
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.core import (kinetics_from_pH, PARAM_DIM, NOMINAL_MAX_ADS,
                             NOMINAL_K_ADS, _PH as NOMINAL_PH)


# ─── Reactor Configuration ────────────────────────────────────────────────────
N_REACTORS = 3

# Feed pump capacity of the 1x reactor, kg/s. Full-throttle feeding through
# stages 0-1 delivers ~212,000 kg into it on top of the 10,000 kg charge, so
# its overflow is reachable only at maximum feed (or a slowed clock). The 4x
# reactor overflows at ~25% feed; the 0.5x reactor cannot fill in one co-feed
# at a unit clock.
BASE_FLOW_LF = 2487.0 / 3600.0
BASE_FLOW_C5 = 628.0 / 3600.0
REACTOR_FLOW_SCALE = (1.0, 4.0, 0.5)


def get_reactor_config():
    """Define structural capacities for the heterogeneous 3-reactor setup.

    Returns:
        Dictionary containing length-``N_REACTORS`` arrays for:
            - Mmax: Maximum reactor mass limits (kg).
            - max_flow_LF: Maximum LF feed flow rates (kg/s).
            - max_flow_C5: Maximum C5 feed flow rates (kg/s).
    """
    scale = np.asarray(REACTOR_FLOW_SCALE, dtype=np.float64)
    return {
        'Mmax': np.full(N_REACTORS, 220000.0, dtype=np.float64),
        'max_flow_LF': BASE_FLOW_LF * scale,
        'max_flow_C5': BASE_FLOW_C5 * scale,
    }


# ─── Global Resource Pool ─────────────────────────────────────────────────────
# Shared by all three reactors and drawn in reactor order (0, 1, 2) each hour.
# Ordinary operation (~2.3 batches per vessel) draws ~1.2e6 kg LF and ~0.3e6
# kg C5; every pump at full throttle could draw ~2.2e6 kg LF, so depletion is
# reachable under greedy feeding but not under ordinary operation.
INITIAL_LF = 2000000.0
INITIAL_C5 = 600000.0

# ─── Central Storage Tank ─────────────────────────────────────────────────────
# Every reactor discharges into this tank, which feeds downstream at a fixed
# S_OUT_RATE (900,000 kg over a 600 h episode). It starts 87.5% full and runs
# dry by ~230 h without inflow, so the first unload must come before then
# (storage underflow), and it holds only S_MAX, so the plant can ship at most
# S_MAX - S_INITIAL + 900,000 = 950,000 kg in an episode without overflowing
# it -- about 4.5 full batches against the ~7 three free-running vessels would
# make. The tank is therefore the plant's bottleneck and couples the reactors:
# their unloads must be staggered, and a finished batch sometimes held.
S_MAX = 400000.0
S_MIN = 5000.0
S_OUT_RATE = 1500.0 / 3600.0
S_INITIAL = 350000.0

# ─── Tracked Species Definition ───────────────────────────────────────────────
SPECIES_NAMES = ['CS', 'XS', 'LS', 'C', 'G', 'X', 'F', 'E', 'AC', 'Cell',
                 'Eth', 'CO2', 'ACT', 'HMF', 'Base']
SPECIES_IDX = {name: i for i, name in enumerate(SPECIES_NAMES)}
N_SPECIES = len(SPECIES_NAMES)

# Per-species normalization ceilings (g/L), chosen so trace species are not
# crushed to zero in the observation.
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

I_ETH = SPECIES_IDX['Eth']


# =============================================================================
# OBSERVATION LAYOUT
# =============================================================================
# One block per reactor, then the plant-wide channels, then each reactor's
# off-gas CO2 rate. The CO2 rates come last so the action projection network
# can read a PREFIX of the observation (SHIELD_OBS_DIM channels): the rate is
# an activity measurement, not a constrained quantity, and must never enter
# the shield.
#
#   per-reactor block (REACTOR_OBS_DIM = 24), reactor i at i * 24:
#     0-14  species concentrations / C_MAX_LIMITS     (zeroed when masked)
#     15    M / Mmax
#     16-20 stage one-hot (5)
#     21    stage credit remaining / base_credits[stage]
#     22    lot-in-progress mass / PRODUCT_MASS_SCALE
#     23    lot-in-progress mass-weighted mean titer / TITER_SCALE
#   plant-wide block (GLOBAL_OBS_DIM = 3), at N_REACTORS * 24:
#     0     central storage mass / S_MAX
#     1     LF feedstock remaining / INITIAL_LF
#     2     C5 feedstock remaining / INITIAL_C5
#   CO2 tail (N_REACTORS), one per reactor:
#     i     reactor i's off-gas CO2 evolution rate / CER_SCALE (zeroed when
#           not observed)
R_MASS = 15
R_STAGE = 16
R_CREDIT = 21
R_LOT_MASS = 22
R_LOT_TITER = 23
REACTOR_OBS_DIM = 24
G_STORAGE = 0
G_LF = 1
G_C5 = 2
GLOBAL_OBS_DIM = 3


# =============================================================================
# PRODUCT QUALITY SPECIFICATION
# =============================================================================
# The lean study's two titer specs, applied to every reactor:
#
#   HARVEST TITER FLOOR (violation id 3, per inner step while a reactor drains)
#     Outlet ethanol must be >= MIN_HARVEST_TITER.
#
#   PRODUCT PURITY SPEC (violation id 4, per lot)
#     Each batch's discharge is one lot, released when that reactor's batch
#     completes if its mass-weighted mean titer is >= MIN_PRODUCT_TITER. Only
#     released lots count as on-spec product; a rejected lot is worth nothing.
#     Lots are per reactor batch, not the storage tank's contents, so a single
#     diluted batch does not void the good batches the other vessels shipped
#     (the reason the lean study moved from one product tank to lots). The tank
#     is a physical buffer only: its constraints are inventory constraints.
#
# Both values shape the TRAINING reward. Changing either one invalidates every
# trained checkpoint, and -- when concentrations are observable -- the APN.
MIN_HARVEST_TITER = 65.0    # g/L in the outlet stream while draining
MIN_PRODUCT_TITER = 75.0    # g/L, mass-weighted mean over a lot

# Normalises a reactor's lot mass. One batch's lot is at most a vessel
# (~0.55); without the stage mask a batch may never complete, and its lot can
# grow past it.
PRODUCT_MASS_SCALE = 400000.0

# Shares the ethanol C_MAX so the lot titer and the in-reactor ethanol channel
# are on one scale.
TITER_SCALE = float(C_MAX_LIMITS[I_ETH])


# ─── Feed Flow Properties ─────────────────────────────────────────────────────
def get_C_C5():
    """Build the baseline C5 feed composition vector.

    Returns:
        NumPy array of length 15 containing species concentrations for C5 feed.
    """
    arr = np.zeros(15)
    vals = {'CS': 1.2, 'XS': 0.5, 'LS': 0.7, 'G': 10.0, 'X': 29.7, 'F': 0.5,
            'AC': 4.1, 'ACT': 0.1, 'HMF': 0.3}
    for k, v in vals.items():
        arr[SPECIES_IDX[k]] = v
    return arr


def get_C_LF():
    """Build the baseline LF feed composition vector.

    Returns:
        NumPy array of length 15 containing species concentrations for LF feed.
    """
    arr = np.zeros(15)
    vals = {'CS': 50.0, 'XS': 1.0, 'LS': 78.0, 'G': 98.0, 'X': 59.0, 'F': 0.2,
            'E': 4.9, 'AC': 16.0, 'ACT': 0.1, 'HMF': 0.1, 'Base': 8.6}
    for k, v in vals.items():
        arr[SPECIES_IDX[k]] = v
    return arr


C_C5_BASE = get_C_C5()
C_LF_BASE = get_C_LF()

# ─── Time Credit and Flow Integration ─────────────────────────────────────────
# Stage credits in hours: [0] charge, [1] co-feed, [2] ferment, [3] unload,
# [4] clean. One nominal batch is 262 h, so a 600 h episode at unit time
# multiplier completes ~2.3 batches per reactor.
BASE_CREDITS = np.array([10.0, 60.0, 120.0, 70.0, 2.0])
TIME_STEP_DURATION = 3600.0
MAX_FLOW_OUT_REAC = 5000.0 / 3600.0


# =============================================================================
# INSTRUMENTATION / SENSOR MODEL
# =============================================================================
# The lean study's instruments, on every reactor:
#
#   ONLINE (continuous, 0.5% error)    reactor masses, lot masses, storage
#                                      mass, feedstock tanks
#   AT-LINE (3% RSD, no delay)         every dissolved species and each lot's
#                                      mean titer (HPLC assay)
#   OFF-GAS (3% error)                 each reactor's CO2 evolution rate
#   NOT MEASURED                       delivered feed strength, each pump's
#                                      gain, each vessel's leak rate -- the
#                                      latent theta
#   EXACT                              recipe stage and credit
#
# Constraints, reward and every reported metric are computed from TRUE state.
# baseline and saferl TRAIN on exact observation (ArmSpec.train_sensor_noise =
# False); meta trains with the noise; every arm is EVALUATED with it.
SENSOR_MODEL_ENABLED = True

# Grab-sample period and draw-to-result lag, in inner steps (1 h): every
# channel is freshly assayed each hour and reported at once (the lean study's
# setting since 2026-09-28).
ANALYZER_SAMPLE_PERIOD_H = 1
ANALYZER_REPORT_LAG_H = 0

# Analyzer error in normalized units: relative SD plus an absolute floor so a
# near-zero reading still carries error.
ANALYZER_NOISE_RSD = 0.03
ANALYZER_NOISE_FLOOR = 0.002

ONLINE_NOISE_RSD = 0.005

# Base seed for the per-episode sensor RNG, separate from any shared RNG so
# deterministic evaluation passes stay reproducible.
SENSOR_SEED = 20261016

# In-situ assays of a lignocellulosic slurry are not available in practice.
# When True every reactor's species channels are zeroed, leaving the gauges,
# the lot titers and the CO2 rates as the only composition signals. This also
# reduces the APN to the inventory constraints (see apn.data.safety_margins).
MASK_INTERNAL_CONCENTRATIONS = True

# ─── Off-gas CO2 evolution rate (online, the observation's tail) ──────────────
# The RATE, not the dissolved CO2 level (which env.core forms by the same term
# as ethanol, so it would be a live titer meter). It reveals titer only once
# integrated over time, i.e. to an agent that keeps a history. False zeroes
# the channels without changing the observation width.
OBSERVE_CO2_RATE = True
CER_SCALE = 20.0
CER_NOISE_RSD = 0.03


# =============================================================================
# UNCERTAINTY / DOMAIN-RANDOMIZATION TASK DISTRIBUTION
# =============================================================================
# A task fixes one realization of the latent theta, sampled once per episode
# and never observed by the agent. NOT randomized: the chemistry, held at the
# PPO values on every task and in every reactor (see MetaRL_Lean/env_config
# for why: scope, and the pH bumps' dead tasks).
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
# the meta arm's task-aware critics. kinetic_activity is 1.0 and contam_rate 0
# in the prior; they move only when a biochemical fault or a contamination
# fires in that reactor. downstream_rate is the storage tank's outflow as a
# multiple of S_OUT_RATE, turnaround_mult the turnaround time and
# storage_capacity the tank's usable volume as multiples of nominal, all 1.0
# in the prior; the fault of the same name moves each.
TASK_CONTEXT_NAMES = (["feed_solids", "feed_gx"]
                      + _per_reactor("feed_gain")
                      + _per_reactor("leak_frac")
                      + _per_reactor("kinetic_activity")
                      + _per_reactor("contam_rate")
                      + ["downstream_rate", "turnaround_mult",
                         "storage_capacity"])
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

# Standardization for the auxiliary regression target (the lean study's
# per-quantity centers and scales, repeated per reactor).
TASK_CONTEXT_CENTER = np.array(
    [1.00, 1.00] + [1.00] * N_REACTORS + [0.00075] * N_REACTORS
    + [1.00] * N_REACTORS + [0.0] * N_REACTORS + [1.00, 1.00, 1.00],
    dtype=np.float64)
TASK_CONTEXT_SCALE = np.array(
    [0.12, 0.12] + [0.09] * N_REACTORS + [0.00045] * N_REACTORS
    + [0.20] * N_REACTORS + [0.001] * N_REACTORS + [0.20, 0.50, 0.10],
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


def build_param_vector(eta, ph_g, ph_x, max_ads, k_ads, tox):
    """Assemble the length-``PARAM_DIM`` kinetic vector for the rate kernel.

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


# The one kinetic parameter vector every task and every reactor uses.
NOMINAL_ETA, NOMINAL_PH_G, NOMINAL_PH_X = kinetics_from_pH(NOMINAL_PH)
NOMINAL_PARAMS = build_param_vector(
    NOMINAL_ETA, NOMINAL_PH_G, NOMINAL_PH_X,
    NOMINAL_MAX_ADS, NOMINAL_K_ADS, np.ones(3))
NOMINAL_PARAMS.flags.writeable = False


def _context(solids, gx, gains, leaks):
    """A prior task's descriptor: no fault has acted yet."""
    return np.concatenate([[solids, gx], gains, leaks,
                           np.ones(N_REACTORS), np.zeros(N_REACTORS),
                           [1.0, 1.0, 1.0]])


def nominal_task():
    """Return the nominal task theta_0 (no randomization).

    The sole training task for the single-regime arms (``baseline``,
    ``saferl``). Its flow is ideal -- every pump gain 1.0, no leak, unscaled
    feed -- so those arms never see a miscalibrated pump, a leaking vessel or
    off-spec feedstock until evaluation.

    Returns:
        Task dict with keys: params, C_C5, C_LF, flow, context, meta.
    """
    gains = np.ones(N_REACTORS)
    leaks = np.zeros(N_REACTORS)
    return {
        "params": np.array(NOMINAL_PARAMS, dtype=np.float64),
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

    Only flow quantities vary; ``params`` is always ``NOMINAL_PARAMS``. The
    feed concentration is one draw for the plant; each reactor's pump gain
    and leak are independent draws.

    Args:
        rng: Optional ``numpy.random.Generator``; a fresh one when None.
        ood: Draw from priors widened by ``OOD_WIDEN_FACTOR``.

    Returns:
        Task dict with keys:
            - params: length-PARAM_DIM kinetic vector (always NOMINAL_PARAMS).
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
    return {"params": np.array(NOMINAL_PARAMS, dtype=np.float64),
            "C_C5": C_C5, "C_LF": C_LF,
            "flow": {"feed_gain": gains, "leak_frac": leaks},
            "context": _context(solids, gx, gains, leaks), "meta": meta}
