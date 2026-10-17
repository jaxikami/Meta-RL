"""Settings of the three-arm study.

Task pool and collection, agent and network sizes, meta-training faults,
evaluation scenarios, checkpoint selection, warm start and output paths.
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.layout import ETHANOL_CHANNELS, ACTION_DIM

MODE = "all"
SEED = int(os.environ.get("PLANT_SEED", 0))
# Train and evaluate the unconstrained baseline arm (off by default, as in
# the lean study; SAFETY is then not reported).
INCLUDE_BASELINE = False

# ─── Task pool and collection ─────────────────────────────────────────────────
# Multitask arms draw a FRESH prior task for every training episode; the
# N_TRAIN_TASKS buckets only shard the replay.
N_TRAIN_TASKS = 12
META_ITERATIONS = 4000
COLLECT_EPISODES_PER_ITER = 3
UPDATES_PER_ITER = 25
META_BATCH = 4
BATCH_SIZE = 512
MIN_BUFFER = 256
# Training window of the recurrent encoder: at least the longest episode, so
# every context reaches back to its episode start. An episode is 120 control
# steps plus one short step per batch boundary of ANY reactor: at most four
# batches per reactor even at the fastest clock (4 x 131 h < 600 h), so 132.
# the agent refuses a context that fills it; smoke_test checks the bound.
CONTEXT_SIZE = 160
WARMUP_EPISODES_PER_TASK = 3
# Evaluation tasks per scenario. The pool is drawn in order from one seed, so
# a larger value extends it. Each plant episode costs ~3x a lean one.
EVAL_EPISODES = 100

# ─── Held-out deterministic evaluation (checkpoint selection + early stop) ────
HELDOUT_EVAL_EVERY = 10
HELDOUT_EVAL_TASKS = 8
HELDOUT_TASK_SEED = 777
# The most overflow-prone prior draws (``corner_risk``), appended to the
# random held-out tasks.
HELDOUT_CORNER_TASKS = 2
HELDOUT_CORNER_SEED = 31337
HELDOUT_CORNER_CANDIDATES = 512
HELDOUT_WORKERS = int(os.environ.get(
    "PLANT_HELDOUT_WORKERS",
    min(HELDOUT_EVAL_TASKS + HELDOUT_CORNER_TASKS, (os.cpu_count() or 2) // 2)))

EVAL_Z_MODE = "mean"
HELDOUT_Z_MODE = "mean"

NORMALIZE_REWARDS = True
REWARD_SCALE = 1.0
RESET_LAMBDAS_PER_EPISODE = True
APN_PENALTY_WEIGHT = 0.01

# ─── Agent: posterior-collapse countermeasures (the lean study's) ────────────
# AUX_MODE = "task" regresses the TRUE task descriptor from z: privileged,
# simulation-only information; results under it must be reported as such.
USE_FILM = True
AUX_MODE = "task"
AUX_WEIGHT = 10.0
AUX_SAMPLE_FRAC = 0.1
# Privileged-state channels the aux head also regresses from z: each
# reactor's broth ethanol titer (C / 150 g/L; ~0.35 mid-batch, ~0.55 at spec).
META_AUX_STATE_TARGETS = tuple((c, 0.35, 0.2) for c in ETHANOL_CHANNELS)

# ─── Networks (SHARED by every arm) ──────────────────────────────────────────
# Token networks (agent.networks): each reactor is a token of its 24 process
# channels, its CO2 rate and the 3 plant-wide channels, mixed by
# MIXER_LAYERS pre-LN self-attention blocks. The actor runs one FiLM residual
# trunk shared by the three tokens; each critic flattens its mixed tokens
# into a wide FiLM residual trunk. Compared with the lean study (actor
# 3 x 256 ~151k, critics 2 x 3 x 512 ~1.1M, GRU 128 ~108k): actor ~1.3M,
# critics ~7.9M (9.2M with meta's privileged conditioning), GRU 256 ~0.44M.
# Changing any of these invalidates every checkpoint.
NET_ACTOR_HIDDEN = 256          # actor token width and shared-trunk width
NET_CRITIC_HIDDEN = 1024        # each critic's flattened trunk width
NET_CRITIC_TOKEN_HIDDEN = 256   # each critic's token width
NET_MIXER_LAYERS = 2
NET_MIXER_HEADS = 4
NET_DEPTH = 3
NET_LAYER_NORM = True
NET_RESIDUAL = True
NET_OBS_NORM = True
NET_ENCODER_LAYER_NORM = True
NET_ENCODER_HIDDEN = 256
NET_WEIGHT_DECAY = 1e-2

# ─── Disturbances in the META-TRAINING task distribution ─────────────────────
META_DISTURB_PROB = 0.45
META_DISTURB_KINDS = (
    "leak", "feed_gain", "feed_conc", "disturb_kinetics_decay",
    "toxin_spike", "cell_viability", "contamination", "downstream_rate",
)

# A per-reactor fault (every kind but feed_conc and downstream_rate) acts on
# one uniformly drawn reactor, or with this probability on all three at once.
META_PLANTWIDE_FRAC = 0.2

META_DISTURB_ONSET_FRAC = (0.25, 0.75)
# The lean study's calibrated ranges (its eval magnitudes in the comments),
# plus the storage tank's downstream_rate. They were calibrated against the
# lean saferl; recalibrate on the plant's saferl before trusting a row.
META_DISTURB_MAG_RANGES = {
    "leak": (0.004, 0.016),                  # eval: 0.010
    "feed_gain": (0.60, 1.45),               # eval: 1.30
    "feed_conc": (0.30, 1.40),               # eval: recurrent dilution only
    "disturb_kinetics_decay": (0.15, 0.70),  # eval: 0.20
    "toxin_spike": (2.0, 8.0),               # eval: 6.50
    "cell_viability": (0.30, 0.70),          # eval: 0.50
    "contamination": (0.0005, 0.004),        # not evaluated (lean: plant-bound)
    "downstream_rate": (0.40, 1.60),         # eval: 0.50  (NEW, uncalibrated)
}
META_DISTURB_STRATA = 5
META_DISTURB_TAIL_WEIGHT = 2.0
META_FAULT_PATTERNS = {
    "single": 0.55,
    "drift": 0.15,
    "compound": 0.15,
    "lot_change": 0.15,
}
META_DRIFT_TREADS = 6
META_COMPOUND_PAIRS = (("leak", "feed_conc"), ("feed_gain", "feed_conc"),
                       ("downstream_rate", "feed_gain"))
META_COMPOUND_PAIR_FRAC = 0.5

# ─── Evaluation disturbance scenarios ────────────────────────────────────────
# One instantaneous fault each, at mid-episode. A per-reactor fault ROTATES
# over the reactors with the episode index (episode k faults reactor k % 3),
# so every reactor is faulted on a third of the (paired) tasks.
EVAL_DISTURBANCES = {
    "disturb_leak": {"kind": "leak", "magnitude": 0.010, "reactor": "rotate"},
    "disturb_feed_gain": {"kind": "feed_gain", "magnitude": 1.30,
                          "reactor": "rotate"},
    "disturb_kinetics_decay": {"kind": "disturb_kinetics_decay",
                               "magnitude": 0.20, "reactor": "rotate"},
    "disturb_toxin_spike": {"kind": "toxin_spike", "magnitude": 6.50,
                            "reactor": "rotate"},
    "disturb_cell_viability": {"kind": "cell_viability", "magnitude": 0.50,
                               "reactor": "rotate"},
    # Downstream distillation at half rate: the tank fills ~0.75 t/h slower.
    "disturb_downstream_rate": {"kind": "downstream_rate", "magnitude": 0.50,
                                "reactor": None},
}
# Realistic fault patterns: (kind, onset_inner_step, magnitude[, reactor]).
EVAL_FAULT_SCENARIOS = {
    "recurrent_dilution": [("feed_conc", 150, 0.55), ("feed_conc", 280, 1.00),
                           ("feed_conc", 420, 0.50)],
}
# Scenario -> (disturbance_probe key, label, actuator index within a reactor):
# the quantity the fault acts on and the actuator that answers it. For a
# per-reactor fault both are the faulted reactor's; for the storage fault the
# probe is the tank and the actuator the mean discharge over the reactors.
DISTURB_PROBES = {
    "disturb_leak": ("mass", "faulted reactor's mass (kg)", 1),
    "disturb_feed_gain": ("feed_gain", "faulted pump's gain", 1),
    "disturb_kinetics_decay": ("kinetic_activity",
                               "faulted reactor's kinetic activity", 0),
    "disturb_toxin_spike": ("inhibitors", "faulted reactor's inhibitors (g/L)", 0),
    "disturb_cell_viability": ("cells", "faulted reactor's cells (g/L)", 0),
    "disturb_downstream_rate": ("storage", "storage mass (kg)", 3),
}

# ─── Constraint identities (per-constraint violation accounting) ─────────────
CONSTRAINT_NAMES = {
    1: "overflow",
    2: "underflow",
    3: "harvest_titer",
    4: "product_purity",
    5: "instability",
    6: "mass_balance",
    7: "storage_overflow",
    8: "storage_underflow",
}
DEPTH_CONSTRAINTS = (1, 2, 3, 4, 7, 8)

# ─── Latent-informativeness gate ─────────────────────────────────────────────
Z_INFORM_MIN_SNR = 1.0
Z_INFORM_MIN_ACTION_DELTA = 1e-3
Z_INFORM_MIN_PRIOR_SENS = 1e-2
Z_INFORM_MIN_METRIC_SPREAD = 0.05
Z_INFORM_EPISODES = 12
Z_INFORM_PROBE_STATES = 24
Z_INFORM_DISTURBANCES = (
    ("feed_conc", 0.40, None),
    ("feed_gain", 1.30, "rotate"),
    ("downstream_rate", 0.50, None),
)

# ─── Output paths ─────────────────────────────────────────────────────────────
RUN_TAG = os.environ.get("PLANT_RUN_TAG", "")
POLICY_DIR = os.path.join("policy", RUN_TAG) if RUN_TAG else "policy"
LOG_DIR = (os.path.join("results", "log", RUN_TAG) if RUN_TAG
           else os.path.join("results", "log"))
PLOT_DIR = (os.path.join("results", "plot", RUN_TAG) if RUN_TAG
            else os.path.join("results", "plot"))

TITER_PANEL_FLOOR_PAD = 3.0

# ─── Early stopping (units are HELD-OUT CHECKS) ──────────────────────────────
EARLY_STOP_ENABLED = True
EARLY_STOP_MIN_ITERS = 20
EARLY_STOP_PATIENCE = 50
EARLY_STOP_MIN_REL_DELTA = 1e-3
EARLY_STOP_REL_FLOOR = 1.0

# ─── Checkpoint selection ────────────────────────────────────────────────────
# Feasible: mean held-out violation EVENTS <= SELECT_MAX_VIOL and no single
# episode above SELECT_MAX_WORST_VIOL. The lean study's 2 / 8 for one
# reactor, scaled for three reactors and a tank (each event is one vessel or
# the tank in one hour). A first guess: calibrate on a trained saferl.
SELECT_MAX_VIOL = 3.0
SELECT_MAX_WORST_VIOL = 12.0
SELECT_VIOL_WEIGHT = 1.0
# Panel branches that must pass the absolute gate on their own: the faults
# that move an ONLINE measurement (reactor masses, the storage mass), so an
# adaptive arm can avoid them.
GATE_FAULT_KINDS = ("leak", "feed_gain", "downstream_rate")

# ─── Disturbed held-out panel (selection + early stop) ───────────────────────
# Magnitudes bracket the evaluation's without equalling them. Per-reactor
# faults rotate over the held-out tasks, as in evaluation.
HELDOUT_DISTURBANCES = (
    ("leak", 0.008, "rotate"),
    ("feed_gain", 1.40, "rotate"),
    ("feed_conc", 0.55, None),
    ("disturb_kinetics_decay", 0.17, "rotate"),
    ("toxin_spike", 5.0, "rotate"),
    ("cell_viability", 0.55, "rotate"),
    ("contamination", 0.0012, "rotate"),
    ("downstream_rate", 0.60, None),
)
HELDOUT_DISTURB_WEIGHT = 0.5
EARLY_STOP_DIVERGE_FACTOR = 5.0
EARLY_STOP_DIVERGE_VIOL_FLOOR = 15.0
EARLY_STOP_DIVERGE_REL_DROP = 0.5
EARLY_STOP_DIVERGE_PATIENCE = 12
EARLY_STOP_DIVERGE_MIN_CHECKS = 15

# ─── SAC / encoder schedule ───────────────────────────────────────────────────
ALPHA_MIN = 0.15
META_KL_LAMBDA = 0.05
KL_ANNEAL_ITERS = 150
META_ENCODER_LR = 5e-4

# ─── Warm start, anchor and adaptation gate (the lean study's) ───────────────
META_ACTOR_WARMUP_ITERS = 300
HELDOUT_WARMUP_EVAL_EVERY = 50
META_ANCHOR_WEIGHT = 5.0
META_ANCHOR_FAULT_FRAC = 0.1
# Per action dimension (12: [clock, LF, C5, out] x 3 reactors), the share of
# the post-fault gap the anchor counts.
META_ANCHOR_FAULT_DIMS = (1.0,) * ACTION_DIM
GATE_LOSS_WEIGHT = 1.0
GATE_SHARPNESS = 4.0
GATE_INIT_BIAS = -4.0
META_ANCHOR_UP = 2.0
META_ANCHOR_DOWN = 0.85
META_ANCHOR_MIN = 0.5
META_ANCHOR_MAX = 50.0
META_EXPLORE_NOISE_BETA = 1.0

# ─── No-regression selection (warm-started arms) ─────────────────────────────
NO_REGRESS_ETH_REL = 0.04
NO_REGRESS_VIOL_ABS = 0.4
NO_REGRESS_VIOL_REL = 0.15
NO_REGRESS_TAIL_REL = 0.5
NO_REGRESS_ETH_REL_CLEAN = 0.01
NO_REGRESS_CLEAN_VIOL = 0.1

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}
PRACTICAL_MIN_REL_EFFECT = 0.01
