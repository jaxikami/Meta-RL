"""Settings of the two-arm study (saferl vs meta).

Task pool and collection, agent and network sizes, meta-training faults,
evaluation scenarios, checkpoint selection, warm start and output paths.
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.layout import LOT_TITER_PRIV_CHANNELS, ACTION_DIM
from metarl.env.disturbances import (EVAL_DISTURBANCES, EVAL_ONSET,
                                     EVAL_SPIKE_H, EVAL_GRADUAL_H,
                                     EVAL_RECURRENT_WINDOWS, FAULT_PATTERNS)

MODE = "all"
SEED = int(os.environ.get("PLANT_SEED", 0))

# ─── Task pool and collection ─────────────────────────────────────────────────
# Multitask arms draw a FRESH prior task for every training episode; the
# N_TRAIN_TASKS buckets only shard the replay.
N_TRAIN_TASKS = 12
META_ITERATIONS = int(os.environ.get("PLANT_META_ITERS", 4000))
COLLECT_EPISODES_PER_ITER = 3
# The meta arm's episodes per iteration and how often its encoder steps.
# At 3 episodes and an encoder step on every update, each stored transition
# fed ~140 encoder steps and the encoder memorised the replay: on its own
# batches it named the fault kind almost perfectly (class-balanced loss 0.08,
# gate open on 88% of faulted steps), on fresh episodes of the same
# distribution it detected little (gate 0.01-0.5 by kind). 9 episodes run in
# parallel in the worker pool during the updates, and an encoder step on one
# update in 3 brings that to ~15.
META_COLLECT_EPISODES_PER_ITER = 9
META_ENCODER_UPDATE_EVERY = 3
UPDATES_PER_ITER = 25
META_BATCH = 4
BATCH_SIZE = 512
MIN_BUFFER = 256
# Training window of the recurrent encoder: longer than any episode, so every
# context reaches back to its episode start. An episode is 120 control steps
# (600 h at 5 h; the environment does not end a step early at a batch
# boundary) or fewer when the integrator fails; the agent refuses a context
# that fills the window, and smoke_test checks the bound.
CONTEXT_SIZE = 128
WARMUP_EPISODES_PER_TASK = 3
# Evaluation tasks per scenario. The pool is drawn in order from one seed, so
# a larger value extends it. Each plant episode costs ~3x a lean one.
EVAL_EPISODES = int(os.environ.get("PLANT_EVAL_EPISODES", 100))
# Worker processes for evaluation: one job per (arm, task), each running every
# scenario of its task from one undisturbed prefix. 0 or 1 runs in-process.
EVAL_WORKERS = int(os.environ.get("PLANT_EVAL_WORKERS", os.cpu_count() or 1))

# ─── Held-out deterministic evaluation (checkpoint selection + early stop) ────
HELDOUT_EVAL_EVERY = 10
HELDOUT_EVAL_TASKS = 8
HELDOUT_TASK_SEED = 777
# The most overflow-prone prior draws (``corner_risk``), appended to the
# random held-out tasks.
HELDOUT_CORNER_TASKS = 2
HELDOUT_CORNER_SEED = 31337
HELDOUT_CORNER_CANDIDATES = 512
# One worker per held-out task: the pool gets one job per task, so fewer
# workers run the check in several waves (8 workers for 10 tasks: two) and
# more sit idle. The main process only waits during a check, so every logical
# CPU is available to the workers.
HELDOUT_WORKERS = int(os.environ.get(
    "PLANT_HELDOUT_WORKERS",
    min(HELDOUT_EVAL_TASKS + HELDOUT_CORNER_TASKS, os.cpu_count() or 2)))

# Training episodes run in the held-out worker pool, concurrently with the
# GPU updates (each iteration's episodes use the weights from before its
# updates, and enter the replay after them). PLANT_COLLECT_WORKERS=0 collects
# in-process. Profiled: collection was ~25% of an iteration, the updates ~65%.
COLLECT_IN_WORKERS = os.environ.get("PLANT_COLLECT_WORKERS", "1") != "0"

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
# reactor's lot mean titer (/ 150 g/L; 0.5 at the purity spec), the quality
# state the observation does not carry.
META_AUX_STATE_TARGETS = tuple((c, 0.35, 0.2) for c in LOT_TITER_PRIV_CHANNELS)

# ─── Networks (SHARED by every arm) ──────────────────────────────────────────
# Token networks (agent.networks): each reactor is a token of its observation
# row (25 process and capacity channels, 3 plant-wide), folded into a GRU per
# reactor with cross-reactor attention over its outputs (NET_HISTORY_*). The
# actor runs one FiLM residual trunk shared by the three reactors; each
# critic mixes the reactors' (history, action) tokens (NET_MIXER_*) and
# flattens them into a wide FiLM residual trunk. Changing any of these
# invalidates every checkpoint.
NET_ACTOR_HIDDEN = 256          # actor token width and shared-trunk width
NET_CRITIC_HIDDEN = 1024        # each critic's flattened trunk width
NET_CRITIC_TOKEN_HIDDEN = 256   # each critic's token width
NET_MIXER_LAYERS = 2
NET_MIXER_HEADS = 4
# The policy's per-reactor history (the Fermentation-PPO policy's encoder): a
# 2-layer GRU of width 256 shared by the reactors (each keeps its own state),
# cross-reactor attention over its outputs in a 64-wide Q/K/V space with 4
# heads, and a 128-wide skip embedding of the current reading. The actor and
# both critics read it; the critics' mixer (NET_MIXER_*) then mixes those
# features with each reactor's action.
NET_HISTORY_HIDDEN = 256
NET_HISTORY_LAYERS = 2
NET_HISTORY_ATTN_DIM = 64
NET_HISTORY_ATTN_HEADS = 4
NET_HISTORY_SKIP = 128
NET_DEPTH = 3
NET_LAYER_NORM = True
NET_RESIDUAL = True
NET_OBS_NORM = True
NET_ENCODER_LAYER_NORM = True
# Meta's inference module is kept small so the arms match in size: with every
# arm's critics on the same conditioning (z, task, privileged state) and the
# same latent width, the context encoder and the aux / fault-kind heads are
# all meta adds (~2% of its parameters; see README "Networks"). 128 is the
# lean study's encoder width.
NET_ENCODER_HIDDEN = 128
NET_HEAD_HIDDEN = 128
NET_WEIGHT_DECAY = 1e-2

# ─── Disturbances in the META-TRAINING task distribution ─────────────────────
# Share of training episodes with a fault. At 0.45, with ten kinds and five
# patterns, each kind's faulted steps were ~2% of the replay and the encoder
# never learned the biochemical and storage kinds (its decoded kinetic
# activity read 0.95 against a true 0.2). The gate keeps saferl acting on
# undisturbed steps, and every faulted episode is undisturbed up to its onset,
# so ~60% of the steps stay undisturbed at 0.7.
META_DISTURB_PROB = 0.7
META_DISTURB_KINDS = (
    "leak", "feed_gain", "feed_conc", "disturb_kinetics_decay",
    "toxin_spike", "cell_viability", "contamination", "downstream_rate",
    "turnaround_delay", "storage_capacity", "cooling_capacity",
)

# A per-reactor fault (every kind but the plant-wide ones) acts on
# one uniformly drawn reactor, or with this probability on all three at once.
META_PLANTWIDE_FRAC = 0.2

META_DISTURB_ONSET_FRAC = (0.25, 0.75)
# The lean study's calibrated ranges (its eval magnitudes in the comments),
# plus the storage tank's downstream_rate. They were calibrated against the
# lean saferl; recalibrate on the plant's saferl before trusting a row.
META_DISTURB_MAG_RANGES = {
    "leak": (0.004, 0.016),                  # eval: 0.010
    "feed_gain": (0.60, 1.45),               # eval: 1.30
    "feed_conc": (0.30, 1.40),               # eval: 0.55 ("dilution")
    "disturb_kinetics_decay": (0.15, 0.70),  # eval: 0.20
    "toxin_spike": (2.0, 8.0),               # eval: 6.50
    "cell_viability": (0.30, 0.70),          # eval: 0.50
    "contamination": (0.0005, 0.004),        # eval: 0.002 (Fermentation-PPO's)
    "downstream_rate": (0.40, 1.60),         # eval: 0.50  (NEW, uncalibrated)
    # Calibrated on the previous plant's saferl: a 0.90 tank (360 t) sat
    # above any level it held at mid-episode, so its overflows are
    # avoidable, yet it overflowed ~140 h.
    "turnaround_delay": (1.5, 5.0),          # eval: 4.0
    "storage_capacity": (0.75, 0.97),        # eval: 0.90
    # The heat-load cap (Fermentation-PPO's constraint 9), uncalibrated.
    "cooling_capacity": (0.65, 0.95),        # eval: 0.80
}
META_DISTURB_STRATA = 5
META_DISTURB_TAIL_WEIGHT = 2.0
# Every kind's four temporal patterns (FAULT_PATTERNS, built by
# tasks.pattern_schedule), plus training-only "compound": two kinds stepping
# at one onset.
META_FAULT_PATTERNS = {
    "step": 0.35,
    "spike": 0.15,
    "gradual": 0.15,
    "recurrent": 0.20,
    "compound": 0.15,
}
# Training timings, in inner steps (hours), drawn per episode: a spike's
# length, a gradual ramp's length (reached in META_DRIFT_TREADS treads), and
# a recurrent fault's on / off windows (repeated to the end of the episode,
# at most META_RECURRENT_MAX times).
META_SPIKE_H = (12, 48)
META_GRADUAL_FRAC = 0.42
META_DRIFT_TREADS = 6
META_RECURRENT_ON_H = (40, 100)
META_RECURRENT_OFF_H = (40, 100)
META_RECURRENT_MAX = 3
META_COMPOUND_PAIRS = (("leak", "feed_conc"), ("feed_gain", "feed_conc"),
                       ("downstream_rate", "feed_gain"))
META_COMPOUND_PAIR_FRAC = 0.5

# ─── Evaluation disturbance scenarios ────────────────────────────────────────
# Fermentation-PPO's (metarl.env.disturbances): every evaluated kind under
# every FAULT_PATTERNS entry, scenario "<label>_<pattern>" (e.g.
# "dilution_recurrent"), at its fixed timings (EVAL_ONSET for step, spike and
# gradual, EVAL_SPIKE_H, EVAL_GRADUAL_H, EVAL_RECURRENT_WINDOWS). A
# per-reactor fault ROTATES over the reactors with the episode index (episode
# k faults reactor k % 3), so every reactor is faulted on a third of the
# (paired) tasks. Imported above: EVAL_DISTURBANCES, EVAL_ONSET, EVAL_SPIKE_H,
# EVAL_GRADUAL_H, EVAL_RECURRENT_WINDOWS.
# Kind label -> (disturbance_probe key, label, actuator index within a
# reactor): the quantity the fault acts on and the actuator that answers it,
# traced on the label's "step" scenario. For a per-reactor fault both are the
# faulted reactor's; for a plant-wide fault the actuator is the mean over the
# reactors.
DISTURB_PROBES = {
    "leak": ("mass", "faulted reactor's mass (kg)", 1),
    "feed_gain": ("feed_gain", "faulted pump's gain", 1),
    "dilution": ("feed_conc", "feed concentration (x task)", 1),
    "kinetics_decay": ("kinetic_activity",
                       "faulted reactor's kinetic activity", 0),
    "toxin": ("inhibitors", "faulted reactor's inhibitors (g/L)", 0),
    "cell_viability": ("cells", "faulted reactor's cells (g/L)", 0),
    "contamination": ("ethanol", "faulted reactor's ethanol (g/L)", 3),
    "downstream_rate": ("storage", "storage mass (kg)", 3),
    "turnaround_delay": ("turnaround", "turnaround time (x nominal)", 0),
    "storage_capacity": ("storage", "storage mass (kg)", 3),
    "cooling_capacity": ("heat_load", "plant heat load (kg CO2/h)", 1),
}

# ─── Constraint identities (per-constraint violation accounting) ─────────────
# The environment's ids (metarl.env.safe).
CONSTRAINT_NAMES = {
    1: "overflow",
    2: "underflow",
    3: "storage_overflow",
    4: "storage_underflow",
    5: "instability",
    6: "mass_balance",
    7: "harvest_titer",
    8: "product_purity",
    9: "heat_load",
}
DEPTH_CONSTRAINTS = (1, 2, 3, 4, 7, 8, 9)

# ─── Latent-informativeness gate ─────────────────────────────────────────────
Z_INFORM_MIN_SNR = 1.0
Z_INFORM_MIN_ACTION_DELTA = 1e-3
Z_INFORM_MIN_PRIOR_SENS = 1e-2
Z_INFORM_MIN_METRIC_SPREAD = 0.05
Z_INFORM_EPISODES = 12
Z_INFORM_PROBE_STATES = 24
# Faults where meta can differ from saferl; feed_gain and downstream_rate cost
# saferl nothing, so a verdict there cannot attribute ADAPT to inference.
Z_INFORM_DISTURBANCES = (
    ("leak", 0.010, "rotate"),
    ("disturb_kinetics_decay", 0.20, "rotate"),
    ("toxin_spike", 6.50, "rotate"),
    ("feed_conc", 0.40, None),
    ("storage_capacity", 0.90, None),
    ("cooling_capacity", 0.80, None),
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
# saferl, trained from scratch, still improves after long plateaus: at 30
# checks it stopped at 6.06e7 held-out on-spec ethanol, against 6.86e7 for
# the previous saferl run at 50. PLANT_EARLY_STOP_PATIENCE overrides it.
EARLY_STOP_PATIENCE = int(os.environ.get("PLANT_EARLY_STOP_PATIENCE", 50))
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
    ("turnaround_delay", 3.5, None),
    ("storage_capacity", 0.88, None),
    ("cooling_capacity", 0.82, None),
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
# Training rollouts and the update act through the actor on every step (the
# gate fully open); held-out checks and evaluation act through the learned
# gate. Acting through the learned gate in training, the actor explored and
# learned only where the detector already fired, so where it stayed shut
# saferl acted and meta learned nothing new (forcing the gate open at onset
# changed nothing on any fault but leak). And the executed action told the
# encoder whether the gate was open (exploration noise passes only through an
# open gate), so it learned to read the gate rather than the plant: the gate
# opened on 85-91% of faulted training steps and stayed shut on the held-out
# ones. Opening it on the privileged fault flag alone leaked that flag the
# same way. With the actor acting everywhere the actions carry neither, and
# the anchor holds it on saferl's policy before a fault.
META_TRAIN_GATE_OPEN = True
# Class-balanced fault-kind classifier on z (privileged target, training
# only), giving the encoder a gradient per kind; 0 disables it.
META_FAULT_CLS_WEIGHT = 1.0
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

# ─── Hold rule fine-tune (privileged training signal; see agent.learning) ────
# Penalty on opening the outlet of a batch below the harvest spec (x margin),
# and stalls also train the gate. 0 disables. PLANT_FINETUNE_FROM resumes the
# meta arm from that checkpoint (its saferl source stays the warm start).
HOLD_PENALTY_WEIGHT = float(os.environ.get("PLANT_HOLD_WEIGHT", 0.0))
HOLD_MARGIN = 0.03
HOLD_GATE_TARGET = True
FINETUNE_FROM = os.environ.get("PLANT_FINETUNE_FROM", "")
