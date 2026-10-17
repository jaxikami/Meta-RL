"""Meta-training and evaluation entrypoint for the PEARL fermentation agent —
SINGLE RUN (one seed, one pass through train_meta -> train_baseline -> evaluate).

This script trains the uncertainty-aware PEARL meta-policy against a distribution
of simulated fermentation tasks (proposal §2.3) and evaluates it against the
non-adaptive Safe-RL baseline on in-distribution tasks, out-of-distribution (OOD)
tasks, and the three mid-episode disturbance stress scenarios (proposal §3.2).

Both agents share the identical safety stack (SERL stage mask + the environment's
Lagrangian constraint penalties); the ONLY difference is that the meta agent
infers a latent task variable z from recent transitions while the baseline runs
with z ≡ 0 and is trained on the nominal task only.  This isolates meta-learning
as the single experimental variable.

NOTE ON VARIANCE: this pipeline is high-variance run-to-run (two runs of
identical code produced baseline eval violations of 0.25 and 46.1 — the
meta-vs-baseline gap is smaller than that between-seed noise). A single run of
THIS script is fine for iterating on a fix (does the training curve look
healthy, did KL/|z| stay bounded, etc.), but is NOT sufficient evidence for a
meta-vs-baseline verdict. For that, use the sibling driver
`train_pearl_seed_sweep.py`, which calls this module's train_meta/
train_baseline/evaluate under several seeds and reports the paired-difference
distribution instead of one noisy point.

Modes (set ``MODE`` below or pass as argv[1]):
    'train_meta'     — meta-train the PEARL agent, save policy/pearl_meta.pth
    'train_baseline' — train the non-adaptive Safe-RL baseline, save
                       policy/pearl_baseline.pth
    'evaluate'       — load both checkpoints and run the comparison suite
    'eval_z_probe'   — measure all z-draw modes on a trained checkpoint
    'z_informativeness'
                     — RUN THIS BEFORE ANY SEVERITY SWEEP. Asks whether the
                       latent z carries task information the actor acts on, via
                       three probes: across-task separability of the posterior,
                       action sensitivity to a swapped latent, and the paired
                       cost of rolling out under a deliberately WRONG latent.
                       An uninformative z pins the headroom_test INFER column at
                       zero no matter how the sweep is configured, so a null
                       there is uninterpretable until this passes.
    'severity_sweep' — escalate each disturbance's magnitude until the agent(s)
                       actually fail, to find where the eval has any headroom
                       (see the SEVERITY_SWEEP_* block below). Optional second
                       argv selects the arms:
                           python train_pearl_single_run.py severity_sweep \
                               baseline,meta,meta_noz
    'headroom_test'  — the follow-up to severity_sweep: all three arms
                       (meta / meta_noz / baseline) restricted to
                       SEVERITY_CLEAN_WINDOW, the only severity region the sweep
                       found to be BOTH discriminating (the baseline actually
                       fails) and artifact-free (the mass clamp does not bind).
                       Prints the task-inference vs multi-task decomposition.
    'all'            — train_meta -> train_baseline -> evaluate
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import json
import os
import sys
from collections import deque

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch
import tqdm

# ─── Local ────────────────────────────────────────────────────────────────────
from agent_pearl import PEARLAgent, MultiTaskReplayBuffer, device
from env_config import get_main_reactor_config, sample_task, nominal_task
from env_safe import FermentationEnv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ─── Run configuration ────────────────────────────────────────────────────────
MODE = "all"
# Global RNG seed. Previously NOTHING seeded torch/np.random globally (only the
# task pool used a fixed seed), so every run drew different network inits and
# replay/collection randomness — and the pipeline turned out to be HIGH
# VARIANCE: two runs of identical code gave baseline eval violations of 0.25
# vs 46.1. Single-run meta-vs-baseline numbers are therefore not trustworthy.
# Seeding makes a run reproducible; sweep SEED over several values (via
# train_pearl_seed_sweep.py) and compare the DISTRIBUTIONS, with error bars,
# not single points. Both agents are seeded identically for a paired comparison.
SEED = 0
# TRAINING-DENSITY DIAGNOSTIC (temporary, compare against the n=40 result already
# saved under policy/pearl_eval.json). Baseline collects COLLECT_EPISODES_PER_ITER
# episodes per iteration on ONE task; meta collects the same total episodes
# spread across N_TRAIN_TASKS tasks, so at 40 tasks meta sees ~1/40th the
# per-task data density baseline does for the same gradient budget. Dropping to
# 8 tasks gives meta 5x the data per task at an unchanged iteration count and
# unchanged META_ITERATIONS/EARLY_STOP_* budget, to test whether the
# meta-vs-baseline gap in the last honest run (baseline 0 violations everywhere,
# meta 5-40) was actually an undertraining artifact rather than a limitation of
# task inference itself. Nothing else changes: eval/held-out task pools are
# sampled independently of N_TRAIN_TASKS (different seeds), so this comparison
# stays apples-to-apples with the n=40 run.
N_TRAIN_TASKS = 8           # size of the fixed meta-training task pool
META_ITERATIONS = 2000      # outer meta-training iterations
# Collection (CPU numba env) is the wall-clock bottleneck and the GPU sits idle
# during it. NOT every knob here is "free" on a bigger GPU, though — only
# BATCH_SIZE is: the per-task update loop does 3 separate backward passes
# (critic, actor, alpha) plus small parameter-loop ops, and that whole sequence
# runs META_BATCH * UPDATES_PER_ITER times per outer iteration.  These tiny
# networks are latency-bound (dominated by per-kernel-launch/Python overhead),
# not compute-bound, so raising META_BATCH/UPDATES_PER_ITER multiplies wall
# time roughly linearly regardless of VRAM — that's what made 8*30=240
# task-updates/iter ~3x slower than 4*20=80.  BATCH_SIZE, in contrast, only
# widens the matmuls inside each of those ops, which a 16GB GPU chews through
# for free at this model size.  So: big batch (real GPU win, better gradient
# estimates), modest meta_batch/updates_per_iter (keeps the *sequential*
# op count, the actual wall-clock lever, in check).
COLLECT_EPISODES_PER_ITER = 2   # rollouts collected per iteration (CPU-bound)
UPDATES_PER_ITER = 20       # SAC/encoder gradient steps per iteration
META_BATCH = 4              # tasks sampled per gradient step
BATCH_SIZE = 1024           # RL minibatch per task — big batch, cheap on 16GB
MIN_BUFFER = 256            # transitions per task before it is trainable
                            # (< BATCH_SIZE is fine; sampling is with replacement)
CONTEXT_SIZE = 64
EVAL_EPISODES = 20

# ── Held-out deterministic evaluation (checkpoint selection + early stopping) ──
# The old progress signal was a rolling mean of avg_violations over the last 50
# STOCHASTIC TRAINING episodes, and `best_viol` was its argmin over up to 2000
# iterations. That is selection on noise, and the gap it produced was large:
# meta reported best 2.4 in training and scored 24.7 at eval (seed 0); baseline
# reported 0.3 and scored 1.5 (seed 1). Worse, it hid the divergence — the
# argmin always landed at iter ~57-68, so every reported "best" checkpoint was a
# lucky early snapshot of a run that then got 30x worse.
#
# Instead: every HELDOUT_EVAL_EVERY iterations, run HELDOUT_EVAL_TASKS
# DETERMINISTIC episodes on tasks that are NOT in the training pool, and use
# that as both the checkpoint-selection metric and the early-stopping metric.
#
# Every 10 iterations on 8 tasks (was every 25 on 5). The previous run's entire
# useful signal was 6 data points, and its best checkpoint landed on the very
# first non-trivial one (iteration 25, 0 violations) — far too coarse to tell a
# real optimum from a lucky grid point, and too coarse to see where the collapse
# begins. Cost is 8/10 = 0.8 episodes per iteration on top of
# COLLECT_EPISODES_PER_ITER=2, i.e. ~+40% collection time; worth it, because
# every downstream conclusion rests on which checkpoint gets picked.
HELDOUT_EVAL_EVERY = 10
HELDOUT_EVAL_TASKS = 8
HELDOUT_TASK_SEED = 777     # disjoint from the training pool seed (0)

# How the meta agent draws z from its posterior during DETERMINISTIC evaluation.
# 'sample' was the only prior behaviour and it redrew z every control step, so
# meta's "deterministic" eval was not deterministic at all while the baseline's
# (z == 0) was — a confound worth more than the entire meta-vs-baseline gap.
# 'mean' (posterior mean, no sampling) is the correct analogue of the actor's
# deterministic mode; 'hold' is PEARL's own scheme (one hypothesis per
# trajectory). Use MODE='eval_z_probe' to measure all three on a trained
# checkpoint without retraining.
EVAL_Z_MODE = "mean"
# z_mode used for the periodic held-out eval during training (kept the same as
# EVAL_Z_MODE so the checkpoint-selection metric matches the final eval).
HELDOUT_Z_MODE = "mean"

# Warm-up episodes collected *per task* before any gradient step.  The update
# readiness check needs BATCH_SIZE transitions in a single task's replay
# bucket; collecting COLLECT_EPISODES_PER_ITER rollouts spread randomly over
# N_TRAIN_TASKS fills 40 buckets far too slowly, so early iterations do no
# learning at all.  Seeding every bucket up front makes the first iteration
# already trainable and removes the long dead-start.
WARMUP_EPISODES_PER_TASK = 3

# Reward handling.  The env's per-step reward is penalty-dominated and *large*
# in magnitude (order 1-10, not small), so a fixed reward scale > 1 blows the
# Q-target up (huge critic loss) and drowns the KL bottleneck + entropy terms.
# Instead we standardise the reward online by a running std inside the agent
# (NORMALIZE_REWARDS), which auto-scales the Bellman target to ~O(1/(1-gamma))
# no matter the raw magnitude, and leave REWARD_SCALE at 1.0.  Applied
# identically to both agents, so the meta-vs-baseline comparison stays fair.
NORMALIZE_REWARDS = True
REWARD_SCALE = 1.0

# Reset the environment's Lagrange multipliers at every episode.  They decay
# with a ~830-step half-life — longer than one episode (~120 control steps) —
# so with the default (no reset) a violation on one task leaves the multiplier
# pinned near its cap and that penalty bleeds into the next, unrelated task's
# episode.  In a meta-RL setting where each episode is a fresh task draw, the
# multipliers must start clean each episode.  Both agents share this.
RESET_LAMBDAS_PER_EPISODE = True

# ── Disturbances in the META-TRAINING task distribution ───────────────────────
# ROOT CAUSE THIS FIXES: collect_rollout() only ever called env.set_task(), so
# meta-training exposed the encoder exclusively to STATIC parameter draws (pH,
# toxin tolerance, enzyme lot, feed composition). It never saw a mid-episode
# regime change — yet evaluate() tests exactly that, on three disturbance
# scenarios. The encoder was asked to represent a kind of task variation that
# was absent from its training distribution.
#
# The evidence that this is the binding constraint (policy_n8/pearl_eval.json,
# N_TRAIN_TASKS=8): meta and baseline both hit 0 violations on the four
# static-ish scenarios, and meta collapses ONLY on disturb_mass_drift (28.10
# violations, reward -149, vs baseline 0 / +0.25) — the one scenario that
# requires detecting change rather than identifying a fixed regime. The latent
# diagnostic agrees: z-spread under mass_drift (0.0220) is indistinguishable
# from in-distribution (0.0218), i.e. the encoder does not register that
# anything happened.
#
# ONLY the meta agent's collection is affected (collect_rollout(disturb=True),
# called from train_meta only). train_baseline() stays nominal-task-only with no
# disturbances — that is the isolating variable of the whole experiment — and
# evaluate()'s scenarios are untouched, so eval remains a fixed held-out test.
META_DISTURB_PROB = 0.45    # P(a meta-training episode carries a disturbance)
META_DISTURB_KINDS = ("inlet_step", "enzyme_drop", "mass_drift")
# Onset as a FRACTION of env.max_steps (inner steps, the unit
# env.schedule_disturbance expects). Eval fires at exactly max_steps // 2, so
# this brackets it on both sides: the agent must handle an onset it has not
# memorised the timing of.
META_DISTURB_ONSET_FRAC = (0.25, 0.75)
# Magnitude ranges: STRICTLY BROADER THAN, AND CONTAINING, the eval literals
# (inlet_step 0.7, enzyme_drop 0.5, mass_drift -0.005 — see evaluate()). This is
# deliberate and load-bearing. Training on the exact eval configuration would
# make a good disturb_mass_drift number meaningless: it would demonstrate
# memorisation of one disturbance, not adaptation to a regime change. Drawing
# continuously from a wider interval means the exact eval triple is measure-zero
# in the training distribution while remaining inside its support. Sign/direction
# per kind matches the env's semantics in env_safe._apply_disturbances (feeds and
# enzyme params are MULTIPLIED by magnitude, so <1 = degradation; mass_drift is a
# per-step multiplicative rate, so <0 = shrinking reactor mass).
# smoke_test_pearl.check_meta_disturbance_injection asserts the containment and
# the non-degeneracy, so a later edit cannot quietly break either property.
META_DISTURB_MAG_RANGES = {
    "inlet_step": (0.50, 0.90),      # eval: 0.7
    "enzyme_drop": (0.30, 0.70),     # eval: 0.5
    "mass_drift": (-0.008, -0.002),  # eval: -0.005
}

# ── Disturbance-severity sweep (MODE='severity_sweep') ───────────────────────
# WHY THIS EXISTS. After disturbances entered the meta-training distribution,
# EVERY arm — meta, meta_noz, and the non-adaptive baseline trained on the
# nominal task ONLY, with no disturbance ever — scored 0.00 violations on all
# three eval disturbance scenarios (policy_n8_dist/pearl_eval.json). Reward
# deltas were 0.026-0.119 against a per-scenario SEM of ~0.24 (std ~1.0-1.4 over
# EVAL_EPISODES=20), and the actuator columns were near-identical across
# in-distribution, OOD and all three disturbances. So: the violation metric is
# saturated at its floor, the reward differences are well inside noise, and the
# eval as configured cannot discriminate any policy from any other — let alone
# show that task inference helps.
#
# A scenario the NON-ADAPTIVE baseline already solves perfectly cannot be
# evidence for adaptation, no matter what the meta agent does on it. So before
# any further agent work, this sweep asks the prior question: at what
# disturbance severity does the non-adaptive baseline actually fail? That
# failure point, if it exists, is the only region where a meta-vs-baseline
# comparison has room to mean anything.
#
# If NO severity breaks the baseline, the shared safety stack (SERL stage mask +
# the env's Lagrangian penalties) is supplying feasibility independently of the
# policy, and no policy-level adaptation claim is testable in this environment
# without changing the constraint structure. That is a finding, not a null
# result, and it would redirect the work away from the encoder entirely.
#
# Level 0 of every kind is the eval literal, so each sweep contains the current
# eval result as its first row. Onset is held at max_steps // 2 exactly as in
# evaluate(), so SEVERITY IS THE ONLY VARIABLE. Higher levels deliberately
# escalate PAST META_DISTURB_MAG_RANGES — past meta's training support — because
# beyond level ~1 neither agent has seen the regime, which is the fair place to
# test generalisation rather than mere exposure.
SEVERITY_SWEEP_LEVELS = {
    # Feeds are MULTIPLIED by magnitude (env_safe._apply_disturbances), so
    # SMALLER = more severe. 0.05 = a 95% inlet cut.
    "inlet_step": (0.70, 0.50, 0.30, 0.15, 0.05),
    # Hydrolysis eta and enzyme adsorption caps MULTIPLIED, so smaller = worse.
    "enzyme_drop": (0.50, 0.30, 0.15, 0.05, 0.01),
    # Per-step multiplicative reactor-mass drift, so MORE NEGATIVE = worse.
    # At -0.05 over ~60 post-onset inner steps the mass falls to ~5% of nominal.
    "mass_drift": (-0.005, -0.010, -0.020, -0.035, -0.050),
}
# Which arms to run at every level. The baseline ALONE answers the question this
# sweep exists to answer, at 1/3 the wall clock; add "meta" and "meta_noz" once a
# break point is known, to compare the arms AT that severity (which is the only
# comparison that could carry the adaptation claim).
SEVERITY_SWEEP_AGENTS = ("baseline",)
# A level counts as BREAKING an agent if any of these trip. Violations and
# infeasibility are outright failures. The ethanol floor catches the degenerate
# escape the productivity table already warns about: an agent can hold
# violations at zero by shutting the plant down, and that IS a failure mode a
# genuine adaptation advantage could legitimately exploit — so it must count.
SEVERITY_BREAK_VIOL = 0.5           # mean violations/episode
SEVERITY_BREAK_FEASIBLE = 1.0       # feasible_rate below this = broke
SEVERITY_BREAK_ETHANOL_FRAC = 0.5   # ethanol below this x its UNDISTURBED value

# ── What the first full sweep found (MODE='headroom_test') ────────────────────
# RESULT (policy_n8_dist/pearl_severity_sweep.json, baseline arm):
#
#   inlet_step   NO BREAK down to 0.05 (a 95% inlet cut): 0.00 violations, and
#                only -13% ethanol.
#   enzyme_drop  NO BREAK down to 0.01 (a 99% enzyme knockout): 0.00 violations,
#                -4% ethanol.
#   mass_drift   breaks at -0.010 (12.60 viol), then 81.4 / 125.1 / 164.9 at
#                -0.020 / -0.035 / -0.050. Clean monotone dose-response.
#
# WHY the first two can never break, structurally: every constraint in this env
# is an INVENTORY constraint (see the enumeration at env_safe.py's Lagrangian
# block — reactor overflow/underflow, storage overflow/underflow), i.e. a
# function of self.M and self.S_mass only. inlet_step perturbs feed COMPOSITION
# (C_C5/C_LF) and enzyme_drop perturbs KINETICS (eta, adsorption caps); neither
# appears in any constraint, and neither changes the mass flow F_ins, which is
# set by the feed actions. So both can only ever move the productivity term
# (~0.015/step at most) against Lagrangian penalties reaching ~10/reactor/step.
# Escalating them further is wasted compute, not a missing severity.
# (Note also that inlet_step scales composition WITHOUT scaling mass flow, so it
# models "same feed mass, diluted with inert", not "less feed". A feed-RATE
# disturbance would hit the inventory constraints and would be a genuine second
# discriminating axis; this one is not.)
#
# WHY the two most extreme mass_drift levels are NOT trustworthy: drift is
# applied per inner step as M *= (1 + mag) from onset, and env_safe clamps with
# np.maximum(1.0, ...) while crediting the UNCLAMPED delta to
# episode_mass_discharged (mass_balance_abs_tol is 5.0 kg). With M0 = 10000
# kg/reactor and max_steps - onset = 300 post-onset steps:
#     -0.005 -> 2231 kg    -0.010 -> 490 kg    -0.020 -> 23 kg    (clamp clear)
#     -0.035 -> 0.25 kg    -0.050 -> 0.002 kg                     (CLAMP BINDS)
# so at -0.035/-0.050 some of the violation count is a mass-ledger artifact
# rather than a control failure.
#
# CONCLUSION: mass_drift in [-0.020, -0.010] is the ONLY region of this eval
# where the non-adaptive baseline genuinely fails and the measurement is clean —
# therefore the only region where the adaptation claim is testable at all. Note
# that META_DISTURB_MAG_RANGES['mass_drift'] = (-0.008, -0.002) lies entirely
# inside the region where the baseline is already perfect, so the meta agent has
# never trained anywhere near this window.
SEVERITY_CLEAN_WINDOW = {"mass_drift": (-0.010, -0.015, -0.020)}
# All three arms, because in this window meta vs meta_noz is the decisive test:
# meta_noz shares meta's weights and training but has z ablated, so a meta-only
# advantage over baseline here would still not distinguish task inference from
# robustness picked up by training on disturbances at all.
SEVERITY_CLEAN_ARMS = ("baseline", "meta", "meta_noz")
# Predicted post-onset reactor mass below which the M >= 1.0 clamp is close
# enough to binding that the mass-ledger discrepancy can contribute violations.
# 10 kg is 10x the floor and 2x mass_balance_abs_tol (5.0 kg).
SEVERITY_CLAMP_RISK_KG = 10.0

# ── Latent-informativeness gates (MODE='z_informativeness') ──────────────────
# WHY THESE EXIST. The adaptation decomposition's INFER column (noz - meta) is
# the proposal's entire claim, but it is only INTERPRETABLE if z is carrying task
# information in the first place. If the encoder returns effectively the same
# latent for every task, or the actor's output is insensitive to the latent it is
# handed, then INFER is mathematically pinned near zero no matter how the sweep
# is configured — and reading "INFER = +0.70, headroom yes" as evidence about
# adaptation is then a category error, not a weak result. The first headroom_test
# run showed exactly this pattern (INFER +0.70 / +0.65 / -0.20, sign-flipping and
# far inside the paired noise), which is the signature of a null-by-construction
# latent rather than of a real effect too small to see.
#
# These thresholds turn that into a pass/fail gate that runs BEFORE any further
# sweep work, so a collapsed encoder is diagnosed as a collapsed encoder instead
# of being reported as a null adaptation result.
#
# Separability: across-task std of the posterior MEAN, as a fraction of the
# posterior's own std. Below 1.0 the task-to-task movement of the belief is
# smaller than the width of the belief itself, i.e. tasks are not resolved.
Z_INFORM_MIN_SNR = 1.0
# Behavioural sensitivity: mean |action delta| (post-tanh, so on a [-1, 1]
# scale, per action dim) when the SAME states are re-evaluated under a latent
# inferred on a DIFFERENT task. Below this the actor is ignoring its z input.
Z_INFORM_MIN_ACTION_DELTA = 1e-3
# Actor sensitivity at PRIOR scale: d|action| / d|z|, measured by perturbing z at
# the N(0, I) scale the actor was trained on rather than at whatever tiny scale a
# collapsed encoder happens to produce. THIS IS THE PROBE THAT DECIDES THE FIX.
# A swapped-latent action delta near zero is uninformative when the two latents
# differ by ~0.008 (as they do under a collapsed encoder) — it is confounded with
# the collapse. This probe asks the counterfactual instead: if the encoder were
# repaired and emitted well-separated unit-scale latents, would the policy's
# output move? If it would not, the actor has learned to ignore z and an
# encoder-only fix cannot possibly work.
Z_INFORM_MIN_PRIOR_SENS = 1e-2
# Across-task standard deviation of violations, for a FIXED policy, below which
# the metric is treated as task-INDEPENDENT: tasks are then indistinguishable in
# the quantity INFER is measured in, so no task inference can improve it. Set at
# the same 0.5 violations/episode used by SEVERITY_BREAK_VIOL, since a spread
# smaller than the break threshold itself cannot carry a comparison.
Z_INFORM_MIN_VIOL_SPREAD = 0.5
# Episodes used by the z-swap rollout arm. Kept independent of EVAL_EPISODES
# because this diagnostic runs 3 rollout passes and is the one thing that should
# be cheap enough to run before every sweep.
Z_INFORM_EPISODES = 12

# Separate directory so this diagnostic run's checkpoints/eval/plots don't
# overwrite the n=40 result already saved under policy/ — you need both to
# compare. Set back to "policy" once this test is resolved and you return to
# the default N_TRAIN_TASKS=40 configuration.
# RUN_TAG additionally separates training-DISTRIBUTION variants at a fixed
# N_TRAIN_TASKS, so this run does not overwrite either existing comparison point
# (policy/ at n=40, policy_n8/ at n=8, both trained WITHOUT disturbances).
# "dist" = disturbances are part of the meta-training distribution
# (META_DISTURB_PROB above). Set RUN_TAG = "" to write to the plain directory.
RUN_TAG = "dist"
_BASE_MODEL_DIR = "policy" if N_TRAIN_TASKS == 40 else f"policy_n{N_TRAIN_TASKS}"
MODEL_DIR = f"{_BASE_MODEL_DIR}_{RUN_TAG}" if RUN_TAG else _BASE_MODEL_DIR
META_PATH = os.path.join(MODEL_DIR, "pearl_meta.pth")
BASE_PATH = os.path.join(MODEL_DIR, "pearl_baseline.pth")
# Separate "best-seen" checkpoints, tracked by the DETERMINISTIC HELD-OUT
# violation count (see HELDOUT_EVAL_EVERY below). Training in this setting is
# not monotonic: on one run the meta agent reached avg_violations of 8.5 at
# iteration 57, then regressed to 283 by iteration 2000 without ever recovering.
# META_PATH/BASE_PATH always hold the *latest* snapshot (useful for resuming /
# inspecting a trend); BEST_META_PATH/BEST_BASE_PATH hold whichever snapshot had
# the lowest held-out violation count, so a later regression can never silently
# overwrite a genuinely better policy. evaluate() loads the best checkpoints when
# present.
# NOTE: this used to be the argmin of a rolling mean over 50 STOCHASTIC TRAINING
# episodes, which is selection on noise — it reported "best 2.4" for a checkpoint
# that scored 24.7 at eval, and its argmin always landed near iteration 60,
# concealing the fact that every run then diverged by 30x.
BEST_META_PATH = os.path.join(MODEL_DIR, "pearl_meta_best.pth")
BEST_BASE_PATH = os.path.join(MODEL_DIR, "pearl_baseline_best.pth")

# Early stopping.  Raw reward here is penalty-dominated (a do-nothing policy
# already scores ~-24,000/episode from the Lagrangian constraint penalty
# alone), so it is a poor convergence signal — the violation count is the
# metric that actually reflects learning progress (it runs from ~590/episode
# for a random policy down toward 0).  The metric is the DETERMINISTIC HELD-OUT
# violation count (see HELDOUT_EVAL_EVERY), not the rolling mean of stochastic
# training episodes it used to be.
#   - target: held-out violations stay at/below EARLY_STOP_TARGET_VIOL for
#     EARLY_STOP_TARGET_PATIENCE checks (fast exit once already good, without
#     waiting out a full plateau window).
#   - plateau: held-out violations haven't improved by EARLY_STOP_MIN_DELTA from
#     the best-seen value for EARLY_STOP_PATIENCE checks (catches training that
#     has stalled even above the target).
#   - diverge: see EARLY_STOP_DIVERGE_* below.
#
# UNITS: every patience/min value below counts HELD-OUT EVAL CHECKS, not raw
# iterations, because the stopper is only consulted when a fresh held-out number
# exists. Checks land at iteration 1 and then every HELDOUT_EVAL_EVERY, so check
# k is at iteration HELDOUT_EVAL_EVERY*(k-1) — i.e. multiply by
# HELDOUT_EVAL_EVERY (=10) and subtract one interval for the iteration
# equivalent.
EARLY_STOP_ENABLED = True
EARLY_STOP_MIN_ITERS = 30           # checks (~iter 290) before any stop
EARLY_STOP_PATIENCE = 30            # checks w/o improvement -> plateau stop
EARLY_STOP_MIN_DELTA = 1.0          # min violation-count improvement to count
EARLY_STOP_TARGET_VIOL = 3.0        # "good enough" avg violations/episode
EARLY_STOP_TARGET_PATIENCE = 5      # checks sustained at/under target -> stop
# Divergence trip: if the metric climbs to DIVERGE_FACTOR x its best-seen value
# for DIVERGE_PATIENCE consecutive checks, the run is DIVERGING, not
# plateauing, and there is no point burning the remaining patience window. The
# old stopper could not tell these apart: it sat through 700 iterations of
# monotone worsening (7.9 -> 262 violations) and then reported "plateaued".
#
# Set EARLY_STOP_DIVERGE_ENABLED = False for ONE diagnostic run when you want to
# observe the whole curve rather than stop at the first confirmed collapse. The
# previous run tripped at iteration 125 of 2000, which was correct behaviour but
# left open whether it would have recovered later (the pre-fix run did show a
# partial secondary recovery around iteration 400). The best-checkpoint logic is
# independent of this, so nothing is lost by letting a diagnostic run continue.
EARLY_STOP_DIVERGE_ENABLED = True
EARLY_STOP_DIVERGE_FACTOR = 5.0
EARLY_STOP_DIVERGE_PATIENCE = 5
EARLY_STOP_DIVERGE_MIN_CHECKS = 15
# Absolute held-out-violation excursion that counts as divergence WHEN the
# best-seen score is ~0 and the diverge_factor ratio is therefore undefined.
# Without this, a single held-out reading of exactly 0.00 turns the divergence
# test into "above 5.0 violations" — which is what killed the policy_n8_dist run
# at iteration 140/2000 while it was still improving. Real divergence in this
# project went 20 -> 20745, so 50 still catches a genuine blow-up in 1-2 checks.
EARLY_STOP_ZERO_BEST_FLOOR = 50.0

# ── SAC exploration schedule ──────────────────────────────────────────────────
# Floor on the SAC temperature alpha, applied to BOTH agents so the comparison
# stays fair. Evidence for needing one (policy/pearl_meta_log.json, post-fix
# run): alpha decayed monotonically 0.976 -> 0.055 over 125 iterations with no
# rebound, held-out violations hit 0.0 at iteration 25 while alpha was ~0.55,
# then went 56.2 -> 93.0 as alpha fell through 0.31 and 0.17. Alpha only falls
# while policy entropy EXCEEDS the target, so a monotone decay means the target
# demands more determinism than a 40-task problem can afford this early: the
# policy locks in before it generalises, with no exploration left to escape.
# 0.15 sits just under the alpha at the observed optimum.
ALPHA_MIN = 0.15
# Alternative, smoother lever on the same problem — entropy target per FREE
# action dimension (None = the standard -1.0 heuristic). Less negative demands
# more entropy and slows alpha's decay without a hard floor. Change ONE of
# ALPHA_MIN / TARGET_ENTROPY_PER_DIM at a time, not both, or you cannot tell
# which did the work.
TARGET_ENTROPY_PER_DIM = None

# Meta gets a longer runway than the baseline values above. Evidence from a
# real run (policy/pearl_meta_log.json): avg_violations dipped to 8.3 at
# iteration 62 (alpha=0.237, real exploration still active), then got steadily
# worse and never recovered anywhere close to that level as alpha decayed
# further (down to 0.019 by iteration 341) — the policy locked into a
# low-entropy, near-deterministic regime before it had learned to generalize
# across the 40-task pool, with no residual exploration noise left to escape
# it. Baseline's single, easy, fixed task converges fine under the same
# entropy-annealing schedule; meta's much harder, more diverse problem needs
# more sustained exploration and more patience before a plateau should be
# trusted as genuine convergence. See META_TARGET_ENTROPY below for the
# complementary fix (slower alpha decay in the first place).
META_EARLY_STOP_MIN_ITERS = 70      # checks (~iter 690)
META_EARLY_STOP_PATIENCE = 70       # checks (~700 iters of no improvement)

# SAC target entropy. Defaults to -action_dim*n_reactors (=-12 here) in
# PEARLAgent if left at None -- the standard heuristic, and what the baseline
# keeps. A prior run set this to -6.0 to slow alpha decay, on the theory that
# meta was failing from premature entropy collapse. That was DISPROVEN by the
# diagnose_meta_log.py run: meta doesn't stall from low entropy, its critic
# *diverges* (QL 20->20,745 after iter 58) driven by an unregularized context
# encoder (see META_KL_LAMBDA). The entropy target was a red herring, and -6.0
# also made the meta-vs-baseline comparison unfair (baseline uses -12). Back
# to None (=-12, the standard heuristic, matching baseline).
META_TARGET_ENTROPY = None

# KL information-bottleneck weight on the context encoder (PEARLAgent default
# 0.1). The diagnosis: at the violation minimum (iter 58) the posterior was
# healthy (KL=0.50, QL=20); afterward the encoder ran away (KL->41.5) because
# the bottleneck penalty kl_lambda*KL was only ~0.02% of the critic loss --
# effectively unregularized. An unbounded z makes the critic's target
# nonstationary -> QL diverges -> policy collapses to ~random. The baseline
# (no encoder) keeps QL~20 throughout, confirming the encoder is the culprit.
# Raising kl_lambda holds z near the N(0,I) prior. TRADEOFF: too high collapses
# z to the prior entirely (z uninformative -> meta degenerates to baseline, no
# adaptation), too low is today's runaway.
#
# RESULT: the 1.0 probe (10x) was NOT enough -- verified against
# pearl_meta_log.json from that run: healthy era (iter 50-70) KL~0.01-0.10,
# |z|~0.5-1.0; by iter 250-300 KL had grown to 3.7-12.6, |z| to 0.9-2.1, and
# QL to 331-1983 (avg_violations back up to ~500+, near the ~590 random-policy
# floor) -- same unrecovered-runaway shape as the kl_lambda=0.1 run, just
# slower. The bottleneck/critic ratio only reached ~0.66% at iter 300 (vs
# 0.02% at kl_lambda=0.1) -- still nowhere near large enough to actually
# anchor the encoder. (Also ruled out the entropy-offset mechanism as a
# contributor here: alpha stayed low, ~0.07, the whole run, since
# META_TARGET_ENTROPY is back to -12 -- so this is purely the encoder/critic
# feedback loop, not alpha*target_entropy inflating Q.)
#
# UPDATE: kl_lambda=3.0 (with ENCODER_LR below) proved BISTABLE across runs —
# one run kept mild structure (KL=0.55, usable posterior), the next COLLAPSED
# the posterior to the prior (KL=0.001, z carries no task info -> meta
# degenerates into a worse-trained baseline). That is the over-regularization
# failure mode. Dropping kl_lambda to 0.5 was tried next, expecting it to
# relieve the collapse -- instead a 3-SEED SWEEP showed it collapsed (KL~0.001)
# in EVERY seed, more consistently than at 3.0. Diagnosis: with ENCODER_LR
# slowed to 3e-5, EVERY gradient into the encoder is slow -- including the
# informative one (learn to predict theta / reduce critic loss), not just the
# runaway-causing one. The KL penalty gradient competes on the same slow
# clock and wins before the informative signal establishes itself: this is
# the standard VAE "posterior collapse" failure mode. (Also caught a flawed
# diagnostic here: the OLD |z| metric was a SAMPLE mu+sigma*noise, so at full
# collapse (mu~=0, sigma~=1) it read ~0.8-0.9 -- just E|N(0,1)| noise -- and
# looked "healthy" when the encoder was actually fully dead. Now logging |mu|,
# the posterior MEAN, which only stays large when genuinely informative; see
# agent_pearl.py's update() for the fix. KL remains the primary indicator.)
#
# FIX: KL ANNEALING (the standard fix for posterior collapse) instead of a
# fixed kl_lambda from iteration 1. Start the penalty near 0 so the informative
# signal gets a real head start, then ramp to the target strength by
# KL_ANNEAL_ITERS below -- BEFORE the divergence window (empirically iter
# ~150-300 across every run so far) opens. See seed_everything/train_meta for
# where this is applied (agent.kl_lambda is set per-iteration, not fixed at
# construction). META_KL_LAMBDA below is now the ANNEAL TARGET (end-of-ramp
# value), not the value used from iteration 1.
META_KL_LAMBDA = 0.5
# Outer iterations over which kl_lambda ramps linearly from 0 to
# META_KL_LAMBDA. Anchored to two empirical facts that have held across every
# run regardless of hyperparameters: (a) the single best avg_violations point
# has always landed around iter 57-62, so the ramp must still be gentle there;
# (b) divergence/collapse dynamics only become visible from ~iter 150 onward,
# so kl_lambda should be at (or near) full strength well before that. 150
# threads both: soft during the critical early-learning window, fully engaged
# before the danger zone.
KL_ANNEAL_ITERS = 150

# Learning rate for the context encoder specifically (see docstring in
# agent_pearl.PEARLAgent.__init__ for the full rationale). Actor/critic/alpha
# keep the shared 3e-4 (lr default); the encoder gets 10x slower so its
# per-step drift in z can't outpace the critic tracking it -- directly
# targets the runaway mechanism without relying solely on kl_lambda. NOTE:
# this is also the reason kl_lambda alone couldn't fix collapse (see above) --
# slowing the encoder slows BOTH its informative and its regularizing
# gradients equally, so KL annealing (order of application, not just
# magnitude) is now the primary lever for the timescale conflict.
#
# REVERTED TO 3e-4 (the shared lr). The 10x slowdown was introduced to fight an
# "encoder runaway" in which KL and critic_loss diverged together. That runaway
# has since been traced to a DIFFERENT cause entirely: the reward-scale estimator
# had an unbounded sample count, froze at its random-policy value within ~100
# iterations, and made the Bellman target shrink as the policy degraded (see
# RunningMeanStd in agent_pearl.py). With that fixed, critic_loss now stays in a
# healthy 2-15 band instead of decaying to 0.46.
#
# So the slowdown is now pure cost: it throttles the encoder's INFORMATIVE
# gradient (reduce critic loss / identify the task) just as much as the runaway
# one, and it is the documented reason KL annealing was needed in the first place
# — the KL penalty competed on the same slow clock and won, collapsing the
# posterior. The last run's evidence that the encoder is starved rather than
# runaway: final KL 0.356 and |mu| 0.021, and at evaluation the meta agent's
# actuator positions were constant to three decimals across in-distribution,
# OOD, and all three disturbances. z was carrying essentially no task
# information. If the runaway returns at 3e-4, that is now a real finding rather
# than a rediscovery of the reward-scale bug.
META_ENCODER_LR = 3e-4


class EarlyStopper:
    """Plateau / target / divergence early stopping on a metric (lower is better).

    Every count below is in CHECKS (calls to ``should_stop``), not raw outer
    iterations, because the caller only has a fresh held-out number every
    HELDOUT_EVAL_EVERY iterations.

    Three independent triggers, any of which stops training once `min_checks`
    checks have happened:
      - target: the metric stays at or below `target` for `target_patience`
        consecutive checks — a fast exit once training is already "good
        enough", without waiting for a full plateau to establish itself.
      - plateau: the metric hasn't improved by at least `min_delta` from its
        best-seen value for `patience` consecutive checks — catches training
        that has stalled, even above the target (or when no target is set).
      - diverge: the metric has climbed to `diverge_factor` x its best-seen
        value for `diverge_patience` consecutive checks, AND is not currently
        trending down.  A PLATEAU AND A DIVERGENCE ARE NOT THE SAME EVENT, and
        an early version conflated them: on every seed of an earlier sweep it sat
        through 700 iterations of monotone worsening (violations 7.9 -> 262) and
        then reported "plateaued", which is a large part of why the collapse
        stayed invisible.

    `zero_best_floor` exists because the watched metric (held-out violations) can
    legitimately reach exactly 0.00, which makes the `diverge_factor` ratio
    undefined and silently turned the divergence test into a bare absolute
    threshold.  See the detailed note in ``should_stop`` — this is what
    terminated the policy_n8_dist run at iteration 140 of 2000 while its held-out
    score was still improving.
    """

    def __init__(self, min_checks=12, patience=12, min_delta=1.0,
                 target=None, target_patience=2,
                 diverge_factor=5.0, diverge_patience=3,
                 diverge_min_checks=6, zero_best_floor=50.0):
        self.min_checks = min_checks
        self.patience = patience
        self.min_delta = min_delta
        self.target = target
        self.target_patience = target_patience
        self.diverge_factor = diverge_factor
        self.diverge_patience = diverge_patience
        # Guard against an early lucky check setting a tiny `best` and killing an
        # otherwise healthy run: the divergence trip needs some history first.
        # Still fires far earlier than the plateau window would.
        self.diverge_min_checks = diverge_min_checks
        self.zero_best_floor = float(zero_best_floor)
        self.best = float("inf")
        self.plateau_count = 0
        self.target_count = 0
        self.diverge_count = 0
        self.n_checks = 0
        # Short window of recent readings, for the trend guard in should_stop.
        # Only the last 3 are ever consulted.
        self._recent = deque(maxlen=3)

    def should_stop(self, it, value):
        """Check the stopping condition for the latest metric reading.

        Args:
            it: Current outer iteration (1-indexed); used in messages only.
            value: Latest metric value (lower is better).

        Returns:
            A tuple (stop, reason); reason is "" when stop is False.
        """
        self.n_checks += 1
        # Trend history first: both the plateau and the divergence tests below
        # consult it. Only the last 3 readings are ever used.
        self._recent.append(value)
        improving = (len(self._recent) >= 3
                     and self._recent[-1] < self._recent[-3])

        # ── Plateau bookkeeping ──────────────────────────────────────────────
        # THE `improving` GUARD IS LOAD-BEARING, for the same reason as in the
        # divergence test. `best` locks to 0.00 as soon as one held-out check
        # scores zero violations (which happens early and often — violations are
        # a saturating, largely task-independent metric here). After that,
        # `value < best - min_delta` requires a violation count below -1.0, which
        # is impossible, so plateau_count would increment on EVERY subsequent
        # check regardless of what training was doing and fire the plateau stop
        # a fixed `patience` checks later — on the policy_n8_dist trace that was
        # iteration ~320 of 2000, while the held-out score was still trending
        # down (17.6 -> 8.0 -> 5.1). A metric that is falling has not plateaued.
        #
        # Consequence to be aware of: on a noisy held-out signal this makes the
        # plateau trigger much harder to fire, so runs will tend to go to the
        # META_ITERATIONS cap. That is the intended trade here — the `target`
        # trigger still exits early on genuine success and `diverge` still
        # catches a real blow-up.
        if value < self.best - self.min_delta:
            self.best = value
            self.plateau_count = 0
        elif improving:
            self.plateau_count = 0
        else:
            self.plateau_count += 1

        if self.target is not None and value <= self.target:
            self.target_count += 1
        else:
            self.target_count = 0

        # ── Divergence test ──────────────────────────────────────────────────
        # THIS BLOCK KILLED THE policy_n8_dist RUN AT ITERATION 140 OF 2000, AND
        # IT WAS WRONG TO. Two defects, both triggered by a metric that can
        # legitimately hit exactly 0.00:
        #
        # 1. DEGENERATE THRESHOLD. The old test was
        #        value > max(best * factor, best + 5 * min_delta)
        #    With best == 0.0 the first term vanishes and the second collapses to
        #    a bare absolute ceiling of 5.0 violations. A single lucky held-out
        #    reading of 0.00 (iterations 20 and 30 in that run) therefore
        #    permanently redefined "diverging" as "above 5 violations" — nothing
        #    to do with divergence. Because violations here are a saturating
        #    metric (see the note on inventory constraints in the severity-sweep
        #    block: they are largely task-independent and bottom out at 0), that
        #    zero is not a fluke to be tolerated but the common case.
        # 2. NO TREND GUARD. Divergence must mean "getting worse and staying
        #    worse", not "sitting above a threshold". That run's last three
        #    held-out readings were 17.6 -> 8.0 -> 5.125: monotonically
        #    IMPROVING, and it was stopped anyway, 0.125 violations above the
        #    accidental ceiling, with the KL anneal schedule (150 iters) not even
        #    finished.
        #
        # Fix: fall back to a generous absolute floor when `best` is too small for
        # a ratio to mean anything, and never count a check as diverging while the
        # metric is trending down.
        # NOTE THE GUARD ORDER: diverge_factor is None whenever divergence
        # detection is switched off (EARLY_STOP_DIVERGE_ENABLED = False, the
        # documented diagnostic path), so the threshold must not be computed
        # before that check — `self.best * None` is a TypeError.
        diverge_thresh = float("inf")
        if self.diverge_factor is not None and np.isfinite(self.best):
            if self.best > self.min_delta:
                diverge_thresh = max(self.best * self.diverge_factor,
                                     self.best + 5.0 * self.min_delta)
            else:
                # best is ~0: the ratio is undefined, so require a large absolute
                # excursion instead. Real divergence in this project went
                # 20 -> 20745, so this still trips within 1-2 checks on a genuine
                # blow-up.
                diverge_thresh = self.zero_best_floor
        diverging = not improving and value > diverge_thresh
        self.diverge_count = self.diverge_count + 1 if diverging else 0

        # Gated on diverge_min_checks (much smaller than min_checks) rather than
        # on min_checks: a run that has already blown up should not burn the
        # whole plateau window, but it also should not be killed off one lucky
        # early reading.
        if (self.n_checks >= self.diverge_min_checks
                and self.diverge_count >= self.diverge_patience):
            # Report the ratio only when it is meaningful. A best-seen of exactly
            # 0.0 (which happens — a deterministic held-out episode really can
            # score zero violations) makes value/best meaningless; the earlier
            # version printed "38000000000.0x its best-seen 0.00".
            if self.best > self.min_delta:
                mag = f"{value / self.best:.1f}x its best-seen {self.best:.2f}"
            else:
                mag = f"{value:.2f} above its best-seen {self.best:.2f}"
            return True, (f"DIVERGING (not plateaued): metric {value:.2f} is "
                          f"{mag} for {self.diverge_count} consecutive checks")
        if self.n_checks < self.min_checks:
            return False, ""
        if self.target is not None and self.target_count >= self.target_patience:
            return True, (f"metric {value:.2f} <= target {self.target:.2f} "
                          f"sustained for {self.target_count} checks")
        if self.plateau_count >= self.patience:
            return True, (f"metric plateaued at {value:.2f} "
                          f"(best {self.best:.2f}) for {self.plateau_count} "
                          f"checks with no improvement")
        return False, ""


# ─── Environment / task helpers ───────────────────────────────────────────────


def seed_everything(seed):
    """Seed all RNGs that affect a training run, for reproducible comparisons.

    Covers Python's ``random``, NumPy's global RNG (used by rollout task
    selection and replay sampling), and torch CPU+CUDA (network init, action
    sampling, encoder z-sampling). Note: the task pool is drawn from its own
    seeded generator (make_task_pool), so it is already fixed independently of
    this; this seeds the *stochastic training dynamics* on top of that.
    """
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env():
    """Instantiate the heterogeneous 3-reactor fermentation environment.

    Returns:
        A FermentationEnv configured with the standard reactor limits.
    """
    cfg = get_main_reactor_config()
    env = FermentationEnv(
        n_reactors=3, Mmax=cfg["Mmax"],
        max_flow_LF=cfg["max_flow_LF"], max_flow_C5=cfg["max_flow_C5"])
    env.reset_lambdas_per_episode = RESET_LAMBDAS_PER_EPISODE
    return env


def make_task_pool(n, seed=0, ood=False):
    """Sample a fixed pool of tasks theta ~ p(theta).

    Args:
        n: Number of tasks.
        seed: RNG seed for reproducibility.
        ood: If True, sample from widened out-of-distribution priors.

    Returns:
        List of task dicts (see env_config.sample_task).
    """
    rng = np.random.default_rng(seed)
    return [sample_task(rng, ood=ood) for _ in range(n)]


def maybe_schedule_disturbance(env, prob=None, rng=None):
    """Clear disturbances, then with probability ``prob`` schedule a random one.

    This is what puts mid-episode regime changes INTO the meta-training task
    distribution (see the META_DISTURB_* block at the top for why the absence of
    them was the root cause of meta's disturb_mass_drift collapse). Called only
    from the meta agent's collection path.

    Always calls ``env.clear_disturbances()`` first, so (a) a no-disturbance draw
    yields a genuinely clean episode rather than inheriting the previous one's
    schedule, and (b) nothing leaks into the held-out eval that shares this env.

    Args:
        env: FermentationEnv. Only ``max_steps`` is read (onset is expressed as a
            fraction of it) plus the two disturbance methods.
        prob: Disturbance probability; defaults to META_DISTURB_PROB. Pass 0.0
            for a guaranteed-clean episode.
        rng: Object exposing ``random()``/``choice()``/``uniform()`` — either the
            ``np.random`` module (the default, so draws follow the global SEED
            like the rest of collection) or a ``np.random.Generator`` for tests.

    Returns:
        The scheduled disturbance dict {'kind', 'onset', 'magnitude'}, or None if
        this episode was left undisturbed.
    """
    env.clear_disturbances()
    p = META_DISTURB_PROB if prob is None else float(prob)
    rng = np.random if rng is None else rng
    # Short-circuit at p <= 0 so a 0%-probability config consumes no randomness
    # and can never disturb an episode (asserted in the smoke test).
    if p <= 0.0 or float(rng.random()) >= p:
        return None
    kind = str(rng.choice(META_DISTURB_KINDS))
    lo, hi = META_DISTURB_MAG_RANGES[kind]
    magnitude = float(rng.uniform(lo, hi))
    f_lo, f_hi = META_DISTURB_ONSET_FRAC
    # Keep the onset strictly inside the episode: an onset at step 0 gives the
    # encoder no pre-change context to contrast against, and one at max_steps
    # never fires at all. Either would be a wasted episode.
    onset = int(round(float(rng.uniform(f_lo, f_hi)) * env.max_steps))
    onset = max(1, min(int(env.max_steps) - 1, onset))
    env.schedule_disturbance(kind, onset, magnitude)
    return {"kind": kind, "onset": onset, "magnitude": magnitude}


def collect_rollout(env, agent, replay, task_id, task, add_to_replay=True,
                    disturb=False):
    """Run one episode under a fixed task, updating the posterior online.

    Args:
        env: FermentationEnv.
        agent: PEARLAgent used to select actions.
        replay: MultiTaskReplayBuffer (may be None if add_to_replay is False).
        task_id: Integer task identifier for the replay buffer.
        task: Task dict to install via env.set_task.
        add_to_replay: If True, store transitions in the replay buffer.
        disturb: If True, this episode may carry a randomly drawn mid-episode
            disturbance (see maybe_schedule_disturbance). Set only by
            train_meta; train_baseline leaves it False so the baseline stays
            nominal-task-only with no disturbances — the isolating variable.

    Returns:
        A tuple (total_reward, total_violations, n_steps).
    """
    # Unconditional: an episode collected with disturb=False must be clean even
    # if the previous episode scheduled something on this same env instance.
    if disturb:
        maybe_schedule_disturbance(env)
    else:
        env.clear_disturbances()
    env.set_task(task)
    state = env.reset()
    agent.z_mode = "sample"      # training rollouts explore the posterior
    agent.clear_z()
    done = False
    total_reward = 0.0
    total_viol = 0
    steps = 0
    step_rewards = []
    while not done:
        safe_action, _ = agent.select_action(state, deterministic=False)
        next_state, reward, done, info = env.step(safe_action)
        r_sum = float(np.sum(reward))
        if add_to_replay and replay is not None:
            replay.add(task_id, state, safe_action, r_sum, next_state, done)
        agent.update_context(state, safe_action, r_sum, next_state)
        state = next_state
        total_reward += r_sum
        total_viol += int(info.get("violation_count", 0))
        step_rewards.append(r_sum)
        steps += 1
    if add_to_replay:
        # Fold this episode's rewards into the agent's reward-scale estimator
        # exactly ONCE each. This used to happen inside agent.update() on
        # replayed minibatches, which counted every transition thousands of
        # times and froze the reward scale within ~100 iterations (see
        # RunningMeanStd in agent_pearl.py). Gated on add_to_replay so
        # evaluation rollouts can never mutate the agent's learned scale.
        agent.observe_rewards(step_rewards)
    return total_reward, total_viol, steps


def rollout_deterministic(env, agent, task, z_mode="mean"):
    """Run one DETERMINISTIC episode without touching any training state.

    Used for the periodic held-out evaluation that drives checkpoint selection
    and early stopping. Deliberately does not write to the replay buffer and does
    not call ``observe_rewards``, so it cannot contaminate training. The
    environment's Lagrange multipliers are reset by ``env.reset()`` under
    RESET_LAMBDAS_PER_EPISODE, so they do not leak either way.

    Args:
        env: FermentationEnv.
        agent: PEARLAgent.
        task: Task dict to install.
        z_mode: Latent-draw mode for the meta agent
            ('mean'/'hold'/'sample'/'zero'); ignored by the baseline, whose z is
            always zero.

    Returns:
        A tuple (total_reward, total_violations).
    """
    prev_mode = agent.z_mode
    agent.z_mode = z_mode
    try:
        # The held-out eval shares its env with meta collection, which now may
        # schedule disturbances (see maybe_schedule_disturbance). Clear them so
        # the checkpoint-selection metric stays the clean, static-task measure it
        # has always been — otherwise a disturbance from the last training
        # episode would leak in and make held-out scores incomparable to the
        # earlier runs.
        env.clear_disturbances()
        env.set_task(task)
        state = env.reset()
        agent.clear_z()
        done = False
        total_reward = 0.0
        total_viol = 0
        while not done:
            safe_action, _ = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(safe_action)
            r_sum = float(np.sum(reward))
            agent.update_context(state, safe_action, r_sum, next_state)
            state = next_state
            total_reward += r_sum
            total_viol += int(info.get("violation_count", 0))
    finally:
        agent.z_mode = prev_mode
    return total_reward, total_viol


def heldout_score(env, agent, tasks, z_mode="mean"):
    """Mean deterministic violations over a held-out task list (lower is better).

    Args:
        env: FermentationEnv.
        agent: PEARLAgent.
        tasks: Held-out task dicts (must not be in the training pool).
        z_mode: Latent-draw mode for the meta agent.

    Returns:
        A tuple (mean_violations, mean_reward).
    """
    v, r = [], []
    for task in tasks:
        ep_r, ep_v = rollout_deterministic(env, agent, task, z_mode=z_mode)
        r.append(ep_r)
        v.append(ep_v)
    return float(np.mean(v)), float(np.mean(r))


# ─── Meta-training ────────────────────────────────────────────────────────────


def train_meta():
    """Meta-train the PEARL agent across the task distribution.

    Returns:
        The trained PEARLAgent (also checkpointed to META_PATH).
    """
    print(f"--- Meta-training PEARL (device: {device}, seed: {SEED}) ---")
    seed_everything(SEED)
    env = make_env()
    pool = make_task_pool(N_TRAIN_TASKS, seed=0)
    agent = PEARLAgent(env.stateDim, env.actionDim, n_reactors=env.n_reactors,
                       use_context=True, use_safeguard=True,
                       reward_scale=REWARD_SCALE,
                       normalize_rewards=NORMALIZE_REWARDS,
                       kl_lambda=META_KL_LAMBDA,
                       encoder_lr=META_ENCODER_LR,
                       target_entropy=META_TARGET_ENTROPY,
                       # Bound the online posterior to the same number of
                       # transitions the encoder is trained on, so the z the
                       # actor is deployed with matches the z it was trained
                       # with (see PEARLAgent's posterior_window docstring).
                       posterior_window=CONTEXT_SIZE,
                       # Shared with the baseline, so the comparison is unchanged.
                       alpha_min=ALPHA_MIN,
                       target_entropy_per_dim=TARGET_ENTROPY_PER_DIM)
    replay = MultiTaskReplayBuffer(env.stateDim, env.actionDim, env.n_reactors)

    # Held-out tasks: same prior as the training pool, disjoint draws. These
    # drive checkpoint selection and early stopping (see HELDOUT_EVAL_EVERY).
    heldout = make_task_pool(HELDOUT_EVAL_TASKS, seed=HELDOUT_TASK_SEED)

    log = {"iter": [], "avg_reward": [], "avg_violations": [], "critic_loss": [],
           "actor_loss": [], "kl": [], "alpha": [], "z_norm": [],
           "heldout_iter": [], "heldout_violations": [], "heldout_reward": []}
    reward_hist = []
    viol_hist = []

    # ── Warm-up: seed every task's replay bucket ──────────────────────────────
    # Fill all N_TRAIN_TASKS buckets so the readiness check (MIN_BUFFER
    # transitions per bucket) passes on iteration 1 instead of after dozens of
    # empty iterations.
    print(f"[Meta] Warm-up: {WARMUP_EPISODES_PER_TASK} episode(s) x "
          f"{N_TRAIN_TASKS} tasks "
          f"(disturbance rate {META_DISTURB_PROB:.0%}) ...")
    for tid in range(N_TRAIN_TASKS):
        for _ in range(WARMUP_EPISODES_PER_TASK):
            # disturb=True from the warm-up onward: the encoder's very first
            # gradient signal should already contain disturbed episodes,
            # otherwise the replay buffer starts out with the same
            # static-tasks-only distribution this fix exists to remove.
            r, v, _s = collect_rollout(env, agent, replay, tid, pool[tid],
                                       disturb=True)
            reward_hist.append(r); viol_hist.append(v)

    stopper = EarlyStopper(
        min_checks=META_EARLY_STOP_MIN_ITERS, patience=META_EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA, target=EARLY_STOP_TARGET_VIOL,
        target_patience=EARLY_STOP_TARGET_PATIENCE,
        diverge_factor=(EARLY_STOP_DIVERGE_FACTOR
                        if EARLY_STOP_DIVERGE_ENABLED else None),
        diverge_patience=EARLY_STOP_DIVERGE_PATIENCE,
        diverge_min_checks=EARLY_STOP_DIVERGE_MIN_CHECKS,
        zero_best_floor=EARLY_STOP_ZERO_BEST_FLOOR) \
        if EARLY_STOP_ENABLED else None
    best_viol = float("inf")
    best_reward = float("-inf")   # secondary key, used when violations tie at 0
    best_iter = None
    heldout_v = float("nan")

    pbar = tqdm.trange(1, META_ITERATIONS + 1)
    for it in pbar:
        # KL annealing: ramp agent.kl_lambda linearly from 0 to
        # META_KL_LAMBDA over KL_ANNEAL_ITERS outer iterations (see the
        # META_KL_LAMBDA / KL_ANNEAL_ITERS comments above for why a FIXED
        # kl_lambda from iteration 1 caused posterior collapse with the slow
        # encoder_lr). agent.kl_lambda is a plain attribute read fresh inside
        # update(), so mutating it here takes effect immediately.
        agent.kl_lambda = META_KL_LAMBDA * min(1.0, it / KL_ANNEAL_ITERS)

        # ── Collect ─────────────────────────────────────────────────────────
        for _ in range(COLLECT_EPISODES_PER_ITER):
            tid = int(np.random.randint(N_TRAIN_TASKS))
            r, v, _s = collect_rollout(env, agent, replay, tid, pool[tid],
                                       disturb=True)
            reward_hist.append(r); viol_hist.append(v)

        # ── Optimize ────────────────────────────────────────────────────────
        stats = None
        for _ in range(UPDATES_PER_ITER):
            stats = agent.update(replay, meta_batch=META_BATCH,
                                 batch_size=BATCH_SIZE, context_size=CONTEXT_SIZE,
                                 min_buffer=MIN_BUFFER)

        # NOTE: with META_DISTURB_PROB > 0 these rolling training means now pool
        # clean AND disturbed episodes, so they read worse than the pre-fix run's
        # curve by construction and are not directly comparable to it. The
        # comparable numbers are the held-out score below (always clean, see
        # rollout_deterministic) and the final eval table.
        avg_r = float(np.mean(reward_hist[-50:])) if reward_hist else 0.0
        avg_v = float(np.mean(viol_hist[-50:])) if viol_hist else 0.0

        # ── Held-out deterministic eval ─────────────────────────────────────
        # Checkpoint selection and early stopping run off THIS number, not the
        # rolling mean of stochastic training episodes. The old argmin-over-
        # rolling-training-mean selected on noise: it reported best 2.4 while the
        # very same checkpoint scored 24.7 at eval, and it always landed at iter
        # ~60, hiding the fact that the run then diverged 30x.
        fresh_heldout = False
        if it == 1 or it % HELDOUT_EVAL_EVERY == 0:
            heldout_v, heldout_r = heldout_score(
                env, agent, heldout, z_mode=HELDOUT_Z_MODE)
            log["heldout_iter"].append(it)
            log["heldout_violations"].append(heldout_v)
            log["heldout_reward"].append(heldout_r)
            fresh_heldout = True
            # TIE-BREAK ON REWARD. Held-out violations SATURATE at 0.00 here (they
            # are largely task-independent — see the inventory-constraint note in
            # the severity-sweep block), so a bare `<` on violations hands the
            # best-checkpoint slot to whichever snapshot reached 0.00 FIRST and
            # never revisits the decision. In the policy_n8_dist run that was
            # iteration 20 of a 2000-iteration budget: every downstream
            # diagnostic then ran on an essentially untrained encoder. Reward
            # still discriminates when violations are tied, so use it as the
            # secondary key.
            better = (heldout_v < best_viol
                      or (heldout_v <= best_viol and heldout_r > best_reward))
            if better:
                best_viol = heldout_v
                best_reward = heldout_r
                best_iter = it
                os.makedirs(MODEL_DIR, exist_ok=True)
                agent.save(BEST_META_PATH)

        if stats is not None:
            log["iter"].append(it)
            log["avg_reward"].append(avg_r)
            log["avg_violations"].append(avg_v)
            log["critic_loss"].append(stats["critic_loss"])
            log["actor_loss"].append(stats["actor_loss"])
            log["kl"].append(stats["kl"])
            log["alpha"].append(stats["alpha"])
            log["z_norm"].append(stats["z_norm"])
            # Violations are the true learning signal here: the reward is
            # dominated by a large constraint penalty, so watch Viol fall (from
            # ~590/episode for a random policy toward 0), not the raw reward.
            # |mu| (posterior MEAN magnitude, log key "z_norm" kept for
            # continuity) is watched alongside KL per the encoder diagnosis
            # (see META_KL_LAMBDA/KL_ANNEAL_ITERS above). CORRECTED from an
            # earlier version that tracked a SAMPLED z: a sample is noisy even
            # at full collapse (mu~=0, sigma~=1 gives E|sample|~=0.8), which
            # made a fully-dead encoder look "healthy". |mu| only stays
            # nonzero when the encoder is actually informative. KL is still
            # the primary collapse/divergence indicator (distance from the
            # prior directly); |mu| corroborates it. No "healthy band" is
            # established yet for |mu| under annealing -- this run is the
            # first with the fix, so treat its own early behavior as the new
            # reference point rather than comparing to old sampled-z numbers.
            pbar.set_postfix({
                "Rwd": f"{avg_r:.1f}",
                "Viol": f"{avg_v:.0f}",
                # hoViol is the held-out DETERMINISTIC violation count — the
                # only number here that checkpointing and early stopping act
                # on, and the only one comparable to the final eval table.
                "hoViol": f"{heldout_v:.1f}",
                "QL": f"{stats['critic_loss']:.3f}",
                "KL": f"{stats['kl']:.3f}",
                "|mu|": f"{stats['z_norm']:.3f}",
                "klW": f"{agent.kl_lambda:.3f}",
                "a": f"{stats['alpha']:.3f}",
                "rStd": f"{agent.ret_rms.std:.2f}" if agent.ret_rms else "-",
            })

        # Checkpoint and flush the log periodically so a run can be stopped
        # early and still leave a usable agent + a visible reward trend.
        if it % 100 == 0:
            os.makedirs(MODEL_DIR, exist_ok=True)
            agent.save(META_PATH)
            with open(os.path.join(MODEL_DIR, "pearl_meta_log.json"), "w") as f:
                json.dump(log, f)

        # Only consult the stopper when a FRESH held-out number exists, so its
        # patience counters are in units of held-out checks (see EarlyStopper).
        if stopper is not None and fresh_heldout:
            stop, reason = stopper.should_stop(it, heldout_v)
            if stop:
                pbar.close()
                print(f"\n[Meta] Early stop at iter {it}/{META_ITERATIONS}: {reason}")
                break

    agent.save(META_PATH)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "pearl_meta_log.json"), "w") as f:
        json.dump(log, f)
    final_v = log["avg_violations"][-1] if log["avg_violations"] else float("nan")
    print(f"[Meta] Saved final PEARL agent (iter {log['iter'][-1] if log['iter'] else 0}, "
          f"train_avg_violations={final_v:.1f}, "
          f"heldout_violations={heldout_v:.1f}) -> {META_PATH}")
    print(f"[Meta] Best HELD-OUT snapshot: violations={best_viol:.1f} at iter "
          f"{best_iter} -> {BEST_META_PATH}")
    return agent


def train_baseline():
    """Train the non-adaptive Safe-RL baseline (z=0, nominal task only).

    Returns:
        The trained baseline PEARLAgent (also checkpointed to BASE_PATH).
    """
    print(f"--- Training non-adaptive Safe-RL baseline (device: {device}, "
          f"seed: {SEED}) ---")
    seed_everything(SEED)
    env = make_env()
    nominal = nominal_task()
    agent = PEARLAgent(env.stateDim, env.actionDim, n_reactors=env.n_reactors,
                       use_context=False, use_safeguard=True,
                       reward_scale=REWARD_SCALE,
                       normalize_rewards=NORMALIZE_REWARDS,
                       # Identical exploration schedule to the meta agent, so
                       # meta-learning stays the only experimental variable.
                       alpha_min=ALPHA_MIN,
                       target_entropy_per_dim=TARGET_ENTROPY_PER_DIM)
    replay = MultiTaskReplayBuffer(env.stateDim, env.actionDim, env.n_reactors)

    reward_hist = []
    viol_hist = []

    # Warm-up the single nominal bucket so updates begin on iteration 1.
    print(f"[Baseline] Warm-up: {WARMUP_EPISODES_PER_TASK} episode(s) ...")
    for _ in range(WARMUP_EPISODES_PER_TASK):
        r, v, _s = collect_rollout(env, agent, replay, 0, nominal)
        reward_hist.append(r); viol_hist.append(v)

    # Held-out set for the baseline: THE SAME 8 TASKS THE META AGENT IS SELECTED
    # ON, not the nominal task it trains on.
    #
    # This is a fairness fix, and it mattered. When the baseline was selected on a
    # single nominal episode it reached 0.0 held-out violations at iteration 80 —
    # and that very checkpoint scored 48.15 violations on the 20-task evaluation
    # set. Selecting the snapshot that best fits nominal and then testing it on
    # varied tasks handicaps the baseline through the selection rule rather than
    # through the experimental variable, which would invalidate the comparison.
    #
    # The baseline still TRAINS on the nominal task only — that remains the sole
    # experimental difference. It is only the checkpoint-selection and
    # early-stopping metric that is now identical across the two agents, which is
    # what makes "meta beats baseline" a statement about training rather than
    # about model selection.
    heldout = make_task_pool(HELDOUT_EVAL_TASKS, seed=HELDOUT_TASK_SEED)

    stopper = EarlyStopper(
        min_checks=EARLY_STOP_MIN_ITERS, patience=EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA, target=EARLY_STOP_TARGET_VIOL,
        target_patience=EARLY_STOP_TARGET_PATIENCE,
        diverge_factor=(EARLY_STOP_DIVERGE_FACTOR
                        if EARLY_STOP_DIVERGE_ENABLED else None),
        diverge_patience=EARLY_STOP_DIVERGE_PATIENCE,
        diverge_min_checks=EARLY_STOP_DIVERGE_MIN_CHECKS,
        zero_best_floor=EARLY_STOP_ZERO_BEST_FLOOR) \
        if EARLY_STOP_ENABLED else None
    best_viol = float("inf")
    best_reward = float("-inf")   # secondary key, used when violations tie at 0
    best_iter = None
    heldout_v = float("nan")

    pbar = tqdm.trange(1, META_ITERATIONS + 1)
    for it in pbar:
        for _ in range(COLLECT_EPISODES_PER_ITER):
            # Single nominal task -> a single replay bucket (task id 0).
            r, v, _s = collect_rollout(env, agent, replay, 0, nominal)
            reward_hist.append(r); viol_hist.append(v)
        stats = None
        for _ in range(UPDATES_PER_ITER):
            stats = agent.update(replay, meta_batch=1,
                                 batch_size=BATCH_SIZE, context_size=CONTEXT_SIZE,
                                 min_buffer=MIN_BUFFER)
        avg_r = float(np.mean(reward_hist[-50:])) if reward_hist else 0.0
        avg_v = float(np.mean(viol_hist[-50:])) if viol_hist else 0.0

        # Checkpoint on the DETERMINISTIC held-out number, not the rolling mean
        # of stochastic training episodes (see the meta loop for why).
        fresh_heldout = False
        if it == 1 or it % HELDOUT_EVAL_EVERY == 0:
            # heldout_r was previously discarded as `_heldout_r`; it is now the
            # tie-break key for checkpoint selection, so it must be bound.
            heldout_v, heldout_r = heldout_score(env, agent, heldout)
            fresh_heldout = True
            # Same reward tie-break as train_meta — see the note there. The
            # baseline's held-out violations saturate at 0.00 even more readily,
            # so without this its "best" checkpoint is also an early snapshot.
            better = (heldout_v < best_viol
                      or (heldout_v <= best_viol and heldout_r > best_reward))
            if better:
                best_viol = heldout_v
                best_reward = heldout_r
                best_iter = it
                os.makedirs(MODEL_DIR, exist_ok=True)
                agent.save(BEST_BASE_PATH)

        if stats is not None:
            pbar.set_postfix({"Rwd": f"{avg_r:.1f}", "Viol": f"{avg_v:.0f}",
                              "hoViol": f"{heldout_v:.1f}",
                              "QL": f"{stats['critic_loss']:.3f}",
                              "a": f"{stats['alpha']:.3f}",
                              "rStd": f"{agent.ret_rms.std:.2f}"
                                      if agent.ret_rms else "-"})
        if it % 200 == 0:
            agent.save(BASE_PATH)

        if stopper is not None and fresh_heldout:
            stop, reason = stopper.should_stop(it, heldout_v)
            if stop:
                pbar.close()
                print(f"\n[Baseline] Early stop at iter {it}/{META_ITERATIONS}: {reason}")
                break
    agent.save(BASE_PATH)
    print(f"[Baseline] Saved final agent (iter {it}, "
          f"train_avg_violations={avg_v:.1f}, "
          f"heldout_violations={heldout_v:.1f}) -> {BASE_PATH}")
    print(f"[Baseline] Best HELD-OUT snapshot: violations={best_viol:.1f} at iter "
          f"{best_iter} -> {BEST_BASE_PATH}")
    return agent


# ─── Evaluation ───────────────────────────────────────────────────────────────


def _recovery_metrics(reward_traj, viol_traj, onset_outer, win=10):
    """Summarise the disturbance response from mean per-step trajectories.

    Rather than requiring a return to the (possibly no-longer-achievable)
    pre-disturbance reward, the headline comparison is the post-onset mean
    reward and violation rate — meta should hold higher reward and fewer
    violations than the non-adaptive baseline.  ``recovery_steps`` (outer steps
    for smoothed reward to climb back to the pre-onset level) is reported as a
    secondary adaptation-speed signal.

    Args:
        reward_traj: Mean per-outer-step reward, shape [T].
        viol_traj: Mean per-outer-step violation count, shape [T].
        onset_outer: Disturbance onset in outer steps.
        win: Pre-onset averaging window (outer steps).

    Returns:
        Dict of pre/post reward and violation summaries plus recovery_steps.
    """
    n = len(reward_traj)
    onset_outer = int(min(max(onset_outer, 1), n - 1))
    pre = reward_traj[max(0, onset_outer - win):onset_outer]
    post = reward_traj[onset_outer:]
    pre_r = float(np.nanmean(pre)) if len(pre) else float("nan")

    # Smoothed post reward for the recovery test.
    if len(post):
        k = min(5, len(post))
        post_sm = np.convolve(np.nan_to_num(post), np.ones(k) / k, mode="same")
    else:
        post_sm = post
    tol = 0.05 * abs(pre_r) + 1e-6
    recovery = None
    for i, v in enumerate(post_sm):
        if v >= pre_r - tol:
            recovery = int(i)
            break

    pre_v = float(np.nanmean(viol_traj[max(0, onset_outer - win):onset_outer])) \
        if onset_outer > 0 else float("nan")
    post_v = float(np.nanmean(viol_traj[onset_outer:])) if onset_outer < n else float("nan")
    return {
        "pre_reward": pre_r,
        "post_mean_reward": float(np.nanmean(post)) if len(post) else float("nan"),
        "post_min_reward": float(np.nanmin(post)) if len(post) else float("nan"),
        "recovery_steps": recovery,
        "pre_viol_rate": pre_v,
        "post_viol_rate": post_v,
    }


def _eval_agent_on_tasks(env, agent, tasks, disturbance=None, z_mode=None):
    """Evaluate an agent over a list of tasks, optionally with a disturbance.

    Records per-outer-step reward/violation trajectories (averaged across tasks)
    so the disturbance response can be plotted and summarised.

    Args:
        env: FermentationEnv.
        agent: Trained PEARLAgent.
        tasks: List of task dicts.
        disturbance: Optional dict {kind, onset, magnitude} scheduled mid-episode.
        z_mode: How the meta agent draws z ('mean'/'hold'/'sample'/'zero');
            defaults to EVAL_Z_MODE. 'zero' is the ablation arm — the trained
            meta agent with its latent switched off, which is what isolates the
            contribution of task inference from that of multi-task training.
            Ignored by the baseline (z is always zero there). With
            'sample' the meta agent's policy is NOT deterministic even under
            ``deterministic=True``, because z is redrawn every control step —
            that was the old behaviour and it is not comparable to the
            baseline's genuinely deterministic rollout.

    Returns:
        Dict of aggregate metrics, mean per-step trajectories, and (when a
        disturbance is given) recovery metrics.
    """
    agent.z_mode = z_mode if z_mode is not None else EVAL_Z_MODE
    rewards, viols, feasible = [], [], []
    reward_trajs, viol_trajs = [], []
    # Productivity instrumentation. WITHOUT THIS THE HEADLINE METRICS ARE
    # UNFALSIFIABLE: reward here is penalty-dominated (the Lagrangian term can
    # reach 10.0 per reactor per step while the productivity term maxes out
    # around 0.015), so "produce nothing, violate nothing" scores close to the
    # optimum. An agent reporting 0.00 violations and a reward just above zero is
    # indistinguishable from an agent that has learned to sit still — unless
    # ethanol output and actuator activity are measured directly.
    ethanol, feed_used, act_means = [], [], []
    # End-of-episode posterior mean per task. The across-task spread of this is
    # the most direct possible test of whether the encoder distinguishes tasks:
    # if z_mu is nearly the same vector for every task, the latent cannot be
    # carrying task identity no matter what the ablation deltas look like, and no
    # amount of policy tuning will produce adaptation.
    z_mus = []
    # End-of-episode posterior VARIANCE per task, kept alongside the mean so the
    # across-task spread of z_mu can be read as a signal-to-noise ratio rather
    # than an absolute number: a spread of 0.05 means something very different
    # when the posterior's own width is 0.02 than when it is 1.0 (the prior).
    z_sigma_sqs = []
    for task in tasks:
        env.clear_disturbances()
        if disturbance is not None:
            env.schedule_disturbance(
                disturbance["kind"], disturbance["onset"],
                disturbance["magnitude"])
        env.set_task(task)
        state = env.reset()
        agent.clear_z()
        done = False
        ep_r = 0.0
        ep_v = 0
        ep_eth = 0.0
        ep_acts = []
        r_traj, v_traj = [], []
        while not done:
            safe_action, _ = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(safe_action)
            r_sum = float(np.sum(reward))
            agent.update_context(state, safe_action, r_sum, next_state)
            state = next_state
            v_step = int(info.get("violation_count", 0))
            ep_r += r_sum
            ep_v += v_step
            # eth_harvested is already accumulated across the inner ticks of this
            # outer step by the env, so summing it over outer steps gives total
            # episode ethanol.
            ep_eth += float(info.get("eth_harvested", 0.0))
            # actual_a_reshaped is the POST-RESCALE action in [0, 1]; columns are
            # [time_mult, LF feed, C5 feed, outflow]. A do-nothing policy sits at
            # ~0.0 on columns 1-3.
            act = info.get("actual_a_reshaped")
            if act is not None:
                ep_acts.append(np.asarray(act, dtype=np.float64)[:, 1:4].mean(axis=0))
            r_traj.append(r_sum)
            v_traj.append(v_step)
        rewards.append(ep_r)
        viols.append(ep_v)
        ethanol.append(ep_eth)
        z_mus.append(agent.z_mu.detach().cpu().numpy().copy())
        z_sigma_sqs.append(agent.z_sigma_sq.detach().cpu().numpy().copy())
        # Feedstock actually drawn down over the episode, read off the env's
        # remaining-stock counters (no env modification needed).
        lf_used = float(getattr(env, "initial_lf", 0.0) - getattr(env, "total_lf", 0.0))
        c5_used = float(getattr(env, "initial_c5", 0.0) - getattr(env, "total_c5", 0.0))
        feed_used.append(lf_used + c5_used)
        act_means.append(np.mean(np.stack(ep_acts), axis=0) if ep_acts
                         else np.full(3, np.nan))
        feasible.append(1.0 if env.time_step_count >= env.max_steps else 0.0)
        reward_trajs.append(np.asarray(r_traj, dtype=np.float64))
        viol_trajs.append(np.asarray(v_traj, dtype=np.float64))
    env.clear_disturbances()

    # Aggregate variable-length trajectories: pad to max length with NaN, nanmean.
    L = max(len(t) for t in reward_trajs)

    def _mean_traj(trajs):
        M = np.full((len(trajs), L), np.nan)
        for i, t in enumerate(trajs):
            M[i, :len(t)] = t
        return np.nanmean(M, axis=0)

    mean_r = _mean_traj(reward_trajs)
    mean_v = _mean_traj(viol_trajs)

    out = {
        "z_mode": agent.z_mode if agent.use_context else "n/a (z=0)",
        "reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "violations": float(np.mean(viols)),
        "violations_std": float(np.std(viols)),
        "feasible_rate": float(np.mean(feasible)),
        # ── Productivity: is this agent actually running the plant? ──────────
        "ethanol": float(np.mean(ethanol)),
        "ethanol_std": float(np.std(ethanol)),
        "feed_used": float(np.mean(feed_used)),
        "act_lf": float(np.nanmean([a[0] for a in act_means])),
        "act_c5": float(np.nanmean([a[1] for a in act_means])),
        "act_out": float(np.nanmean([a[2] for a in act_means])),
        # Across-task spread of the end-of-episode posterior mean, averaged over
        # latent dims, alongside the mean |z_mu| for scale. A spread near zero
        # means the encoder returns the same latent for every task — i.e. it is
        # not identifying anything, whatever the ablation deltas say.
        "z_spread": float(np.mean(np.std(np.stack(z_mus), axis=0))),
        "z_abs_mean": float(np.mean(np.abs(np.stack(z_mus)))),
        # Mean posterior std, i.e. the encoder's OWN uncertainty about z at the
        # end of an episode. z_spread / z_post_std is the separability SNR: below
        # ~1 the task-to-task movement of the posterior mean is smaller than the
        # width of the posterior it is drawn from, so z cannot be resolving task
        # identity even if z_spread is nonzero in absolute terms.
        "z_post_std": float(np.mean(np.sqrt(np.stack(z_sigma_sqs)))),
        # ── Per-episode arrays: REQUIRED for honest arm-vs-arm comparison ─────
        # Every arm in a sweep is evaluated on the SAME task pool in the SAME
        # order (see severity_sweep's single `tasks` list), so episode i of arm A
        # and episode i of arm B are the same task and the difference between
        # them is PAIRED. Reporting only means and per-arm stds throws that
        # pairing away and compares two noisy aggregates instead, which is how a
        # 0.7-violation INFER delta ends up printed next to a per-arm std of 8
        # with no indication that the delta is inside the noise. With these
        # arrays the decomposition can report a paired CI (see _paired_delta).
        "viol_per_ep": [float(v) for v in viols],
        "reward_per_ep": [float(r) for r in rewards],
        "ethanol_per_ep": [float(e) for e in ethanol],
        # Per-task posterior mean, so a diagnostic can recompute between- vs
        # within-task structure without re-running the rollouts.
        "z_mu_per_ep": [[float(x) for x in zm] for zm in z_mus],
        "reward_traj": mean_r.tolist(),
        "viol_traj": mean_v.tolist(),
    }
    if disturbance is not None:
        onset_outer = int(disturbance["onset"] // env.control_action_interval)
        out["onset_outer"] = onset_outer
        out["recovery"] = _recovery_metrics(mean_r, mean_v, onset_outer)
    return out


def _plot_comparison(results):
    """Render meta-vs-baseline comparison figures from an eval results dict.

    Produces two PNGs under MODEL_DIR:
      - pearl_compare_bars.png: grouped bars of mean reward and mean violations
        per scenario (with reward std error bars).
      - pearl_disturbance_response.png: per-outer-step reward trajectories for
        the three disturbance scenarios, with the onset marked, showing the
        non-adaptive baseline collapsing while the meta agent adapts.

    Args:
        results: The dict returned by evaluate().
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"[Plot] matplotlib unavailable ({exc}); skipping figures.")
        return

    names = list(results.keys())
    x = np.arange(len(names))
    w = 0.27
    m_r = [results[n]["meta"]["reward"] for n in names]
    n_r = [results[n]["meta_noz"]["reward"] for n in names]
    b_r = [results[n]["baseline"]["reward"] for n in names]
    m_re = [results[n]["meta"]["reward_std"] for n in names]
    n_re = [results[n]["meta_noz"]["reward_std"] for n in names]
    b_re = [results[n]["baseline"]["reward_std"] for n in names]
    m_v = [results[n]["meta"]["violations"] for n in names]
    n_v = [results[n]["meta_noz"]["violations"] for n in names]
    b_v = [results[n]["baseline"]["violations"] for n in names]
    labels = [n.replace("disturb_", "Δ ").replace("_", " ") for n in names]

    # Three bars, because the middle one is what makes the comparison
    # interpretable: meta vs meta-ablated isolates task inference, and
    # meta-ablated vs baseline isolates multi-task training.
    fig, (axr, axv) = plt.subplots(1, 2, figsize=(15, 6))
    axr.bar(x - w, m_r, w, yerr=m_re, capsize=3, label="meta (PEARL)", color="steelblue")
    axr.bar(x, n_r, w, yerr=n_re, capsize=3, label="meta, z ablated to 0",
            color="mediumpurple")
    axr.bar(x + w, b_r, w, yerr=b_re, capsize=3, label="baseline (safe RL)", color="tomato")
    axr.set_ylabel("Episode reward"); axr.set_title("Reward — higher is better")
    axr.set_xticks(x); axr.set_xticklabels(labels, rotation=30, ha="right")
    axr.legend(); axr.grid(True, axis="y", alpha=0.3)

    axv.bar(x - w, m_v, w, label="meta (PEARL)", color="steelblue")
    axv.bar(x, n_v, w, label="meta, z ablated to 0", color="mediumpurple")
    axv.bar(x + w, b_v, w, label="baseline (safe RL)", color="tomato")
    axv.set_ylabel("Constraint-violation steps"); axv.set_title("Violations — lower is better")
    axv.set_xticks(x); axv.set_xticklabels(labels, rotation=30, ha="right")
    axv.legend(); axv.grid(True, axis="y", alpha=0.3)
    plt.suptitle("Meta-RL vs non-adaptive Safe-RL under uncertainty", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "pearl_compare_bars.png"), dpi=150)
    plt.close()

    dist = [n for n in names if n.startswith("disturb_")]
    if dist:
        fig, axes = plt.subplots(1, len(dist), figsize=(6 * len(dist), 5), squeeze=False)
        for ax, n in zip(axes[0], dist):
            mr = np.array(results[n]["meta"]["reward_traj"], dtype=float)
            nr = np.array(results[n]["meta_noz"]["reward_traj"], dtype=float)
            br = np.array(results[n]["baseline"]["reward_traj"], dtype=float)
            onset = results[n]["meta"].get("onset_outer")
            ax.plot(mr, label="meta (PEARL)", color="steelblue")
            ax.plot(nr, label="meta, z ablated to 0", color="mediumpurple")
            ax.plot(br, label="baseline (safe RL)", color="tomato")
            if onset is not None:
                ax.axvline(onset, color="k", ls="--", lw=1, label="disturbance")
            ax.set_title(n.replace("disturb_", "").replace("_", " "))
            ax.set_xlabel("control step"); ax.set_ylabel("reward / step")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.suptitle("Disturbance response (per-step reward)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "pearl_disturbance_response.png"), dpi=150)
        plt.close()
    print(f"[Plot] Saved comparison figures -> {MODEL_DIR}/pearl_compare_bars.png, "
          f"{MODEL_DIR}/pearl_disturbance_response.png")


def evaluate():
    """Compare the meta agent against the non-adaptive Safe-RL baseline.

    Runs both agents on in-distribution tasks, genuinely out-of-distribution
    (widened-prior) tasks, and the three mid-episode disturbance stress
    scenarios, then prints a side-by-side table, a per-scenario verdict, and the
    disturbance recovery metrics, saves policy/pearl_eval.json, and renders
    comparison figures.

    Returns:
        The results dict.
    """
    print(f"--- Evaluating PEARL vs non-adaptive baseline (device: {device}) ---")
    env = make_env()

    # posterior_window MUST match training (see PEARLAgent.posterior_window):
    # an unbounded online posterior at eval time deploys a much sharper z than
    # the actor was ever trained on.
    meta = PEARLAgent(env.stateDim, env.actionDim, n_reactors=env.n_reactors,
                      use_context=True, use_safeguard=True,
                      posterior_window=CONTEXT_SIZE)
    base = PEARLAgent(env.stateDim, env.actionDim, n_reactors=env.n_reactors,
                      use_context=False, use_safeguard=True)

    def _load_preferring_best(agent, best_path, final_path, label):
        # Training is not monotonic here (see BEST_META_PATH comment near the
        # top): prefer the best-by-violations snapshot over the final one,
        # since the final iteration can be a regression from an earlier,
        # genuinely better policy.
        if os.path.exists(best_path):
            agent.load(best_path)
            print(f"[Eval] Loaded {label} from BEST checkpoint -> {best_path}")
        elif os.path.exists(final_path):
            agent.load(final_path)
            print(f"[Eval] {best_path} not found; loaded {label} from final "
                  f"checkpoint -> {final_path}")
        else:
            print(f"WARNING: neither {best_path} nor {final_path} found; "
                  f"{label} agent is untrained.")

    _load_preferring_best(meta, BEST_META_PATH, META_PATH, "meta")
    _load_preferring_best(base, BEST_BASE_PATH, BASE_PATH, "baseline")

    # In-distribution tasks (same prior as training, unseen draws) and genuinely
    # out-of-distribution tasks (widened priors, extending beyond training ranges).
    id_tasks = make_task_pool(EVAL_EPISODES, seed=12345, ood=False)
    ood_tasks = make_task_pool(EVAL_EPISODES, seed=99999, ood=True)

    onset = env.max_steps // 2  # disturbance strikes mid-episode
    scenarios = {
        "in_distribution": {"tasks": id_tasks, "dist": None},
        "out_of_distribution": {"tasks": ood_tasks, "dist": None},
        "disturb_inlet_step": {
            "tasks": id_tasks,
            "dist": {"kind": "inlet_step", "onset": onset, "magnitude": 0.7}},
        "disturb_enzyme_drop": {
            "tasks": id_tasks,
            "dist": {"kind": "enzyme_drop", "onset": onset, "magnitude": 0.5}},
        "disturb_mass_drift": {
            "tasks": id_tasks,
            "dist": {"kind": "mass_drift", "onset": onset, "magnitude": -0.005}},
    }

    print(f"[Eval] meta z_mode = '{EVAL_Z_MODE}' (baseline z == 0 always)")
    results = {}
    for name, spec in scenarios.items():
        results[name] = {
            "meta": _eval_agent_on_tasks(env, meta, spec["tasks"], spec["dist"]),
            # ABLATION: the SAME trained meta agent, re-run with z forced to
            # zero. This is the decisive test of the project's central claim.
            # meta vs meta_noz isolates the contribution of TASK INFERENCE;
            # meta_noz vs baseline isolates the contribution of MULTI-TASK
            # TRAINING (both then run with an identical z == 0). If
            # meta ~= meta_noz, the advantage over baseline is not adaptation —
            # it is just that training across 40 tasks yields a more robust
            # controller than training on one, which is a different claim.
            "meta_noz": _eval_agent_on_tasks(env, meta, spec["tasks"],
                                             spec["dist"], z_mode="zero"),
            "baseline": _eval_agent_on_tasks(env, base, spec["tasks"], spec["dist"]),
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print(f"  {'Scenario':<22}{'Agent':<10}{'Reward':>14}{'Viol':>10}{'Feasible':>12}")
    print("  " + "-" * 80)
    for name, res in results.items():
        for who in ("meta", "meta_noz", "baseline"):
            r = res[who]
            print(f"  {name if who=='meta' else '':<22}{who:<10}"
                  f"{r['reward']:>9.3f}±{r['reward_std']:<4.2f}"
                  f"{r['violations']:>10.2f}{r['feasible_rate']:>12.1%}")
        print("  " + "-" * 80)
    print("=" * 84)

    # ── Productivity table (the sanity check on the verdict) ──────────────────
    # A zero-violation agent that harvests no ethanol and leaves its feedstock
    # untouched has not solved the control problem — it has found the degenerate
    # optimum of a penalty-dominated reward. Read this table BEFORE the verdict.
    print("\n  PRODUCTIVITY (is the plant actually running?):")
    print(f"    {'Scenario':<22}{'Agent':<10}{'Ethanol':>13}{'Feed used':>12}"
          f"{'LF act':>9}{'C5 act':>9}{'Out act':>9}")
    for name, res in results.items():
        for who in ("meta", "meta_noz", "baseline"):
            r = res[who]
            print(f"    {name if who == 'meta' else '':<22}{who:<10}"
                  f"{r['ethanol']:>13.4g}{r['feed_used']:>12.4g}"
                  f"{r['act_lf']:>9.3f}{r['act_c5']:>9.3f}{r['act_out']:>9.3f}")
    print("    Actions are post-rescale in [0, 1]; ~0.0 on LF/C5/Out means the\n"
          "    agent is not feeding or draining, i.e. it is doing nothing. Zero\n"
          "    violations with ~zero ethanol is a degenerate result, not a win.\n"
          "    Also check whether the actuator columns VARY across scenarios: a\n"
          "    controller whose actions are constant across in-distribution, OOD\n"
          "    and every disturbance is not adapting, whatever its score.")

    # ── Ablation verdict: is task inference doing the work? ────────────────────
    print("\n  ABLATION (does the latent z contribute anything?):")
    print(f"    {'Scenario':<22}{'meta viol':>11}{'noz viol':>10}{'base viol':>11}"
          f"{'meta rwd':>10}{'noz rwd':>10}{'base rwd':>10}")
    d_infer, d_multitask = [], []
    for name, res in results.items():
        m, n, b = res["meta"], res["meta_noz"], res["baseline"]
        d_infer.append(n["violations"] - m["violations"])
        d_multitask.append(b["violations"] - n["violations"])
        print(f"    {name:<22}{m['violations']:>11.2f}{n['violations']:>10.2f}"
              f"{b['violations']:>11.2f}{m['reward']:>10.3f}{n['reward']:>10.3f}"
              f"{b['reward']:>10.3f}")
    # Per-scenario PAIRED deltas with 95% CIs. The arms share one task pool per
    # scenario, so these are paired differences and the CI is the scale on which
    # to judge them; the pooled means printed below are a summary, NOT a test.
    print(f"\n    {'Scenario':<22}{'INFER (noz-meta)':>30}"
          f"{'MULTI (base-noz)':>30}")
    for name, res in results.items():
        pi = _paired_delta(res["meta_noz"].get("viol_per_ep", []),
                           res["meta"].get("viol_per_ep", []))
        pm = _paired_delta(res["baseline"].get("viol_per_ep", []),
                           res["meta_noz"].get("viol_per_ep", []))
        print(f"    {name:<22}{_fmt_delta(pi):>30}{_fmt_delta(pm):>30}")
    print("    'ns' = CI includes zero, i.e. consistent with no effect.\n"
          f"    'tiny(x%)' = statistically real but under "
          f"{PRACTICAL_MIN_REL_EFFECT:.0%} of the arm's own value\n"
          "    (meta and meta_noz share weights, so their paired CI is very\n"
          "    tight and trivial gaps DO reach significance) — also null.\n"
          "    Neither is evidence that task inference helps a little; run\n"
          "    `z_informativeness` to check whether z is informative at all.")

    print(f"\n    mean violations attributable to TASK INFERENCE  "
          f"(noz - meta): {np.mean(d_infer):+7.2f}")
    print(f"    mean violations attributable to MULTI-TASK TRAINING "
          f"(base - noz): {np.mean(d_multitask):+7.2f}")
    mi, mm = float(np.mean(d_infer)), float(np.mean(d_multitask))
    # A share-of-total is only meaningful when both components push the SAME way.
    # With opposite signs there is no "gap" to apportion, and printing a
    # percentage there produced a meaningless "-23.4%".
    if mi > 0 and mm > 0:
        print(f"    -> task inference accounts for "
              f"{100.0 * mi / (mi + mm):.1f}% of the meta-baseline gap")
    elif mi + mm < 0:
        print("    -> NO meta-baseline gap to decompose: the baseline is better "
              "overall.\n       Task inference helping (+) here only means meta "
              "beats its own ablation,\n       not that it beats the baseline.")
    else:
        print("    -> components have opposite signs; no share-of-total is "
              "meaningful.")
    print("    The first number is the project's actual claim. If it is ~0 while\n"
          "    the second is large, the result is real but it is NOT about\n"
          "    adaptation, and the framing has to change.")

    # Upstream of the ablation: does the encoder even tell the tasks apart?
    print("\n  LATENT DIAGNOSTIC (does the encoder distinguish tasks?):")
    print(f"    {'Scenario':<22}{'z spread across tasks':>24}{'mean |z_mu|':>14}")
    for name, res in results.items():
        m = res["meta"]
        print(f"    {name:<22}{m['z_spread']:>24.5f}{m['z_abs_mean']:>14.5f}")
    print("    z spread is the std of the end-of-episode posterior mean ACROSS\n"
          "    tasks. Near zero means one latent for every task — the encoder is\n"
          "    not identifying the task, so adaptation is impossible upstream of\n"
          "    any policy issue. Compare against mean |z_mu| for scale.")

    # ── Per-scenario verdict (does meta beat the non-adaptive baseline?) ───────
    print("\n  VERDICT (meta − baseline):")
    for name, res in results.items():
        d_r = res["meta"]["reward"] - res["baseline"]["reward"]
        d_v = res["baseline"]["violations"] - res["meta"]["violations"]
        win = "meta better" if (d_r > 0 and d_v >= 0) else (
            "mixed" if (d_r > 0 or d_v > 0) else "baseline better")
        # Flag the degenerate case explicitly rather than letting a do-nothing
        # policy be reported as a win: fewer violations is only meaningful if the
        # agent is producing on the same order as its comparator.
        m_eth = res["meta"]["ethanol"]
        b_eth = res["baseline"]["ethanol"]
        if win == "meta better" and m_eth < 0.5 * max(b_eth, 1e-12):
            win += "  [!] but meta harvests <50% of baseline's ethanol"
        print(f"    {name:<22} Δreward={d_r:+8.3f}  fewer_viol={d_v:+7.2f}  -> {win}")

    # ── Disturbance recovery metrics ───────────────────────────────────────────
    print("\n  DISTURBANCE RESPONSE (post-onset):")
    print(f"    {'Scenario':<22}{'Agent':<10}{'post_reward':>13}{'post_viol':>12}{'recov_steps':>13}")
    for name, res in results.items():
        if "recovery" not in res["meta"]:
            continue
        for who in ("meta", "meta_noz", "baseline"):
            rec = res[who]["recovery"]
            rs = rec["recovery_steps"]
            rs_str = "n/a" if rs is None else str(rs)
            print(f"    {name if who=='meta' else '':<22}{who:<10}"
                  f"{rec['post_mean_reward']:>13.3f}{rec['post_viol_rate']:>12.3f}"
                  f"{rs_str:>13}")
    print()

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "pearl_eval.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Eval] Saved results -> {os.path.join(MODEL_DIR, 'pearl_eval.json')}")
    _plot_comparison(results)
    return results


def eval_z_probe():
    """Measure how much of the meta-vs-baseline gap is just z sampling noise.

    NO TRAINING REQUIRED — this loads the EXISTING best checkpoints and re-runs
    evaluation with the meta agent's latent draw set to each of the three modes:

        sample : redraw z ~ q(z|c) every control step (the old behaviour)
        mean   : z = posterior mean, no sampling
        hold   : draw once after z_hold_after steps, then freeze for the episode

    The baseline (z == 0) is unaffected and is printed once as the reference.

    WHY THIS IS THE FIRST EXPERIMENT TO RUN: the encoder in the last sweep was
    fully collapsed to the prior (KL ~ 1e-3, |mu| ~ 0.01), which means z carried
    no task information whatsoever. Under 'sample' the meta policy therefore had
    a 5-dim noise vector re-drawn into its conditioning input 120 times per
    episode while the baseline had a constant zero — so the comparison partly
    measured "clean controller vs jittered controller" rather than
    "adaptive vs non-adaptive". If meta's violations drop substantially under
    'mean'/'hold', that confound was real and any retraining must fix the z
    handling first. If they barely move, the gap is genuinely about policy
    quality and the reward-scale / action-mask fixes are where the leverage is.

    Returns:
        Dict mapping z-mode (plus 'baseline') -> {scenario -> metrics}.
    """
    print(f"--- z-mode probe (device: {device}) ---")
    env = make_env()
    meta = PEARLAgent(env.stateDim, env.actionDim, n_reactors=env.n_reactors,
                      use_context=True, use_safeguard=True,
                      posterior_window=CONTEXT_SIZE)
    base = PEARLAgent(env.stateDim, env.actionDim, n_reactors=env.n_reactors,
                      use_context=False, use_safeguard=True)
    for agent, best, final, label in (
            (meta, BEST_META_PATH, META_PATH, "meta"),
            (base, BEST_BASE_PATH, BASE_PATH, "baseline")):
        path = best if os.path.exists(best) else final
        if not os.path.exists(path):
            raise SystemExit(f"No checkpoint for {label} ({best} / {final}). "
                             f"Run train_meta/train_baseline first.")
        agent.load(path)
        print(f"[Probe] loaded {label} -> {path}")

    id_tasks = make_task_pool(EVAL_EPISODES, seed=12345, ood=False)
    ood_tasks = make_task_pool(EVAL_EPISODES, seed=99999, ood=True)
    onset = env.max_steps // 2
    scenarios = {
        "in_distribution": (id_tasks, None),
        "out_of_distribution": (ood_tasks, None),
        "disturb_mass_drift": (
            id_tasks,
            {"kind": "mass_drift", "onset": onset, "magnitude": -0.005}),
    }

    out = {}
    for mode_name in ("sample", "mean", "hold"):
        out[mode_name] = {
            sc: _eval_agent_on_tasks(env, meta, tasks, dist, z_mode=mode_name)
            for sc, (tasks, dist) in scenarios.items()
        }
    out["baseline"] = {
        sc: _eval_agent_on_tasks(env, base, tasks, dist)
        for sc, (tasks, dist) in scenarios.items()
    }

    print("\n" + "=" * 78)
    print(f"  {'Scenario':<22}{'z mode':<12}{'Reward':>14}{'Viol':>10}")
    print("  " + "-" * 74)
    for sc in scenarios:
        for who in ("sample", "mean", "hold", "baseline"):
            r = out[who][sc]
            label = who if who != "baseline" else "baseline(z=0)"
            print(f"  {sc if who == 'sample' else '':<22}{label:<12}"
                  f"{r['reward']:>9.3f}±{r['reward_std']:<4.2f}"
                  f"{r['violations']:>10.2f}")
        print("  " + "-" * 74)
    print("=" * 78)
    print("\n  Read it like this: 'sample' is the OLD number. If 'mean'/'hold'\n"
          "  are much better than 'sample', the previous meta-vs-baseline gap\n"
          "  was inflated by per-step z resampling noise, not by policy quality.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "pearl_z_probe.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Probe] Saved -> {path}")
    return out


def _z_rollout(env, agent, task, disturbance=None, z_mode="mean",
               fixed_z=None, keep_states=0):
    """Run one deterministic episode and return outcome + latent diagnostics.

    A thinner sibling of ``_eval_agent_on_tasks`` used by
    ``z_informativeness``: it works one task at a time (so the caller can pair
    passes task-by-task), can deploy a caller-supplied latent, and can retain a
    strided sample of the visited states for the off-policy action-sensitivity
    probe.

    Args:
        env: FermentationEnv.
        agent: Trained PEARLAgent (meta, use_context=True).
        task: Task dict.
        disturbance: Optional {kind, onset, magnitude}.
        z_mode: Latent draw mode ('mean' / 'zero' / 'fixed' / ...).
        fixed_z: Latent to install when z_mode == 'fixed'.
        keep_states: If > 0, retain up to this many evenly-strided states.

    Returns:
        Dict with 'violations', 'reward', 'ethanol', 'z_mu', 'z_sigma_sq' (the
        posterior at episode end, which keeps updating even under 'fixed'/'zero'
        so the belief can be inspected independently of what was deployed), and
        'states'.
    """
    env.clear_disturbances()
    if disturbance is not None:
        env.schedule_disturbance(disturbance["kind"], disturbance["onset"],
                                 disturbance["magnitude"])
    env.set_task(task)
    state = env.reset()
    # Restored before returning: leaving the agent in whatever mode the last
    # pass used would silently ablate the latent for any later caller that
    # forgets to set it (the 'zero' pass would be the one left behind).
    prev_mode = agent.z_mode
    agent.z_mode = z_mode
    if z_mode == "fixed":
        agent.set_fixed_z(fixed_z)
    agent.clear_z()
    done = False
    ep_r = 0.0
    ep_v = 0
    ep_eth = 0.0
    states = []
    while not done:
        if keep_states:
            states.append(np.asarray(state, dtype=np.float32).copy())
        safe_action, _ = agent.select_action(state, deterministic=True)
        next_state, reward, done, info = env.step(safe_action)
        r_sum = float(np.sum(reward))
        agent.update_context(state, safe_action, r_sum, next_state)
        state = next_state
        ep_r += r_sum
        ep_v += int(info.get("violation_count", 0))
        ep_eth += float(info.get("eth_harvested", 0.0))
    env.clear_disturbances()
    agent.z_mode = prev_mode
    if keep_states and len(states) > keep_states:
        idx = np.linspace(0, len(states) - 1, keep_states).astype(int)
        states = [states[i] for i in idx]
    return {
        "violations": float(ep_v),
        "reward": float(ep_r),
        "ethanol": float(ep_eth),
        "z_mu": agent.z_mu.detach().cpu().numpy().copy(),
        "z_sigma_sq": agent.z_sigma_sq.detach().cpu().numpy().copy(),
        "states": states,
    }


def _actions_under_z(agent, states, z_vec):
    """Deterministic actions on fixed states under a fixed latent.

    Off-policy on purpose: the SAME states are re-scored under different latents,
    so the resulting difference isolates the actor's sensitivity to z from the
    divergence in visited states that a full rollout would confound it with.

    Args:
        agent: Trained PEARLAgent.
        states: List of observations, each shape [n_reactors, obs_dim].
        z_vec: Latent to condition on.

    Returns:
        A tuple (safe_actions, raw_actions), each shape [len(states), nR, aDim].
        'raw' is the policy output; 'safe' is what reaches the plant after the
        SERL stage mask. Reported separately because the mask overwrites
        structurally locked dims with constants: a policy whose z-response lives
        entirely in locked dims would show a nonzero raw delta and a zero safe
        delta, which is behaviourally identical to ignoring z.
    """
    prev_mode = agent.z_mode
    agent.z_mode = "fixed"
    agent.set_fixed_z(z_vec)
    safe, raw = [], []
    for s in states:
        a_safe, a_raw = agent.select_action(s, deterministic=True)
        safe.append(a_safe)
        raw.append(a_raw)
    agent.z_mode = prev_mode
    return np.stack(safe), np.stack(raw)


def z_informativeness(episodes=None):
    """Test whether the latent z carries task information the actor acts on.

    THIS RUNS BEFORE ANY FURTHER SEVERITY SWEEP. The adaptation decomposition's
    INFER column (noz - meta) is the proposal's claim, but INFER is pinned near
    zero whenever the latent is uninformative — so an all-'ns' INFER column has
    two incompatible readings ("meta adapted no better than its own ablation" vs
    "there was never a latent to ablate") and the sweep alone cannot separate
    them. The first headroom_test run produced exactly that ambiguity: INFER of
    +0.70 / +0.65 / -0.20, sign-flipping and far inside the paired noise.

    Five probes. 1, 2 and 2b test the MACHINERY (is there a latent, and does the
    policy respond to it); 3 tests the CONSEQUENCE; 4 tests whether the metric
    being scored is task-dependent at all, and outranks the rest.

      1. SEPARABILITY. Across-task std of the posterior mean, divided by the
         posterior's own std. Below 1 the belief moves less between tasks than it
         is uncertain within one, so it is not identifying tasks. This is the
         encoder-side question and it is decisive on its own: if it fails,
         nothing downstream can succeed.
      2. SENSITIVITY. Re-score one task's states under ANOTHER task's latent and
         measure the action change. This is the actor-side question, and it is
         separate: an encoder can be perfectly informative while the actor has
         learned to ignore its z input (the trunk can simply zero those weights),
         which produces the same null.
      3. CONSEQUENCE. Full rollouts with a DELIBERATELY WRONG latent (another
         task's), paired against the correct one. This is the only probe whose
         units are the ones the proposal is scored in. It is also strictly
         stronger than the existing 'zero' ablation: z=0 is a valid, in-support
         input the actor may well be robust to, whereas a confidently wrong task
         hypothesis is what actual mis-inference looks like at deployment.

    Run under both no disturbance and the clean-window mass_drift severity, since
    task inference has the most to offer exactly where the regime has shifted.

    Args:
        episodes: Tasks per scenario; defaults to Z_INFORM_EPISODES.

    Returns:
        Dict of per-scenario diagnostics (also saved to
        MODEL_DIR/pearl_z_informativeness.json).
    """
    n_ep = Z_INFORM_EPISODES if episodes is None else int(episodes)
    if n_ep < 2:
        # The swap probe pairs each task with a DIFFERENT task's latent; with one
        # task the cyclic shift hands a task its own latent back and probes 2/3
        # silently measure nothing.
        raise SystemExit("z_informativeness needs at least 2 tasks (the swap "
                         f"probe pairs distinct tasks); got {n_ep}.")
    print(f"--- Latent-informativeness diagnostic (device: {device}) ---")
    env = make_env()
    meta = PEARLAgent(env.stateDim, env.actionDim, n_reactors=env.n_reactors,
                      use_context=True, use_safeguard=True,
                      posterior_window=CONTEXT_SIZE)
    _load_ckpt_preferring_best(meta, BEST_META_PATH, META_PATH, "meta")

    # Same pool/seed as evaluate() and severity_sweep, so these tasks are the
    # ones the headroom numbers were measured on.
    tasks = make_task_pool(n_ep, seed=12345, ood=False)
    onset = env.max_steps // 2
    # Probe 3 needs a severity where the baseline actually fails, which is what
    # SEVERITY_CLEAN_WINDOW records. Read its first entry generically so editing
    # that constant to a different kind reconfigures this diagnostic instead of
    # raising KeyError on a hardcoded 'mass_drift'.
    if not SEVERITY_CLEAN_WINDOW:
        raise SystemExit("SEVERITY_CLEAN_WINDOW is empty: no severity is known "
                         "to break the baseline, so probe 3 has no regime to "
                         "run in. Run severity_sweep first.")
    clean_kind = next(iter(SEVERITY_CLEAN_WINDOW))
    clean_mag = SEVERITY_CLEAN_WINDOW[clean_kind][0]
    scenarios = {
        "undisturbed": None,
        f"{clean_kind}@{clean_mag:g}": {"kind": clean_kind, "onset": onset,
                                        "magnitude": clean_mag},
    }

    out = {}
    for sc_name, dist in scenarios.items():
        print(f"\n  [{sc_name}]  {n_ep} tasks")

        # ── Pass 1: correct latent (the normal deployment) ───────────────────
        matched = [_z_rollout(env, meta, t, dist, z_mode="mean", keep_states=24)
                   for t in tasks]
        z_mu = np.stack([r["z_mu"] for r in matched])          # [n_ep, latent]
        z_post_std = float(np.mean(np.sqrt(np.stack(
            [r["z_sigma_sq"] for r in matched]))))
        # Per-dim across-task spread of the posterior mean, then averaged.
        z_spread = float(np.mean(np.std(z_mu, axis=0)))
        snr = z_spread / z_post_std if z_post_std > 0 else float("nan")

        # Pair each task with a DIFFERENT task's latent. A cyclic shift by one
        # guarantees every task gets a wrong-but-real latent (never its own), and
        # is deterministic, so this diagnostic is reproducible.
        swap_idx = [(i + 1) % len(tasks) for i in range(len(tasks))]

        # ── Pass 2: actor sensitivity on fixed states (off-policy) ───────────
        d_safe, d_raw, d_zero_safe = [], [], []
        for i, r in enumerate(matched):
            if not r["states"]:
                continue
            a_own, raw_own = _actions_under_z(meta, r["states"], z_mu[i])
            a_oth, raw_oth = _actions_under_z(meta, r["states"],
                                              z_mu[swap_idx[i]])
            a_zero, _ = _actions_under_z(meta, r["states"],
                                         np.zeros_like(z_mu[i]))
            d_safe.append(float(np.mean(np.abs(a_own - a_oth))))
            d_raw.append(float(np.mean(np.abs(raw_own - raw_oth))))
            d_zero_safe.append(float(np.mean(np.abs(a_own - a_zero))))
        act_d_safe = float(np.mean(d_safe)) if d_safe else float("nan")
        act_d_raw = float(np.mean(d_raw)) if d_raw else float("nan")
        act_d_zero = float(np.mean(d_zero_safe)) if d_zero_safe else float("nan")
        # How far apart the swapped latents actually were. Without this the probe
        # above is unreadable: a tiny action delta means one thing if the latents
        # differed by 1.0 and nothing at all if they differed by 0.008.
        swap_dz = float(np.mean([
            np.mean(np.abs(z_mu[i] - z_mu[swap_idx[i]]))
            for i in range(len(tasks))]))

        # ── Pass 2b: actor sensitivity at PRIOR scale (decides the fix) ───────
        # Perturb z at the N(0, I) scale the actor was trained on, and normalise
        # by the actual latent displacement, giving a finite-difference
        # sensitivity d|a|/d|z| that is comparable regardless of encoder health.
        # This is the counterfactual "would a REPAIRED encoder help?" probe.
        rng = np.random.default_rng(0)
        latent_dim = int(z_mu.shape[1])
        d_prior = []
        for r in matched:
            if not r["states"]:
                continue
            z1 = rng.standard_normal(latent_dim).astype(np.float32)
            z2 = rng.standard_normal(latent_dim).astype(np.float32)
            a1, _ = _actions_under_z(meta, r["states"], z1)
            a2, _ = _actions_under_z(meta, r["states"], z2)
            dz = float(np.mean(np.abs(z1 - z2)))
            if dz > 0:
                d_prior.append(float(np.mean(np.abs(a1 - a2))) / dz)
        act_sens_prior = float(np.mean(d_prior)) if d_prior else float("nan")

        # ── Pass 3: rollouts under a wrong latent, and under z=0 ─────────────
        swapped = [_z_rollout(env, meta, t, dist, z_mode="fixed",
                              fixed_z=z_mu[swap_idx[i]])
                   for i, t in enumerate(tasks)]
        zeroed = [_z_rollout(env, meta, t, dist, z_mode="zero") for t in tasks]

        v_m = [r["violations"] for r in matched]
        v_s = [r["violations"] for r in swapped]
        v_z = [r["violations"] for r in zeroed]
        # Positive = the wrong/absent latent is WORSE, i.e. the correct latent
        # was buying something.
        swap_cost = _paired_delta(v_s, v_m)
        zero_cost = _paired_delta(v_z, v_m)

        # ── Probe 4: is the SCORED metric even task-dependent? ────────────────
        # The last question, and the one that outranks the other three: INFER is
        # measured in VIOLATIONS. If a fixed policy's violation count barely
        # varies across tasks, then tasks are indistinguishable in that metric and
        # NO amount of task inference can improve it — the quantity being scored
        # is not a function of the thing z is trying to infer. Reported next to
        # ethanol, which is where this environment's task-dependence actually
        # lives (theta multiplies kinetic rates, so it moves productivity, while
        # every constraint here is an inventory constraint on observed mass).
        eth_m = [r["ethanol"] for r in matched]

        def _cv(xs):
            """Across-task coefficient of variation, NaN when the mean is 0."""
            xs = np.asarray(xs, dtype=np.float64)
            m = float(np.mean(xs))
            return float(np.std(xs) / abs(m)) if m != 0.0 else float("nan")

        viol_cv, eth_cv = _cv(v_m), _cv(eth_m)
        viol_spread_tasks = float(np.std(v_m))

        print(f"    1. SEPARABILITY  z_spread={z_spread:.4f}  "
              f"post_std={z_post_std:.4f}  SNR={snr:.3f}  "
              f"(want > {Z_INFORM_MIN_SNR:g})")
        print(f"       mean |z_mu| = {float(np.mean(np.abs(z_mu))):.4f}")
        print(f"    2. SENSITIVITY   |dA| swapped-z: safe={act_d_safe:.2e}  "
              f"raw={act_d_raw:.2e}   (want > {Z_INFORM_MIN_ACTION_DELTA:g})")
        print(f"       ...but the swapped latents differed by only "
              f"|dz|={swap_dz:.2e}, so read 2b instead:")
        print(f"    2b. PRIOR-SCALE  d|A|/d|z| at z~N(0,I) = "
              f"{act_sens_prior:.2e}   (want > {Z_INFORM_MIN_PRIOR_SENS:g})")
        print(f"       -> would a REPAIRED encoder move the policy at all?")
        print(f"    3. CONSEQUENCE   violations, paired vs correct latent:")
        print(f"       wrong latent - correct = {_fmt_delta(swap_cost)}")
        print(f"       z=0          - correct = {_fmt_delta(zero_cost)}")
        print(f"    4. METRIC        across-task spread, fixed policy: "
              f"violations sd={viol_spread_tasks:.2f} (CV={viol_cv:.3f})  "
              f"ethanol CV={eth_cv:.3f}")

        # ── Verdict ──────────────────────────────────────────────────────────
        if np.isfinite(snr) and snr < Z_INFORM_MIN_SNR:
            if (np.isfinite(act_sens_prior)
                    and act_sens_prior < Z_INFORM_MIN_PRIOR_SENS):
                verdict = ("FAIL(encoder AND actor): the encoder returns the same "
                           "latent for every task, AND the actor barely responds "
                           "to z even at prior scale. Repairing the encoder alone "
                           "cannot produce a nonzero INFER — the policy has no "
                           "z-dependence left to exploit.")
            else:
                verdict = ("FAIL(encoder only): the encoder returns essentially "
                           "the same latent for every task, so INFER is null by "
                           "construction — but the actor DOES respond to z at "
                           "prior scale, so an encoder-side fix is worth "
                           "attempting. More sweeps cannot help.")
        elif (np.isfinite(act_d_safe)
                and act_d_safe < Z_INFORM_MIN_ACTION_DELTA):
            if (np.isfinite(act_d_raw)
                    and act_d_raw >= Z_INFORM_MIN_ACTION_DELTA):
                verdict = ("FAIL(masked): the actor DOES respond to z, but only "
                           "on dims the SERL stage mask overwrites, so no "
                           "z-response reaches the plant.")
            else:
                verdict = ("FAIL(sensitivity): z separates tasks but the actor's "
                           "output barely moves with it — the policy has learned "
                           "to ignore its latent input. Actor-side fix required.")
        elif not (swap_cost["significant"] and not swap_cost.get("negligible")):
            # 'negligible' must be excluded here, not just 'not significant':
            # meta and meta_noz share weights, so a paired CI can certify a
            # 0.01% difference as real. Calling that PASS would be the exact
            # statistical-vs-practical confusion PRACTICAL_MIN_REL_EFFECT exists
            # to stop.
            why = ("costs nothing measurable"
                   if not swap_cost["significant"]
                   else f"costs a statistically real but negligible "
                        f"{swap_cost['rel'] * 100.0:+.2f}% of the arm's value")
            verdict = ("INCONCLUSIVE: z is informative and the actor uses it, "
                       f"but deploying a WRONG latent {why}. "
                       "Task inference has no consequence in this regime; look "
                       "for a scenario where the correct regime actually "
                       "dictates a different control law.")
        else:
            verdict = ("PASS: a wrong latent measurably hurts, so task inference "
                       "is doing real work here and INFER is a fair test at this "
                       "severity.")
        print(f"    => {verdict}")
        # Printed independently of the verdict above, because it can invalidate
        # the whole measurement regardless of how healthy the latent is.
        if (np.isfinite(viol_spread_tasks)
                and viol_spread_tasks < Z_INFORM_MIN_VIOL_SPREAD):
            print("    !! METRIC WARNING: a fixed policy's violation count is "
                  "essentially\n       CONSTANT across tasks here, so violations "
                  "are not a function of\n       task identity and task inference "
                  "cannot improve them even in\n       principle. INFER measured "
                  "in violations is the wrong test; score\n       it on ethanol / "
                  "productivity, where this env's task-dependence\n       "
                  "actually lives.")

        out[sc_name] = {
            "n_tasks": n_ep,
            "disturbance": dist,
            "z_spread": z_spread,
            "z_post_std": z_post_std,
            "z_snr": snr,
            "z_abs_mean": float(np.mean(np.abs(z_mu))),
            "z_mu_per_task": z_mu.tolist(),
            "action_delta_swapped_safe": act_d_safe,
            "action_delta_swapped_raw": act_d_raw,
            "action_delta_zero_safe": act_d_zero,
            "swap_latent_distance": swap_dz,
            "action_sens_prior_scale": act_sens_prior,
            "viol_cv_across_tasks": viol_cv,
            "viol_sd_across_tasks": viol_spread_tasks,
            "ethanol_cv_across_tasks": eth_cv,
            "ethanol_matched": eth_m,
            "viol_matched": v_m,
            "viol_swapped": v_s,
            "viol_zeroed": v_z,
            "swap_cost_viol": swap_cost,
            "zero_cost_viol": zero_cost,
            "reward_matched": [r["reward"] for r in matched],
            "reward_swapped": [r["reward"] for r in swapped],
            "reward_zeroed": [r["reward"] for r in zeroed],
            "verdict": verdict,
        }

    print("\n  HOW TO READ THIS: probes 1, 2 and 2b are about the MACHINERY and\n"
          "  need no headroom to be meaningful — they are valid even where the\n"
          "  baseline is at its constraint ceiling. Probe 3 needs headroom, so\n"
          "  read it at the disturbed scenario. A FAIL on 1 or 2b means the\n"
          "  INFER column in headroom_test is uninterpretable, not merely small.\n"
          "  Probe 2 alone is NOT enough: when the encoder has collapsed, the\n"
          "  swapped latents are nearly identical, so a null action delta there\n"
          "  is confounded with probe 1 — 2b is the unconfounded version.\n"
          "  Probe 4 outranks all of them: if a fixed policy's violation count\n"
          "  does not vary across tasks, violations are not a function of task\n"
          "  identity and no encoder or actor fix can make INFER positive on\n"
          "  that metric. Score the claim on ethanol in that case.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "pearl_z_informativeness.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[z-inform] Saved -> {path}")
    return out


def _load_ckpt_preferring_best(agent, best_path, final_path, label):
    """Load an agent from its best-held-out checkpoint, else the final one.

    Args:
        agent: PEARLAgent to load into.
        best_path: Best-by-held-out-violations checkpoint path.
        final_path: Final-iteration checkpoint path.
        label: Name used in the log line ('meta' / 'baseline').

    Raises:
        SystemExit: If neither checkpoint exists — a severity sweep over an
            untrained agent would silently produce meaningless numbers.
    """
    path = best_path if os.path.exists(best_path) else final_path
    if not os.path.exists(path):
        raise SystemExit(f"No checkpoint for {label} ({best_path} / "
                         f"{final_path}). Run train_meta/train_baseline first.")
    agent.load(path)
    print(f"[Sweep] loaded {label} -> {path}")


def _drift_clamp_risk(env, mag, onset, m0=10000.0):
    """Predicted post-onset reactor mass under a mass_drift, and whether the
    ``M >= 1.0`` clamp is close enough to binding to corrupt the measurement.

    env_safe applies drift as ``M *= (1 + mag)`` each inner step from onset and
    clamps at 1.0 kg, but credits the UNCLAMPED delta to
    ``episode_mass_discharged``. Once the clamp binds, the mass ledger
    over-credits and the end-of-episode balance guard can fire, so violations at
    that severity are partly an artifact of the disturbance implementation rather
    than a control failure.

    Args:
        env: FermentationEnv (reads max_steps).
        mag: Per-step drift rate (negative shrinks mass).
        onset: Inner-step index at which drift begins.
        m0: Nominal per-reactor starting mass in kg.

    Returns:
        A tuple (predicted_final_mass_kg, at_risk_bool).
    """
    steps = max(0, int(env.max_steps) - int(onset))
    final = float(m0) * (1.0 + float(mag)) ** steps
    return final, final < SEVERITY_CLAMP_RISK_KG


# Two-sided 95% Student-t critical values by degrees of freedom. Tabulated rather
# than imported because scipy is not a dependency of this project, and hardcoding
# 1.96 (the normal limit) understates the interval by ~7% at the n=20 this eval
# actually runs — precisely the regime where the question "is this delta inside
# the noise" is being decided.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def _t95(df):
    """Two-sided 95% t critical value, falling back to the normal limit."""
    if df < 1:
        return float("nan")
    return _T95.get(int(df), 1.960)


# Minimum RELATIVE effect size that counts as practically meaningful, as a
# fraction of the compared arm's own magnitude. WHY THIS IS NEEDED: pairing makes
# the CI very tight, and meta vs meta_noz are nearly the same policy (they share
# weights and differ only in whether z is zeroed), so their paired difference has
# almost no variance. That combination will stamp "SIG" on differences of a few
# thousand out of 2.8e7 ethanol — a 0.004% effect that is statistically real and
# scientifically meaningless. Statistical significance without practical
# significance is exactly the failure mode this whole reporting overhaul was
# meant to prevent, so the label has to carry both.
PRACTICAL_MIN_REL_EFFECT = 0.01     # 1% of the compared arm's value


def _paired_delta(a, b, ref=None):
    """Paired difference (a - b) with a 95% CI, for two arms on the SAME tasks.

    THIS IS THE POINT: every arm in a severity sweep is evaluated on one shared
    task pool, traversed in the same order, so element i of each arm's
    per-episode array is the same task under a different policy. The paired
    difference therefore cancels the task-to-task variation that dominates the
    per-arm standard deviation, and its CI is the correct scale on which to judge
    whether an INFER or MULTI delta is real. Comparing the two arm MEANS against
    their separate stds instead (the previous reporting) both overstates the noise
    for a real effect and, more damagingly, gives a bare point estimate like
    "+0.70" no visible uncertainty at all.

    Args:
        a: Per-episode values for the first arm (array-like).
        b: Per-episode values for the second arm, same length and task order.
        ref: Scale against which to judge PRACTICAL significance. Defaults to
            |mean(b)|, i.e. the difference is expressed as a fraction of the arm
            it is being compared to.

    Returns:
        Dict with 'mean' (a - b), 'se', 'ci95' (half-width), 'n',
        'significant' (CI excludes zero), 'rel' (mean / |ref| — so its SIGN is
        the sign of the difference, not of the ratio) and 'negligible'
        (statistically significant but |rel| < PRACTICAL_MIN_REL_EFFECT).

        Degenerate inputs: mismatched lengths or an empty array give an all-NaN
        'mean'/'se'/'ci95' with significant=False; a single pair gives a finite
        'mean' and 'rel' but NaN 'se'/'ci95' and significant=False. A caller can
        always print the row.

        CAVEAT worth knowing: when mean(b) is 0 (e.g. the meta arm scores 0
        violations on every task, which is common here) or NaN, 'rel' is NaN and
        the practical-significance guard cannot apply — such a row can still show
        'SIG' on a tiny absolute difference. Pass an explicit ``ref`` when a
        meaningful scale exists.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    nan = float("nan")
    base = {"rel": nan, "negligible": False}
    if a.size != b.size or a.size == 0:
        return {"mean": nan, "se": nan, "ci95": nan, "n": int(min(a.size, b.size)),
                "significant": False, "paired": False, **base}
    d = a - b
    mean = float(np.mean(d))
    n = int(d.size)
    scale = abs(float(np.mean(b))) if ref is None else abs(float(ref))
    rel = mean / scale if scale > 0 else nan
    if n < 2:
        return {"mean": mean, "se": nan, "ci95": nan, "n": n,
                "significant": False, "paired": True,
                "rel": rel, "negligible": False}
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    ci95 = float(_t95(n - 1) * se)
    # A zero-variance difference (both arms identical on every task) is not
    # "significant"; it is no difference at all.
    sig = bool(se > 0.0 and abs(mean) > ci95)
    return {"mean": mean, "se": se, "ci95": ci95, "n": n,
            "significant": sig, "paired": True, "rel": rel,
            "negligible": bool(sig and np.isfinite(rel)
                               and abs(rel) < PRACTICAL_MIN_REL_EFFECT)}


def _fmt_delta(d):
    """Render a _paired_delta dict as '+0.70 ±2.41 ns' for a table cell.

    Three labels, not two:
        SIG   - statistically significant AND at least
                PRACTICAL_MIN_REL_EFFECT of the compared arm's magnitude.
        tiny  - statistically significant but a negligible fraction of the
                arm's value. Reporting this as an effect would be misleading:
                with paired episodes and two near-identical policies the CI can
                shrink far below any difference that matters.
        ns    - the CI includes zero.
    """
    if not np.isfinite(d["mean"]):
        return "n/a"
    if not np.isfinite(d["ci95"]):
        return f"{d['mean']:+.2f}"
    if d.get("negligible"):
        flag = f"tiny({d['rel'] * 100.0:+.2f}%)"
    elif d["significant"]:
        flag = "SIG"
    else:
        flag = "ns"
    return f"{d['mean']:+.2f}±{d['ci95']:.2f} {flag}"


def severity_sweep(kinds=None, arms=None, episodes=None, levels=None,
                   out_name="pearl_severity_sweep.json"):
    """Escalate each disturbance's magnitude until the agent(s) actually fail.

    Answers the question that has to be settled before any further agent work:
    does this environment have ANY headroom in which adaptation could show up?
    See the SEVERITY_SWEEP_* block at the top for the full rationale — in short,
    all three arms score 0.00 violations on every eval disturbance scenario, so
    the eval currently cannot discriminate between policies at all.

    Holds onset fixed at ``max_steps // 2`` (exactly as evaluate() does) and
    varies ONLY magnitude, over the same task pool evaluate() uses, so level 0 of
    each kind reproduces the corresponding row of the eval table.

    Args:
        kinds: Disturbance kinds to sweep; defaults to all of
            SEVERITY_SWEEP_LEVELS.
        arms: Which agents to run ('baseline'/'meta'/'meta_noz'); defaults to
            SEVERITY_SWEEP_AGENTS.
        episodes: Episodes per level; defaults to EVAL_EPISODES (kept equal so
            the level-0 rows are directly comparable to evaluate()).
        levels: Optional {kind: (magnitude, ...)} overriding
            SEVERITY_SWEEP_LEVELS — e.g. SEVERITY_CLEAN_WINDOW to run only the
            region where the baseline genuinely fails and the measurement is
            clean. When given, ``kinds`` defaults to its keys.
        out_name: Filename written under MODEL_DIR, so a focused run does not
            overwrite the full sweep's results.

    Returns:
        Dict with 'reference', 'sweep', 'break_points' and 'config' entries
        (also saved to MODEL_DIR/pearl_severity_sweep.json).
    """
    all_levels = SEVERITY_SWEEP_LEVELS if levels is None else dict(levels)
    kinds = tuple(all_levels) if kinds is None else tuple(kinds)
    arms = tuple(SEVERITY_SWEEP_AGENTS) if arms is None else tuple(arms)
    n_ep = EVAL_EPISODES if episodes is None else int(episodes)
    print(f"--- Disturbance-severity sweep (device: {device}) ---")
    print(f"[Sweep] arms={list(arms)} kinds={list(kinds)} episodes={n_ep}")

    env = make_env()
    meta = base = None
    if any(a.startswith("meta") for a in arms):
        # posterior_window MUST match training, as in evaluate().
        meta = PEARLAgent(env.stateDim, env.actionDim, n_reactors=env.n_reactors,
                          use_context=True, use_safeguard=True,
                          posterior_window=CONTEXT_SIZE)
        _load_ckpt_preferring_best(meta, BEST_META_PATH, META_PATH, "meta")
    if "baseline" in arms:
        base = PEARLAgent(env.stateDim, env.actionDim, n_reactors=env.n_reactors,
                          use_context=False, use_safeguard=True)
        _load_ckpt_preferring_best(base, BEST_BASE_PATH, BASE_PATH, "baseline")

    # SAME pool and seed as evaluate()'s id_tasks, so this is not a different
    # experiment with a coincidentally similar table.
    tasks = make_task_pool(n_ep, seed=12345, ood=False)
    onset = env.max_steps // 2

    def _run(arm, dist):
        if arm == "meta":
            return _eval_agent_on_tasks(env, meta, tasks, dist)
        if arm == "meta_noz":
            return _eval_agent_on_tasks(env, meta, tasks, dist, z_mode="zero")
        return _eval_agent_on_tasks(env, base, tasks, dist)

    def _broke(res, ref_res):
        """Return the list of tripped break criteria (empty = survived)."""
        why = []
        if res["violations"] > SEVERITY_BREAK_VIOL:
            why.append(f"viol {res['violations']:.2f}")
        if res["feasible_rate"] < SEVERITY_BREAK_FEASIBLE:
            why.append(f"feasible {res['feasible_rate']:.0%}")
        if (ref_res["ethanol"] > 0.0
                and res["ethanol"] < SEVERITY_BREAK_ETHANOL_FRAC
                * ref_res["ethanol"]):
            why.append(f"ethanol {res['ethanol'] / ref_res['ethanol']:.0%} "
                       f"of undisturbed")
        return why

    # Undisturbed reference per arm: the ethanol denominator for the break test,
    # and a self-check — these numbers must reproduce evaluate()'s
    # 'in_distribution' rows, since it is the same agent on the same tasks.
    print("\n[Sweep] Undisturbed reference (must match evaluate()'s "
          "in_distribution rows):")
    ref = {}
    for arm in arms:
        ref[arm] = _run(arm, None)
        print(f"    {arm:<10} reward={ref[arm]['reward']:>7.3f}  "
              f"viol={ref[arm]['violations']:>6.2f}  "
              f"ethanol={ref[arm]['ethanol']:.3e}")

    sweep = {}
    break_points = {}
    clamp_info = {}
    for kind in kinds:
        mags = all_levels[kind]
        lo, hi = META_DISTURB_MAG_RANGES.get(kind, (float("nan"),) * 2)
        print(f"\n  {kind}  (meta training support: {lo:g} .. {hi:g}; "
              f"onset fixed at {onset})")
        print(f"    {'mag':>8}{'agent':<12}{'reward':>15}{'viol':>9}"
              f"{'feas':>8}{'ethanol':>12}{'%undist':>9}  broke")
        print("    " + "-" * 76)
        sweep[kind] = {}
        for mag in mags:
            dist = {"kind": kind, "onset": onset, "magnitude": mag}
            key = f"{mag:g}"
            sweep[kind][key] = {}
            # Flag levels where the disturbance implementation itself, not the
            # controller, may be generating the violations (see
            # _drift_clamp_risk). Printed once per level, above its arm rows.
            # Kept in a PARALLEL dict so sweep[kind][mag] stays a clean
            # {arm: metrics} mapping for downstream parsing.
            if kind == "mass_drift":
                m_final, at_risk = _drift_clamp_risk(env, mag, onset)
                clamp_info.setdefault(kind, {})[key] = {
                    "predicted_final_mass_kg": m_final,
                    "clamp_at_risk": bool(at_risk)}
                if at_risk:
                    print(f"    !! {mag:g}: predicted post-onset mass "
                          f"{m_final:.3g} kg trips the M>=1.0 clamp; violations "
                          f"here are partly a mass-ledger artifact, NOT a "
                          f"control failure. Do not use this level.")
            for arm in arms:
                res = _run(arm, dist)
                why = _broke(res, ref[arm])
                res["broke"] = why
                sweep[kind][key][arm] = res
                if why and kind not in break_points.get(arm, {}):
                    break_points.setdefault(arm, {})[kind] = {
                        "magnitude": mag, "why": why}
                frac = (res["ethanol"] / ref[arm]["ethanol"]
                        if ref[arm]["ethanol"] > 0 else float("nan"))
                in_supp = "" if (lo <= mag <= hi) else " *"
                print(f"    {mag:>8g}{arm:<12}"
                      f"{res['reward']:>9.3f}±{res['reward_std']:<5.2f}"
                      f"{res['violations']:>9.2f}"
                      f"{res['feasible_rate']:>8.0%}"
                      f"{res['ethanol']:>12.3e}{frac:>9.0%}"
                      f"  {', '.join(why) if why else '-'}{in_supp}")
            print("    " + "-" * 76)
    print("    * = magnitude outside META_DISTURB_MAG_RANGES, i.e. outside "
          "meta's\n      training support (both arms are extrapolating there).")

    # ── Break points ─────────────────────────────────────────────────────────
    print("\n  BREAK POINTS (first magnitude tripping viol > "
          f"{SEVERITY_BREAK_VIOL}, feasible < "
          f"{SEVERITY_BREAK_FEASIBLE:.0%}, or ethanol < "
          f"{SEVERITY_BREAK_ETHANOL_FRAC:.0%} of undisturbed):")
    any_break = False
    for arm in arms:
        for kind in kinds:
            bp = break_points.get(arm, {}).get(kind)
            if bp is None:
                print(f"    {arm:<10} {kind:<14} NO BREAK at any level tested "
                      f"(worst: {all_levels[kind][-1]:g})")
            else:
                any_break = True
                print(f"    {arm:<10} {kind:<14} breaks at "
                      f"{bp['magnitude']:g}  ({', '.join(bp['why'])})")

    # ── Verdict ──────────────────────────────────────────────────────────────
    print("\n  HOW TO READ THIS:")
    if any_break and "baseline" in arms:
        print("    The baseline DOES break somewhere. Those break magnitudes are\n"
              "    the only severities at which a meta-vs-baseline comparison has\n"
              "    headroom. Re-run with arms=('baseline','meta','meta_noz') at\n"
              "    (and just below) each break magnitude — and note that meta vs\n"
              "    meta_noz there is still the decisive test, since only that\n"
              "    isolates task inference from robustness gained by training on\n"
              "    disturbances at all.")
    elif "baseline" in arms:
        print("    The baseline does NOT break at any severity tested. A\n"
              "    non-adaptive controller trained on ONE task therefore holds\n"
              "    the constraint ceiling under every disturbance this\n"
              "    environment can pose, which means the shared safety stack\n"
              "    (SERL mask + Lagrangian penalties), not the policy, is\n"
              "    supplying feasibility. No policy-level adaptation claim is\n"
              "    testable here as configured. The next move is then structural\n"
              "    — e.g. re-run with use_safeguard=False to see what the policy\n"
              "    itself is worth, tighten the constraints, or raise the\n"
              "    productivity weight so there is a real speed/safety tradeoff\n"
              "    that requires knowing the regime — NOT more encoder work.")

    # ── Adaptation decomposition, when all three arms are present ────────────
    # This is the payoff table of the whole exercise, and it only means anything
    # at severities where the baseline actually BREAKS — a level the baseline
    # already solves at 0.00 violations cannot support any claim either way.
    decomposition = {}
    if {"meta", "meta_noz", "baseline"} <= set(arms):
        print("\n  ADAPTATION DECOMPOSITION (violations; lower is better):")
        # Column width 28, not 20: a 'tiny(+0.04%)' label is 12 chars against 3
        # for 'SIG', and Python's >N pads but never truncates, so a narrow column
        # silently shifts every column to its right out from under the header.
        print(f"    {'kind':<12}{'mag':>7}{'base':>7}{'noz':>7}{'meta':>7}"
              f"  {'INFER (noz-meta)':>28}  {'MULTI (base-noz)':>28}  headroom")
        print("    " + "-" * 110)
        for kind in kinds:
            lo, hi = META_DISTURB_MAG_RANGES.get(kind, (float("nan"),) * 2)
            for mag in all_levels[kind]:
                cell = sweep[kind][f"{mag:g}"]
                b = cell["baseline"]["violations"]
                n = cell["meta_noz"]["violations"]
                m = cell["meta"]["violations"]
                # INFER: what ablating z costs, i.e. task inference (noz - meta).
                # MULTI: what multi-task training buys with z held at 0 in both
                # arms (base - noz). Same decomposition evaluate() prints.
                #
                # Both are now PAIRED over the shared task pool and carry a 95%
                # CI (see _paired_delta). Without this the table printed a bare
                # "+0.70" beside per-arm stds of ~8 and left the reader to
                # assume it meant something.
                infer = _paired_delta(cell["meta_noz"].get("viol_per_ep", []),
                                      cell["meta"].get("viol_per_ep", []))
                multi = _paired_delta(cell["baseline"].get("viol_per_ep", []),
                                      cell["meta_noz"].get("viol_per_ep", []))
                # Headroom now has THREE outcomes, not two. The old two-valued
                # version said "yes" whenever the baseline broke, which labelled
                # every row of the first headroom_test run as usable even though
                # all three magnitudes sat outside META_DISTURB_MAG_RANGES —
                # i.e. every arm was extrapolating and meta was being asked about
                # a regime it never trained in. That caveat was printed as a '*'
                # in the sweep table above and then silently dropped here, in the
                # one table anybody actually reads for the verdict.
                in_supp = bool(lo <= mag <= hi) if np.isfinite(lo) else False
                # Reuse the SAME break test the sweep table and break_points
                # above are built from, rather than re-deriving headroom from the
                # violation count alone: _broke also trips on infeasibility and
                # on the ethanol floor, so a level where the baseline "survived"
                # by shutting the plant down IS headroom, and judging it by
                # violations only would contradict the table printed just above.
                if not cell["baseline"].get("broke"):
                    headroom = "NO (baseline at ceiling)"
                elif in_supp:
                    headroom = "yes (in meta's support)"
                else:
                    headroom = "yes, but EXTRAPOLATING"
                decomposition.setdefault(kind, {})[f"{mag:g}"] = {
                    "baseline_viol": b, "noz_viol": n, "meta_viol": m,
                    "infer": infer, "multi": multi,
                    "in_meta_support": in_supp, "headroom": headroom,
                }
                print(f"    {kind:<12}{mag:>7g}{b:>7.2f}{n:>7.2f}{m:>7.2f}"
                      f"  {_fmt_delta(infer):>28}  {_fmt_delta(multi):>28}"
                      f"  {headroom}")
        print("    " + "-" * 110)
        print("    INFER = task inference (noz - meta). MULTI = multi-task\n"
              "    training (base - noz). Only INFER is the proposal's claim.\n"
              "    Deltas are PAIRED over the shared task pool, +-95% CI.\n"
              "    'ns' = the CI includes zero, i.e. this row is consistent with\n"
              "    NO effect and must not be reported as one.\n"
              f"    'tiny(x%)' = statistically real but under "
              f"{PRACTICAL_MIN_REL_EFFECT:.0%} of the arm's own\n"
              "    value; ALSO treat as null. Ignore any row whose headroom\n"
              "    column says NO.")

        # ── Same decomposition, scored on ETHANOL ────────────────────────────
        # WHY BOTH. Every constraint in this env is an inventory constraint on
        # observed mass (M, S_mass, the raw-material ledger), none of which
        # depends on the task parameters theta. theta multiplies kinetic rates,
        # so it moves PRODUCTIVITY. Violations can therefore be nearly
        # task-independent, and an INFER measured in violations then asks task
        # inference to improve a quantity that is not a function of the task —
        # unwinnable by construction, and indistinguishable in the table above
        # from an encoder that simply failed. Ethanol is where the
        # task-dependence lives, so it is the metric on which a task-inference
        # claim can actually be won or lost.
        # NOTE THE FLIPPED SIGN: more ethanol is BETTER, so these deltas are
        # (meta - noz) and (noz - base), keeping "positive = the thing being
        # credited helped" consistent with the violations table.
        print("\n  SAME DECOMPOSITION, SCORED ON ETHANOL (higher is better):")
        print(f"    {'kind':<12}{'mag':>7}{'base':>12}{'noz':>12}{'meta':>12}"
              f"  {'INFER (meta-noz)':>30}  {'MULTI (noz-base)':>30}")
        print("    " + "-" * 118)
        for kind in kinds:
            for mag in all_levels[kind]:
                cell = sweep[kind][f"{mag:g}"]
                eb = cell["baseline"]["ethanol"]
                en = cell["meta_noz"]["ethanol"]
                em = cell["meta"]["ethanol"]
                ei = _paired_delta(cell["meta"].get("ethanol_per_ep", []),
                                   cell["meta_noz"].get("ethanol_per_ep", []))
                em_ = _paired_delta(cell["meta_noz"].get("ethanol_per_ep", []),
                                    cell["baseline"].get("ethanol_per_ep", []))
                decomposition.setdefault(kind, {}).setdefault(
                    f"{mag:g}", {}).update({
                        "baseline_ethanol": eb, "noz_ethanol": en,
                        "meta_ethanol": em,
                        "infer_ethanol": ei, "multi_ethanol": em_})
                print(f"    {kind:<12}{mag:>7g}{eb:>12.3e}{en:>12.3e}"
                      f"{em:>12.3e}  {_fmt_delta(ei):>30}"
                      f"  {_fmt_delta(em_):>30}")
        print("    " + "-" * 118)
        print("    Deltas are absolute ethanol, paired, +-95% CI. If the\n"
              "    violations table above is all 'ns' but this one is not, the\n"
              "    adaptation claim should be argued on productivity, not safety.\n"
              f"    'tiny(x%)' = statistically real but under "
              f"{PRACTICAL_MIN_REL_EFFECT:.0%} of the arm's own value; treat as\n"
              "    null. meta and meta_noz share weights, so their paired CI is\n"
              "    tiny and trivial gaps WILL reach significance — read the %.")

        # ── Reading gate: is INFER even interpretable here? ──────────────────
        # A latent that does not separate tasks pins INFER near zero by
        # construction, so an all-'ns' INFER column has two very different
        # explanations and the z diagnostics are what tell them apart. Surfacing
        # them here (they were already computed and written to JSON, but never
        # printed) stops "no measurable adaptation" from being read as "meta
        # tried and failed" when the real state is "z was never informative".
        mref = ref.get("meta", {})
        spread = mref.get("z_spread", float("nan"))
        post_std = mref.get("z_post_std", float("nan"))
        snr = spread / post_std if post_std and np.isfinite(post_std) else float("nan")
        print(f"\n  LATENT CHECK (meta, undisturbed): across-task spread of z_mu"
              f" = {spread:.4f}\n"
              f"    posterior std = {post_std:.4f}  ->  separability SNR = "
              f"{snr:.3f}  (want > {Z_INFORM_MIN_SNR:g})")
        # 'ns everywhere' must consider BOTH metrics now that ethanol is scored:
        # a significant ethanol INFER alongside a null violations INFER is the
        # expected shape of a real result in this env, and the gate below must not
        # tell the reader to go fix the encoder in that case.
        # A 'tiny' row counts as null for this purpose: statistically real but
        # practically negligible is not a positive result.
        def _real(delta):
            return bool(delta.get("significant")
                        and not delta.get("negligible"))

        all_ns = all(not _real(c["infer"])
                     and not _real(c.get("infer_ethanol", {}))
                     for k in decomposition for c in decomposition[k].values())
        if np.isfinite(snr) and snr < Z_INFORM_MIN_SNR:
            print("    z_mu moves LESS across tasks than the posterior's own\n"
                  "    width, so the encoder is not resolving task identity and\n"
                  "    INFER cannot be anything but noise at ANY severity. Fix\n"
                  "    the encoder before running more sweeps; run\n"
                  "    `python train_pearl_single_run.py z_informativeness`\n"
                  "    for the behavioural confirmation (z-swap test).")
        elif all_ns:
            print("    The latent looks separable but every INFER row is ns:\n"
                  "    either the actor ignores z (check the z-swap action delta\n"
                  "    in z_informativeness) or the effect is smaller than this\n"
                  "    episode count can resolve — raise `episodes`, or use\n"
                  "    train_pearl_seed_sweep.py for a multi-seed paired test.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    out = {
        "reference": ref,
        "sweep": sweep,
        "break_points": break_points,
        "clamp_info": clamp_info,
        # Paired INFER/MULTI deltas with CIs, empty unless all three arms ran.
        # Persisted so a later analysis does not have to re-derive the pairing
        # (and cannot accidentally re-derive it UNpaired).
        "decomposition": decomposition,
        "config": {
            "arms": list(arms), "kinds": list(kinds), "episodes": n_ep,
            "onset": int(onset), "levels": {k: list(all_levels[k])
                                            for k in kinds},
            "break_viol": SEVERITY_BREAK_VIOL,
            "break_feasible": SEVERITY_BREAK_FEASIBLE,
            "break_ethanol_frac": SEVERITY_BREAK_ETHANOL_FRAC,
            "meta_train_ranges": {k: list(v) for k, v
                                  in META_DISTURB_MAG_RANGES.items()},
        },
    }
    path = os.path.join(MODEL_DIR, out_name)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Sweep] Saved -> {path}")
    return out


# ─── Entrypoint ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else MODE
    if mode == "train_meta":
        train_meta()
    elif mode == "train_baseline":
        train_baseline()
    elif mode == "evaluate":
        evaluate()
    elif mode == "eval_z_probe":
        eval_z_probe()
    elif mode == "z_informativeness":
        z_informativeness()
    elif mode == "severity_sweep":
        # Optional extra argv: comma-separated arms, e.g.
        #   python train_pearl_single_run.py severity_sweep baseline,meta,meta_noz
        sweep_arms = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
        severity_sweep(arms=sweep_arms)
    elif mode == "headroom_test":
        # All three arms, restricted to the ONE region the full sweep showed to
        # be both discriminating and artifact-free (see SEVERITY_CLEAN_WINDOW).
        severity_sweep(arms=SEVERITY_CLEAN_ARMS, levels=SEVERITY_CLEAN_WINDOW,
                       out_name="pearl_headroom_test.json")
    elif mode == "all":
        train_meta()
        train_baseline()
        evaluate()
    else:
        raise SystemExit(f"Unknown MODE '{mode}'. Use train_meta|train_baseline|"
                         "evaluate|eval_z_probe|z_informativeness|"
                         "severity_sweep|headroom_test|all.")
