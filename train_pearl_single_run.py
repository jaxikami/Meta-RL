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
    'all'            — train_meta -> train_baseline -> evaluate
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import json
import os
import sys

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
N_TRAIN_TASKS = 40          # size of the fixed meta-training task pool
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
MODEL_DIR = "policy"
META_PATH = os.path.join(MODEL_DIR, "pearl_meta.pth")
BASE_PATH = os.path.join(MODEL_DIR, "pearl_baseline.pth")
# Separate "best-seen" checkpoints, tracked by avg_violations (the true
# progress signal here — see EARLY_STOP_* comments below). Training in this
# setting is not monotonic: on one run the meta agent reached avg_violations
# of 8.5 at iteration 57, then regressed to 283 by iteration 2000 without ever
# recovering. META_PATH/BASE_PATH always hold the *latest* snapshot (useful
# for resuming / inspecting a trend); BEST_META_PATH/BEST_BASE_PATH hold
# whichever snapshot had the lowest avg_violations ever observed, so a later
# regression can never silently overwrite a genuinely better policy.
# evaluate() loads the best checkpoints when present.
BEST_META_PATH = os.path.join(MODEL_DIR, "pearl_meta_best.pth")
BEST_BASE_PATH = os.path.join(MODEL_DIR, "pearl_baseline_best.pth")

# Early stopping.  Raw reward here is penalty-dominated (a do-nothing policy
# already scores ~-24,000/episode from the Lagrangian constraint penalty
# alone), so it is a poor convergence signal — the violation count is the
# metric that actually reflects learning progress (it runs from ~590/episode
# for a random policy down toward 0).  Two independent triggers, either of
# which stops a run once EARLY_STOP_MIN_ITERS outer iterations have elapsed:
#   - target: avg_violations stays at/below EARLY_STOP_TARGET_VIOL for
#     EARLY_STOP_TARGET_PATIENCE iterations (fast exit once already good,
#     without waiting out a full plateau window).
#   - plateau: avg_violations hasn't improved by EARLY_STOP_MIN_DELTA from its
#     best-seen value for EARLY_STOP_PATIENCE iterations (catches training
#     that has stalled even above the target).
# avg_violations is a rolling mean over the last 50 episodes (~25 iterations
# of new data at COLLECT_EPISODES_PER_ITER=2), so it moves slowly — MIN_ITERS
# and the patience windows are sized in outer iterations, generous enough that
# several rolling-window refreshes happen before a stop decision is made.
EARLY_STOP_ENABLED = True
EARLY_STOP_MIN_ITERS = 300          # never stop before this many iterations
EARLY_STOP_PATIENCE = 300           # iters w/o improvement -> plateau stop
EARLY_STOP_MIN_DELTA = 1.0          # min violation-count improvement to count
EARLY_STOP_TARGET_VIOL = 3.0        # "good enough" avg violations/episode
EARLY_STOP_TARGET_PATIENCE = 50     # iters sustained at/under target -> stop

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
META_EARLY_STOP_MIN_ITERS = 700
META_EARLY_STOP_PATIENCE = 700

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
META_ENCODER_LR = 3e-5


class EarlyStopper:
    """Plateau + target early stopping on a rolling metric (lower is better).

    Two independent triggers, either of which stops training once `min_iters`
    outer iterations have elapsed:
      - target: the metric stays at or below `target` for `target_patience`
        consecutive checks — a fast exit once training is already "good
        enough", without waiting for a full plateau to establish itself.
      - plateau: the metric hasn't improved by at least `min_delta` from its
        best-seen value for `patience` consecutive checks — catches training
        that has stalled, even above the target (or when no target is set).
    """

    def __init__(self, min_iters=300, patience=300, min_delta=1.0,
                 target=None, target_patience=50):
        self.min_iters = min_iters
        self.patience = patience
        self.min_delta = min_delta
        self.target = target
        self.target_patience = target_patience
        self.best = float("inf")
        self.plateau_count = 0
        self.target_count = 0

    def should_stop(self, it, value):
        """Check the stopping condition for the latest metric reading.

        Args:
            it: Current outer iteration (1-indexed).
            value: Latest rolling-metric value (lower is better).

        Returns:
            A tuple (stop, reason); reason is "" when stop is False.
        """
        if value < self.best - self.min_delta:
            self.best = value
            self.plateau_count = 0
        else:
            self.plateau_count += 1

        if self.target is not None and value <= self.target:
            self.target_count += 1
        else:
            self.target_count = 0

        if it < self.min_iters:
            return False, ""
        if self.target is not None and self.target_count >= self.target_patience:
            return True, (f"metric {value:.2f} <= target {self.target:.2f} "
                          f"sustained for {self.target_count} iterations")
        if self.plateau_count >= self.patience:
            return True, (f"metric plateaued at {value:.2f} "
                          f"(best {self.best:.2f}) for {self.plateau_count} "
                          f"iterations with no improvement")
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


def collect_rollout(env, agent, replay, task_id, task, add_to_replay=True):
    """Run one episode under a fixed task, updating the posterior online.

    Args:
        env: FermentationEnv.
        agent: PEARLAgent used to select actions.
        replay: MultiTaskReplayBuffer (may be None if add_to_replay is False).
        task_id: Integer task identifier for the replay buffer.
        task: Task dict to install via env.set_task.
        add_to_replay: If True, store transitions in the replay buffer.

    Returns:
        A tuple (total_reward, total_violations, n_steps).
    """
    env.set_task(task)
    state = env.reset()
    agent.clear_z()
    done = False
    total_reward = 0.0
    total_viol = 0
    steps = 0
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
        steps += 1
    return total_reward, total_viol, steps


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
                       target_entropy=META_TARGET_ENTROPY)
    replay = MultiTaskReplayBuffer(env.stateDim, env.actionDim, env.n_reactors)

    log = {"iter": [], "avg_reward": [], "avg_violations": [], "critic_loss": [],
           "actor_loss": [], "kl": [], "alpha": [], "z_norm": []}
    reward_hist = []
    viol_hist = []

    # ── Warm-up: seed every task's replay bucket ──────────────────────────────
    # Fill all N_TRAIN_TASKS buckets so the readiness check (MIN_BUFFER
    # transitions per bucket) passes on iteration 1 instead of after dozens of
    # empty iterations.
    print(f"[Meta] Warm-up: {WARMUP_EPISODES_PER_TASK} episode(s) x "
          f"{N_TRAIN_TASKS} tasks ...")
    for tid in range(N_TRAIN_TASKS):
        for _ in range(WARMUP_EPISODES_PER_TASK):
            r, v, _s = collect_rollout(env, agent, replay, tid, pool[tid])
            reward_hist.append(r); viol_hist.append(v)

    stopper = EarlyStopper(
        min_iters=META_EARLY_STOP_MIN_ITERS, patience=META_EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA, target=EARLY_STOP_TARGET_VIOL,
        target_patience=EARLY_STOP_TARGET_PATIENCE) if EARLY_STOP_ENABLED else None
    best_viol = float("inf")
    best_iter = None

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
            r, v, _s = collect_rollout(env, agent, replay, tid, pool[tid])
            reward_hist.append(r); viol_hist.append(v)

        # ── Optimize ────────────────────────────────────────────────────────
        stats = None
        for _ in range(UPDATES_PER_ITER):
            stats = agent.update(replay, meta_batch=META_BATCH,
                                 batch_size=BATCH_SIZE, context_size=CONTEXT_SIZE,
                                 min_buffer=MIN_BUFFER)

        avg_r = float(np.mean(reward_hist[-50:])) if reward_hist else 0.0
        avg_v = float(np.mean(viol_hist[-50:])) if viol_hist else 0.0

        # Training here is not monotonic (see BEST_META_PATH comment above) —
        # save the strictly-best-seen snapshot separately so a later
        # regression can never overwrite it.
        if avg_v < best_viol:
            best_viol = avg_v
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
                "QL": f"{stats['critic_loss']:.3f}",
                "KL": f"{stats['kl']:.3f}",
                "|mu|": f"{stats['z_norm']:.3f}",
                "klW": f"{agent.kl_lambda:.3f}",
                "a": f"{stats['alpha']:.3f}",
            })

        # Checkpoint and flush the log periodically so a run can be stopped
        # early and still leave a usable agent + a visible reward trend.
        if it % 100 == 0:
            os.makedirs(MODEL_DIR, exist_ok=True)
            agent.save(META_PATH)
            with open(os.path.join(MODEL_DIR, "pearl_meta_log.json"), "w") as f:
                json.dump(log, f)

        if stopper is not None and viol_hist:
            stop, reason = stopper.should_stop(it, avg_v)
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
          f"avg_violations={final_v:.1f}) -> {META_PATH}")
    print(f"[Meta] Best-seen snapshot: avg_violations={best_viol:.1f} at iter "
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
                       normalize_rewards=NORMALIZE_REWARDS)
    replay = MultiTaskReplayBuffer(env.stateDim, env.actionDim, env.n_reactors)

    reward_hist = []
    viol_hist = []

    # Warm-up the single nominal bucket so updates begin on iteration 1.
    print(f"[Baseline] Warm-up: {WARMUP_EPISODES_PER_TASK} episode(s) ...")
    for _ in range(WARMUP_EPISODES_PER_TASK):
        r, v, _s = collect_rollout(env, agent, replay, 0, nominal)
        reward_hist.append(r); viol_hist.append(v)

    stopper = EarlyStopper(
        min_iters=EARLY_STOP_MIN_ITERS, patience=EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA, target=EARLY_STOP_TARGET_VIOL,
        target_patience=EARLY_STOP_TARGET_PATIENCE) if EARLY_STOP_ENABLED else None
    best_viol = float("inf")
    best_iter = None

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

        # Save the strictly-best-seen snapshot separately (see BEST_BASE_PATH
        # comment above) so a later regression can't overwrite it.
        if avg_v < best_viol:
            best_viol = avg_v
            best_iter = it
            os.makedirs(MODEL_DIR, exist_ok=True)
            agent.save(BEST_BASE_PATH)

        if stats is not None:
            pbar.set_postfix({"Rwd": f"{avg_r:.1f}", "Viol": f"{avg_v:.0f}",
                              "QL": f"{stats['critic_loss']:.3f}",
                              "a": f"{stats['alpha']:.3f}"})
        if it % 200 == 0:
            agent.save(BASE_PATH)

        if stopper is not None and viol_hist:
            stop, reason = stopper.should_stop(it, avg_v)
            if stop:
                pbar.close()
                print(f"\n[Baseline] Early stop at iter {it}/{META_ITERATIONS}: {reason}")
                break
    agent.save(BASE_PATH)
    print(f"[Baseline] Saved final agent (iter {it}, avg_violations={avg_v:.1f}) "
          f"-> {BASE_PATH}")
    print(f"[Baseline] Best-seen snapshot: avg_violations={best_viol:.1f} at iter "
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


def _eval_agent_on_tasks(env, agent, tasks, disturbance=None):
    """Evaluate an agent over a list of tasks, optionally with a disturbance.

    Records per-outer-step reward/violation trajectories (averaged across tasks)
    so the disturbance response can be plotted and summarised.

    Args:
        env: FermentationEnv.
        agent: Trained PEARLAgent.
        tasks: List of task dicts.
        disturbance: Optional dict {kind, onset, magnitude} scheduled mid-episode.

    Returns:
        Dict of aggregate metrics, mean per-step trajectories, and (when a
        disturbance is given) recovery metrics.
    """
    rewards, viols, feasible = [], [], []
    reward_trajs, viol_trajs = [], []
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
            r_traj.append(r_sum)
            v_traj.append(v_step)
        rewards.append(ep_r)
        viols.append(ep_v)
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
        "reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "violations": float(np.mean(viols)),
        "violations_std": float(np.std(viols)),
        "feasible_rate": float(np.mean(feasible)),
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
    w = 0.38
    m_r = [results[n]["meta"]["reward"] for n in names]
    b_r = [results[n]["baseline"]["reward"] for n in names]
    m_re = [results[n]["meta"]["reward_std"] for n in names]
    b_re = [results[n]["baseline"]["reward_std"] for n in names]
    m_v = [results[n]["meta"]["violations"] for n in names]
    b_v = [results[n]["baseline"]["violations"] for n in names]
    labels = [n.replace("disturb_", "Δ ").replace("_", " ") for n in names]

    fig, (axr, axv) = plt.subplots(1, 2, figsize=(15, 6))
    axr.bar(x - w/2, m_r, w, yerr=m_re, capsize=3, label="meta (PEARL)", color="steelblue")
    axr.bar(x + w/2, b_r, w, yerr=b_re, capsize=3, label="baseline (safe RL)", color="tomato")
    axr.set_ylabel("Episode reward"); axr.set_title("Reward — higher is better")
    axr.set_xticks(x); axr.set_xticklabels(labels, rotation=30, ha="right")
    axr.legend(); axr.grid(True, axis="y", alpha=0.3)

    axv.bar(x - w/2, m_v, w, label="meta (PEARL)", color="steelblue")
    axv.bar(x + w/2, b_v, w, label="baseline (safe RL)", color="tomato")
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
            br = np.array(results[n]["baseline"]["reward_traj"], dtype=float)
            onset = results[n]["meta"].get("onset_outer")
            ax.plot(mr, label="meta (PEARL)", color="steelblue")
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

    meta = PEARLAgent(env.stateDim, env.actionDim, n_reactors=env.n_reactors,
                      use_context=True, use_safeguard=True)
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

    results = {}
    for name, spec in scenarios.items():
        results[name] = {
            "meta": _eval_agent_on_tasks(env, meta, spec["tasks"], spec["dist"]),
            "baseline": _eval_agent_on_tasks(env, base, spec["tasks"], spec["dist"]),
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print(f"  {'Scenario':<22}{'Agent':<10}{'Reward':>14}{'Viol':>10}{'Feasible':>12}")
    print("  " + "-" * 80)
    for name, res in results.items():
        for who in ("meta", "baseline"):
            r = res[who]
            print(f"  {name if who=='meta' else '':<22}{who:<10}"
                  f"{r['reward']:>9.3f}±{r['reward_std']:<4.2f}"
                  f"{r['violations']:>10.2f}{r['feasible_rate']:>12.1%}")
        print("  " + "-" * 80)
    print("=" * 84)

    # ── Per-scenario verdict (does meta beat the non-adaptive baseline?) ───────
    print("\n  VERDICT (meta − baseline):")
    for name, res in results.items():
        d_r = res["meta"]["reward"] - res["baseline"]["reward"]
        d_v = res["baseline"]["violations"] - res["meta"]["violations"]
        win = "meta better" if (d_r > 0 and d_v >= 0) else (
            "mixed" if (d_r > 0 or d_v > 0) else "baseline better")
        print(f"    {name:<22} Δreward={d_r:+8.3f}  fewer_viol={d_v:+7.2f}  -> {win}")

    # ── Disturbance recovery metrics ───────────────────────────────────────────
    print("\n  DISTURBANCE RESPONSE (post-onset):")
    print(f"    {'Scenario':<22}{'Agent':<10}{'post_reward':>13}{'post_viol':>12}{'recov_steps':>13}")
    for name, res in results.items():
        if "recovery" not in res["meta"]:
            continue
        for who in ("meta", "baseline"):
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


# ─── Entrypoint ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else MODE
    if mode == "train_meta":
        train_meta()
    elif mode == "train_baseline":
        train_baseline()
    elif mode == "evaluate":
        evaluate()
    elif mode == "all":
        train_meta()
        train_baseline()
        evaluate()
    else:
        raise SystemExit(f"Unknown MODE '{mode}'. Use train_meta|train_baseline|"
                         "evaluate|all.")
