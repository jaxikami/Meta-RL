"""PEARL meta-RL agent for uncertainty-aware multi-reactor fermentation control.

This module implements PEARL (Probabilistic Embeddings for Actor-critic RL,
Rakelly et al. 2019) on top of the *same* safety stack used by the non-adaptive
Safe-RL agent, so the only variable between the two is meta-learning
(proposal §3.1).

Pipeline overview::

    Context transitions (o, a, r, o')  ->  permutation-invariant encoder q_psi(z|c)
        ->  Gaussian posterior (product of Gaussians)  ->  sample latent task z
        ->  SAC actor pi(a | o, z)  ->  tanh squash  ->  SERL stage mask
        ->  environment

Key components:
    ContextEncoder: maps a *set* of recent transitions from the current task to a
        Gaussian posterior over the latent task variable z.  Permutation-invariant
        via product-of-Gaussians aggregation, matching PEARL.
    TanhGaussianActor: SAC policy pi(a|o,z) producing a squashed-Gaussian action
        per reactor, conditioned on the current observation and the inferred z.
    FlatCritic: twin soft Q-networks Q(o, a, z) for off-policy SAC updates.
    PEARLAgent: orchestrates posterior inference, safe action selection, the
        off-policy SAC update with a KL information bottleneck on the encoder, and
        target-network Polyak averaging.

Safety reuse:
    The SERL stage mask (FermentationEnv.get_action_mask) and the environment's
    Lagrangian constraint penalties are shared verbatim with the Safe-PPO agent.
    (An earlier learned Action Projection Network layer has been removed.)

Non-adaptive baseline:
    Constructing the agent with ``use_context=False`` fixes z ≡ 0, disables the
    encoder, and (when trained only on the nominal task) reproduces the proposal's
    non-adaptive Safe-RL baseline: identical safety architecture, no task inference.
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os
from collections import deque

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ─── Local ────────────────────────────────────────────────────────────────────
from env_safe import FermentationEnv
from agent_safe import CrossReactorAttention, device

# ─── PEARL / SAC Hyperparameters ──────────────────────────────────────────────

# Dimensionality of the inferred latent task variable z.  Small (the true latent
# theta has ~10 physically meaningful degrees of freedom) so the information
# bottleneck stays tight and the posterior is easy to infer online.
LATENT_DIM = 5

# Numerical bounds on the tanh-Gaussian policy log-std.  The textbook SAC range
# is [-20, 2], but -20 (std ~ 2e-9) lets the policy collapse to a near-delta
# function.  For a tanh-squashed Gaussian the log-density diverges as std -> 0
# (it is -log(std) + O(1) per dimension), and with 12 action dims (4 x 3
# reactors) that reaches into the hundreds and gets multiplied by alpha
# straight into the critic's Bellman target — this is what produced critic
# losses in the hundreds of thousands here.  -5 (std ~ 0.0067) is still a very
# confident, near-deterministic action but keeps the per-dimension log-density
# ceiling around 5 instead of 20, a 4x reduction in the worst case.
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

# Hard bound on the pre-squash (pre-tanh) value.  tanh saturates to within
# 1e-8 of +/-1 well before |u|=9, so this costs no expressiveness, but it caps
# the tanh log-prob correction (which grows ~linearly in |u| once saturated)
# instead of leaving it open to whatever magnitude the unconstrained mean_head
# output drifts to.
PRE_TANH_CLIP = 9.0

# Safety bounds on the SAC temperature.  Without this, a pathological entropy
# signal (e.g. from the log-std/log-prob dynamics above) can push log_alpha to
# grow without bound, as observed (alpha climbing past 1.0 and continuing to
# rise). This clamps alpha to a still-wide but finite range.
LOG_ALPHA_MIN, LOG_ALPHA_MAX = -5.0, 3.0

# Clip on the entropy term (alpha * logp) injected into the critic's Bellman
# target.  A last line of defense: even with the bounds above, this guarantees
# no single pathological sample can inject an arbitrarily large perturbation
# into the TD target.
ENTROPY_TERM_CLIP = 50.0


# ─── Latent-Variable Helpers ──────────────────────────────────────────────────


def product_of_gaussians(mus, sigmas_sq, include_prior=True):
    """Combine independent Gaussian factors into one via product of Gaussians.

    Implements PEARL's permutation-invariant aggregation over a set of
    per-transition Gaussian factors N(mu_i, sigma_i^2).  The product of Gaussians
    is itself Gaussian with precision-weighted mean.

    Args:
        mus: Tensor of factor means, shape [N, latent_dim].
        sigmas_sq: Tensor of factor variances, shape [N, latent_dim] (positive).
        include_prior: If True, fold the N(0, I) prior in as an additional
            factor (precision 1, mean 0).  This is PEARL's formulation and it
            matters for two reasons here.  (a) It bounds the posterior variance
            at 1.0, so a posterior can never be *wider* than the prior — with
            the prior omitted, a single high-variance factor produced a
            posterior with sigma^2 >> 1, i.e. a z sample far outside anything
            the actor was trained on.  (b) It makes the zero-factor case fall
            out of the same formula instead of being special-cased, which is
            what keeps the training-time posterior (``ContextEncoder.forward``
            over a context minibatch) and the rollout-time incremental
            posterior (``PEARLAgent.update_context``) algebraically identical.
            They must match: the actor is trained on z drawn from the former
            and deployed on z drawn from the latter.

    Returns:
        A tuple (mu, sigma_sq) of the combined posterior, each shape [latent_dim].
    """
    sigmas_sq = torch.clamp(sigmas_sq, min=1e-7)
    precision = torch.sum(1.0 / sigmas_sq, dim=0)
    weighted_mu = torch.sum(mus / sigmas_sq, dim=0)
    if include_prior:
        precision = precision + 1.0
    sigma_sq = 1.0 / precision
    mu = sigma_sq * weighted_mu
    return mu, sigma_sq


# ─── Reward Normalisation ─────────────────────────────────────────────────────


class RunningMeanStd:
    """Bounded-window running mean/variance estimator for reward scaling.

    Used to standardise the SAC reward target to roughly unit scale.  The env's
    per-step reward is penalty-dominated and large in magnitude (order 1-10),
    which pushes Q-values (and hence the Bellman loss) into a range where the
    KL bottleneck and entropy terms become negligible.  Dividing the reward by a
    running standard deviation keeps the critic target O(1/(1-gamma)) regardless
    of the raw penalty magnitude, so the other loss terms stay meaningful.

    ``max_count`` IS THE FIX FOR THE PRIMARY DIVERGENCE MECHANISM in this
    project.  Previously ``count`` was unbounded AND ``update`` was called from
    the SAC minibatch path (meta_batch * updates_per_iter = 80 calls of 1024
    samples = ~82k samples per outer iteration).  Two things went wrong:

      1. ``count`` reached ~8M by iteration 100 and ~57M by iteration 700, so
         each new batch moved the estimate by <1% and then <0.15%.  The reward
         scale FROZE at the value it had during early, random-policy rollouts —
         exactly when the Lagrangian penalties are largest.
      2. Because the frozen std was large, ``r / std`` shrank as the policy got
         worse, shrinking the Bellman target, shrinking every gradient, and
         removing the signal needed to recover.  This is visible in
         policy/pearl_meta_log.json: from iter 60 to 762 avg_reward went
         -27 -> -2438 (90x WORSE) while critic_loss went 10.9 -> 0.46 (24x
         BETTER).  A critic fitting ever more perfectly while the policy
         collapses is the signature of a vanishing target scale, not of a
         diverging critic (which was the earlier, mistaken reading).
         Downstream: with critic_loss ~ 0.46 the encoder's only informative
         gradient (d critic_loss / d z) also vanished, so the KL term won by
         default and the posterior collapsed (KL -> 1e-3).

    Capping ``count`` turns this into an exponentially-weighted estimator with
    an effective window of ``max_count`` samples, so the scale keeps tracking
    the reward distribution of recent data.  It is only meaningful in
    combination with the other half of the fix: ``update`` is now called ONCE
    PER COLLECTED ENV STEP (via ``PEARLAgent.observe_rewards``) rather than once
    per replayed minibatch sample, so ``count`` measures env steps and the
    window is interpretable (1e5 env steps ~= 830 episodes ~= 400 outer
    iterations at COLLECT_EPISODES_PER_ITER=2).
    """

    def __init__(self, eps=1e-4, max_count=1e5):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps
        self.max_count = float(max_count)

    def update(self, x):
        """Fold a batch of scalar rewards into the running statistics."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return
        # Cast to plain Python floats/ints immediately: x.mean()/x.var() return
        # numpy.float64 scalars, and once one of those touches self.mean/
        # self.var via +=, numpy's type promotion keeps them numpy.float64
        # from then on.  torch.load's weights_only=True safe-unpickler (the
        # default since torch 2.6) doesn't allowlist numpy scalar types, so a
        # checkpoint carrying them fails to load — this must be plain floats
        # all the way through, not just cast at save time.
        b_mean, b_var, b_count = float(x.mean()), float(x.var()), int(x.size)
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean += delta * b_count / tot
        m_a = self.var * self.count
        m_b = b_var * b_count
        self.var = (m_a + m_b + delta * delta * self.count * b_count / tot) / tot
        # Cap the effective sample count so this stays a bounded-window
        # estimator that keeps tracking recent data (see the class docstring:
        # an unbounded count froze the reward scale at its random-policy value
        # and starved every gradient in the run).
        self.count = min(tot, self.max_count)

    @property
    def std(self):
        return float(np.sqrt(self.var)) + 1e-6


# ─── Context Encoder ──────────────────────────────────────────────────────────


class ContextEncoder(nn.Module):
    """Permutation-invariant probabilistic task-inference network q_psi(z | c).

    Each context transition (o, a, r, o') is mapped independently to a Gaussian
    factor over z; the factors are combined by product of Gaussians, yielding a
    posterior that is invariant to the ordering of the transitions (PEARL).

    Attributes:
        latent_dim: Width of the latent task variable z.
        net: Shared per-transition MLP producing [mu, sigma_raw] factors.
    """

    def __init__(self, transition_dim, latent_dim=LATENT_DIM, hidden=128):
        """Initialise the context encoder.

        Args:
            transition_dim: Width of a single flattened context transition.
            latent_dim: Latent task-variable dimensionality.
            hidden: Hidden width of the shared per-transition MLP.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(transition_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * latent_dim),
        )

    def forward(self, context):
        """Infer the Gaussian posterior over z from a set of transitions.

        Args:
            context: Tensor of flattened transitions, shape [N, transition_dim].
                An empty first dimension yields the standard-normal prior.

        Returns:
            A tuple (mu, sigma_sq) of the posterior, each shape [latent_dim].
        """
        if context is None or context.shape[0] == 0:
            zero = torch.zeros(self.latent_dim, device=self._device())
            return zero, torch.ones(self.latent_dim, device=self._device())
        mus, sigmas_sq = self.factor(context)
        return product_of_gaussians(mus, sigmas_sq)

    def factor(self, context):
        """Compute per-transition Gaussian factors without combining them.

        Exposed separately from ``forward`` so callers that maintain a running
        posterior online (see ``PEARLAgent.update_context``) can encode just the
        newest transition and fold it into an incremental product-of-Gaussians
        accumulator, instead of re-encoding and recombining the entire context
        set from scratch on every step.

        Args:
            context: Tensor of flattened transitions, shape [N, transition_dim].

        Returns:
            A tuple (mus, sigmas_sq), each shape [N, latent_dim].
        """
        out = self.net(context)                       # [N, 2*latent]
        mus = out[:, :self.latent_dim]                # [N, latent]
        sigmas_sq = F.softplus(out[:, self.latent_dim:]) + 1e-6
        return mus, sigmas_sq

    def _device(self):
        return next(self.parameters()).device


# ─── SAC Actor ────────────────────────────────────────────────────────────────


class TanhGaussianActor(nn.Module):
    """SAC squashed-Gaussian policy pi(a | o, z) over per-reactor actions.

    Each reactor's observation is enriched by cross-reactor self-attention (reused
    from the Safe-PPO architecture), concatenated with the broadcast latent task
    context z, and mapped to a per-reactor Gaussian over the 4 action dims, then
    tanh-squashed to [-1, 1].

    Attributes:
        n_reactors: Number of reactors.
        obs_dim: Per-reactor observation width.
        action_dim: Per-reactor action width.
        latent_dim: Latent task-variable width.
    """

    def __init__(self, obs_dim, action_dim, latent_dim, n_reactors=3, hidden=256):
        """Initialise the actor.

        Args:
            obs_dim: Per-reactor observation width (28).
            action_dim: Per-reactor action width (4).
            latent_dim: Latent task-variable width.
            n_reactors: Number of reactors.
            hidden: Hidden width of the per-reactor head.
        """
        super().__init__()
        self.n_reactors = n_reactors
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.attn = CrossReactorAttention(
            feature_dim=obs_dim, n_heads=2, n_reactors=n_reactors)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def _features(self, obs, z):
        """Compute per-reactor pre-head features.

        Args:
            obs: Observation tensor, shape [B, n_reactors, obs_dim].
            z: Latent context, shape [B, latent_dim].

        Returns:
            Feature tensor, shape [B, n_reactors, hidden].
        """
        B = obs.shape[0]
        # CrossReactorAttention attends over its input dim-0.  To attend across
        # reactors (per env-batch sample), put reactors on dim-0 and the batch on
        # dim-1: [B, nR, D] -> [nR, B, D] -> attn -> back to [B, nR, D].
        xt = obs.transpose(0, 1)                         # [nR, B, obs_dim]
        enriched, _ = self.attn(xt)                      # [nR, B, obs_dim]
        x = enriched.transpose(0, 1)                     # [B, nR, obs_dim]
        z_b = z.unsqueeze(1).expand(B, self.n_reactors, self.latent_dim)
        return self.trunk(torch.cat([x, z_b], dim=-1))   # [B, nR, hidden]

    def forward(self, obs, z):
        """Return the per-reactor Gaussian parameters (pre-squash).

        Args:
            obs: Observation tensor, shape [B, n_reactors, obs_dim].
            z: Latent context, shape [B, latent_dim].

        Returns:
            A tuple (mean, log_std), each shape [B, n_reactors, action_dim].
        """
        feat = self._features(obs, z)
        mean = self.mean_head(feat)
        log_std = torch.clamp(self.log_std_head(feat), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs, z, deterministic=False, with_logprob=True, mask=None):
        """Sample a tanh-squashed action and its log-probability.

        Args:
            obs: Observation tensor, shape [B, n_reactors, obs_dim].
            z: Latent context, shape [B, latent_dim].
            deterministic: If True, return tanh(mean) with no sampling noise.
            with_logprob: If True, also return the summed log-probability.
            mask: Optional SERL stage mask, shape [B, n_reactors, action_dim],
                1.0 where the dim is structurally free.  When given, masked dims
                contribute ZERO to the returned log-probability.  This matters:
                a structurally locked dim is overwritten downstream by
                ``PEARLAgent._apply_safety``, so it has no effect on the
                environment, yet its only gradient came from the entropy term —
                which drove its log_std to LOG_STD_MAX and inflated logp by a
                large, meaningless constant.  That constant then (a) corrupted
                automatic alpha tuning and (b) was injected into the Bellman
                target as alpha*logp, which is the pathology ENTROPY_TERM_CLIP
                was papering over.

        Returns:
            A tuple (action, logp) where action has shape
            [B, n_reactors, action_dim] in [-1, 1] and logp has shape [B] (or None
            when with_logprob is False).
        """
        mean, log_std = self.forward(obs, z)
        std = log_std.exp()
        if deterministic:
            u = mean
        else:
            u = mean + std * torch.randn_like(std)
        # Bound the pre-squash value: tanh is saturated to within 1e-8 of +/-1
        # well before |u|=9, so this costs no expressiveness, but it caps the
        # log-prob correction below (which otherwise grows with |u| once
        # saturated) instead of leaving it open to an unconstrained mean_head
        # output drifting to extreme values.
        u = torch.clamp(u, -PRE_TANH_CLIP, PRE_TANH_CLIP)
        a = torch.tanh(u)
        if not with_logprob:
            return a, None
        dist = Normal(mean, std)
        # Numerically stable form of log(1 - tanh(u)^2), via the identity
        #   log(1 - tanh(u)^2) = 2 * (log(2) - u - softplus(-2u))
        # The naive `log(1 - tanh(u)^2 + eps)` is only *forward*-stable (the
        # eps floors it), but its *gradient* blows up near saturation, which
        # combined with an unbounded mean_head let the actor's log_std / mean
        # drift into a regime where per-sample log-prob reached the hundreds —
        # multiplied by alpha straight into the critic's Bellman target, this
        # produced the exploding critic loss seen in training.
        log_one_minus_a_sq = 2.0 * (np.log(2.0) - u - F.softplus(-2.0 * u))
        logp = dist.log_prob(u) - log_one_minus_a_sq      # [..., nR, action_dim]
        if mask is not None:
            # Structurally locked dims never reach the environment; exclude them
            # from the entropy accounting entirely (see the `mask` arg docstring).
            logp = logp * mask
        # Sum over action dims and reactors -> per-sample scalar log-prob.
        logp = logp.sum(dim=(-1, -2))                     # [B]
        return a, logp


# ─── SAC Critic ───────────────────────────────────────────────────────────────


class FlatCritic(nn.Module):
    """Twin soft Q-networks Q(o, a, z) over the flattened multi-reactor system.

    The whole reactor bank is treated as a single agent whose reward is the sum
    of per-reactor rewards; the critic therefore outputs a single scalar Q from
    the concatenated observation, action, and latent context.

    Attributes:
        q1: First Q-network.
        q2: Second Q-network (twin, reduces overestimation).
    """

    def __init__(self, obs_dim, action_dim, latent_dim, n_reactors=3, hidden=256):
        """Initialise the twin critic.

        Args:
            obs_dim: Per-reactor observation width.
            action_dim: Per-reactor action width.
            latent_dim: Latent task-variable width.
            n_reactors: Number of reactors.
            hidden: Hidden width.
        """
        super().__init__()
        in_dim = n_reactors * (obs_dim + action_dim) + latent_dim

        def make():
            return nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        self.q1 = make()
        self.q2 = make()

    def forward(self, obs, action, z):
        """Evaluate both Q-networks.

        Args:
            obs: Observation tensor, shape [B, n_reactors, obs_dim].
            action: Action tensor, shape [B, n_reactors, action_dim].
            z: Latent context, shape [B, latent_dim].

        Returns:
            A tuple (q1, q2), each shape [B].
        """
        B = obs.shape[0]
        x = torch.cat([obs.reshape(B, -1), action.reshape(B, -1), z], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


# ─── Replay Buffer ────────────────────────────────────────────────────────────


class MultiTaskReplayBuffer:
    """Per-task ring buffers of executed transitions for off-policy PEARL/SAC.

    Each task index owns an independent buffer.  ``sample_batch`` draws an RL
    minibatch (for the Bellman update) and ``sample_context`` draws a set of
    recent transitions (for task inference).

    Attributes:
        obs_dim: Per-reactor observation width.
        action_dim: Per-reactor action width.
        n_reactors: Number of reactors.
    """

    def __init__(self, obs_dim, action_dim, n_reactors=3, capacity=100000):
        """Initialise the multi-task buffer.

        Args:
            obs_dim: Per-reactor observation width.
            action_dim: Per-reactor action width.
            n_reactors: Number of reactors.
            capacity: Max transitions retained per task.
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_reactors = n_reactors
        self.capacity = capacity
        self._tasks = {}

    def _ensure(self, task_id):
        if task_id not in self._tasks:
            self._tasks[task_id] = deque(maxlen=self.capacity)

    def add(self, task_id, obs, act_exec, reward, next_obs, done):
        """Store one executed transition under a task.

        Args:
            task_id: Integer task identifier.
            obs: Observation array, shape [n_reactors, obs_dim].
            act_exec: Executed (projected) action, shape [n_reactors, action_dim].
            reward: Scalar summed reward.
            next_obs: Next observation, shape [n_reactors, obs_dim].
            done: Terminal flag.
        """
        self._ensure(task_id)
        self._tasks[task_id].append((
            np.asarray(obs, dtype=np.float32),
            np.asarray(act_exec, dtype=np.float32),
            np.float32(reward),
            np.asarray(next_obs, dtype=np.float32),
            np.float32(1.0 if done else 0.0),
        ))

    def task_ids(self):
        """Return the list of task ids that currently hold transitions."""
        return [t for t, b in self._tasks.items() if len(b) > 0]

    def size(self, task_id):
        """Return the number of stored transitions for a task."""
        return len(self._tasks.get(task_id, ()))

    def _stack(self, items):
        o, a, r, o2, d = zip(*items)
        to = lambda arr: torch.as_tensor(np.stack(arr)).to(device)
        return (to(o), to(a),
                torch.as_tensor(np.array(r)).to(device),
                to(o2),
                torch.as_tensor(np.array(d)).to(device))

    def sample_batch(self, task_id, batch_size):
        """Sample a random RL minibatch of transitions for a task.

        Args:
            task_id: Task identifier.
            batch_size: Number of transitions to draw (with replacement if small).

        Returns:
            A tuple (obs, act, reward, next_obs, done) of device tensors.
        """
        buf = self._tasks[task_id]
        n = len(buf)
        idx = np.random.randint(0, n, size=batch_size)
        return self._stack([buf[i] for i in idx])

    def sample_context(self, task_id, context_size):
        """Sample a set of recent transitions to infer the task posterior.

        Args:
            task_id: Task identifier.
            context_size: Number of transitions to include in the context set.

        Returns:
            A tuple (obs, act, reward, next_obs, done) of device tensors.
        """
        buf = self._tasks[task_id]
        n = len(buf)
        take = min(context_size, n)
        # Bias toward recent transitions (most informative about the regime).
        start = max(0, n - max(take, min(n, 2 * context_size)))
        pool = list(buf)[start:]
        idx = np.random.choice(len(pool), size=take, replace=False)
        return self._stack([pool[i] for i in idx])


# ─── PEARL Agent ──────────────────────────────────────────────────────────────


class PEARLAgent:
    """Off-policy PEARL meta-RL agent with the shared SERL + Lagrangian safety stack.

    Attributes:
        use_context: If False, z is fixed to zeros and the encoder is unused
            (non-adaptive Safe-RL baseline).
        latent_dim: Latent task-variable dimensionality.
        gamma: Discount factor.
        tau: Polyak averaging coefficient for target critics.
        kl_lambda: Weight on the KL(q(z|c) || N(0,I)) information bottleneck.
    """

    def __init__(self, obs_dim, action_dim, n_reactors=3, latent_dim=LATENT_DIM,
                 gamma=0.99, tau=5e-3, lr=3e-4, kl_lambda=0.1, encoder_lr=None,
                 reward_scale=1.0, normalize_rewards=True,
                 use_context=True, use_safeguard=True,
                 target_entropy=None, posterior_window=64,
                 reward_window=1e5, alpha_min=None,
                 target_entropy_per_dim=None):
        """Initialise the PEARL agent and its optimizers.

        Args:
            obs_dim: Per-reactor observation width (28).
            action_dim: Per-reactor action width (4).
            n_reactors: Number of reactors.
            latent_dim: Latent task-variable width.
            gamma: Discount factor.
            tau: Target-network Polyak coefficient.
            lr: Learning rate for actor/critic/alpha.
            kl_lambda: KL information-bottleneck weight on the encoder.
            encoder_lr: Learning rate for the context encoder specifically.
                Defaults to `lr` if None. A slower encoder_lr is a second,
                complementary lever to kl_lambda for the encoder-runaway
                failure mode observed in this project's meta-training (KL and
                critic_loss diverging together after an early good dip): even
                a well-anchored KL penalty can't help if the encoder's raw
                gradient step size lets z move faster than the critic can
                track it, so the critic's regression target stays
                nonstationary. Slowing the encoder directly reduces that
                per-step drift, independent of how much weight the KL term
                carries — pushing kl_lambda arbitrarily higher instead risks
                collapsing z to the prior entirely (meta degenerates to the
                encoder-free baseline, losing task adaptation).
            reward_scale: Multiplier applied to summed rewards for SAC.
            use_context: Enable task inference (True) or fix z=0 (baseline, False).
            use_safeguard: Apply the SERL stage mask (structural action locking).
            target_entropy: SAC target entropy; defaults to -action_dim*n_reactors.
                Interpreted PER ACTIVE ACTION DIMENSION: the alpha update scales
                it by the fraction of dims the SERL mask leaves free at each
                state (see ``update``).  With no mask, or with every dim active,
                this reduces exactly to the standard heuristic.
            posterior_window: Number of most-recent transitions the ONLINE
                rollout posterior aggregates (None = unbounded, the old
                behaviour).  Must match the training-time ``context_size`` in
                ``update``: the incremental product-of-Gaussians in
                ``update_context`` previously accumulated EVERY step of the
                episode, so by step 120 the precision sum was ~120 factors and
                sigma^2 ~ 1/120, while the actor had only ever been trained on z
                drawn from a 64-factor posterior.  Deploying a systematically
                sharper (and differently located) z than training produced is a
                train/deploy mismatch independent of whether the encoder itself
                is any good.
            reward_window: Effective sample window for the reward-scale
                estimator, in ENV STEPS (see RunningMeanStd).
            alpha_min: Hard FLOOR on the SAC temperature (None = only the
                numerical LOG_ALPHA_MIN bound applies).  Directly targets the
                collapse observed in policy/pearl_meta_log.json: alpha decayed
                monotonically 0.976 -> 0.055 over 125 iterations, and the
                held-out optimum (0 violations at iteration 25) sat exactly
                where alpha was passing 0.55 — after which held-out violations
                went 0 -> 56 -> 93 as alpha fell through 0.3 and 0.17.  Because
                alpha only falls while the policy's entropy exceeds the target,
                a monotone decay with no rebound means the target is demanding
                more determinism than this 40-task problem can afford: the
                policy locks in before it has learned to generalise, with no
                exploration noise left to escape.  A floor keeps a minimum
                exploration pressure alive through the window where the
                collapse happens.
            target_entropy_per_dim: Entropy target PER FREE ACTION DIMENSION.
                Takes precedence over ``target_entropy`` when given.  Less
                negative = more entropy demanded = alpha decays more slowly, so
                this is the smoother alternative to a hard ``alpha_min`` floor.
                The standard SAC heuristic is -1.0 per dim.
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_reactors = n_reactors
        self.latent_dim = latent_dim if use_context else 0
        self.gamma = gamma
        self.tau = tau
        self.kl_lambda = kl_lambda
        self.reward_scale = reward_scale
        self.normalize_rewards = normalize_rewards
        self.ret_rms = (RunningMeanStd(max_count=reward_window)
                        if normalize_rewards else None)
        self.use_context = use_context
        self.use_safeguard = use_safeguard
        self.posterior_window = (int(posterior_window)
                                 if posterior_window else None)

        # How z is drawn from the running posterior during a rollout.  See
        # ``_resample_z`` — this is a *deployment* knob, not a training one.
        #   'sample' : redraw z ~ q(z|c) every control step (the old, and only,
        #              behaviour). PEARL does NOT do this.
        #   'mean'   : z = posterior mean, no sampling. The correct analogue of
        #              actor `deterministic=True` for evaluation.
        #   'hold'   : draw once after `z_hold_after` steps of belief-forming,
        #              then hold z fixed for the rest of the episode — PEARL's
        #              actual posterior-sampling scheme (one task hypothesis per
        #              trajectory).
        #   'zero'   : z == 0 regardless of the posterior. ABLATION ONLY, for
        #              evaluating a TRAINED meta agent with its task inference
        #              switched off. This is the decisive test of the project's
        #              central claim: if a meta agent scores the same with z
        #              forced to zero, then task inference contributes nothing
        #              and its advantage over the baseline comes purely from
        #              having trained across many tasks rather than from
        #              adapting to the one it is in. It matches the baseline's
        #              z exactly, so the two differ only in training.
        self.z_mode = "sample"
        self.z_hold_after = 10

        # Effective latent width fed to networks (>=1 so tensors are well-formed;
        # in baseline mode z is a constant zero vector of width 1).
        self._z_dim = max(1, self.latent_dim)

        # ── Networks ────────────────────────────────────────────────────────
        # Transition = [o_flat, a_flat, r, o2_flat].
        trans_dim = n_reactors * obs_dim + n_reactors * action_dim + 1 \
            + n_reactors * obs_dim
        self.encoder = ContextEncoder(
            trans_dim, latent_dim=self._z_dim).to(device)
        self.actor = TanhGaussianActor(
            obs_dim, action_dim, self._z_dim, n_reactors).to(device)
        self.critic = FlatCritic(
            obs_dim, action_dim, self._z_dim, n_reactors).to(device)
        self.critic_target = FlatCritic(
            obs_dim, action_dim, self._z_dim, n_reactors).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # ── Optimizers ──────────────────────────────────────────────────────
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        if self.use_context:
            self.encoder_opt = torch.optim.Adam(
                self.encoder.parameters(),
                lr=encoder_lr if encoder_lr is not None else lr)

        # Automatic temperature (entropy) tuning.
        self.target_entropy = (target_entropy if target_entropy is not None
                               else -float(action_dim * n_reactors))
        # Per-active-dimension entropy target.  The alpha update multiplies this
        # by the number of dims the SERL mask leaves free, so a state where most
        # dims are structurally locked does not demand the same total entropy as
        # a fully-free state.  Identical to the old behaviour when nothing is
        # masked.
        if target_entropy_per_dim is not None:
            self._entropy_per_dim = float(target_entropy_per_dim)
            self.target_entropy = (self._entropy_per_dim
                                   * float(action_dim * n_reactors))
        else:
            self._entropy_per_dim = (self.target_entropy
                                     / float(action_dim * n_reactors))
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        # Effective lower bound on log_alpha: the numerical guard, raised to the
        # caller's alpha floor when one is given (see the alpha_min docstring).
        self.alpha_min = alpha_min
        self._log_alpha_min = (
            LOG_ALPHA_MIN if alpha_min is None
            else max(LOG_ALPHA_MIN, float(np.log(alpha_min))))

        # Safety layer: the SERL stage mask (applied in _apply_safety) plus the
        # environment's Lagrangian penalties.  No learned projection network.
        # Cached default (post-tanh) action for structurally locked dims: LF/C5/
        # outflow off, time multiplier at its physical default.  Precomputed
        # because _apply_safety now runs inside the gradient loop, not just in
        # action selection.
        _default_latent = torch.zeros(action_dim, device=device)
        _default_latent[0] = -0.34657359
        _default_latent[1:] = -10.0
        self._default_a = torch.tanh(_default_latent)     # [action_dim]

        # ── Live posterior state (maintained during a rollout) ──────────────
        # Accumulators start at the N(0, I) PRIOR (precision 1, weighted mean 0)
        # so the zero-transition posterior is exactly the prior and the
        # incremental form matches product_of_gaussians(..., include_prior=True).
        self.z_mu = torch.zeros(self._z_dim, device=device)
        self.z_sigma_sq = torch.ones(self._z_dim, device=device)
        self.z = torch.zeros(self._z_dim, device=device)
        self._z_precision_sum = torch.ones(self._z_dim, device=device)
        self._z_weighted_mu_sum = torch.zeros(self._z_dim, device=device)
        # Sliding window of per-transition factors, so the online posterior
        # aggregates the last `posterior_window` transitions instead of the
        # whole episode (see the posterior_window arg docstring).  Exact, O(1):
        # the evicted factor's contribution is subtracted from the accumulators.
        self._z_factors = (deque(maxlen=self.posterior_window)
                           if self.posterior_window else None)
        self._z_steps = 0

    # ── Latent-context management ───────────────────────────────────────────

    def clear_z(self):
        """Reset the running posterior to the N(0, I) prior and resample z.

        Called at the start of each rollout episode: the agent begins from the
        prior and refines its belief as it accumulates task transitions.
        """
        self.z_mu = torch.zeros(self._z_dim, device=device)
        self.z_sigma_sq = torch.ones(self._z_dim, device=device)
        # Running precision-weighted accumulators for the incremental
        # product-of-Gaussians posterior (see update_context), seeded with the
        # N(0, I) prior factor so zero transitions == the prior exactly.
        self._z_precision_sum = torch.ones(self._z_dim, device=device)
        self._z_weighted_mu_sum = torch.zeros(self._z_dim, device=device)
        if self._z_factors is not None:
            self._z_factors.clear()
        self._z_steps = 0
        self._resample_z()

    def _resample_z(self):
        """Draw z from the current posterior per ``self.z_mode``.

        In baseline mode (use_context=False) z is always the zero vector, which
        makes the baseline's policy genuinely deterministic under
        ``select_action(deterministic=True)``.  The meta agent's was NOT: with
        z_mode='sample' (the only prior behaviour) z was redrawn on every single
        control step, so 'deterministic' evaluation still injected fresh
        N(z_mu, z_sigma) noise into the policy conditioning 120 times per
        episode.  Since the encoder had collapsed to the prior (KL ~ 1e-3), that
        noise carried no task information at all — it was pure action jitter,
        and the meta-vs-baseline eval was comparing a jittered controller
        against a clean one.  'mean' and 'hold' remove that confound.
        """
        if not self.use_context or self.z_mode == "zero":
            self.z = torch.zeros(self._z_dim, device=device)
            return
        if self.z_mode == "mean":
            self.z = self.z_mu.clone()
            return
        std = torch.sqrt(torch.clamp(self.z_sigma_sq, min=1e-8))
        self.z = self.z_mu + std * torch.randn_like(std)

    @staticmethod
    def _flatten_transition(obs, act, reward, next_obs):
        """Flatten one transition into the encoder's input vector."""
        return np.concatenate([
            np.asarray(obs, dtype=np.float32).reshape(-1),
            np.asarray(act, dtype=np.float32).reshape(-1),
            np.array([np.float32(reward)]),
            np.asarray(next_obs, dtype=np.float32).reshape(-1),
        ])

    def update_context(self, obs, act, reward, next_obs):
        """Fold one new transition into the running online posterior.

        The context encoder's product-of-Gaussians aggregation is associative:
        combining n independent per-transition factors all at once gives the
        exact same posterior as combining them one at a time via a running
        precision-weighted accumulator. This lets us encode ONLY the newest
        transition each step (O(1) work) instead of re-encoding and
        re-combining the entire growing episode context from scratch on every
        step (O(episode length) work, quadratic over an episode) — the same
        posterior, computed incrementally.

        Args:
            obs: Observation, shape [n_reactors, obs_dim].
            act: Executed action, shape [n_reactors, action_dim].
            reward: Scalar summed reward.
            next_obs: Next observation, shape [n_reactors, obs_dim].
        """
        if not self.use_context:
            return
        # Normalise the reward feature the same way it is normalised for the
        # Bellman target (divide by the running std; read-only here — the
        # running stats are updated ONLY from freshly collected env steps via
        # observe_rewards(), which the evaluation paths never call, so running
        # this during an evaluation rollout cannot contaminate them).  Without
        # this, the encoder's per-transition input mixes a normalized obs/
        # action (O(1)) with a raw, penalty-dominated reward (O(10-200)),
        # letting that one feature dominate the factor network and destabilize
        # the posterior — independent of whatever normalization is applied to
        # the critic target.
        reward_n = reward / self.ret_rms.std if self.ret_rms is not None else reward
        trans = self._flatten_transition(obs, act, reward_n, next_obs)
        trans_t = torch.as_tensor(
            trans, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mu_i, sigma_i_sq = self.encoder.factor(trans_t)
        mu_i = mu_i.squeeze(0)
        sigma_i_sq = torch.clamp(sigma_i_sq.squeeze(0), min=1e-7)
        prec_i = 1.0 / sigma_i_sq

        # Sliding window: evict the oldest factor's contribution before adding
        # the new one, so the posterior always reflects the last
        # `posterior_window` transitions — the same number the encoder is
        # trained on.  Exact (product of Gaussians is associative *and*
        # invertible in precision space) and O(1) per step.
        if self._z_factors is not None:
            if len(self._z_factors) == self._z_factors.maxlen:
                old_mu, old_prec = self._z_factors[0]
                self._z_precision_sum = self._z_precision_sum - old_prec
                self._z_weighted_mu_sum = (self._z_weighted_mu_sum
                                           - old_mu * old_prec)
            self._z_factors.append((mu_i, prec_i))

        self._z_precision_sum = self._z_precision_sum + prec_i
        self._z_weighted_mu_sum = self._z_weighted_mu_sum + mu_i * prec_i
        # Precision can only be >= 1 (the prior factor seeds it), so this is
        # always well-conditioned and sigma^2 <= 1.
        self.z_sigma_sq = 1.0 / torch.clamp(self._z_precision_sum, min=1e-6)
        self.z_mu = self.z_sigma_sq * self._z_weighted_mu_sum

        self._z_steps += 1
        # 'hold' mode: let the belief form for z_hold_after steps (resampling as
        # it sharpens), then freeze the drawn hypothesis for the rest of the
        # episode.  z_mu/z_sigma_sq keep updating either way so diagnostics stay
        # live; only the *drawn* z is frozen.
        if not (self.z_mode == "hold" and self._z_steps > self.z_hold_after):
            self._resample_z()

    # ── Safe action selection ───────────────────────────────────────────────

    def _stage_mask(self, obs):
        """SERL stage mask for an observation of any batch shape.

        Args:
            obs: Observation tensor, shape [..., n_reactors, obs_dim].

        Returns:
            Mask tensor, shape [..., n_reactors, action_dim], or None when the
            safeguard is disabled.
        """
        if not self.use_safeguard:
            return None
        return FermentationEnv.get_action_mask(obs[..., :self.obs_dim]).to(device)

    def _apply_safety(self, state_t, raw_squashed, mask=None):
        """Apply the SERL stage mask to a squashed action.

        Locks structurally inactive action dimensions to their physical defaults
        (e.g. LF feed = 0 outside the feed stages).  This is the structural half
        of the safety stack; the environment's Lagrangian penalties provide the
        soft-constraint half.

        Shape-agnostic in the leading dimensions (broadcasts a [action_dim]
        default), so it can be applied to a single [n_reactors, action_dim]
        action during rollout AND to a [B, n_reactors, action_dim] minibatch
        inside the gradient loop.  The masking is a plain multiply by 0/1, so
        gradients flow only through the dims that actually reach the env — which
        is exactly the desired behaviour.

        Args:
            state_t: State tensor, shape [..., n_reactors, obs_dim].
            raw_squashed: Tanh-squashed action, shape [..., n_reactors, action_dim].
            mask: Precomputed stage mask; derived from state_t when None.

        Returns:
            Stage-masked action tensor, same shape as raw_squashed.
        """
        if not self.use_safeguard:
            return raw_squashed
        if mask is None:
            mask = self._stage_mask(state_t)
        return raw_squashed * mask + self._default_a * (1.0 - mask)

    def _safe_sample(self, obs, z, deterministic=False, with_logprob=True):
        """Sample an action, mask it, and mask its entropy accounting.

        THE POINT OF THIS HELPER is that the executed action stored in the replay
        buffer is the MASKED one, so the critic only ever learns Q(o, a) for
        actions whose locked dims sit at the SERL defaults.  Previously the actor
        loss evaluated ``critic(o, a_pi)`` with an UNMASKED ``a_pi``, and the
        Bellman target evaluated ``critic_target(o2, a2)`` with an unmasked
        ``a2`` — both querying the critic far off the data distribution on every
        locked dimension.  The actor was therefore optimising an extrapolated
        Q-surface for dims the environment ignores, in both agents.  Routing
        every policy sample that touches the critic through here fixes that.

        Args:
            obs: Observation tensor, shape [..., n_reactors, obs_dim].
            z: Latent context, shape [..., latent_dim].
            deterministic: If True, use the policy mean.
            with_logprob: If True, also return the masked log-probability.

        Returns:
            A tuple (action, logp, n_active) where action is stage-masked, logp
            excludes locked dims (None when with_logprob is False), and n_active
            is the per-sample count of free action dims (shape obs.shape[:-2]).
        """
        mask = self._stage_mask(obs)
        a, logp = self.actor.sample(obs, z, deterministic=deterministic,
                                    with_logprob=with_logprob, mask=mask)
        a = self._apply_safety(obs, a, mask=mask)
        if mask is None:
            n_active = torch.full(obs.shape[:-2],
                                  float(self.action_dim * self.n_reactors),
                                  device=device)
        else:
            n_active = mask.sum(dim=(-1, -2))
        return a, logp, n_active

    def select_action(self, state, deterministic=False):
        """Select a safe action for one control step using the current latent z.

        Args:
            state: Normalized observation, shape [n_reactors, obs_dim].
            deterministic: If True, use the policy mean (evaluation).

        Returns:
            A tuple (safe_action, raw_action) as NumPy arrays of shape
            [n_reactors, action_dim].  ``safe_action`` is sent to the environment;
            ``raw_action`` is the pre-safety policy sample (stored for diagnostics).
        """
        with torch.no_grad():
            state_t = torch.as_tensor(
                state, dtype=torch.float32, device=device)
            z_b = self.z.unsqueeze(0)                       # [1, z_dim]
            raw, _ = self.actor.sample(
                state_t.unsqueeze(0), z_b,
                deterministic=deterministic, with_logprob=False)
            raw = raw.squeeze(0)                            # [nR, action_dim]
            safe = self._apply_safety(state_t, raw)
        return safe.cpu().numpy(), raw.cpu().numpy()

    # ── Reward-scale tracking ───────────────────────────────────────────────

    def observe_rewards(self, rewards):
        """Fold freshly COLLECTED env rewards into the reward-scale estimate.

        Must be called from the data-collection path (once per env step, with
        each reward counted exactly once), NOT from the SAC update path.  The
        update path replays the same transitions thousands of times, which
        inflated RunningMeanStd's sample count by ~82k per outer iteration and
        froze the reward scale within ~100 iterations — see the RunningMeanStd
        docstring for why that was the primary divergence mechanism here.

        Do NOT call this during evaluation rollouts: the reward scale is part of
        the agent's learned state and evaluation must not mutate it.

        Args:
            rewards: Scalar or sequence of summed per-step env rewards.
        """
        if self.ret_rms is not None:
            self.ret_rms.update(rewards)

    # ── Off-policy SAC + encoder update ─────────────────────────────────────

    @property
    def alpha(self):
        """Current SAC entropy temperature."""
        return self.log_alpha.exp()

    def _encode_context_batch(self, ctx):
        """Encode a context minibatch (o, a, r, o') into a posterior and sample z.

        Args:
            ctx: Tuple (obs, act, reward, next_obs, done) of device tensors.

        Returns:
            A tuple (z, mu, sigma_sq): sampled latent and posterior parameters.
        """
        o, a, r, o2, _ = ctx
        r = r.reshape(-1, 1).float()
        # Same std-only normalization as the Bellman-target reward (see
        # update()), applied here so the encoder's reward input feature sits at
        # the same O(1) scale as the normalized obs/action features instead of
        # the raw, penalty-dominated magnitude.  Read-only: the running std is
        # updated only from freshly collected env steps (observe_rewards).
        if self.ret_rms is not None:
            r = r / self.ret_rms.std
        trans = torch.cat([
            o.reshape(o.shape[0], -1),
            a.reshape(a.shape[0], -1),
            r,
            o2.reshape(o2.shape[0], -1),
        ], dim=-1)
        mu, sigma_sq = self.encoder(trans)
        std = torch.sqrt(torch.clamp(sigma_sq, min=1e-8))
        z = mu + std * torch.randn_like(std)
        return z, mu, sigma_sq

    def update(self, replay, meta_batch=4, batch_size=256, context_size=64,
               min_buffer=None):
        """Run one PEARL/SAC gradient update over a meta-batch of tasks.

        For each sampled task the three SAC components are updated with SEPARATE
        zero/backward/step passes so that the actor loss (which backprops through
        the Q-networks) cannot corrupt the critic parameters:

            1. Critic (+ encoder + KL bottleneck): the twin soft-Bellman loss.
               Gradients flow into the context encoder via the latent z, and the
               KL(q(z|c) || N(0,I)) term regularizes the posterior.
            2. Actor: maximizes min-Q minus the entropy term, with z detached
               (PEARL convention — the actor does not shape task inference).
            3. Temperature alpha: automatic entropy tuning toward target entropy.

        Args:
            replay: MultiTaskReplayBuffer with per-task transitions.
            meta_batch: Number of tasks sampled per update.
            batch_size: RL minibatch size per task.
            context_size: Number of context transitions per task.
            min_buffer: Minimum transitions a task needs before it is eligible
                for an update.  Defaults to batch_size.  Because sample_batch
                draws with replacement, this can be set well below batch_size so
                training starts before every bucket holds a full batch.

        Returns:
            Dict of scalar diagnostics (losses, alpha), or None if no task has
            enough data yet.
        """
        thresh = batch_size if min_buffer is None else min_buffer
        ready = [t for t in replay.task_ids() if replay.size(t) >= thresh]
        if not ready:
            return None
        tasks = np.random.choice(
            ready, size=min(meta_batch, len(ready)), replace=False)

        # Accumulate as device tensors and defer the host sync (.item()) to a
        # single call after the loop — calling .item() per task forces a
        # CUDA synchronization on every one of the meta_batch task updates,
        # which serializes what would otherwise be async GPU kernel launches.
        critic_loss_acc = torch.zeros((), device=device)
        actor_loss_acc = torch.zeros((), device=device)
        kl_acc = torch.zeros((), device=device)
        alpha_loss_acc = torch.zeros((), device=device)
        # Mean |mu| (posterior MEAN magnitude) per update, tracked alongside
        # KL. IMPORTANT CORRECTION: this used to track the SAMPLED z
        # (mu + sigma*noise), which is a bad collapse indicator — when the
        # posterior fully collapses to the prior (mu~=0, sigma~=1), a sample
        # is just standard-normal noise, and E|N(0,1)| ~= 0.8. That produced
        # a real false signal: a run with KL=0.001 (posterior == prior, z
        # carries NO task info) still showed |z|~0.8-0.9 from the sampled
        # metric, looking "healthy" when the encoder was actually fully
        # collapsed. Tracking |mu| instead measures the actual posterior
        # location (the signal), not sampling noise, so it only stays large
        # when the encoder is genuinely encoding something task-dependent.
        # KL remains the primary/most reliable collapse indicator (it directly
        # measures distance from the prior); |mu| is a complementary, more
        # honest proxy than the old sampled-z version.
        z_norm_acc = torch.zeros((), device=device)

        for t in tasks:
            o, a, r, o2, d = replay.sample_batch(t, batch_size)
            o = o.float(); a = a.float(); o2 = o2.float()
            r = r.float()
            # Standardise the reward (by a running std) before scaling, so the
            # Bellman target stays O(1/(1-gamma)) regardless of the raw
            # penalty-dominated magnitude.  Std-only (no mean subtraction) keeps
            # the sign/ranking of rewards intact.
            #
            # READ-ONLY here.  The running statistics are now updated exclusively
            # from freshly collected env steps via observe_rewards(); updating
            # them from replayed minibatches (as this line used to) counted every
            # transition thousands of times, froze the scale at its
            # random-policy value, and starved every gradient in the run.  It
            # also cost a GPU->CPU sync per task per update.
            if self.ret_rms is not None:
                r = r / self.ret_rms.std
            r = r * self.reward_scale
            d = d.float()

            # ── Task inference (encoder) ────────────────────────────────────
            if self.use_context:
                ctx = replay.sample_context(t, context_size)
                z_sample, mu, sigma_sq = self._encode_context_batch(ctx)
                z = z_sample.unsqueeze(0).expand(o.shape[0], self._z_dim)
                # KL(q(z|c) || N(0, I)) information bottleneck.
                kl = 0.5 * torch.sum(
                    sigma_sq + mu.pow(2) - 1.0 - torch.log(sigma_sq + 1e-8))
                mu_for_log = mu
            else:
                z = torch.zeros(o.shape[0], self._z_dim, device=device)
                kl = torch.zeros((), device=device)
                mu_for_log = z[0]

            # ── 1. Critic (+ encoder + KL) update ───────────────────────────
            with torch.no_grad():
                # a2 is STAGE-MASKED before it reaches the critic, matching the
                # executed actions the buffer actually holds (see _safe_sample).
                a2, logp2, _ = self._safe_sample(o2, z.detach())
                q1_t, q2_t = self.critic_target(o2, a2, z.detach())
                # Clip the entropy term injected into the Bellman target. The
                # bounds on log_std/pre-tanh value above keep logp well-behaved
                # in the normal case, but this is a last-resort circuit breaker
                # so a single pathological sample can't inject an arbitrarily
                # large perturbation into the TD target regardless of cause.
                entropy_term = torch.clamp(
                    self.alpha.detach() * logp2, -ENTROPY_TERM_CLIP, ENTROPY_TERM_CLIP)
                q_t = torch.min(q1_t, q2_t) - entropy_term
                target = r + (1.0 - d) * self.gamma * q_t
            q1, q2 = self.critic(o, a, z)   # gradients flow to encoder via z
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            critic_obj = critic_loss + (self.kl_lambda * kl
                                        if self.use_context else 0.0)

            self.critic_opt.zero_grad()
            if self.use_context:
                self.encoder_opt.zero_grad()
            critic_obj.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
            self.critic_opt.step()
            if self.use_context:
                nn.utils.clip_grad_norm_(self.encoder.parameters(), 5.0)
                self.encoder_opt.step()

            # ── 2. Actor update (z detached; critic held fixed) ─────────────
            # actor_loss backprops through the (frozen-for-this-step) critic to
            # reach the actor; because we zero and step ONLY the actor optimizer,
            # the incidental critic gradients are discarded.
            # a_pi is STAGE-MASKED, so critic(o, a_pi) is queried on the same
            # action distribution the critic was trained on instead of
            # extrapolating on every structurally locked dimension.
            a_pi, logp_pi, n_active = self._safe_sample(o, z.detach())
            q1_pi, q2_pi = self.critic(o, a_pi, z.detach())
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha.detach() * logp_pi - q_pi).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
            self.actor_opt.step()

            # ── 3. Temperature update ───────────────────────────────────────
            # Entropy target scales with the number of dims the SERL mask leaves
            # FREE at each state.  logp_pi now excludes locked dims, so pairing
            # it with a fixed -12 total would demand the full 12-dim entropy
            # budget from however few dims happen to be active — driving alpha
            # up spuriously in late stages.  Reduces to the old fixed target
            # whenever all dims are active.
            target_ent = self._entropy_per_dim * n_active.detach()
            alpha_loss = -(self.log_alpha
                           * (logp_pi.detach() + target_ent)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            # Defensive circuit breaker: without this, a pathological entropy
            # signal can push log_alpha to grow without bound (observed here —
            # alpha climbing past 1.0 and continuing to rise every iteration).
            with torch.no_grad():
                # Lower bound is the configurable alpha floor when set, else the
                # bare numerical guard (see the alpha_min arg docstring for why
                # an unfloored alpha collapsed exploration here).
                self.log_alpha.clamp_(self._log_alpha_min, LOG_ALPHA_MAX)

            # ── Polyak target update (per task step) ────────────────────────
            # Fused multi-tensor ops (_foreach_*) issue ONE kernel for the whole
            # parameter list instead of one kernel per tensor.  This loop runs
            # meta_batch * updates_per_iter times per outer iteration — with
            # ~12 critic parameter tensors that was ~24 tiny sequential CUDA
            # launches per task-update; a bigger GPU doesn't help here since
            # each op is latency-bound (kernel-launch overhead), not
            # compute-bound.  Fusing cuts it to 2 launches regardless of
            # meta_batch, which is the actual lever for this loop's cost.
            with torch.no_grad():
                p_list = list(self.critic.parameters())
                tp_list = list(self.critic_target.parameters())
                torch._foreach_mul_(tp_list, 1.0 - self.tau)
                torch._foreach_add_(tp_list, p_list, alpha=self.tau)

            critic_loss_acc = critic_loss_acc + critic_loss.detach()
            actor_loss_acc = actor_loss_acc + actor_loss.detach()
            alpha_loss_acc = alpha_loss_acc + alpha_loss.detach()
            kl_acc = kl_acc + kl.detach()
            # mu_for_log is the posterior MEAN (not a sample) — see the
            # comment at z_norm_acc's declaration for why this replaced the
            # old sampled-z version. Always defined (zeros in baseline mode).
            z_norm_acc = z_norm_acc + mu_for_log.detach().abs().mean()

        n = len(tasks)
        # Single host sync for all five metrics instead of 5*meta_batch.
        critic_loss_v, actor_loss_v, alpha_loss_v, kl_v, z_norm_v = (
            torch.stack([critic_loss_acc, actor_loss_acc,
                        alpha_loss_acc, kl_acc, z_norm_acc]) / n).tolist()
        return {
            "critic_loss": critic_loss_v,
            "actor_loss": actor_loss_v,
            "alpha_loss": alpha_loss_v,
            "kl": kl_v,
            "z_norm": z_norm_v,
            "alpha": float(self.alpha.item()),
        }

    # ── Checkpointing ───────────────────────────────────────────────────────

    def save(self, path):
        """Save agent networks to a checkpoint file.

        Args:
            path: Destination file path (parent dirs are created).
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "use_context": self.use_context,
            # Recorded so a mismatched deployment window is caught on load
            # rather than silently reintroducing the train/deploy z mismatch
            # this field exists to prevent.
            "posterior_window": self.posterior_window,
        }
        if self.ret_rms is not None:
            # Cast explicitly: RunningMeanStd.update() already keeps these as
            # plain Python floats internally, but casting again here is a
            # free, harmless second guarantee that no numpy scalar type ever
            # reaches the pickle (those trip torch's weights_only=True default
            # unpickler on load — see _torch_load_safe).
            ckpt["ret_rms"] = (float(self.ret_rms.mean), float(self.ret_rms.var),
                               float(self.ret_rms.count))
        if self.use_context:
            ckpt["encoder"] = self.encoder.state_dict()
        torch.save(ckpt, path)

    @staticmethod
    def _torch_load_safe(path):
        """Load a checkpoint, handling both this class's current format and
        ones saved before the RunningMeanStd fix.

        Checkpoints written by the current code contain only plain Python
        floats/ints and tensors, so they load fine under torch's default
        ``weights_only=True`` safe-unpickler.  Checkpoints written before that
        fix embedded ``numpy.float64`` scalars in the ``ret_rms`` tuple (a
        side effect of ``ndarray.mean()``/``.var()`` returning numpy scalar
        types), which trips the safe-unpickler because that type isn't on its
        default allowlist.

        Falls back, in order:
          1. ``weights_only=True`` — the safe default; succeeds for anything
             saved by the current code.
          2. ``weights_only=True`` with a narrow ``safe_globals`` allowlist for
             the specific numpy scalar type an old-format checkpoint could
             contain (the officially recommended fix for this exact error) —
             still refuses anything else unexpected.
          3. ``weights_only=False``, as a last resort. Safe to do here because
             this is always a checkpoint this same codebase created locally,
             never an externally downloaded file.
        """
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            # Older torch without the weights_only kwarg at all.
            return torch.load(path, map_location=device)
        except Exception:
            pass

        try:
            allowed = []
            for mod_name, attr in (("numpy._core.multiarray", "scalar"),
                                   ("numpy.core.multiarray", "scalar")):
                try:
                    mod = __import__(mod_name, fromlist=[attr])
                    allowed.append(getattr(mod, attr))
                except (ImportError, AttributeError):
                    continue
            if allowed:
                with torch.serialization.safe_globals(allowed):
                    return torch.load(path, map_location=device, weights_only=True)
        except Exception:
            pass

        print(f"[Checkpoint] {path}: falling back to weights_only=False to "
              f"load a pre-fix checkpoint with embedded numpy scalars "
              f"(a locally-created file from this codebase, not an external "
              f"download).")
        return torch.load(path, map_location=device, weights_only=False)

    def load(self, path):
        """Load agent networks from a checkpoint file.

        Args:
            path: Source checkpoint file path.
        """
        ckpt = self._torch_load_safe(path)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"].to(device))
        if self.ret_rms is not None and "ret_rms" in ckpt:
            self.ret_rms.mean, self.ret_rms.var, self.ret_rms.count = ckpt["ret_rms"]
            # Checkpoints written before the bounded-window fix carry an enormous
            # count (~5e7, because the estimator was updated from every replayed
            # minibatch sample). Left as-is, the first few post-load updates
            # would be weighted to nothing. Clamp on load so the window is
            # honoured immediately.
            self.ret_rms.count = min(float(self.ret_rms.count),
                                     self.ret_rms.max_count)
        if self.use_context and "encoder" in ckpt:
            self.encoder.load_state_dict(ckpt["encoder"])
        # The actor was trained on z drawn from a posterior over exactly
        # `posterior_window` transitions; deploying it with a different window
        # hands it a systematically sharper (or flatter) z than it ever saw.
        # Adopt the trained value and say so, rather than silently keeping
        # whatever the caller constructed this agent with.
        ck_window = ckpt.get("posterior_window", "absent")
        if ck_window != "absent" and ck_window != self.posterior_window:
            print(f"[Checkpoint] {path}: posterior_window {self.posterior_window}"
                  f" -> {ck_window} (adopting the trained value; the actor's z "
                  f"distribution depends on it).")
            self.posterior_window = ck_window
            self._z_factors = (deque(maxlen=self.posterior_window)
                               if self.posterior_window else None)
            self.clear_z()
