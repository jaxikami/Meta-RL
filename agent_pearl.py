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


def product_of_gaussians(mus, sigmas_sq):
    """Combine independent Gaussian factors into one via product of Gaussians.

    Implements PEARL's permutation-invariant aggregation over a set of
    per-transition Gaussian factors N(mu_i, sigma_i^2).  The product of Gaussians
    is itself Gaussian with precision-weighted mean.

    Args:
        mus: Tensor of factor means, shape [N, latent_dim].
        sigmas_sq: Tensor of factor variances, shape [N, latent_dim] (positive).

    Returns:
        A tuple (mu, sigma_sq) of the combined posterior, each shape [latent_dim].
    """
    sigmas_sq = torch.clamp(sigmas_sq, min=1e-7)
    sigma_sq = 1.0 / torch.sum(1.0 / sigmas_sq, dim=0)
    mu = sigma_sq * torch.sum(mus / sigmas_sq, dim=0)
    return mu, sigma_sq


# ─── Reward Normalisation ─────────────────────────────────────────────────────


class RunningMeanStd:
    """Online (Welford) running mean/variance estimator for reward scaling.

    Used to standardise the SAC reward target to roughly unit scale.  The env's
    per-step reward is penalty-dominated and large in magnitude (order 1-10),
    which pushes Q-values (and hence the Bellman loss) into a range where the
    KL bottleneck and entropy terms become negligible.  Dividing the reward by a
    running standard deviation keeps the critic target O(1/(1-gamma)) regardless
    of the raw penalty magnitude, so the other loss terms stay meaningful.
    """

    def __init__(self, eps=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

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
        self.count = tot

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

    def sample(self, obs, z, deterministic=False, with_logprob=True):
        """Sample a tanh-squashed action and its log-probability.

        Args:
            obs: Observation tensor, shape [B, n_reactors, obs_dim].
            z: Latent context, shape [B, latent_dim].
            deterministic: If True, return tanh(mean) with no sampling noise.
            with_logprob: If True, also return the summed log-probability.

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
        # Sum over action dims and reactors -> per-sample scalar log-prob.
        logp = dist.log_prob(u) - log_one_minus_a_sq
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
                 target_entropy=None):
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
        self.ret_rms = RunningMeanStd() if normalize_rewards else None
        self.use_context = use_context
        self.use_safeguard = use_safeguard

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
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        # Safety layer: the SERL stage mask (applied in _apply_safety) plus the
        # environment's Lagrangian penalties.  No learned projection network.

        # ── Live posterior state (maintained during a rollout) ──────────────
        self.z_mu = torch.zeros(self._z_dim, device=device)
        self.z_sigma_sq = torch.ones(self._z_dim, device=device)
        self.z = torch.zeros(self._z_dim, device=device)
        self._z_precision_sum = torch.zeros(self._z_dim, device=device)
        self._z_weighted_mu_sum = torch.zeros(self._z_dim, device=device)

    # ── Latent-context management ───────────────────────────────────────────

    def clear_z(self):
        """Reset the running posterior to the N(0, I) prior and resample z.

        Called at the start of each rollout episode: the agent begins from the
        prior and refines its belief as it accumulates task transitions.
        """
        self.z_mu = torch.zeros(self._z_dim, device=device)
        self.z_sigma_sq = torch.ones(self._z_dim, device=device)
        # Running precision-weighted accumulators for the incremental
        # product-of-Gaussians posterior (see update_context).
        self._z_precision_sum = torch.zeros(self._z_dim, device=device)
        self._z_weighted_mu_sum = torch.zeros(self._z_dim, device=device)
        self._resample_z()

    def _resample_z(self):
        """Sample z from the current posterior (or fix z=0 in baseline mode)."""
        if not self.use_context:
            self.z = torch.zeros(self._z_dim, device=device)
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
        # running stats are only *updated* from training minibatches in
        # update(), never from update_context, so calling this during
        # evaluation rollouts cannot contaminate the statistics).  Without
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
        self._z_precision_sum = self._z_precision_sum + prec_i
        self._z_weighted_mu_sum = self._z_weighted_mu_sum + mu_i * prec_i
        self.z_sigma_sq = 1.0 / self._z_precision_sum
        self.z_mu = self.z_sigma_sq * self._z_weighted_mu_sum
        self._resample_z()

    # ── Safe action selection ───────────────────────────────────────────────

    def _apply_safety(self, state_t, raw_squashed):
        """Apply the SERL stage mask to a squashed action.

        Locks structurally inactive action dimensions to their physical defaults
        (e.g. LF feed = 0 outside the feed stages).  This is the structural half
        of the safety stack; the environment's Lagrangian penalties provide the
        soft-constraint half.

        Args:
            state_t: State tensor, shape [n_reactors, obs_dim].
            raw_squashed: Tanh-squashed action, shape [n_reactors, action_dim].

        Returns:
            Stage-masked action tensor, shape [n_reactors, action_dim].
        """
        if not self.use_safeguard:
            return raw_squashed
        mask = FermentationEnv.get_action_mask(
            state_t[..., :self.obs_dim]).to(device)
        default_latent = torch.zeros_like(raw_squashed)
        default_latent[..., 0] = -0.34657359
        default_latent[..., 1:] = -10.0
        default_squashed = torch.tanh(default_latent[0])
        return raw_squashed * mask + default_squashed.unsqueeze(0) * (1 - mask)

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
        # the raw, penalty-dominated magnitude.  Read-only: the running std was
        # already updated this task-step from the RL minibatch in update().
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
            if self.ret_rms is not None:
                self.ret_rms.update(r.detach().cpu().numpy())
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
                a2, logp2 = self.actor.sample(o2, z.detach())
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
            a_pi, logp_pi = self.actor.sample(o, z.detach())
            q1_pi, q2_pi = self.critic(o, a_pi, z.detach())
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha.detach() * logp_pi - q_pi).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
            self.actor_opt.step()

            # ── 3. Temperature update ───────────────────────────────────────
            alpha_loss = -(self.log_alpha
                           * (logp_pi.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            # Defensive circuit breaker: without this, a pathological entropy
            # signal can push log_alpha to grow without bound (observed here —
            # alpha climbing past 1.0 and continuing to rise every iteration).
            with torch.no_grad():
                self.log_alpha.clamp_(LOG_ALPHA_MIN, LOG_ALPHA_MAX)

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
        if self.use_context and "encoder" in ckpt:
            self.encoder.load_state_dict(ckpt["encoder"])
