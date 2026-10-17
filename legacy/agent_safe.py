"""Safe PPO agent for multi-reactor fermentation control.

This module implements a Proximal Policy Optimisation agent whose safety comes
from two layers: a structural SERL stage mask that locks physically inactive
action dimensions, and the environment's Lagrangian constraint penalties that
shape the reward away from violations.  (An earlier learned Action Projection
Network layer has been removed.)

Pipeline overview::

    Actor (GRU policy)  ->  tanh squash  ->  SERL stage mask  ->  environment

Key components:
    ActorCritic: Recurrent GRU actor-critic (DRPO) shared between actor and
        value heads.  Each reactor is an independent batch element with its
        own hidden state.
    PPO_Agent: Orchestrates rollout collection, GAE computation, TBPTT PPO
        updates, and policy/old-policy synchronisation.
    RunningMeanStd: Online Welford estimator used to normalise reward batches
        before GAE, reducing Lagrangian spike variance.
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
from collections import deque

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.distributions import Normal

# ─── Local ────────────────────────────────────────────────────────────────────
from env_safe import FermentationEnv

# ─── Recurrent Policy Hyperparameters ─────────────────────────────────────────

# Width of the GRU hidden state.  256 units balance temporal capacity against
# the parameter budget of a single-layer GRU shared by actor and critic heads.
GRU_HIDDEN_DIM = 256

# Number of stacked GRU layers.  2 layers allow the network to build a
# hierarchy of temporal abstractions: layer 1 tracks step-level fluctuations
# while layer 2 integrates these into multi-step trends such as "storage has
# been declining for 10 consecutive control steps" — critical for anticipating
# CV4 (storage drying) before the underflow hard limit is reached.
GRU_NUM_LAYERS = 2

# Sequence length for Truncated Backpropagation Through Time (TBPTT) during
# the PPO update.  Gradients flow through exactly SEQ_LEN GRU steps per chunk;
# shorter windows reduce memory usage; longer windows improve long-range credit
# assignment. 32 control steps = 160 simulated hours of fermentation history.
SEQ_LEN = 32

# ─── Backend Configuration ────────────────────────────────────────────────────

# Apply backend fixes for AMD ROCm/HIP on Windows to prevent C++ compiler failure
torch.backends.cudnn.enabled = False
if hasattr(torch.backends, 'miopen'):
    torch.backends.miopen.enabled = False

# Detect hardware acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Utility Classes ──────────────────────────────────────────────────────────


class RunningMeanStd:
    """Online reward-variance tracker using Welford's algorithm.

    Keeps a running mean and variance over all reward samples seen so far.
    Call ``update(x)`` each rollout, then ``normalize(x)`` to whiten rewards
    before GAE computation.  Whitened rewards reduce penalty-spike variance
    without destroying the reward signal direction.

    Attributes:
        mean: Running mean of observed samples.
        var: Running variance of observed samples.
        count: Total effective sample count (includes epsilon offset).
    """

    def __init__(self, epsilon=1e-4):
        """Initialise running statistics.

        Args:
            epsilon: Small constant added to initial count for numerical stability.
        """
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        """Incorporate a new batch of reward samples into the running statistics.

        Uses Welford's online algorithm for numerically stable one-pass
        mean and variance estimation.

        Args:
            x: Reward array of any shape. Flattened internally before update.
        """
        x_flat = x.flatten().astype(np.float64)
        n = len(x_flat)
        batch_mean = x_flat.mean()
        batch_var = x_flat.var()
        total = self.count + n
        delta = batch_mean - self.mean
        self.mean += delta * n / total
        m2 = self.var * self.count + batch_var * n + delta ** 2 * self.count * n / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x, clip=10.0):
        """Return a whitened copy of x using the running mean and variance.

        Output is clipped to [-clip, clip] to guard against outlier rewards
        from Lagrangian penalty spikes early in training.

        Args:
            x: Array to normalise. Must be broadcastable with self.mean.
            clip: Symmetric clipping bound applied after whitening.

        Returns:
            Normalised array of the same shape as x.
        """
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8), -clip, clip)


class PPOMemory:
    """Rollout buffer that accumulates one trajectory segment for the PPO update.

    When reserve() is called with known capacity and shapes, the four main
    tensors (states, actions, safe_actions, logprobs) are backed by contiguous
    pre-allocated NumPy arrays and filled via direct index assignment.  This
    eliminates per-step Python list growth and the expensive np.array(list)
    reconstruction at learn() time.

    rewards and is_terminals are kept as plain lists because their shapes
    vary (per-reactor reward vectors vs scalar bools) and they are processed
    differently in GAE computation.

    Call clear() after learn() consumes the buffer.

    Attributes:
        states: List or pre-allocated buffer of observation arrays.
        actions: List or pre-allocated buffer of latent 'z' from the actor (pre-tanh).
        safe_actions: List or pre-allocated buffer of projected safe actions sent to env.
        logprobs: List or pre-allocated buffer of log-probabilities.
        rewards: List of per-step reward arrays.
        is_terminals: List of terminal flags per step.
    """

    def __init__(self):
        """Initialise empty rollout buffer."""
        self.states = []
        self.actions = []       # Latent 'z' from the actor (pre-tanh)
        self.safe_actions = []  # Projected safe action actually sent to env
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self._preallocated = False
        self._ptr = 0

    def reserve(self, capacity, n_reactors, state_dim, action_dim):
        """Switch to preallocated array-backed storage.

        Must be called once before the first rollout starts.  Subsequent
        clear() calls reset the pointer without re-allocating.

        Args:
            capacity: Maximum number of steps before learn() is called.
            n_reactors: Number of reactors (batch dimension B).
            state_dim: Observation width per reactor.
            action_dim: Action width per reactor.
        """
        self._cap = capacity
        self._n_reactors = n_reactors
        self._states_buf    = np.empty((capacity, n_reactors, state_dim),  dtype=np.float32)
        self._actions_buf   = np.empty((capacity, n_reactors, action_dim), dtype=np.float32)
        self._safe_acts_buf = np.empty((capacity, n_reactors, action_dim), dtype=np.float32)
        self._logprobs_buf  = np.empty((capacity, n_reactors),             dtype=np.float32)
        self._preallocated = True
        self._ptr = 0

    def append(self, state, action, logprob, safe_action, reward, is_terminal):
        """Append one step; uses array indexing when preallocated.

        Args:
            state: Observation array for this step.
            action: Raw latent intent from the actor.
            logprob: Log-probability of the action under the old policy.
            safe_action: Stage-masked action sent to the environment.
            reward: Reward array for this step.
            is_terminal: Whether this step terminates the episode.
        """
        if self._preallocated:
            self._states_buf[self._ptr]    = state
            self._actions_buf[self._ptr]   = action
            self._logprobs_buf[self._ptr]  = logprob
            self._safe_acts_buf[self._ptr] = safe_action
            self._ptr += 1
        else:
            self.states.append(state)
            self.actions.append(action)
            self.logprobs.append(logprob)
            self.safe_actions.append(safe_action)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)

    def clear(self):
        """Reset the buffer; pointer-reset only when preallocated."""
        if self._preallocated:
            self._ptr = 0
        else:
            del self.states[:]
            del self.actions[:]
            del self.safe_actions[:]
            del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


# ─── Actor-Critic Architecture ────────────────────────────────────────────────


class CrossReactorAttention(nn.Module):
    """Cross-reactor scaled dot-product self-attention.

    Each reactor attends to all reactors' raw state features at every timestep,
    enabling the policy to capture coordination dynamics — for example, reactor 1
    unloading in Stage 3 raises central storage, so reactor 2 may safely increase
    feed.  Without this, each reactor is an independent batch element that cannot
    observe its siblings.

    Per-timestep data flow::

        [n_reactors, state_dim] -> Q/K/V projections -> multi-head attention
        -> output projection -> residual add -> LayerNorm -> [n_reactors, state_dim]

    Attention weights are stored in ``last_attn_weights`` for interpretability.

    Attributes:
        n_heads: Number of attention heads.
        n_reactors: Number of reactors (sequence length for attention).
        head_dim: Dimensionality per attention head.
        scale: Scaling factor for dot-product attention.
        last_attn_weights: Most recent attention weights [T, n_heads, B, B].
    """

    def __init__(self, feature_dim, n_heads=2, n_reactors=3):
        """Initialise the cross-reactor attention layer.

        Args:
            feature_dim: Input feature dimensionality (must be divisible by n_heads).
            n_heads: Number of attention heads.
            n_reactors: Number of reactors in the system.

        Raises:
            AssertionError: If feature_dim is not divisible by n_heads.
        """
        super().__init__()
        assert feature_dim % n_heads == 0, (
            f"feature_dim ({feature_dim}) must be divisible by n_heads ({n_heads})")

        self.n_heads = n_heads
        self.n_reactors = n_reactors
        self.head_dim = feature_dim // n_heads
        self.scale = self.head_dim ** 0.5

        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

        # Zero-init output projection so residual starts as identity — the model
        # begins with pre-attention behaviour and learns to deviate.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.last_attn_weights = None  # [T, n_heads, B, B] — set each forward

    def forward(self, x):
        """Enrich each reactor's features with cross-reactor context.

        Args:
            x: Input tensor of shape [B=n_reactors, T, feature_dim].

        Returns:
            A tuple of:
                - enriched: Tensor of shape [B, T, feature_dim] (cross-reactor aware).
                - attn_weights: Tensor of shape [T, n_heads, B, B] (detached, for
                  inspection).
        """
        B, T, D = x.shape
        H, d = self.n_heads, self.head_dim

        # [B, T, D] -> [T, B, D]: treat timesteps as batch, reactors as sequence
        xt = x.transpose(0, 1)

        # Project and reshape: [T, B, D] -> [T, H, B, d]
        Q = self.q_proj(xt).view(T, B, H, d).transpose(1, 2)
        K = self.k_proj(xt).view(T, B, H, d).transpose(1, 2)
        V = self.v_proj(xt).view(T, B, H, d).transpose(1, 2)

        # Scaled dot-product attention across reactor dimension
        attn = torch.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) / self.scale, dim=-1)
        self.last_attn_weights = attn.detach()

        # Combine heads: [T, H, B, d] -> [T, B, D]
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(T, B, D)

        # Residual + LayerNorm, then transpose back to [B, T, D]
        return self.norm(xt + self.out_proj(out)).transpose(0, 1), attn.detach()


class ActorCritic(nn.Module):
    """Recurrent Actor-Critic with cross-reactor attention (DRPO).

    Augments the standard GRU-based policy with a cross-reactor self-attention
    layer that allows each reactor to observe its siblings' states before the
    recurrent and skip-connection encoders.  This captures inter-reactor
    coordination dynamics that are invisible when reactors are processed as
    independent batch elements.

    Architecture::

        cross_reactor_attn  CrossReactorAttention(state_dim, n_heads=2)
                            Scaled dot-product across reactor dim -> residual + LN
        embed_gru           Linear(state_dim -> 256) -> LayerNorm -> ELU   [GRU path]
        embed_skip          Linear(state_dim -> 128) -> LayerNorm -> ELU   [skip path]
        gru                 GRU(256 -> 256, 2 layers, batch_first=True)
        gru_ln              LayerNorm(256)
        actor_latent_mean   MLP: 384 -> 128 -> latent_dim
        log_std             nn.Parameter(latent_dim)
        critic              MLP: 384 -> 128 -> 1

    Cross-reactor information flows solely through the attention layer;
    the GRU still maintains independent per-reactor hidden states.

    Interpretability:
        ``self.cross_reactor_attn.last_attn_weights``  [T, n_heads, B, B]
        attn[t, h, i, j] = how much reactor i attends to reactor j at step t.

    Attributes:
        current_log_std_max: Upper bound for log_std clamping (annealed externally).
        LOG_STD_MIN: Lower bound for log_std clamping.
        latent_dim: Width of the actor's Gaussian latent intent space.
        use_safeguard: Whether to apply SERL stage mask in act/evaluate.
        env_state_dim: Raw environment state width (for mask extraction).
        gru_hidden: GRU hidden state width.
    """

    def __init__(self, state_dim, action_dim, latent_dim,
                 use_safeguard=True, env_state_dim=28, n_reactors=3):
        """Initialise the recurrent actor-critic network.

        Args:
            state_dim: Observation dimensionality per reactor.
            action_dim: Number of physical action dimensions per reactor.
            latent_dim: Width of the actor's Gaussian latent intent space.
            use_safeguard: If True, applies SERL stage mask to mean/std.
            env_state_dim: Raw environment state width for mask extraction.
            n_reactors: Number of reactors in the system.
        """
        super().__init__()
        self.current_log_std_max = 0.0
        self.LOG_STD_MIN = -2.0
        self.latent_dim = latent_dim
        self.use_safeguard = use_safeguard
        self.env_state_dim = env_state_dim
        self.gru_hidden = GRU_HIDDEN_DIM

        # ── Cross-reactor attention ────────────────────────────────────────────
        self.cross_reactor_attn = CrossReactorAttention(
            feature_dim=state_dim, n_heads=2, n_reactors=n_reactors)

        # ── Per-step input encoders ────────────────────────────────────────────
        self.embed_gru = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ELU())
        self.embed_skip = nn.Sequential(
            nn.Linear(state_dim, 128), nn.LayerNorm(128), nn.ELU())

        # ── Shared temporal memory ─────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=GRU_HIDDEN_DIM,
            num_layers=GRU_NUM_LAYERS,
            batch_first=True
        )
        self.gru_ln = nn.LayerNorm(GRU_HIDDEN_DIM)

        # Orthogonal init on recurrent weights for stable gradient flow.
        for name, param in self.gru.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # ── Actor and critic heads ─────────────────────────────────────────────
        head_in = GRU_HIDDEN_DIM + 128  # 384
        head_hidden = 128
        self.actor_latent_mean = nn.Sequential(
            nn.Linear(head_in, head_hidden), nn.LayerNorm(head_hidden), nn.ELU(),
            nn.Linear(head_hidden, latent_dim))
        self.log_std = nn.Parameter(torch.zeros(latent_dim))

        self.critic = nn.Sequential(
            nn.Linear(head_in, head_hidden), nn.LayerNorm(head_hidden), nn.ELU(),
            nn.Linear(head_hidden, 1))

    def _encode_sequence(self, state_seq, h_init, done_seq=None):
        """Encode an observation sequence through attention, GRU, and skip paths.

        Args:
            state_seq: Observation tensor of shape [B, T, state_dim] (B = n_reactors).
            h_init: Initial hidden state of shape [num_layers, B, gru_hidden].
            done_seq: Optional float tensor of shape [T] with 1.0 at episode
                boundaries. When provided, the GRU hidden state is reset to
                zeros at each boundary.

        Returns:
            A tuple of:
                - feat: Concatenated features of shape [B, T, gru_hidden + 128].
                - h: Final GRU hidden state of shape [num_layers, B, gru_hidden].
        """
        B, T, _ = state_seq.shape

        # 1. Cross-reactor attention on raw state features
        enriched, _ = self.cross_reactor_attn(state_seq)  # [B, T, D]

        # 2. Per-step embedding (shared across GRU and skip paths)
        flat     = enriched.reshape(B * T, -1)
        emb      = self.embed_gru(flat).reshape(B, T, -1)   # [B, T, 256]
        emb_skip = self.embed_skip(flat).reshape(B, T, -1)  # [B, T, 128]

        if done_seq is not None and T > 1:
            # Segment-based processing: find episode boundaries and run the GRU
            # on each contiguous segment in a single call rather than T individual
            # single-step calls.  With typical rollout sizes the number of
            # boundaries is O(1-3), so this reduces T GRU kernel launches to ~2.
            h = h_init
            outs = []
            # Indices where done_seq[b]==1 -> h must be zeroed before step b+1.
            boundary_steps = (
                (done_seq > 0.5).nonzero(as_tuple=True)[0] + 1
            ).tolist()
            # Filter out out-of-range values (done at last step has no next step).
            boundary_steps = [b for b in boundary_steps if b < T]
            boundary_steps.append(T)   # sentinel: always process to end
            seg_start = 0
            for seg_end in boundary_steps:
                seg_emb = emb[:, seg_start:seg_end, :]     # [B, seg_len, H]
                out_seg, h = self.gru(seg_emb, h)
                outs.append(out_seg)
                seg_start = seg_end
                if seg_start < T:
                    h = torch.zeros_like(h)   # reset at episode boundary
            gru_out = torch.cat(outs, dim=1)  # [B, T, gru_hidden]
        else:
            # Fast path when no episode boundaries need explicit handling.
            gru_out, h = self.gru(emb, h_init)

        feat = self.gru_ln(gru_out)

        # Concatenate skip embedding to GRU output
        feat_with_skip = torch.cat([feat, emb_skip], dim=-1)
        return feat_with_skip, h

    def act(self, state, h=None):
        """Sample a latent intent for the current control step and advance h.

        Args:
            state: Current observation tensor of shape [B, state_dim]
                (B = n_reactors).
            h: Recurrent hidden state of shape [num_layers, B, gru_hidden] or
                None. Initialised to zeros when None (new episode starts).

        Returns:
            A tuple of:
                - z: Sampled raw latent intent of shape [B, latent_dim] (un-squashed).
                - log_prob: Log-probability log pi(z | s, h) of shape [B].
                - h_new: Updated hidden state of shape [num_layers, B, gru_hidden].
        """
        B = state.size(0)
        if h is None:
            h = torch.zeros(GRU_NUM_LAYERS, B, self.gru_hidden, device=state.device)

        # Single-step fast path (no done_seq: episode restarts manage h externally).
        feat, h_new = self._encode_sequence(state.unsqueeze(1), h)
        feat = feat.squeeze(1)   # [B, gru_hidden]

        mean = self.actor_latent_mean(feat)
        std = torch.exp(torch.clamp(
            self.log_std, self.LOG_STD_MIN, self.current_log_std_max))

        if self.use_safeguard:
            mask = FermentationEnv.get_action_mask(
                state[..., :self.env_state_dim])
            default_intent = torch.zeros_like(mean)
            default_intent[..., 0] = -0.34657359
            default_intent[..., 1:] = -10.0
            masked_mean = mean * mask + default_intent * (1 - mask)
            masked_std = std * mask + 1e-8 * (1 - mask)
        else:
            mask = torch.ones_like(mean)
            masked_mean = mean
            masked_std = std

        dist = Normal(masked_mean, masked_std)
        z = dist.sample()
        log_prob = (dist.log_prob(z) * mask).sum(dim=-1)   # [B]

        return z.detach(), log_prob.detach(), h_new.detach()

    def evaluate(self, state_seq, z_seq, h_init, done_seq):
        """Evaluate a rollout chunk under the current (updated) policy.

        Gradients flow back through the GRU for the full SEQ_LEN-step window (TBPTT).

        Args:
            state_seq: Batch-first observation chunk of shape [B, T, state_dim].
            z_seq: Stored raw latent intents of shape [B, T, latent_dim].
            h_init: Chunk initial hidden state of shape [num_layers, B, gru_hidden].
            done_seq: Float tensor of shape [T] with 1.0 at episode boundaries.

        Returns:
            A tuple of:
                - log_probs: Log-probabilities of shape [B, T].
                - state_values: Value estimates of shape [B, T, 1].
                - dist_entropy: Entropy of shape [B, T].
        """
        feat, _ = self._encode_sequence(state_seq, h_init, done_seq)

        mean = self.actor_latent_mean(feat)   # [B, T, latent_dim]
        std = torch.exp(torch.clamp(
            self.log_std, self.LOG_STD_MIN, self.current_log_std_max))

        if self.use_safeguard:
            B, T, _ = state_seq.shape
            raw = state_seq[..., :self.env_state_dim].reshape(
                B * T, self.env_state_dim)
            mask = FermentationEnv.get_action_mask(raw).reshape(B, T, -1)
            default_intent = torch.zeros_like(mean)
            default_intent[..., 0] = -0.34657359
            default_intent[..., 1:] = -10.0
            masked_mean = mean * mask + default_intent * (1 - mask)
            masked_std = std * mask + 1e-8 * (1 - mask)
        else:
            mask = torch.ones_like(mean)
            masked_mean = mean
            masked_std = std

        dist = Normal(masked_mean, masked_std)
        log_probs = (dist.log_prob(z_seq) * mask).sum(dim=-1)   # [B, T]
        dist_entropy = (dist.entropy() * mask).sum(dim=-1)   # [B, T]
        state_values = self.critic(feat)                            # [B, T, 1]

        return log_probs, state_values, dist_entropy


# ─── PPO Agent ────────────────────────────────────────────────────────────────


class PPO_Agent:
    """Safe Recurrent PPO agent for multi-reactor fermentation control.

    Combines three layers of safety enforcement around a recurrent actor-critic:

    1. **SERL stage mask** — hard binary mask derived from the reactor stage
       one-hot.  Locks action dimensions that are physically inactive (e.g.
       LF feed in Stage 0) to their minimum defaults before any network
       computation, making violations structurally impossible for those dims.

    2. **Lagrangian constraint penalties** — applied inside the environment
       reward, they grow when a physical constraint is violated and shape the
       policy gradient away from the constraint boundaries.

    (An earlier learned Action Projection Network layer, with its margin-weighted
    mapping penalty, has been removed.)

    Training uses Deep Recurrent Policy Optimisation (DRPO): GAE is computed
    with recurrent value estimates and gradient updates use Truncated
    Backpropagation Through Time (TBPTT) over SEQ_LEN-step chunks.
    Rewards are normalised online via a Welford running estimator to suppress
    variance from Lagrangian constraint-penalty spikes.

    Attributes:
        gamma: Discount factor for future rewards.
        eps_clip: PPO surrogate clipping range.
        K_epochs: Number of PPO update epochs per learn() call.
        entropy_coeff: Coefficient on the entropy bonus.
        policy: Current actor-critic network.
        policy_old: Old (frozen) actor-critic for rollout collection.
        reward_rms: Running reward normalisation statistics.
        h: Live recurrent hidden state across select_action() calls.
    """

    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma,
                 K_epochs, eps_clip, entropy_coeff, latent_dim=12,
                 use_safeguard=True):
        """Initialise the safe PPO agent.

        Args:
            state_dim: Raw environment observation width (28 per reactor).
            action_dim: Number of physical action dimensions per reactor (4).
            lr_actor: Initial learning rate for the GRU encoder and actor head.
            lr_critic: Initial learning rate for the critic head (kept >= lr_actor * 3
                by the floor logic in main.py).
            gamma: Discount factor for future rewards.
            K_epochs: Number of PPO update epochs per learn() call.
            eps_clip: PPO surrogate clipping range (ratio in [1-eps, 1+eps]).
                Also used as the clipping range for the value loss.
            entropy_coeff: Coefficient on the entropy bonus; annealed by main.py
                from initial_entropy down to min_entropy.
            latent_dim: Width of the actor's Gaussian latent intent space.
                Defaults to action_dim (4) so each dim is independent.
            use_safeguard: If True, applies the SERL stage mask (structural
                action locking) in the actor.
        """
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coeff = entropy_coeff
        self.raw_state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.use_safeguard = use_safeguard

        # The actor-critic operates directly on the raw env state.
        self.aug_state_dim = state_dim

        # 1. Initialize Primary Policy (Predicts Z)
        self.policy = ActorCritic(self.aug_state_dim, action_dim, latent_dim,
                                  use_safeguard, env_state_dim=self.raw_state_dim).to(device)

        self.optimizer = torch.optim.Adam([
            # Group 0: shared encoder (attention + embeddings + GRU) at actor LR.
            {'params': (list(self.policy.cross_reactor_attn.parameters()) +
                        list(self.policy.embed_gru.parameters()) +
                        list(self.policy.embed_skip.parameters()) +
                        list(self.policy.gru.parameters()) +
                        list(self.policy.gru_ln.parameters())), 'lr': lr_actor},
            # Group 1: actor head
            {'params': self.policy.actor_latent_mean.parameters(), 'lr': lr_actor},
            # Group 2: exploration
            {'params': [self.policy.log_std], 'lr': lr_actor},
            # Group 3: critic head (index matches main.py param_groups[3] LR floor).
            {'params': self.policy.critic.parameters(), 'lr': lr_critic,
             'weight_decay': 1e-5}
        ])

        self.policy_old = ActorCritic(
            self.aug_state_dim, action_dim, latent_dim, use_safeguard,
            env_state_dim=self.raw_state_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.valueLoss = nn.SmoothL1Loss()

        # Running stats for reward normalisation (Welford online estimator).
        # Updated once per learn() call with the full rollout reward batch.
        self.reward_rms = RunningMeanStd()

        # ── Recurrent hidden state ─────────────────────────────────────────────
        # self.h         : the live [1, n_reactors, GRU_HIDDEN_DIM] hidden state
        #                  maintained across select_action() calls within an episode.
        #                  Reset to None in reset() so the next act() call starts fresh.
        # self._h_buffer : list of numpy snapshots of self.h captured BEFORE each
        #                  select_action() call.  Consumed by learn() as TBPTT chunk
        #                  initial states. Cleared at the end of each learn() call.
        self.h = None
        self._h_buffer = []

        # AMP: mixed-precision scaler for the PPO update backward pass.
        # Only active when CUDA is available; no-ops gracefully on CPU.
        self._amp_enabled = torch.cuda.is_available()
        self._scaler = GradScaler('cuda', enabled=self._amp_enabled)

    # ── Episode Management ─────────────────────────────────────────────────────

    def reset(self):
        """Clear per-episode recurrent state at episode boundaries.

        Must be called immediately after env.reset() in both the training and
        evaluation loops to prevent GRU hidden-state leakage from the previous
        trajectory. (_h_buffer is NOT cleared here — it spans the full rollout
        and is only consumed inside learn().)
        """
        self.h = None

    # ── Action Selection ───────────────────────────────────────────────────────

    def select_action(self, state):
        """Select a stage-masked action for one control step.

        Executes Actor -> tanh squash -> SERL stage mask, and records the
        hidden-state snapshot needed by learn() for TBPTT warm-starting.

        The SERL stage mask locks structurally inactive action dimensions to
        their physical defaults (e.g. LF feed = 0 outside the feed stages).  The
        raw pre-tanh latent is preserved unmodified for the PPO log-prob ratio.

        Args:
            state: Normalised observation array of shape [n_reactors, state_dim]
                from env.get_state_norm() or env.reset().

        Returns:
            A tuple of:
                - final_action: Stage-masked action in [-1, 1] of shape
                  [n_reactors, action_dim] (sent to the environment).
                - raw_intent: Unmodified pre-tanh latent z of shape
                  [n_reactors, latent_dim]. Stored in memory for PPO log-prob
                  computation in learn().
                - log_prob: Log pi_old(z | s, h) at collection time of shape
                  [n_reactors].
                - state_aug: The tensor-form state that was actually fed to the
                  actor of shape [n_reactors, state_dim] (stored in memory for
                  the TBPTT update).
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(device)

            # A. Actor: sample raw latent intent with recurrent memory.
            # Snapshot h BEFORE this step so learn() can use it as the TBPTT
            # chunk initial state for whichever SEQ_LEN window starts here.
            _n_react = state_t.size(0)
            h_snap = (self.h.cpu().numpy() if self.h is not None
                      else np.zeros((GRU_NUM_LAYERS, _n_react, GRU_HIDDEN_DIM),
                                    dtype=np.float32))
            self._h_buffer.append(h_snap)

            unsafe_action, log_prob, h_new = self.policy_old.act(
                state_t, self.h)
            self.h = h_new

            # B. Tanh squash to [-1, 1] action space.  The raw intent MUST remain
            # unmodified for PPO logprobs — masking is applied only to the squashed
            # action that is sent to the environment.
            raw_intent = unsafe_action.clone()
            final_action = torch.tanh(unsafe_action)

            # C. SERL stage mask: lock structurally inactive dims to their squashed
            # defaults (-0.34657359 -> time_mult ~ 0.5x min; -10.0 -> flow = 0).
            if self.use_safeguard:
                mask = FermentationEnv.get_action_mask(
                    state_t[..., :self.policy.env_state_dim]).to(device)
                default_latent = torch.zeros_like(unsafe_action)
                default_latent[..., 0] = -0.34657359
                default_latent[..., 1:] = -10.0
                default_squashed = torch.tanh(default_latent[0])   # [4] constant
                final_action = (final_action * mask
                                + default_squashed.unsqueeze(0) * (1 - mask))

        return (final_action.cpu().numpy(), raw_intent.cpu().numpy(),
                log_prob.cpu().numpy(), state_t.cpu().numpy())

    # ── PPO Update Internals ───────────────────────────────────────────────────

    def _prepare_tensors(self, memory):
        """Materialise rollout memory as GPU tensors.

        Args:
            memory: PPOMemory instance containing the collected rollout.

        Returns:
            A tuple of (old_states, old_latents, old_logprobs, old_safe_acts,
            done_t, rewards, is_terminals) as device tensors and raw lists.
        """
        if memory._preallocated:
            n = memory._ptr
            old_states    = torch.from_numpy(memory._states_buf[:n]).to(device)
            old_latents   = torch.from_numpy(memory._actions_buf[:n]).to(device)
            old_logprobs  = torch.from_numpy(memory._logprobs_buf[:n]).squeeze().to(device)
            old_safe_acts = torch.from_numpy(memory._safe_acts_buf[:n]).to(device)
        else:
            old_states    = torch.tensor(np.array(memory.states),      dtype=torch.float32).to(device)
            old_latents   = torch.tensor(np.array(memory.actions),     dtype=torch.float32).to(device)
            old_logprobs  = torch.tensor(np.array(memory.logprobs),    dtype=torch.float32).squeeze().to(device)
            old_safe_acts = torch.tensor(np.array(memory.safe_actions),dtype=torch.float32).to(device)

        done_t = torch.tensor(np.array(memory.is_terminals), dtype=torch.float32).to(device)
        return old_states, old_latents, old_logprobs, old_safe_acts, done_t, memory.rewards, memory.is_terminals

    def _compute_advantages(self, old_states, done_t, rewards, is_terminals, lam):
        """Compute Generalised Advantage Estimation (GAE) with recurrent value estimates.

        Args:
            old_states: State tensor of shape [T, B, state_dim].
            done_t: Terminal flag tensor of shape [T].
            rewards: List of reward arrays.
            is_terminals: List of terminal booleans.
            lam: GAE lambda for bias-variance trade-off.

        Returns:
            A tuple of:
                - b_advs: Normalised advantage tensor of shape [T, B].
                - b_returns: Return targets of shape [T, B].
                - values_t: Raw value estimates of shape [B, T].
        """
        T = old_states.shape[0]
        B = old_states.shape[1]

        with torch.no_grad():
            seq = old_states.permute(1, 0, 2)  # [B, T, D]
            h_v = torch.zeros(GRU_NUM_LAYERS, B, GRU_HIDDEN_DIM, device=device)
            feat_v, _ = self.policy._encode_sequence(seq, h_v, done_seq=done_t)
            values_t = self.policy.critic(feat_v).squeeze(-1)  # [B, T]
            values = values_t.permute(1, 0).cpu().numpy()      # [T, B]

            values_padded = np.vstack([values, np.zeros((1, B))])
            rewards_arr = np.array(rewards, dtype=np.float32)

            self.reward_rms.update(rewards_arr)
            rewards_arr = self.reward_rms.normalize(rewards_arr)

            advantages = np.zeros_like(rewards_arr)
            gae = np.zeros(B, dtype=np.float32)

            for t in reversed(range(T)):
                mask_val = 1.0 - is_terminals[t]
                delta = (rewards_arr[t] + self.gamma * values_padded[t + 1] * mask_val
                         - values_padded[t])
                gae = delta + self.gamma * lam * mask_val * gae
                advantages[t] = gae

            returns = advantages + values
            b_advs = torch.tensor(advantages, dtype=torch.float32).to(device)
            b_returns = torch.tensor(returns, dtype=torch.float32).to(device)
            b_advs = (b_advs - b_advs.mean()) / (b_advs.std() + 1e-8)

        return b_advs, b_returns, values_t

    def _build_tbptt_chunks(self, old_states, old_latents, done_t, b_advs,
                            b_returns, old_logprobs, old_safe_acts, values_t):
        """Build TBPTT sequence chunks of size SEQ_LEN.

        Args:
            old_states: State tensor of shape [T, B, state_dim].
            old_latents: Latent tensor of shape [T, B, latent_dim].
            done_t: Terminal flag tensor of shape [T].
            b_advs: Advantage tensor of shape [T, B].
            b_returns: Return tensor of shape [T, B].
            old_logprobs: Log-prob tensor of shape [T, B] or [T].
            old_safe_acts: Safe action tensor of shape [T, B, action_dim].
            values_t: Value tensor of shape [B, T].

        Returns:
            List of chunk tuples, each containing (c_states, c_latents, c_dones,
            c_advs, c_rets, c_old_lp, h_init, c_safe_acts, c_old_vals).
        """
        T = old_states.shape[0]
        B = old_states.shape[1]
        h_stored = self._h_buffer[:T]

        n_chunks = T // SEQ_LEN
        effective_seq = SEQ_LEN if n_chunks > 0 else T
        n_chunks = max(1, n_chunks)

        chunks = []
        for ci in range(n_chunks):
            s = ci * effective_seq
            e = s + effective_seq

            # Reorder back to [B, L, *] for GRU input
            c_states = old_states[s:e].permute(1, 0, 2)
            c_latents = old_latents[s:e].permute(1, 0, 2)
            c_dones = done_t[s:e]
            c_advs = b_advs[s:e].permute(1, 0)
            c_rets = b_returns[s:e].permute(1, 0)

            if old_logprobs.dim() == 2:
                c_old_lp = old_logprobs[s:e].permute(1, 0)
            else:
                c_old_lp = old_logprobs[s:e].unsqueeze(0).expand(B, -1)

            c_safe_acts = old_safe_acts[s:e].permute(1, 0, 2)

            if s < len(h_stored):
                h_init = torch.tensor(h_stored[s], dtype=torch.float32, device=device)
            else:
                h_init = torch.zeros(GRU_NUM_LAYERS, B, GRU_HIDDEN_DIM, device=device)

            c_old_vals = values_t[:, s:e]
            chunks.append((c_states, c_latents, c_dones, c_advs, c_rets,
                           c_old_lp, h_init, c_safe_acts, c_old_vals))

        return chunks

    def _compute_loss(self, chunk, dist_entropy, logprobs, state_values):
        """Compute the composite PPO loss for a single TBPTT chunk.

        Args:
            chunk: Tuple from _build_tbptt_chunks containing all chunk data.
            dist_entropy: Entropy tensor of shape [B, L].
            logprobs: Log-probability tensor of shape [B, L].
            state_values: Value estimates of shape [B, L, 1].

        Returns:
            Scalar composite loss combining clipped surrogate, value, and
            entropy terms.
        """
        _, _, _, c_advs, c_rets, c_old_lp, _, _, c_old_vals = chunk

        ratios = torch.exp(logprobs - c_old_lp.detach())  # [B, L]
        surr1 = ratios * c_advs
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * c_advs

        sv = state_values.squeeze(-1)  # [B, L]
        v_clipped = c_old_vals.detach() + torch.clamp(
            sv - c_old_vals.detach(), -self.eps_clip, self.eps_clip)
        value_loss = torch.max(
            (sv - c_rets).pow(2), (v_clipped - c_rets).pow(2)).mean()

        loss = (
            -torch.min(surr1, surr2).mean()
            + 0.5 * value_loss
            - self.entropy_coeff * dist_entropy.mean()
        )
        return loss

    # ── Main Learning Step ─────────────────────────────────────────────────────

    def learn(self, memory, batch_size=None, lam=0.95):
        """Recurrent PPO update (Deep Recurrent Policy Optimisation).

        Executes the full update pipeline in modular stages:
            1. Tensor materialisation from rollout memory.
            2. Advantage estimation (GAE) with recurrent value network.
            3. Sequence chunking for TBPTT.
            4. SGD epoch loop with clipped surrogate + value + entropy loss.
            5. Policy synchronisation and state cleanup.

        Args:
            memory: PPOMemory instance containing the collected rollout.
            batch_size: Unused (kept for API compatibility).
            lam: GAE lambda for bias-variance trade-off.
        """
        # Step 1: Prepare Tensors
        (old_states, old_latents, old_logprobs, old_safe_acts,
         done_t, rewards, is_terminals) = self._prepare_tensors(memory)

        # Step 2: GAE Computation
        b_advs, b_returns, values_t = self._compute_advantages(
            old_states, done_t, rewards, is_terminals, lam)

        # Step 3: TBPTT Chunking
        chunks = self._build_tbptt_chunks(
            old_states, old_latents, done_t, b_advs, b_returns,
            old_logprobs, old_safe_acts, values_t)

        # Step 4: SGD Epochs
        chunk_indices = np.arange(len(chunks))
        for _ in range(self.K_epochs):
            np.random.shuffle(chunk_indices)
            for ci in chunk_indices:
                chunk = chunks[ci]
                c_states, c_latents, c_dones, _, _, _, h_init, _, _ = chunk

                with autocast(device_type='cuda', enabled=self._amp_enabled):
                    logprobs, state_values, dist_entropy = self.policy.evaluate(
                        c_states, c_latents, h_init.detach(), c_dones)
                    loss = self._compute_loss(
                        chunk, dist_entropy, logprobs, state_values)

                # Gradient step
                self.optimizer.zero_grad()
                self._scaler.scale(loss).backward()
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self._scaler.step(self.optimizer)
                self._scaler.update()

        # Step 5: Sync old policy & Reset State
        self.policy_old.load_state_dict(self.policy.state_dict())
        self._h_buffer = []
        self.h = None
