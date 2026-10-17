"""The agent's networks.

Token actor and critics with cross-reactor attention, the GRU context encoder,
the auxiliary head and the adaptation gate's detector.
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import math

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import N_REACTORS, REACTOR_OBS_DIM, GLOBAL_OBS_DIM
from metarl.env.layout import GLOBAL_START, CER_START
from metarl.agent.common import (LATENT_DIM, LOG_STD_MAX, LOG_STD_MIN,
                                 PRE_TANH_CLIP, MU_CLAMP)


class FiLM(nn.Module):
    """Feature-wise linear modulation ``h * (1 + gamma) + beta`` by z.

    Zero-initialised: identity at start, yet z gets gradient from step one.
    """

    def __init__(self, latent_dim, feature_dim):
        super().__init__()
        self.proj = nn.Linear(latent_dim, 2 * feature_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.feature_dim = feature_dim

    def forward(self, h, z):
        gamma, beta = self.proj(z).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta



def token_index():
    """Observation channels of each reactor's token, shape (N_REACTORS, 28).

    A reactor's token is its own 24-channel block, its off-gas CO2 rate (from
    the observation's tail) and the three plant-wide channels.
    """
    rows = []
    for i in range(N_REACTORS):
        rows.append(list(range(i * REACTOR_OBS_DIM, (i + 1) * REACTOR_OBS_DIM))
                    + [CER_START + i]
                    + list(range(GLOBAL_START, GLOBAL_START + GLOBAL_OBS_DIM)))
    return torch.tensor(rows, dtype=torch.long)


TOKEN_DIM = REACTOR_OBS_DIM + 1 + GLOBAL_OBS_DIM


class _MixerBlock(nn.Module):
    """One pre-LN transformer block over the reactor tokens.

    Both residual branches end in zero-initialised projections, so the block
    starts as the identity and learns how far the reactors should inform each
    other.
    """

    def __init__(self, dim, n_heads, ffn_mult=2):
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"token width {dim} is not divisible by "
                             f"{n_heads} heads")
        self.n_heads = int(n_heads)
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, ffn_mult * dim)
        self.ff2 = nn.Linear(ffn_mult * dim, dim)
        for lin in (self.out, self.ff2):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, h):
        """Mix tokens ``h`` of shape [B, T, D]."""
        B, T, D = h.shape
        H = self.n_heads
        q, k, v = self.qkv(self.ln1(h)).view(B, T, 3, H, D // H).permute(
            2, 0, 3, 1, 4)
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(D // H), dim=-1)
        o = (att @ v).transpose(1, 2).reshape(B, T, D)
        h = h + self.out(o)
        return h + self.ff2(F.relu(self.ff1(self.ln2(h))))


class TokenMixer(nn.Module):
    """A stack of ``_MixerBlock``s with a final LayerNorm."""

    def __init__(self, dim, n_layers=2, n_heads=4):
        super().__init__()
        self.blocks = nn.ModuleList(_MixerBlock(dim, n_heads)
                                    for _ in range(int(n_layers)))
        self.norm = nn.LayerNorm(dim)

    def forward(self, h):
        for b in self.blocks:
            h = b(h)
        return self.norm(h)


def _encoder_inputs(context, obs_dim, obs_norm, delta_norm):
    """Normalise flattened transitions the way the context encoder reads them.

    With ``delta_norm`` each transition is read as (obs, action, reward,
    next_obs - obs), the change standardised on its own scale: the flow
    latents live in those changes (a 0.15%/h leak moves a reactor's mass
    channel by ~1/50 of its spread per control step).
    """
    d = obs_dim
    if delta_norm is not None:
        obs, obs2 = context[:, :d], context[:, -d:]
        return torch.cat([obs_norm(obs) if obs_norm is not None else obs,
                          context[:, d:-d], delta_norm(obs2 - obs)], dim=-1)
    if obs_norm is not None:
        return torch.cat([obs_norm(context[:, :d]), context[:, d:-d],
                          obs_norm(context[:, -d:])], dim=-1)
    return context


class RecurrentContextEncoder(nn.Module):
    """GRU posterior over z from the whole episode so far (VariBAD-style).

    Each transition is normalised, embedded and folded into a GRU state; a
    head maps the state to a diagonal Gaussian over z. With an empty context
    the posterior is the N(0, I) prior.
    """

    def __init__(self, transition_dim, latent_dim=LATENT_DIM, hidden=256,
                 obs_dim=None, obs_norm=None, layer_norm=False,
                 delta_norm=None):
        super().__init__()
        if (obs_norm is not None or delta_norm is not None) and obs_dim is None:
            raise ValueError("obs_norm / delta_norm need obs_dim to find the "
                             "observations")
        self.latent_dim = latent_dim
        self.hidden = int(hidden)
        self.obs_dim = obs_dim
        self.obs_norm = obs_norm
        self.delta_norm = delta_norm
        layers = [nn.Linear(transition_dim, self.hidden)]
        if layer_norm:
            layers.append(nn.LayerNorm(self.hidden))
        layers.append(nn.ReLU())
        self.embed = nn.Sequential(*layers)
        self.cell = nn.GRUCell(self.hidden, self.hidden)
        self.head = nn.Linear(self.hidden, 2 * latent_dim)

    def embed_inputs(self, context):
        """Normalised, embedded transitions, shape [N, hidden]."""
        return self.embed(_encoder_inputs(context, self.obs_dim,
                                          self.obs_norm, self.delta_norm))

    def step(self, context, h):
        """Fold transitions [B, transition_dim] into GRU states [B, hidden]."""
        return self.cell(self.embed_inputs(context), h)

    def posterior(self, h):
        """The Gaussian over z a state stands for: (mu, sigma_sq), each [B, D]."""
        out = self.head(h)
        mu = torch.clamp(out[:, :self.latent_dim], -MU_CLAMP, MU_CLAMP)
        sigma_sq = F.softplus(out[:, self.latent_dim:]) + 1e-6
        return mu, sigma_sq


class MLPTrunk(nn.Module):
    """Hidden layers conditioned on the latent z (the lean agent's trunk).

    Plain: Linear -> LayerNorm -> FiLM(z) -> ReLU per layer. Residual (pre-LN,
    SimBa / BRO): the first Linear embeds the input into a residual stream,
    each further layer adds ``Linear(ReLU(FiLM(LayerNorm(h))))``, and a final
    LayerNorm -> FiLM -> ReLU feeds the heads.
    """

    def __init__(self, in_dim, hidden, depth, latent_dim, use_film=True,
                 layer_norm=True, residual=False):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.use_film = use_film
        self.residual = bool(residual)
        dims = [in_dim + (0 if use_film else latent_dim)] + [hidden] * depth
        self.fcs = nn.ModuleList(nn.Linear(a, b)
                                 for a, b in zip(dims[:-1], dims[1:]))
        self.norms = nn.ModuleList(
            nn.LayerNorm(hidden) if layer_norm else nn.Identity()
            for _ in range(depth))
        self.films = (nn.ModuleList(FiLM(latent_dim, hidden)
                                    for _ in range(depth))
                      if use_film else None)

    def forward(self, x, z):
        if not self.use_film:
            x = torch.cat([x, z], dim=-1)
        if self.residual:
            h = self.fcs[0](x)
            for i in range(1, len(self.fcs)):
                y = self.norms[i - 1](h)
                if self.use_film:
                    y = self.films[i - 1](y, z)
                h = h + self.fcs[i](F.relu(y))
            y = self.norms[-1](h)
            if self.use_film:
                y = self.films[-1](y, z)
            return F.relu(y)
        for i, fc in enumerate(self.fcs):
            h = self.norms[i](fc(x))
            if self.use_film:
                h = self.films[i](h, z)
            x = F.relu(h)
        return x


class TanhGaussianActor(nn.Module):
    """Squashed-Gaussian plant policy over per-reactor tokens.

    Tokens (``token_index``) are embedded with a learned per-reactor slot,
    mixed across reactors (``TokenMixer``), then each token runs the SAME
    FiLM-conditioned trunk and emits its reactor's four actions. z modulates
    every trunk layer of every token.

    Attributes:
        latent_dim: Width of the latent conditioning variable.
        use_film: Whether z modulates hidden features or is concatenated.
        n_reactors: Number of reactor tokens.
    """

    def __init__(self, obs_dim, action_dim, latent_dim, hidden=256, depth=3,
                 use_film=True, layer_norm=True, residual=False,
                 obs_norm=None, mixer_layers=2, mixer_heads=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_film = use_film
        self.obs_norm = obs_norm
        self.obs_dim = int(obs_dim)
        self.n_reactors = N_REACTORS
        self.per = int(action_dim) // N_REACTORS
        if self.per * N_REACTORS != int(action_dim):
            raise ValueError(f"action_dim {action_dim} is not a whole number "
                             f"of actions per reactor")
        if obs_norm is None:
            self.in_norm = nn.LayerNorm(obs_dim)
        self.register_buffer("tok_index", token_index(), persistent=False)
        self.embed = nn.Linear(TOKEN_DIM, hidden)
        self.slot = nn.Parameter(0.02 * torch.randn(N_REACTORS, hidden))
        self.mixer = TokenMixer(hidden, mixer_layers, mixer_heads)
        self.trunk = MLPTrunk(hidden, hidden, depth, latent_dim,
                              use_film=use_film, layer_norm=layer_norm,
                              residual=residual)
        self.mean_head = nn.Linear(hidden, self.per)
        self.log_std_head = nn.Linear(hidden, self.per)

    def _features(self, obs, z):
        """Per-token conditioned features, shape [B * N, hidden]."""
        x = self.in_norm(obs) if self.obs_norm is None else self.obs_norm(obs)
        B = x.shape[0]
        h = self.embed(x[:, self.tok_index]) + self.slot       # [B, N, H]
        h = self.mixer(h)
        zt = z.unsqueeze(1).expand(B, self.n_reactors, z.shape[-1])
        return self.trunk(h.reshape(B * self.n_reactors, -1),
                          zt.reshape(B * self.n_reactors, -1))

    def forward(self, obs, z):
        """Pre-squash Gaussian parameters (mean, log_std), each [B, action_dim]."""
        B = obs.shape[0]
        h = self._features(obs, z)
        mean = self.mean_head(h).reshape(B, -1)
        log_std = torch.clamp(self.log_std_head(h).reshape(B, -1),
                              LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs, z, deterministic=False, with_logprob=True, mask=None,
               noise=None):
        """Sample a squashed action and its log-probability.

        Args:
            obs: Observations, shape [B, obs_dim].
            z: Latent, shape [B, latent_dim].
            deterministic: Return the distribution mean instead of a sample.
            with_logprob: Whether to compute the log-probability.
            mask: Optional structural action mask, shape [B, action_dim];
                locked dimensions contribute zero to the log-probability.
            noise: Optional standard-normal draw used in place of a fresh one
                (colored exploration). Ignored when deterministic.

        Returns:
            Tuple (action, logp); action [B, action_dim] in [-1, 1], logp [B]
            or None.
        """
        mean, log_std = self.forward(obs, z)
        std = log_std.exp()
        if deterministic:
            u = mean
        else:
            eps = torch.randn_like(std) if noise is None else noise
            u = mean + std * eps
        u = torch.clamp(u, -PRE_TANH_CLIP, PRE_TANH_CLIP)
        a = torch.tanh(u)
        if not with_logprob:
            return a, None
        dist = Normal(mean, std)
        log_one_minus_a_sq = 2.0 * (np.log(2.0) - u - F.softplus(-2.0 * u))
        logp = dist.log_prob(u) - log_one_minus_a_sq
        if mask is not None:
            logp = logp * mask
        return a, logp.sum(dim=-1)


class _TokenQ(nn.Module):
    """One Q network: mixed (obs, action) reactor tokens into a wide trunk."""

    def __init__(self, action_dim, cond_dim, hidden, depth, use_film,
                 layer_norm, residual, token_hidden, mixer_layers, mixer_heads):
        super().__init__()
        self.n_reactors = N_REACTORS
        self.per = int(action_dim) // N_REACTORS
        self.register_buffer("tok_index", token_index(), persistent=False)
        self.embed = nn.Linear(TOKEN_DIM + self.per, token_hidden)
        self.slot = nn.Parameter(0.02 * torch.randn(N_REACTORS, token_hidden))
        self.mixer = TokenMixer(token_hidden, mixer_layers, mixer_heads)
        self.trunk = MLPTrunk(N_REACTORS * token_hidden, hidden, depth,
                              cond_dim, use_film=use_film,
                              layer_norm=layer_norm, residual=residual)
        self.head = nn.Linear(hidden, 1)

    def forward(self, obs, action, cond):
        B = obs.shape[0]
        tok = torch.cat([obs[:, self.tok_index],
                         action.reshape(B, self.n_reactors, self.per)], dim=-1)
        h = self.mixer(self.embed(tok) + self.slot)
        return self.head(self.trunk(h.reshape(B, -1), cond)).squeeze(-1)


class Critic(nn.Module):
    """Twin token Q networks conditioned on z (and, for meta, more).

    The conditioning vector is z, with the task descriptor and then the
    plant's privileged state appended when the agent's critics see them
    (``PEARLAgent._critic_cond``). The shared observation normaliser holds
    only buffers, so the Polyak update never touches it.
    """

    def __init__(self, obs_dim, action_dim, latent_dim, hidden=1024, depth=3,
                 use_film=True, layer_norm=True, residual=False,
                 obs_norm=None, token_hidden=256, mixer_layers=2,
                 mixer_heads=4):
        super().__init__()
        self.use_film = use_film
        self.obs_norm = obs_norm
        make = lambda: _TokenQ(action_dim, latent_dim, hidden, depth,  # noqa: E731
                               use_film, layer_norm, residual, token_hidden,
                               mixer_layers, mixer_heads)
        self.q1 = make()
        self.q2 = make()

    def forward(self, obs, action, z):
        """Evaluate both Q functions; each output has shape [B]."""
        if self.obs_norm is not None:
            obs = self.obs_norm(obs)
        return self.q1(obs, action, z), self.q2(obs, action, z)


class AuxHead(nn.Module):
    """Auxiliary prediction head giving the encoder an independent gradient.

    ``"reward"`` predicts the step reward from (obs, action, z); ``"task"``
    regresses the standardized task descriptor (and any privileged state
    targets) from z alone.
    """

    def __init__(self, latent_dim, out_dim, obs_dim=0, action_dim=0,
                 hidden=256, mode="reward"):
        super().__init__()
        self.mode = mode
        in_dim = latent_dim + (obs_dim + action_dim if mode == "reward" else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z, obs=None, action=None):
        if self.mode == "reward":
            return self.net(torch.cat([z, obs, action], dim=-1))
        return self.net(z)


class AdaptGate(nn.Module):
    """Detector of a changed plant, read from the latent z alone.

    Outputs the logit of P(a fault has acted | z). The last layer starts at
    zero weight and a negative bias, so the gate is shut for every z until it
    learns.
    """

    def __init__(self, latent_dim, hidden=128, init_bias=-4.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, float(init_bias))

    def forward(self, z):
        return self.net(z).squeeze(-1)
