"""The agent's networks.

The per-reactor history (a GRU per reactor, then cross-reactor attention over
the GRU outputs: the Fermentation-PPO policy), the token actor and critics
that read it, the GRU context encoder, the auxiliary head and the adaptation
gate's detector.
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
from metarl.env.config import N_REACTORS, REACTOR_OBS_DIM
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

    A reactor's token is its row of the environment's observation: its own
    25 channels and the three plant-wide ones, the per-reactor input of the
    Fermentation-PPO policy.
    """
    rows = [list(range(i * REACTOR_OBS_DIM, (i + 1) * REACTOR_OBS_DIM))
            for i in range(N_REACTORS)]
    return torch.tensor(rows, dtype=torch.long)


TOKEN_DIM = REACTOR_OBS_DIM


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


class CrossReactorAttention(nn.Module):
    """Scaled dot-product self-attention across the reactors (Fermentation-PPO).

    Each reactor attends to every reactor's history features, so it sees where
    its siblings are heading, not just their instantaneous readings. Q/K/V
    are projected down to ``attn_dim`` (three tokens need few dimensions); the
    output projection starts at zero, so the residual block starts as the
    identity. ``last_attn_weights`` keeps the latest [B, heads, N, N] weights.
    """

    def __init__(self, feature_dim, n_heads=4, attn_dim=64):
        super().__init__()
        if attn_dim % n_heads:
            raise ValueError(f"attn_dim {attn_dim} is not divisible by "
                             f"{n_heads} heads")
        self.n_heads = int(n_heads)
        self.attn_dim = int(attn_dim)
        self.head_dim = self.attn_dim // self.n_heads
        self.q_proj = nn.Linear(feature_dim, attn_dim)
        self.k_proj = nn.Linear(feature_dim, attn_dim)
        self.v_proj = nn.Linear(feature_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.last_attn_weights = None

    def forward(self, x):
        """Mix reactor features ``x`` [B, N, D]; returns [B, N, D]."""
        B, N, _ = x.shape
        H, d = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, N, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, d).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d), dim=-1)
        self.last_attn_weights = att.detach()
        o = (att @ v).transpose(1, 2).reshape(B, N, self.attn_dim)
        return self.norm(x + self.out_proj(o))


class ReactorHistory(nn.Module):
    """Each reactor's history, then cross-reactor attention over it.

    The Fermentation-PPO policy's encoder: a reactor's token (``token_index``)
    is embedded (Linear -> LayerNorm -> ELU, plus a learned slot for the
    vessel's identity, which its capacity channels also carry) and folded
    into a GRU whose
    weights every reactor shares and whose state each keeps on its own; the
    GRU output is layer-normalised and the reactors attend to each other's
    (``CrossReactorAttention``). A skip embedding of the current token is
    appended, so each reactor's feature is [history (``hidden``), now
    (``skip``)].

    ``step`` folds one token per reactor into the state [layers, B * N,
    hidden]; ``window`` runs causal windows of replay rows for the update.
    """

    def __init__(self, hidden=256, layers=2, attn_dim=64, heads=4, skip=128):
        super().__init__()
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.n_reactors = N_REACTORS
        self.out_dim = self.hidden + int(skip)
        self.register_buffer("tok_index", token_index(), persistent=False)
        self.embed = nn.Sequential(nn.Linear(TOKEN_DIM, self.hidden),
                                   nn.LayerNorm(self.hidden), nn.ELU())
        self.slot = nn.Parameter(0.02 * torch.randn(N_REACTORS, self.hidden))
        self.cells = nn.ModuleList(nn.GRUCell(self.hidden, self.hidden)
                                   for _ in range(self.layers))
        for cell in self.cells:
            nn.init.orthogonal_(cell.weight_hh)
            nn.init.xavier_uniform_(cell.weight_ih)
            nn.init.zeros_(cell.bias_hh)
            nn.init.zeros_(cell.bias_ih)
        self.gru_ln = nn.LayerNorm(self.hidden)
        self.attn = CrossReactorAttention(self.hidden, heads, attn_dim)
        self.skip = nn.Sequential(nn.Linear(TOKEN_DIM, int(skip)),
                                  nn.LayerNorm(int(skip)), nn.ELU())

    def initial_state(self, batch, device=None):
        """The zero state for ``batch`` episodes, [layers, batch * N, hidden]."""
        return torch.zeros(self.layers, batch * self.n_reactors, self.hidden,
                           device=device)

    def tokens(self, obs_n):
        """Reactor tokens of normalised observations [..., obs_dim] -> [..., N, T]."""
        return obs_n[..., self.tok_index]

    def _fold(self, x, h):
        """One GRU step of embedded inputs x [M, hidden] from h [layers, M, H]."""
        out = []
        for layer, cell in enumerate(self.cells):
            x = cell(x, h[layer])
            out.append(x)
        return torch.stack(out)

    def _read(self, top, tok):
        """Features of top-layer states [B, N, H] and tokens [B, N, T]."""
        return torch.cat([self.attn(self.gru_ln(top)), self.skip(tok)], dim=-1)

    def step(self, tok, h):
        """Fold tokens [B, N, T] into h; returns (features [B, N, out_dim], h)."""
        B = tok.shape[0]
        x = (self.embed(tok) + self.slot).reshape(B * self.n_reactors, -1)
        h = self._fold(x, h)
        return self._read(h[-1].reshape(B, self.n_reactors, -1), tok), h

    def window(self, tok, reset, hi, empty, tok_now, tok_next):
        """Features of replay transitions from their causal windows.

        Each transition's state is the GRU's after the previous transition of
        its episode (row ``hi - 1`` of its window, or zero for an episode's
        first step); its own observation and then the next one are folded
        from there, so both reads are exact at any episode boundary.

        Args:
            tok: Tokens [K, R, N, T] of K windows of R consecutive rows.
            reset: Bool [K, R]: the state is zeroed before folding that row
                (an episode's first row in the window).
            hi: Long [K, L]: each transition's row (its context is the rows
                before it).
            empty: Bool [K, L]: the transition starts its episode.
            tok_now: Tokens [K * L, N, T] of each transition's observation.
            tok_next: Tokens [K * L, N, T] of each transition's next one.

        Returns:
            Tuple (features at the observation, at the next one), each
            [K * L, N, out_dim].
        """
        K, R, N, _ = tok.shape
        x = (self.embed(tok) + self.slot).reshape(K, R, N * self.hidden)
        h = self.initial_state(K, tok.device)
        states = []
        for q in range(R):
            keep = (~reset[:, q]).to(h.dtype).repeat_interleave(N)
            h = h * keep.view(1, -1, 1)
            h = self._fold(x[:, q].reshape(K * N, self.hidden), h)
            states.append(h.view(self.layers, K, N, self.hidden))
        states = torch.stack(states)                  # [R, layers, K, N, H]
        L = hi.shape[1]
        rows_k = torch.arange(K, device=tok.device).repeat_interleave(L)
        prev = states[(hi - 1).clamp(min=0).reshape(-1), :, rows_k]
        prev = prev * (~empty).reshape(-1, 1, 1, 1).to(prev.dtype)
        h0 = prev.permute(1, 0, 2, 3).reshape(self.layers, K * L * N,
                                              self.hidden)
        feat_now, h1 = self.step(tok_now, h0)
        feat_next, _ = self.step(tok_next, h1)
        return feat_now, feat_next


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
    """Squashed-Gaussian plant policy over per-reactor history features.

    Each reactor's history (``ReactorHistory``: its GRU, then attention across
    the reactors) runs the SAME FiLM-conditioned trunk and emits its
    reactor's four actions. z modulates every trunk layer of every reactor.
    The policy is recurrent: ``forward`` / ``sample`` read history features
    [B, N, feature_dim] made by ``history_step`` (acting) or
    ``history_window`` (the update).

    Attributes:
        latent_dim: Width of the latent conditioning variable.
        use_film: Whether z modulates hidden features or is concatenated.
        n_reactors: Number of reactor tokens.
        history: The per-reactor history module.
    """

    def __init__(self, obs_dim, action_dim, latent_dim, hidden=256, depth=3,
                 use_film=True, layer_norm=True, residual=False,
                 obs_norm=None, history_hidden=256, history_layers=2,
                 attn_dim=64, attn_heads=4, skip_dim=128):
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
        self.history = ReactorHistory(history_hidden, history_layers,
                                      attn_dim, attn_heads, skip_dim)
        self.trunk = MLPTrunk(self.history.out_dim, hidden, depth, latent_dim,
                              use_film=use_film, layer_norm=layer_norm,
                              residual=residual)
        self.mean_head = nn.Linear(hidden, self.per)
        self.log_std_head = nn.Linear(hidden, self.per)

    def _norm(self, obs):
        return self.in_norm(obs) if self.obs_norm is None else self.obs_norm(obs)

    def initial_state(self, batch, device=None):
        """The history state at an episode start."""
        return self.history.initial_state(batch, device)

    def history_step(self, obs, h):
        """Fold observations [B, obs_dim]; returns (features, new state)."""
        return self.history.step(self.history.tokens(self._norm(obs)), h)

    def history_window(self, obs_rows, reset, hi, empty, obs, next_obs):
        """``ReactorHistory.window`` over observation rows [K, R, obs_dim]."""
        tok = lambda o: self.history.tokens(self._norm(o))   # noqa: E731
        return self.history.window(tok(obs_rows), reset, hi, empty, tok(obs),
                                   tok(next_obs))

    def single_step_features(self, obs):
        """Features of observations read with no history (a fresh episode)."""
        return self.history_step(
            obs, self.initial_state(obs.shape[0], obs.device))[0]

    def _features(self, feat, z):
        """Per-token conditioned features, shape [B * N, hidden]."""
        B = feat.shape[0]
        zt = z.unsqueeze(1).expand(B, self.n_reactors, z.shape[-1])
        return self.trunk(feat.reshape(B * self.n_reactors, -1),
                          zt.reshape(B * self.n_reactors, -1))

    def forward(self, feat, z):
        """Pre-squash Gaussian parameters (mean, log_std), each [B, action_dim].

        ``feat`` are history features [B, N, feature_dim].
        """
        B = feat.shape[0]
        h = self._features(feat, z)
        mean = self.mean_head(h).reshape(B, -1)
        log_std = torch.clamp(self.log_std_head(h).reshape(B, -1),
                              LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, feat, z, deterministic=False, with_logprob=True, mask=None,
               noise=None):
        """Sample a squashed action and its log-probability.

        Args:
            feat: History features, shape [B, N, feature_dim].
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
        mean, log_std = self.forward(feat, z)
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
    """One Q network: (history feature, action) reactor tokens into a wide trunk."""

    def __init__(self, action_dim, cond_dim, hidden, depth, use_film,
                 layer_norm, residual, token_hidden, mixer_layers, mixer_heads,
                 feature_dim):
        super().__init__()
        self.n_reactors = N_REACTORS
        self.per = int(action_dim) // N_REACTORS
        self.embed = nn.Linear(int(feature_dim) + self.per, token_hidden)
        self.slot = nn.Parameter(0.02 * torch.randn(N_REACTORS, token_hidden))
        self.mixer = TokenMixer(token_hidden, mixer_layers, mixer_heads)
        self.trunk = MLPTrunk(N_REACTORS * token_hidden, hidden, depth,
                              cond_dim, use_film=use_film,
                              layer_norm=layer_norm, residual=residual)
        self.head = nn.Linear(hidden, 1)

    def forward(self, feat, action, cond):
        B = feat.shape[0]
        tok = torch.cat([feat, action.reshape(B, self.n_reactors, self.per)],
                        dim=-1)
        h = self.mixer(self.embed(tok) + self.slot)
        return self.head(self.trunk(h.reshape(B, -1), cond)).squeeze(-1)


class Critic(nn.Module):
    """Twin token Q networks over the actor's history features.

    Each reactor's token is its history feature (``ReactorHistory``, already
    attended across the reactors) and its action; the tokens are mixed (the
    joint action couples the reactors through the tank) and flattened into a
    FiLM trunk conditioned on z, with the task descriptor and then the plant's
    privileged state appended when the agent's critics see them
    (``PEARLAgent._critic_cond``).
    """

    def __init__(self, obs_dim, action_dim, latent_dim, hidden=1024, depth=3,
                 use_film=True, layer_norm=True, residual=False,
                 token_hidden=256, mixer_layers=2, mixer_heads=4,
                 feature_dim=384):
        super().__init__()
        self.use_film = use_film
        make = lambda: _TokenQ(action_dim, latent_dim, hidden, depth,  # noqa: E731
                               use_film, layer_norm, residual, token_hidden,
                               mixer_layers, mixer_heads, feature_dim)
        self.q1 = make()
        self.q2 = make()

    def forward(self, feat, action, z):
        """Evaluate both Q functions on history features; each output is [B]."""
        return self.q1(feat, action, z), self.q2(feat, action, z)


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
