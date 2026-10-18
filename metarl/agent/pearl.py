"""PEARL meta-RL agent (SAC backbone) for the three-reactor plant.

The lean study's agent (``MetaRL_Lean/agent_pearl.py``), with networks built
for a plant of three reactors and a shared storage tank. One class serves
every arm: ``use_context=False`` gives a non-adaptive agent (z == 0, no
encoder), ``use_safeguard`` toggles the structural stage mask, and ``apn`` /
``apn_hard_project`` attach the action-projection shield.

Kept from the lean agent, unchanged in behaviour:

  * posterior-collapse countermeasures: an auxiliary loss on z (``aux_mode``,
    regressed mostly from the posterior mean), FiLM conditioning of every
    hidden layer, critics that also read the true task descriptor and the
    plant's privileged state (asymmetric actor-critic), ``MU_CLAMP``;
  * a GRU posterior over the whole episode (``RecurrentContextEncoder``),
    reading each transition's observation change on its own running scale,
    trained on causal windows that reach back to the episode start;
  * one running observation normaliser shared by every network, AdamW on the
    actor's and critics' weight matrices, pre-LN residual trunks;
  * the warm start from saferl: FiLM weights start at zero, the anchor pulls
    toward the loaded policy before a fault (per fault kind after one), decay
    toward the loaded weights (L2-SP), colored exploration noise;
  * the adaptation gate: ``a = base(s) + g(z) (actor(s, z) - base(s))``, a
    detector on z fitted to the privileged fault flag, every actor-side term
    gate-weighted.

New for the plant (``TokenActor`` / ``TokenCritic``): the 78-channel
observation is read as one TOKEN per reactor -- its own 24 process channels,
its CO2 rate and the 3 plant-wide channels (storage, both feed pools) -- with
a learned slot embedding for its identity (the three vessels' pumps differ
4x / 0.5x). The tokens pass through pre-LN self-attention blocks
(``TokenMixer``), which is where the reactors coordinate over the shared tank
and feed pool. The actor then runs ONE FiLM-conditioned residual trunk shared
by all three tokens and emits each reactor's four actions from its own token,
so every reactor's experience trains the same weights. Each critic flattens
its mixed tokens (obs + that reactor's action) into a wide FiLM residual
trunk: the plant's value is not a sum over reactors, because the tank couples
them. Every arm uses the same architecture, so no contrast comes down to the
network.
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import TASK_CONTEXT_DIM, N_REACTORS
from metarl.env.layout import OBS_LAYOUT_VERSION, ACTION_DIM_PER_REACTOR
from metarl.env.faults import N_DISTURBANCE_KINDS
from metarl.agent.common import (device, ROLLOUT_DEVICE, _on_rollout_device,
                                 LATENT_DIM, LOG_ALPHA_MIN, NET_ARCH,
                                 RunningMeanStd, RunningObsNorm, _adamw)
from metarl.agent.networks import (RecurrentContextEncoder, TanhGaussianActor,
                                   Critic, AuxHead, AdaptGate)
from metarl.agent.acting import ActingMixin
from metarl.agent.learning import LearningMixin
from metarl.agent.checkpoint import CheckpointMixin


class PEARLAgent(ActingMixin, LearningMixin, CheckpointMixin):
    """PEARL meta-RL agent with a SAC backbone and optional safety layers.

    ``use_context=False`` forces z to zero and disables the encoder and the
    auxiliary head; the network architecture is otherwise identical.

    Attributes:
        z_mode: One of 'sample', 'mean', 'hold', 'zero', 'fixed'.
        kl_lambda: Weight on the KL information bottleneck.
    """

    def __init__(self, obs_dim, action_dim, latent_dim=LATENT_DIM,
                 gamma=0.99, tau=5e-3, lr=5e-4, kl_lambda=0.1, encoder_lr=None,
                 reward_scale=1.0, normalize_rewards=True,
                 use_context=True, use_safeguard=True,
                 target_entropy=None,
                 reward_window=1e5, alpha_min=None,
                 target_entropy_per_dim=None,
                 use_film=True, aux_mode="reward", aux_weight=1.0,
                 aux_sample_frac=1.0,
                 apn=None, apn_penalty_weight=0.01, apn_hard_project=True,
                 actor_hidden=256, critic_hidden=1024, net_depth=3,
                 layer_norm=True, residual=True, obs_norm=True,
                 encoder_layer_norm=True, weight_decay=1e-2,
                 critic_task_context=False,
                 critic_privileged_dim=0,
                 adapt_gate=False, gate_weight=1.0, gate_sharpness=2.0,
                 gate_init_bias=-4.0, explore_noise_beta=0.0,
                 aux_state_targets=None, critic_token_hidden=256,
                 mixer_layers=2, mixer_heads=4, encoder_hidden=256):
        """Initialise the agent.

        Arguments are the lean agent's, plus the token networks' shape:
        ``actor_hidden`` is the actor's token width (and its shared trunk's),
        ``critic_token_hidden`` each critic's token width, ``critic_hidden``
        each critic's flattened trunk width, ``mixer_layers`` /
        ``mixer_heads`` the self-attention blocks across reactors, and
        ``encoder_hidden`` the GRU width. See ``MetaRL_Lean/agent_pearl.py``
        for every other argument.
        """
        if float(explore_noise_beta) < 0.0:
            raise ValueError(f"explore_noise_beta must be >= 0, got "
                             f"{explore_noise_beta}")
        self.explore_noise_beta = float(explore_noise_beta)
        self._explore_seq = None
        self._explore_t = 0
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim if use_context else 0
        self._z_dim = max(1, self.latent_dim)
        if critic_task_context and not use_context:
            raise ValueError("critic_task_context needs a context agent")
        if adapt_gate and not use_context:
            raise ValueError("adapt_gate needs a context agent: the gate "
                             "reads the latent")
        if int(critic_privileged_dim) < 0:
            raise ValueError(f"critic_privileged_dim must be >= 0, got "
                             f"{critic_privileged_dim}")
        self.critic_task_context = bool(critic_task_context)
        self.critic_privileged_dim = int(critic_privileged_dim)
        self.aux_state_targets = tuple(
            (int(c), float(m), float(s)) for c, m, s in (aux_state_targets or ()))
        if self.aux_state_targets:
            if aux_mode != "task" or not use_context:
                raise ValueError("aux_state_targets needs a context agent "
                                 "with aux_mode='task'")
            if not self.critic_privileged_dim:
                raise ValueError("aux_state_targets reads the stored "
                                 "privileged state: set critic_privileged_dim")
            bad = [c for c, _, s in self.aux_state_targets
                   if not 0 <= c < self.critic_privileged_dim or s <= 0.0]
            if bad:
                raise ValueError(f"aux_state_targets: channels {bad} are out "
                                 f"of range or have a non-positive scale")
        self.adapt_gate = bool(adapt_gate)
        self.gate_weight = float(gate_weight)
        self.gate_sharpness = float(gate_sharpness)
        self.gate_override = None
        # Hold rule (training only): penalty weight on opening the outlet of a
        # below-spec batch, its margin, and whether stalls also train the gate.
        self.hold_penalty_weight = 0.0
        self.hold_margin = 0.03
        self.hold_gate_target = False
        self.last_gate = float("nan")
        self._critic_c_dim = (self._z_dim
                              + (TASK_CONTEXT_DIM if self.critic_task_context
                                 else 0)
                              + self.critic_privileged_dim)
        self.gamma = gamma
        self.tau = tau
        self.kl_lambda = kl_lambda
        self.reward_scale = reward_scale
        self.normalize_rewards = normalize_rewards
        self.ret_rms = RunningMeanStd(max_count=reward_window) if normalize_rewards else None
        self.use_context = use_context
        self.use_safeguard = use_safeguard
        self.apn = apn
        self.apn_penalty_weight = float(apn_penalty_weight)
        self.apn_hard_project = bool(apn_hard_project)
        self._proj_calls = self._proj_noop = self._proj_steps = 0
        self._proj_moved = 0
        self.use_film = use_film
        self.aux_mode = aux_mode if use_context else "none"
        self.aux_weight = float(aux_weight)
        self.aux_sample_frac = float(aux_sample_frac)
        if not 0.0 <= self.aux_sample_frac <= 1.0:
            raise ValueError(f"aux_sample_frac must be in [0, 1], got "
                             f"{aux_sample_frac}")

        self.z_mode = "sample"
        self.z_hold_after = 10
        self._z_fixed = None
        self.device = device

        # ── Networks ─────────────────────────────────────────────────────────
        trans_dim = obs_dim + action_dim + 1 + obs_dim
        self.trans_dim = trans_dim
        self.net_config = {"arch": NET_ARCH,
                           "n_reactors": int(N_REACTORS),
                           "obs_dim": int(obs_dim),
                           "action_dim": int(action_dim),
                           "actor_hidden": int(actor_hidden),
                           "critic_hidden": int(critic_hidden),
                           "critic_token_hidden": int(critic_token_hidden),
                           "mixer_layers": int(mixer_layers),
                           "mixer_heads": int(mixer_heads),
                           "depth": int(net_depth),
                           "layer_norm": bool(layer_norm),
                           "use_film": bool(use_film),
                           "residual": bool(residual),
                           "obs_norm": bool(obs_norm),
                           "encoder_layer_norm": bool(encoder_layer_norm),
                           "encoder_delta": True,
                           "encoder_recurrent": True,
                           "encoder_hidden": int(encoder_hidden),
                           "latent_dim": int(self._z_dim),
                           "adapt_gate": bool(self.adapt_gate),
                           "task_context_dim": int(TASK_CONTEXT_DIM),
                           "obs_layout": int(OBS_LAYOUT_VERSION)}
        self.obs_norm = RunningObsNorm(obs_dim).to(device) if obs_norm else None
        self.delta_norm = (RunningObsNorm(obs_dim).to(device)
                           if use_context else None)
        self.encoder = RecurrentContextEncoder(
            trans_dim, latent_dim=self._z_dim, hidden=encoder_hidden,
            obs_dim=obs_dim, obs_norm=self.obs_norm,
            layer_norm=encoder_layer_norm,
            delta_norm=self.delta_norm).to(device)
        make_actor = lambda norm: TanhGaussianActor(                # noqa: E731
            obs_dim, action_dim, self._z_dim, hidden=actor_hidden,
            depth=net_depth, use_film=use_film, layer_norm=layer_norm,
            residual=residual, obs_norm=norm, mixer_layers=mixer_layers,
            mixer_heads=mixer_heads)
        self.actor = make_actor(self.obs_norm).to(device)
        make_critic = lambda: Critic(                               # noqa: E731
            obs_dim, action_dim, self._critic_c_dim, hidden=critic_hidden,
            depth=net_depth, use_film=use_film, layer_norm=layer_norm,
            residual=residual, obs_norm=self.obs_norm,
            token_hidden=critic_token_hidden, mixer_layers=mixer_layers,
            mixer_heads=mixer_heads).to(device)
        self.critic = make_critic()
        self.critic_target = make_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())

        # The gate's base policy (the actor's architecture with its own copy
        # of the observation statistics, never trained) and its detector.
        self.base_actor = self.gate = self.gate_opt = None
        if self.adapt_gate:
            self.base_actor = make_actor(
                RunningObsNorm(obs_dim) if obs_norm else None).to(device)
            self.base_actor.load_state_dict(self.actor.state_dict())
            for p in self.base_actor.parameters():
                p.requires_grad_(False)
            self.gate = AdaptGate(self._z_dim,
                                  init_bias=gate_init_bias).to(device)
            self.gate_opt = torch.optim.Adam(self.gate.parameters(), lr=lr)

        self.aux = None
        if self.aux_mode == "reward":
            self.aux = AuxHead(self._z_dim, 1, obs_dim, action_dim,
                               mode="reward").to(device)
        elif self.aux_mode == "task":
            self.aux = AuxHead(self._z_dim, TASK_CONTEXT_DIM
                               + len(self.aux_state_targets),
                               mode="task").to(device)

        # The encoder and aux head stay on plain Adam: the KL bottleneck
        # already regularises the encoder.
        self.weight_decay = float(weight_decay)
        self.actor_opt = _adamw(self.actor, lr, self.weight_decay)
        self.critic_opt = _adamw(self.critic, lr, self.weight_decay)
        self.encoder_opt = torch.optim.Adam(
            self.encoder.parameters(), lr=encoder_lr if encoder_lr else lr)
        self.aux_opt = (torch.optim.Adam(self.aux.parameters(), lr=lr)
                        if self.aux is not None else None)

        # Set by warm_start_from (see the lean agent).
        self.anchor_actor = None
        self.anchor_weight = 0.0
        self.anchor_fault_frac = 1.0
        self.anchor_kind_weights = torch.zeros(N_DISTURBANCE_KINDS + 1,
                                               device=device)
        self._anchor_kind_np = np.zeros(N_DISTURBANCE_KINDS + 1, np.float32)
        self._anchor_on = False
        self.anchor_fault_dim_scale = torch.ones(action_dim, device=device)
        self._actor_decay_ref = None
        self.obs_norm_frozen = False

        # ── Entropy tuning ───────────────────────────────────────────────────
        self.target_entropy = (target_entropy if target_entropy is not None
                               else -float(action_dim))
        if target_entropy_per_dim is not None:
            self._entropy_per_dim = float(target_entropy_per_dim)
            self.target_entropy = self._entropy_per_dim * float(action_dim)
        else:
            self._entropy_per_dim = self.target_entropy / float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.alpha_min = alpha_min
        self._log_alpha_min = (LOG_ALPHA_MIN if alpha_min is None
                               else max(LOG_ALPHA_MIN, float(np.log(alpha_min))))

        # Default action for structurally locked dimensions, per reactor:
        # tanh(-0.34657359) is a time multiplier of exactly 1.0, tanh(-10) ~ -1
        # zero flow.
        _d = torch.full((N_REACTORS, ACTION_DIM_PER_REACTOR), -10.0)
        _d[:, 0] = -0.34657359
        self._default_a = torch.tanh(_d.reshape(-1)).to(device)

        # ── Rollout copies (ROLLOUT_DEVICE) ──────────────────────────────────
        self._actor_r = _on_rollout_device(self.actor)
        self._encoder_r = _on_rollout_device(self.encoder)
        self._apn_r = _on_rollout_device(self.apn)
        self._base_r = _on_rollout_device(self.base_actor)
        self._gate_r = _on_rollout_device(self.gate)
        self._default_a_r = self._default_a.to(ROLLOUT_DEVICE)
        self._roll_stale = False
        self._base_stale = False

        self.clear_z()

    def param_counts(self):
        """Trainable parameter counts per network, for logging."""
        def count(m):
            return 0 if m is None else sum(p.numel() for p in m.parameters())
        return {"actor": count(self.actor), "critic": count(self.critic),
                "encoder": count(self.encoder) if self.use_context else 0,
                "gate": count(self.gate)}

    def _mark_rollout_stale(self):
        self._roll_stale = self._actor_r is not self.actor

    def _mark_base_stale(self):
        self._base_stale = (self.base_actor is not None
                            and self._base_r is not self.base_actor)

    def _sync_rollout(self):
        """Refresh the rollout copies if the training weights changed."""
        if self._roll_stale:
            self._actor_r.load_state_dict(self.actor.state_dict())
            if self.use_context:
                self._encoder_r.load_state_dict(self.encoder.state_dict())
            if self.gate is not None:
                self._gate_r.load_state_dict(self.gate.state_dict())
            self._roll_stale = False
        if self._base_stale:
            self._base_r.load_state_dict(self.base_actor.state_dict())
            self._base_stale = False

    def rollout_weights(self):
        """Everything a rollout needs, as picklable arrays (worker processes)."""
        to_np = lambda m: {k: v.detach().cpu().numpy()          # noqa: E731
                           for k, v in m.state_dict().items()}
        return {
            "actor": to_np(self.actor),
            "encoder": to_np(self.encoder) if self.use_context else None,
            "base_actor": (to_np(self.base_actor) if self.adapt_gate
                           else None),
            "gate": to_np(self.gate) if self.adapt_gate else None,
            "ret_rms": (None if self.ret_rms is None else
                        (self.ret_rms.mean, self.ret_rms.var, self.ret_rms.count)),
        }

    def load_rollout_weights(self, weights):
        """Install a ``rollout_weights`` payload from another agent.

        Raises:
            ValueError: If this agent has the adaptation gate and the payload
                does not.
        """
        as_t = lambda d: {k: torch.from_numpy(v) for k, v in d.items()}  # noqa: E731
        self.actor.load_state_dict(as_t(weights["actor"]))
        if self.use_context and weights["encoder"] is not None:
            self.encoder.load_state_dict(as_t(weights["encoder"]))
        if self.adapt_gate:
            if weights.get("gate") is None or weights.get("base_actor") is None:
                raise ValueError("rollout weights carry no adaptation gate, "
                                 "but this agent has one")
            self.base_actor.load_state_dict(as_t(weights["base_actor"]))
            self.gate.load_state_dict(as_t(weights["gate"]))
            self._mark_base_stale()
        if self.ret_rms is not None and weights["ret_rms"] is not None:
            self.ret_rms.mean, self.ret_rms.var, self.ret_rms.count = (
                float(v) for v in weights["ret_rms"])
        self._mark_rollout_stale()
