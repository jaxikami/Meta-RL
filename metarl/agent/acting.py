"""Acting (``ActingMixin``): the online posterior over z and action selection.

actor -> tanh -> [adaptation gate] -> stage mask -> action projection -> stage
mask.
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.plant import PlantFermentationEnv
from metarl.apn.network import project_to_safe
from metarl.agent.common import (ROLLOUT_DEVICE, PRE_TANH_CLIP,
                                 EXPLORE_NOISE_LEN, powerlaw_noise)


class ActingMixin:
    """Online posterior and action selection of ``PEARLAgent``."""

    def clear_z(self):
        """Reset the online posterior to the prior. Call at every episode start."""
        self._sync_rollout()
        self.z_mu = torch.zeros(self._z_dim, device=ROLLOUT_DEVICE)
        self.z_sigma_sq = torch.ones(self._z_dim, device=ROLLOUT_DEVICE)
        self._h = torch.zeros(1, self.encoder.hidden, device=ROLLOUT_DEVICE)
        self._z_steps = 0
        self._explore_seq = None
        self._explore_t = 0
        self._resample_z()

    def _next_explore_noise(self):
        """The next row of this episode's colored exploration noise, [1, A]."""
        if (self._explore_seq is None
                or self._explore_t >= self._explore_seq.shape[0]):
            self._explore_seq = torch.from_numpy(powerlaw_noise(
                self.explore_noise_beta, EXPLORE_NOISE_LEN,
                self.action_dim)).to(ROLLOUT_DEVICE)
            self._explore_t = 0
        eps = self._explore_seq[self._explore_t].unsqueeze(0)
        self._explore_t += 1
        return eps

    def _resample_z(self):
        """Redraw or recompute z according to ``z_mode``."""
        if not self.use_context or self.z_mode == "zero":
            self.z = torch.zeros(self._z_dim, device=ROLLOUT_DEVICE)
            return
        if self.z_mode == "fixed":
            self.z = (torch.zeros(self._z_dim, device=ROLLOUT_DEVICE)
                      if self._z_fixed is None else self._z_fixed.clone())
            return
        if self.z_mode == "mean":
            self.z = self.z_mu.clone()
            return
        std = torch.sqrt(torch.clamp(self.z_sigma_sq, min=1e-8))
        self.z = self.z_mu + std * torch.randn_like(std)

    def set_fixed_z(self, z_vec):
        """Install a specific latent for ``z_mode='fixed'``."""
        z = torch.as_tensor(z_vec, dtype=torch.float32,
                            device=ROLLOUT_DEVICE).reshape(-1)
        if z.numel() != self._z_dim:
            raise ValueError(
                f"fixed z has {z.numel()} elements, expected {self._z_dim}")
        self._z_fixed = z
        if self.z_mode == "fixed":
            self._resample_z()

    def posterior_state(self):
        """Snapshot the online posterior, so a rollout can be branched."""
        return {
            "z_mu": self.z_mu.clone(), "z_sigma_sq": self.z_sigma_sq.clone(),
            "z": self.z.clone(), "steps": self._z_steps, "h": self._h.clone(),
            "explore_seq": (None if self._explore_seq is None
                            else self._explore_seq.clone()),
            "explore_t": self._explore_t,
        }

    def set_posterior_state(self, state):
        """Restore a ``posterior_state`` snapshot (it stays reusable)."""
        self.z_mu = state["z_mu"].clone()
        self.z_sigma_sq = state["z_sigma_sq"].clone()
        self.z = state["z"].clone()
        self._z_steps = state["steps"]
        self._h = state["h"].clone()
        seq = state.get("explore_seq")
        self._explore_seq = None if seq is None else seq.clone()
        self._explore_t = state.get("explore_t", 0)

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
        """Fold one observed transition into the online posterior (GRU step)."""
        if not self.use_context:
            return
        self._sync_rollout()
        reward_n = reward / self.ret_rms.std if self.ret_rms is not None else reward
        trans = self._flatten_transition(obs, act, reward_n, next_obs)
        trans_t = torch.as_tensor(
            trans, dtype=torch.float32, device=ROLLOUT_DEVICE).unsqueeze(0)
        with torch.no_grad():
            self._h = self._encoder_r.step(trans_t, self._h)
            mu, sigma_sq = self._encoder_r.posterior(self._h)
        self.z_mu, self.z_sigma_sq = mu.squeeze(0), sigma_sq.squeeze(0)
        self._z_steps += 1
        if not (self.z_mode == "hold" and self._z_steps > self.z_hold_after):
            self._resample_z()

    # ---------------------------------------------------------- safety mask

    def _stage_mask(self, obs):
        """Structural action mask [..., action_dim], or None without safeguard."""
        if not self.use_safeguard:
            return None
        return PlantFermentationEnv.get_action_mask(obs[..., :self.obs_dim])

    def _apply_safety(self, obs, raw_squashed, mask=None):
        """Blend a raw action with the masked defaults."""
        if not self.use_safeguard:
            return raw_squashed
        if mask is None:
            mask = self._stage_mask(obs)
        default = (self._default_a if raw_squashed.device == self._default_a.device
                   else self._default_a_r)
        return raw_squashed * mask + default * (1.0 - mask)

    def _gate_open(self, net, z):
        """Openings in [0, 1], shape [B]: sigmoid(sharpness x logit) or the override."""
        if self.gate_override is not None:
            return torch.full(z.shape[:-1], float(self.gate_override),
                              device=z.device)
        return torch.sigmoid(self.gate_sharpness * net(z))

    @staticmethod
    def _gated(base_a, a, gate):
        """``base_a + gate * (a - base_a)``, still in [-1, 1]."""
        return base_a + gate.unsqueeze(-1) * (a - base_a)

    def _base_action(self, net, obs, z):
        """The base policy's deterministic squashed action (it ignores z)."""
        mean = net(obs, torch.zeros_like(z))[0]
        return torch.tanh(torch.clamp(mean, -PRE_TANH_CLIP, PRE_TANH_CLIP))

    def _safe_sample(self, obs, z, deterministic=False, with_logprob=True,
                     gate=None):
        """Sample a stage-masked (optionally gated) action with its log-prob.

        Returns:
            Tuple (action, logp, n_active); n_active counts unmasked action
            dimensions per sample.
        """
        mask = self._stage_mask(obs)
        a, logp = self.actor.sample(obs, z, deterministic=deterministic,
                                    with_logprob=with_logprob, mask=mask)
        if gate is not None:
            with torch.no_grad():
                base_a = self._base_action(self.base_actor, obs, z)
            a = self._gated(base_a, a, gate)
        a = self._apply_safety(obs, a, mask=mask)
        if mask is None:
            n_active = torch.full(obs.shape[:-1], float(self.action_dim),
                                  device=obs.device)
        else:
            n_active = mask.sum(dim=-1)
        return a, logp, n_active

    @staticmethod
    def _shield_obs(net, obs):
        """The leading observation channels an APN was trained on."""
        return obs[..., :net.obs_dim]

    def _project(self, obs, action, mask=None):
        """Push a rollout action onto the APN's safe set, then re-apply the mask.

        Precedence: mask > projection > actor.
        """
        if self._apn_r is None or not self.apn_hard_project:
            return action
        if mask is None:
            mask = self._stage_mask(obs)
        proj, steps = project_to_safe(self._apn_r,
                                      self._shield_obs(self._apn_r, obs),
                                      action, stage_mask=mask)
        out = self._apply_safety(obs, proj, mask=mask)
        self._proj_calls += 1
        self._proj_steps += steps
        if steps == 0:
            self._proj_noop += 1
        if bool((out - action).abs().max() > 1e-6):
            self._proj_moved += 1
        return out

    def proj_stats(self, reset=True):
        """Report and optionally reset the projection's activity counters."""
        calls = max(self._proj_calls, 1)
        out = {"proj_calls": self._proj_calls,
               "proj_noop_frac": self._proj_noop / calls,
               "proj_moved_frac": self._proj_moved / calls,
               "proj_mean_steps": self._proj_steps / calls}
        if reset:
            self._proj_calls = self._proj_noop = self._proj_steps = 0
            self._proj_moved = 0
        return out

    def select_action(self, state, deterministic=False):
        """Choose an action for one environment step.

        Pipeline: actor -> tanh -> [adaptation gate: blend from the base
        policy's action] -> stage mask -> action projection -> stage mask.

        Returns:
            Tuple (executed_action, raw_action), each shape (action_dim,).
        """
        self._sync_rollout()
        obs = torch.as_tensor(state, dtype=torch.float32,
                              device=ROLLOUT_DEVICE).unsqueeze(0)
        z = self.z.unsqueeze(0)
        noise = (self._next_explore_noise()
                 if not deterministic and self.explore_noise_beta > 0.0
                 else None)
        with torch.no_grad():
            raw, _ = self._actor_r.sample(obs, z, deterministic=deterministic,
                                          with_logprob=False, noise=noise)
            if self.adapt_gate:
                g = self._gate_open(self._gate_r, z)
                raw = self._gated(self._base_action(self._base_r, obs, z),
                                  raw, g)
                self.last_gate = float(g)
            mask = self._stage_mask(obs)
            safe = self._apply_safety(obs, raw, mask=mask)
        safe = self._project(obs, safe, mask=mask)
        return (safe.squeeze(0).detach().cpu().numpy(),
                raw.squeeze(0).cpu().numpy())

    def observe_rewards(self, rewards):
        """Update the reward scaler from freshly collected rewards only."""
        if self.ret_rms is not None:
            self.ret_rms.update(rewards)

    def observe_obs(self, observations, next_observations=None):
        """Update the observation normalisers from freshly collected steps only."""
        moved = False
        if self.obs_norm is not None and not self.obs_norm_frozen:
            self.obs_norm.update(observations)
            moved = True
        if self.delta_norm is not None and next_observations is not None:
            self.delta_norm.update(
                np.asarray(next_observations, dtype=np.float32)
                - np.asarray(observations, dtype=np.float32))
            moved = True
        if moved:
            self._mark_rollout_stale()

    def freeze_obs_norm(self):
        """Stop ``observe_obs`` from moving the observation statistics."""
        self.obs_norm_frozen = True

    @property
    def alpha(self):
        """Current SAC temperature."""
        return self.log_alpha.exp()
