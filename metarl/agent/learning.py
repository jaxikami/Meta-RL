"""The meta-gradient update (``LearningMixin``).

Critics, encoder, auxiliary head and gate detector, then the actor (with the
APN mapping penalty and the warm-start anchor), then the temperature.
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import MIN_HARVEST_TITER, TITER_SCALE
from metarl.env.faults import N_DISTURBANCE_KINDS
from metarl.env.layout import ETHANOL_CHANNELS, STAGE_CHANNELS, reactor_action
from metarl.env.plant import DISCHARGE_CUTOFF
from metarl.apn.network import project_to_safe
from metarl.agent.common import device, LOG_ALPHA_MAX, ENTROPY_TERM_CLIP
from metarl.agent.replay import WINDOW_CHUNK_LEN, _to_device, _as_long

# Pre-tanh outlet command whose flow fraction (tanh + 1) / 2 is half the cutoff.
HOLD_PRE_TANH_TARGET = float(np.arctanh(DISCHARGE_CUTOFF - 1.0))


class LearningMixin:
    """The meta-gradient update of ``PEARLAgent``."""

    def _window_posterior(self, rows, lo, hi):
        """Posterior over z for every transition of a windowed batch.

        Runs the GRU along each chunk's rows, zeroing the state at every row
        some transition's context starts at, and reads transition ``p``'s
        posterior from the state after row ``hi - 1`` -- the state the online
        GRU held when the action was taken, provided each context reaches its
        episode start.

        Raises:
            ValueError: If a context fills the whole window.
        """
        o, a, r, o2 = rows
        K, R = r.shape
        L = lo.shape[1]
        W = R - L + 1
        if bool(((hi - lo) >= W).any()):
            raise ValueError(
                f"a context filled the whole {W}-transition window, so it may "
                f"not reach its episode start; the recurrent encoder needs a "
                f"context_size longer than any episode")
        r = r.reshape(K, R, 1).float()
        if self.ret_rms is not None:
            r = r / self.ret_rms.std
        trans = torch.cat([o.float(), a.float(), r, o2.float()], dim=-1)
        enc = self.encoder
        x = enc.embed_inputs(trans.reshape(K * R, -1)).reshape(K, R, -1)
        reset = torch.zeros(K, R, dtype=torch.bool, device=x.device)
        rows_k = torch.arange(K, device=x.device)
        for p in range(L):
            inside = lo[:, p] < R
            reset[rows_k[inside], lo[inside, p]] = True
        h = torch.zeros(K, enc.hidden, device=x.device)
        states = []
        for q in range(R):
            h = torch.where(reset[:, q:q + 1], torch.zeros_like(h), h)
            h = enc.cell(x[:, q], h)
            states.append(h)
        states = torch.stack(states, dim=1)
        last = (hi - 1).clamp(min=0)
        h_sel = states.gather(
            1, last.unsqueeze(-1).expand(-1, -1, enc.hidden))
        mu, sigma_sq = enc.posterior(h_sel.reshape(K * L, enc.hidden))
        empty = (hi <= lo).reshape(-1, 1)
        mu = torch.where(empty, torch.zeros_like(mu), mu)
        sigma_sq = torch.where(empty, torch.ones_like(sigma_sq), sigma_sq)
        return mu, sigma_sq

    def _stall_flags(self, o, priv):
        """[B, N] 1.0 where a reactor is unloading a batch below the harvest spec.

        Reads the privileged exact titer, so it is a training-time signal only.
        """
        titer = priv[:, list(ETHANOL_CHANNELS)] * TITER_SCALE
        unload = o[:, STAGE_CHANNELS[:, 3]] > 0.5
        below = titer < MIN_HARVEST_TITER * (1.0 + self.hold_margin)
        return (unload & below).float()

    def _critic_cond(self, z, ctx, priv=None):
        """The critics' conditioning vector: z [, task] [, privileged state]."""
        parts = [z]
        if self.critic_task_context:
            parts.append(ctx)
        if self.critic_privileged_dim:
            if priv is None or priv.shape[-1] != self.critic_privileged_dim:
                raise ValueError(
                    f"the critics read a {self.critic_privileged_dim}-wide "
                    f"privileged state, got "
                    f"{None if priv is None else tuple(priv.shape)}: build "
                    f"the replay with priv_dim={self.critic_privileged_dim}")
            parts.append(priv)
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def set_anchor_weights(self, pre=None, kinds=None):
        """Set the anchor's weight before a fault and per fault kind after one.

        Args:
            pre: Weight before any fault (``anchor_weight``); None keeps it.
            kinds: Weights after a fault, indexed by fault code (length
                ``N_DISTURBANCE_KINDS + 1``; index 0 mirrors ``pre``), or one
                number for every kind; None sets every kind to ``pre``.
        """
        if pre is not None:
            self.anchor_weight = float(pre)
        w = np.asarray(self.anchor_weight if kinds is None else kinds,
                       dtype=np.float32).reshape(-1)
        if w.size == 1:
            w = np.full(N_DISTURBANCE_KINDS + 1, float(w[0]), np.float32)
        if w.size != N_DISTURBANCE_KINDS + 1:
            raise ValueError(f"expected {N_DISTURBANCE_KINDS + 1} per-kind "
                             f"anchor weights, got {w.size}")
        if self.anchor_weight < 0.0 or np.any(w < 0.0):
            raise ValueError("anchor weights must be >= 0")
        w = w.copy()
        w[0] = self.anchor_weight
        self._anchor_kind_np = w
        self._anchor_on = bool(self.anchor_weight > 0.0 or np.any(w > 0.0))
        self.anchor_kind_weights = torch.tensor(w, device=device)

    def anchor_kind_weights_np(self):
        """The per-kind post-fault anchor weights (index 0 = pre-fault)."""
        return self._anchor_kind_np.copy()

    def _anchor_active(self):
        return self._anchor_on

    @torch.no_grad()
    def _decay_toward_source(self):
        """Turn this step's AdamW decay into decay toward the loaded weights (L2-SP)."""
        for group in self.actor_opt.param_groups:
            step = float(group["lr"]) * float(group.get("weight_decay", 0.0))
            if step == 0.0:
                continue
            for p in group["params"]:
                ref = self._actor_decay_ref.get(id(p))
                if ref is not None and p.grad is not None:
                    p.add_(ref, alpha=step)

    def _anchor_terms(self, o, zd, fault, gate=None, weight=None, stall=None):
        """The warm-start anchor's penalty and its diagnostics.

        The squared gap between this actor's and the loaded policy's mean
        actions on the stage's free dimensions, at each transition's own
        latent; ``anchor_weight`` before a fault, ``anchor_fault_frac`` x the
        acting kind's weight after one (each dimension scaled by
        ``anchor_fault_dim_scale``).

        Returns:
            Tuple (penalty, gap_nominal, gap_fault).
        """
        with torch.no_grad():
            ref_mean, _ = self.anchor_actor(o, zd)
            ref_a = torch.tanh(ref_mean)
            mask_free = self._stage_mask(o)

        g_dims = (torch.tanh(self.actor(o, zd)[0]) - ref_a).pow(2)
        if mask_free is not None:
            g_dims = g_dims * mask_free
        gap = g_dims.sum(dim=-1)
        gap_post = (g_dims * self.anchor_fault_dim_scale.to(o.device)).sum(dim=-1)
        flag = (fault > 0).to(gap.dtype)
        with torch.no_grad():
            kw = self.anchor_kind_weights.to(o.device)
            kind = fault.long().clamp(0, kw.numel() - 1)
            w_gap = torch.where(flag > 0, self.anchor_fault_frac * kw[kind],
                                torch.full_like(gap, self.anchor_weight))
        if stall is not None:
            w_gap = w_gap * (1.0 - stall)
        pen_gap = torch.where(flag > 0, gap_post, gap)
        penalty = ((w_gap * pen_gap).mean() if weight is None
                   else (weight * w_gap * pen_gap).sum())

        with torch.no_grad():
            acted = gap.detach()
            if gate is not None:
                acted = gate.pow(2) * acted
            nan = torch.full((), float("nan"), device=o.device)
            n_f = flag.sum()
            n_n = flag.numel() - n_f
            gap_fault = torch.where(
                n_f > 0, (acted * flag).sum() / n_f.clamp(min=1.0), nan)
            gap_nominal = torch.where(
                n_n > 0, (acted * (1.0 - flag)).sum() / n_n.clamp(min=1.0),
                nan)
        return penalty, gap_nominal, gap_fault

    def update(self, replay, meta_batch=4, batch_size=256, context_size=64,
               min_buffer=None, want_stats=True, train_actor=True):
        """Run one meta-gradient step over a sample of tasks.

        The lean agent's update: the sampled tasks are trained as ONE batch
        (one optimizer step per network and one Polyak update), the critic
        and encoder first (with the KL, auxiliary and gate losses), then the
        actor (SAC loss, APN mapping penalty, warm-start anchor, all
        gate-weighted behind the gate), then the temperature.

        Returns:
            Dict of scalar diagnostics (empty when ``want_stats`` is False),
            or None if no task has enough data.
        """
        thresh = batch_size if min_buffer is None else min_buffer
        ready = [t for t in replay.task_ids() if replay.size(t) >= thresh]
        if not ready:
            return None
        tasks = np.random.choice(ready, size=min(meta_batch, len(ready)),
                                 replace=False)

        if self.use_context:
            chunk = max(1, min([WINDOW_CHUNK_LEN]
                               + [replay.size(t) for t in tasks]))
            parts = [replay.sample_windows_np(t, batch_size, context_size,
                                              chunk_len=chunk)
                     for t in tasks]
            batch = _to_device(np.concatenate([p[0][i] for p in parts])
                               for i in range(len(parts[0][0])))
            rows = _to_device(np.concatenate([p[1][i] for p in parts])
                              for i in range(4))
            lo = _as_long(np.concatenate([p[2] for p in parts]))
            hi = _as_long(np.concatenate([p[3] for p in parts]))
            mu, sigma_sq = self._window_posterior(rows, lo, hi)
            std = torch.sqrt(torch.clamp(sigma_sq, min=1e-8))
            z = mu + std * torch.randn_like(std)
            kl = 0.5 * torch.sum(
                sigma_sq + mu.pow(2) - 1.0 - torch.log(sigma_sq + 1e-8),
                dim=-1).mean()
        else:
            parts = [replay.sample_batch_np(t, batch_size) for t in tasks]
            batch = _to_device(np.concatenate([p[i] for p in parts])
                               for i in range(len(parts[0])))
            z = mu = torch.zeros(batch[0].shape[0], self._z_dim, device=device)
            kl = torch.zeros((), device=device)
        o, a, r, o2, d, ctx_true, fault_code, priv, priv2 = batch
        o, a, o2, d = o.float(), a.float(), o2.float(), d.float()
        priv, priv2 = priv.float(), priv2.float()
        fault_code = fault_code.float()
        ctx_true, fault = ctx_true.float(), (fault_code > 0).float()
        r = r.float()
        if self.ret_rms is not None:
            r = r / self.ret_rms.std
        r = r * self.reward_scale

        gate_open = None
        if self.adapt_gate:
            with torch.no_grad():
                gate_open = self._gate_open(self.gate, z.detach())

        # ── 1. Critic (+ encoder + KL + auxiliary + gate detector) ───────────
        with torch.no_grad():
            a2, logp2, _ = self._safe_sample(o2, z.detach(), gate=gate_open)
            q1_t, q2_t = self.critic_target(
                o2, a2, self._critic_cond(z.detach(), ctx_true, priv2))
            entropy_term = torch.clamp(self.alpha.detach() * logp2,
                                       -ENTROPY_TERM_CLIP, ENTROPY_TERM_CLIP)
            if gate_open is not None:
                entropy_term = gate_open * entropy_term
            target = r + (1.0 - d) * self.gamma * (
                torch.min(q1_t, q2_t) - entropy_term)
        # z is NOT detached here: this is the encoder's Bellman gradient.
        q1, q2 = self.critic(o, a, self._critic_cond(z, ctx_true, priv))
        critic_loss = (F.smooth_l1_loss(q1, target, beta=1.0)
                       + F.smooth_l1_loss(q2, target, beta=1.0))

        aux_loss = torch.zeros((), device=device)
        if self.aux is not None:
            if self.aux_mode == "reward":
                aux_target = r.detach()

                def aux_err(latent):
                    pred = self.aux(latent, o, a).squeeze(-1)
                    return F.mse_loss(pred, aux_target)
            else:
                aux_target = ctx_true
                if self.aux_state_targets:
                    aux_target = torch.cat(
                        [ctx_true] + [((priv[:, c] - m) / s).unsqueeze(-1)
                                      for c, m, s in self.aux_state_targets],
                        dim=-1)

                def aux_err(latent):
                    return F.mse_loss(self.aux(latent), aux_target)
            f = self.aux_sample_frac
            if f < 1.0:
                aux_loss = aux_loss + (1.0 - f) * aux_err(mu)
            if f > 0.0:
                aux_loss = aux_loss + f * aux_err(z)

        # The hold rule reads the privileged titer: only arms that store it.
        hold_on = self.hold_penalty_weight > 0.0 or self.hold_gate_target
        stalled = (self._stall_flags(o, priv) if hold_on
                   else torch.zeros(o.shape[0], 1, device=o.device))
        stall_any = stalled.amax(dim=-1)
        gate_loss = torch.zeros((), device=device)
        if self.adapt_gate:
            gate_target = (torch.maximum(fault, stall_any)
                           if self.hold_gate_target else fault)
            gate_loss = F.binary_cross_entropy_with_logits(self.gate(mu),
                                                           gate_target)

        critic_obj = critic_loss
        if self.use_context:
            critic_obj = critic_obj + self.kl_lambda * kl
        if self.aux is not None:
            critic_obj = critic_obj + self.aux_weight * aux_loss
            self.aux_opt.zero_grad()
        if self.adapt_gate:
            critic_obj = critic_obj + self.gate_weight * gate_loss
            self.gate_opt.zero_grad()
        self.critic_opt.zero_grad()
        if self.use_context:
            self.encoder_opt.zero_grad()
        critic_obj.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_opt.step()
        if self.use_context:
            nn.utils.clip_grad_norm_(self.encoder.parameters(), 5.0)
            self.encoder_opt.step()
        if self.aux is not None:
            nn.utils.clip_grad_norm_(self.aux.parameters(), 5.0)
            self.aux_opt.step()
        if self.adapt_gate:
            nn.utils.clip_grad_norm_(self.gate.parameters(), 5.0)
            self.gate_opt.step()

        actor_loss = torch.full((), float("nan"), device=device)
        alpha_loss = torch.zeros((), device=device)
        map_pen = torch.zeros((), device=device)
        anchor_pen = torch.zeros((), device=device)
        hold_pen = torch.zeros((), device=device)
        nan = torch.full((), float("nan"), device=device)
        gap_nominal = gap_fault = nan
        if train_actor:
            # ── 2. Actor (z detached; the actor does not shape inference) ─────
            zd = z.detach()
            w_gate = (None if gate_open is None
                      else gate_open / gate_open.sum().clamp(min=1.0))

            def batch_mean(x):
                return x.mean() if w_gate is None else (w_gate * x).sum()

            a_pi, logp_pi, n_active = self._safe_sample(o, zd, gate=gate_open)
            q1_pi, q2_pi = self.critic(o, a_pi,
                                       self._critic_cond(zd, ctx_true, priv))
            actor_loss = batch_mean(self.alpha.detach() * logp_pi
                                    - torch.min(q1_pi, q2_pi))

            # ── 2b. Margin-weighted APN mapping penalty ──────────────────────
            if self.apn is not None and self.apn_penalty_weight > 0.0:
                with torch.no_grad():
                    mask_pi = self._stage_mask(o)
                    s_obs = self._shield_obs(self.apn, o)
                    a_safe, _ = project_to_safe(self.apn, s_obs,
                                                a_pi.detach(),
                                                stage_mask=mask_pi)
                    a_safe = self._apply_safety(o, a_safe, mask=mask_pi)
                    margin_pi = self.apn(s_obs, a_pi.detach())
                    w_unsafe = torch.clamp(-margin_pi, min=0.0) + 0.1
                map_pen = batch_mean((a_pi - a_safe).pow(2).mean(dim=-1)
                                     * w_unsafe)
                actor_loss = actor_loss + self.apn_penalty_weight * map_pen

            # ── 2c. Anchor to the warm-start policy ──────────────────────────
            if self.anchor_actor is not None and self._anchor_active():
                anchor_pen, gap_nominal, gap_fault = self._anchor_terms(
                    o, zd, fault_code, gate=gate_open, weight=w_gate,
                    stall=stall_any if self.hold_penalty_weight > 0.0 else None)
                actor_loss = actor_loss + anchor_pen

            # ── 2d. Hold rule: keep the outlet shut on a below-spec batch ────
            # Applied to the ungated actor so the actor itself learns it. Any
            # discharge above DISCHARGE_CUTOFF drains (and violates every inner
            # step), so the hinge acts on the pre-tanh mean until the command
            # is below half the cutoff, averaged over the stalled entries only.
            if self.hold_penalty_weight > 0.0:
                out_dims = [reactor_action(i, 3) for i in range(stalled.shape[1])]
                pre = self.actor(o, zd)[0][:, out_dims]
                excess = F.relu(pre - HOLD_PRE_TANH_TARGET)
                hold_pen = (stalled * excess).sum() / stalled.sum().clamp(min=1.0)
                actor_loss = actor_loss + self.hold_penalty_weight * hold_pen

            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
            self.actor_opt.step()
            if self._actor_decay_ref is not None:
                self._decay_toward_source()

            # ── 3. Temperature (target entropy over ACTIVE dimensions) ───────
            alpha_loss = -batch_mean(self.log_alpha
                                     * (logp_pi.detach()
                                        + self._entropy_per_dim
                                        * n_active.detach()))
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            with torch.no_grad():
                self.log_alpha.clamp_(self._log_alpha_min, LOG_ALPHA_MAX)

        # ── Polyak target update ─────────────────────────────────────────────
        with torch.no_grad():
            p_list = list(self.critic.parameters())
            tp_list = list(self.critic_target.parameters())
            torch._foreach_mul_(tp_list, 1.0 - self.tau)
            torch._foreach_add_(tp_list, p_list, alpha=self.tau)

        self._mark_rollout_stale()
        if not want_stats:
            return {}
        gate_nominal = gate_fault = nan
        if gate_open is not None:
            with torch.no_grad():
                n_f = fault.sum()
                n_n = fault.numel() - n_f
                gate_fault = torch.where(
                    n_f > 0, (gate_open * fault).sum() / n_f.clamp(min=1.0),
                    nan)
                gate_nominal = torch.where(
                    n_n > 0,
                    (gate_open * (1.0 - fault)).sum() / n_n.clamp(min=1.0),
                    nan)
        return {
            "gate_loss": float(gate_loss.detach()),
            "gate_nominal": float(gate_nominal),
            "gate_fault": float(gate_fault),
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "kl": float(kl.detach()),
            "aux_loss": float(aux_loss.detach()),
            "map_loss": float(map_pen.detach()),
            "anchor_loss": float(anchor_pen.detach()),
            "hold_loss": float(hold_pen.detach()),
            "stall_frac": float(stall_any.mean()),
            "anchor_gap_nominal": float(gap_nominal),
            "anchor_gap_fault": float(gap_fault),
            # |posterior mean|, not |sample|.
            "z_norm": float(mu.detach().abs().mean()),
            "alpha": float(self.alpha.detach().item()),
        }
