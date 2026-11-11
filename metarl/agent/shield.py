"""The APN shield (``ShieldMixin``): Safe PPO's multi-head APN around the policy.

L2 projects each rollout action (``_project``), L3 adapts the projection's
buffers and residual within the episode (``observe_step``), the refiner
refreshes KineticNet on the agent's own states (``observe_episode``), and the
update adds the APN's safety gradient and mapping penalty to the actor's loss
(``_apn_actor_terms``). The APN itself (``metarl.apn``) is frozen.
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import copy

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn.functional as F

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import N_REACTORS
from metarl.apn.constants import I_S
from metarl.apn.adaptation import AdaptiveBackoff, AdaptiveResidual
from metarl.agent.common import device, ROLLOUT_DEVICE

# MultiHeadAPN.safety_loss()'s hinge, per sample so the gate can weight it.
SAFETY_SHARPNESS = 50.0
SAFETY_EPS = 1e-3
# Rows per apn.project call in the update. ROCm's batched solve
# (hipblasSgetrsBatched) fails with HIPBLAS_STATUS_ALLOC_FAILED at 2048 rows
# (the meta arm's 4 x 512) on an RX 9070 XT with 17 GB free; 1024 runs.
PROJECT_CHUNK = 1024
# Ceiling of L3's located capacity limits (their floors), per constraint
# group: storage overflow (3) and heat load (9). AdaptiveBackoff's own (0.15
# and 0.25) put the lowest caps the shield can enforce at 0.80 S_MAX and 0.70
# MAX_HEAT_LOAD, enough for Safe PPO's evaluation faults (0.90 tank, 0.80
# cooling) but not for meta's training ranges (down to 0.75 and 0.65) or T1's
# extremes (0.65 and 0.52): with a 0.65 tank the shield kept draining into a
# 0.80 wall 23% past the real cap once calibrate() had located it.
# 0.475 = (1 - buffer) x 0.5 follows either cap down to half its rating.
CAP_OFFSET_MAX = {3: 0.475, 9: 0.475}


class ShieldBackoff(AdaptiveBackoff):
    """AdaptiveBackoff whose located capacity limits may reach CAP_OFFSET_MAX.

    Only the floor goes deeper. The alarm-rate rule keeps Safe PPO's ceiling
    (or the floor, if higher): after a cap drops below the operating point,
    the plant alarms every step until the level or load has come down, and
    those alarms say nothing new. Letting them raise the storage buffer to
    0.475 overshot the 0.33 a 0.65 tank needs, and the fixed-rate recipe lost
    10% on-spec.
    """

    def update(self, violation_counts):
        """AdaptiveBackoff.update, each capacity ceiling at max(its own, floor)."""
        self.DELTA_MAX = {**AdaptiveBackoff.DELTA_MAX,
                          **{g: max(AdaptiveBackoff.DELTA_MAX[g], self.floor[g])
                             for g in CAP_OFFSET_MAX}}
        super().update(violation_counts)

    def _raise_floor(self, g, value):
        """AdaptiveBackoff._raise_floor, a capacity floor up to CAP_OFFSET_MAX."""
        self.DELTA_MAX = {**AdaptiveBackoff.DELTA_MAX, **CAP_OFFSET_MAX}
        super()._raise_floor(g, value)


class ShieldMixin:
    """The APN projection, its L3 adaptation and KineticNet refresh of ``PEARLAgent``."""

    def _init_shield(self, apn, penalty_weight, safety_grad_weight,
                     hard_project, adaptation):
        # The APN (``metarl.apn.apn.MultiHeadAPN``) is frozen; only its
        # KineticNet moves, refreshed by ``refiner`` (training process only).
        self.apn = apn
        if self.apn is not None:
            self.apn.eval()
            for p in self.apn.parameters():
                p.requires_grad_(False)
        self.apn_penalty_weight = float(penalty_weight)
        self.apn_safety_grad_weight = float(safety_grad_weight)
        self.apn_hard_project = bool(hard_project)
        self.apn_adaptation = bool(adaptation)
        self.refiner = None
        self._refine_episodes = 0
        self._shield_backoff = ShieldBackoff()
        self._shield_residual = AdaptiveResidual(N_REACTORS)
        self._shield_last = None
        self._proj_calls = self._proj_noop = self._proj_steps = 0
        self._proj_moved = 0

    def _reset_shield(self):
        """Clear L3's adaptation (every episode start)."""
        self._shield_backoff.reset()
        self._shield_residual.reset()
        self._shield_last = None

    def _shield_state(self):
        return copy.deepcopy((self._shield_backoff, self._shield_residual,
                              self._shield_last))

    def _set_shield_state(self, state):
        (self._shield_backoff, self._shield_residual,
         self._shield_last) = copy.deepcopy(state)

    @staticmethod
    def _rows(x):
        """A flat observation or action [..., n * k] as per-reactor rows [B, n, k]."""
        return x.reshape(-1, N_REACTORS, x.shape[-1] // N_REACTORS)

    def _shield_l3(self):
        """L3's offsets and residual for the APN (none without adaptation)."""
        if not self.apn_adaptation:
            return {}
        return {"offsets": self._shield_backoff.offsets(),
                "residual": self._shield_residual.tensors(ROLLOUT_DEVICE)}

    def _project(self, obs, action, mask=None):
        """L2: project a rollout action with the APN, then re-apply the mask.

        Precedence: mask > projection > actor. L3's residual is first fitted
        to the transition the last executed action caused.
        """
        if self._apn_r is None or not self.apn_hard_project:
            return action
        if mask is None:
            mask = self._stage_mask(obs)
        rows = self._rows(obs)
        if self.apn_adaptation and self._shield_last is not None:
            last_obs, last_act = self._shield_last
            self._shield_residual.update(last_obs, last_act,
                                         rows[0].cpu().numpy())
        proj, stats = self._apn_r.project(rows, self._rows(action),
                                          **self._shield_l3())
        out = self._apply_safety(obs, proj.reshape(action.shape), mask=mask)
        self._shield_last = (rows[0].cpu().numpy(),
                             self._rows(out)[0].cpu().numpy())
        self._proj_calls += 1
        self._proj_steps += stats["iters"]
        if stats["noop"]:
            self._proj_noop += 1
        if bool((out - action).abs().max() > 1e-6):
            self._proj_moved += 1
        return out

    def observe_step(self, next_state, info):
        """L3: learn from the plant's alarms after the last executed action.

        Call after every ``env.step`` of an episode the agent acts in. Raises
        the buffer of every constraint the plant alarmed on and re-locates a
        moved storage or heat-removal limit (Safe PPO's ``observe``).

        Args:
            next_state: Flat observation after the step.
            info: The step's info dict ('violation_counts' and the alarms'
                depths).
        """
        if (self._apn_r is None or not self.apn_hard_project
                or not self.apn_adaptation or self._shield_last is None):
            return
        self._shield_backoff.update(info.get("violation_counts", {}))
        nxt = np.asarray(next_state, dtype=np.float32).reshape(N_REACTORS, -1)
        self._shield_backoff.calibrate(info, float(self._shield_last[0][0, I_S]),
                                       float(nxt[0, I_S]))

    def proj_counts(self):
        """The projection's raw counters: (calls, no-ops, steps, moved)."""
        return (self._proj_calls, self._proj_noop, self._proj_steps,
                self._proj_moved)

    def add_proj_counts(self, counts):
        """Add raw counters from elsewhere (a worker's episode)."""
        calls, noop, steps, moved = counts
        self._proj_calls += calls
        self._proj_noop += noop
        self._proj_steps += steps
        self._proj_moved += moved

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

    # ------------------------------------------------- the APN's KineticNet

    @property
    def kinetic(self):
        """The APN's KineticNet (its learned heads), or None."""
        return None if self.apn is None else self.apn.kinetic

    def install_kinetic(self, state_dict):
        """Load KineticNet weights into the APN and its rollout copy."""
        self.kinetic.load_state_dict(state_dict)
        if self._apn_r is not self.apn:
            self._apn_r.kinetic.load_state_dict(
                {k: v.cpu() for k, v in state_dict.items()})

    def enable_refiner(self, every, episodes_per_step, seed=0):
        """Refresh KineticNet on this agent's own states while it trains.

        Safe PPO's DAgger refresh (``metarl.apn.refiner``): it relabels the
        policy's (state, action) pairs in the vectorized plant and fine-tunes
        KineticNet on them every ``every`` refiner steps. A step is
        ``episodes_per_step`` training episodes, Safe PPO's rollout per PPO
        update. Only the training process refines; its workers receive the
        weights with ``rollout_weights``.
        """
        if self.kinetic is None or every <= 0:
            return
        from metarl.apn.refiner import APNRefiner
        self.refiner = APNRefiner(self.kinetic, device, every=int(every),
                                  seed=seed)
        self._refine_per_step = int(episodes_per_step)
        self._refine_episodes = 0

    def observe_episode(self, observations, actions):
        """Hand one training episode's (state, executed action) to the refiner.

        Args:
            observations: Flat observations the actions were chosen on,
                [T, STATE_DIM].
            actions: Executed actions, [T, ACTION_DIM].

        Returns:
            True if KineticNet was refreshed.
        """
        if self.refiner is None:
            return False
        obs = np.asarray(observations, dtype=np.float32)
        act = np.asarray(actions, dtype=np.float32)
        self.refiner.add(obs.reshape(len(obs), N_REACTORS, -1),
                         act.reshape(len(act), N_REACTORS, -1))
        self._refine_episodes += 1
        if self._refine_episodes % self._refine_per_step:
            return False
        if not self.refiner.step():
            return False
        if self._apn_r is not self.apn:
            self.install_kinetic(self.kinetic.state_dict())
        return True

    # ------------------------------------------------ the actor's APN terms

    def _project_batch(self, s_rows, a_rows, m):
        """``apn.project`` of a whole update batch, in PROJECT_CHUNK-row calls.

        The rows are corrected independently, except that ``project`` returns
        early when every row is safe and otherwise trims the drain of every
        row. A chunk that is all safe while the batch is not is therefore
        trimmed first, so each row gets what one call on the batch gives it.

        Args:
            s_rows: States [B, n, 28].
            a_rows: Actions [B, n, 4], detached.
            m: ``apn.margins(s_rows, a_rows)`` [B, H].

        Returns:
            The projected actions [B, n, 4].
        """
        if len(a_rows) <= PROJECT_CHUNK:
            return self.apn.project(s_rows, a_rows)[0]
        if not bool((m >= 0.0).all()):
            # Idempotent, so the chunks that project trims again are unchanged.
            a_rows = self.apn._trim_drain(s_rows, a_rows)
        return torch.cat([self.apn.project(s, a)[0] for s, a in
                          zip(s_rows.split(PROJECT_CHUNK),
                              a_rows.split(PROJECT_CHUNK))])

    def _apn_actor_terms(self, actor_loss, o, feat, zd, gate, batch_mean):
        """Add Safe PPO's safety gradient and mapping penalty to the actor's loss.

        Both act on the actor's deterministic action as it would be sent
        before projection: the hinge on every head's margin, backpropagated
        through the frozen APN, and the pull toward that action's projection,
        weighted by its worst margin. Off-policy, the projection is recomputed
        here (static buffers: L3 lives in the episode).

        Returns:
            Tuple (actor_loss, safety_pen, map_pen).
        """
        safety_pen = torch.zeros((), device=device)
        map_pen = torch.zeros((), device=device)
        if self.apn is None or not (self.apn_penalty_weight > 0.0
                                    or self.apn_safety_grad_weight > 0.0):
            return actor_loss, safety_pen, map_pen
        a_det, _, _ = self._safe_sample(o, feat, zd, deterministic=True,
                                        with_logprob=False, gate=gate)
        s_rows, a_rows = self._rows(o), self._rows(a_det)
        if self.apn_safety_grad_weight > 0.0:
            m = self.apn.margins(s_rows, a_rows)
            hinge = F.softplus(SAFETY_SHARPNESS * (SAFETY_EPS - m)) / SAFETY_SHARPNESS
            safety_pen = batch_mean(hinge.mean(dim=-1))
            actor_loss = actor_loss + self.apn_safety_grad_weight * safety_pen
        if self.apn_penalty_weight > 0.0:
            with torch.no_grad():
                m = self.apn.margins(s_rows, a_rows.detach())
                w_unsafe = torch.clamp(-m.min(dim=1).values, min=0.0) + 0.1
                a_safe = self._project_batch(s_rows, a_rows.detach(), m)
                a_safe = self._apply_safety(o, a_safe.reshape(a_det.shape))
            map_pen = batch_mean((a_det - a_safe).pow(2).mean(dim=-1)
                                 * w_unsafe)
            actor_loss = actor_loss + self.apn_penalty_weight * map_pen
        return actor_loss, safety_pen, map_pen
