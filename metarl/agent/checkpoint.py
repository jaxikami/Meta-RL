"""Checkpoints (``CheckpointMixin``): save, load, warm start from an arm."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import copy
import os

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.agent.common import device

# net_config entries added after checkpoints already existed, at the value
# those checkpoints were trained with (none yet for the plant).
LEGACY_NET_DEFAULTS = {}

# net_config entries that describe only what a context agent adds; compared
# only when both sides have context, so saferl stays a warm-start source.
ENCODER_NET_KEYS = ("encoder_layer_norm", "encoder_delta", "adapt_gate",
                    "task_context_dim", "encoder_recurrent", "encoder_hidden",
                    "latent_dim")


def _net_mismatch(stored, mine, compare_encoder):
    """Keys on which a checkpoint's network options differ from an agent's.

    Args:
        stored: The checkpoint's ``net`` dict.
        mine: The agent's ``net_config``.
        compare_encoder: Include ``ENCODER_NET_KEYS``; False when either side
            is context-free.

    Returns:
        Sorted list of differing keys, empty when compatible.
    """
    have = {**LEGACY_NET_DEFAULTS, **stored}
    keys = set(have) | set(mine)
    if not compare_encoder:
        keys -= set(ENCODER_NET_KEYS)
    return sorted(k for k in keys if have.get(k) != mine.get(k))


def _film_padded(src, dst, where, z_dim):
    """Fit a source state dict to ``dst``'s shapes for a warm start.

    A FiLM projection WEIGHT reads the conditioning vector, whose first
    ``z_dim`` entries are the latent: those columns come back as zeros, so
    every layer keeps the source's modulation for any z, and the rest (the
    critics' task descriptor and privileged state) is copied. A FiLM weight of
    another conditioning width comes back all zeros.

    Raises:
        ValueError: On a missing key or any other shape mismatch.
    """
    out = {}
    for k, v in dst.items():
        if k not in src:
            raise ValueError(f"{where}: no tensor {k!r}")
        s = src[k]
        film = ".films." in f".{k}" and k.endswith(".proj.weight")
        if s.shape == v.shape:
            out[k] = s
            if film:
                out[k] = s.clone()
                out[k][:, :z_dim] = 0.0
        elif film:
            out[k] = torch.zeros_like(v)
        else:
            raise ValueError(f"{where}: {k!r} has shape {tuple(s.shape)}, "
                             f"this agent needs {tuple(v.shape)}")
    return out


class CheckpointMixin:
    """Checkpoint save / load / warm start of ``PEARLAgent``."""

    def save(self, path):
        """Write a checkpoint."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "use_context": self.use_context,
            "use_film": self.use_film,
            "net": dict(self.net_config),
            "aux_mode": self.aux_mode,
            "aux_sample_frac": self.aux_sample_frac,
            "use_safeguard": self.use_safeguard,
            "uses_apn": self.apn is not None,
            "apn_hard_project": self.apn_hard_project,
            "critic_task_context": self.critic_task_context,
            "critic_privileged_dim": self.critic_privileged_dim,
        }
        if self.ret_rms is not None:
            ckpt["ret_rms"] = (float(self.ret_rms.mean), float(self.ret_rms.var),
                               float(self.ret_rms.count))
        if self.use_context:
            ckpt["encoder"] = self.encoder.state_dict()
        if self.aux is not None:
            ckpt["aux"] = self.aux.state_dict()
        if self.fault_head is not None:
            ckpt["fault_head"] = self.fault_head.state_dict()
        if self.adapt_gate:
            ckpt["base_actor"] = self.base_actor.state_dict()
            ckpt["gate"] = self.gate.state_dict()
            ckpt["gate_sharpness"] = self.gate_sharpness
        torch.save(ckpt, path)

    @staticmethod
    def _torch_load_safe(path):
        """Load a checkpoint, preferring the restricted unpickler."""
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except Exception:
            print(f"[warn] {path}: falling back to weights_only=False")
            return torch.load(path, map_location=device, weights_only=False)

    def _check_net(self, path, ckpt, what):
        """Raise ValueError when a checkpoint's network options differ."""
        stored = ckpt.get("net")
        both_context = (self.use_context
                        and bool(ckpt.get("use_context", self.use_context)))
        diff = (sorted(self.net_config) if stored is None
                else _net_mismatch(stored, self.net_config, both_context))
        if diff:
            have = {**LEGACY_NET_DEFAULTS, **(stored or {})}
            layout = {"obs_layout", "obs_dim", "arch", "n_reactors"} & set(diff)
            raise ValueError(
                f"cannot {what} {path}: it differs from this agent in "
                + ", ".join(f"{k} ({have.get(k)} there, "
                            f"{self.net_config.get(k)} here)" for k in diff)
                + (". The plant's observations or the network architecture "
                   "changed after it was trained; only retraining fixes this."
                   if layout else
                   ". Retrain that arm, or set the NET_* constants in "
                   "metarl/training/config.py to match."))

    def load(self, path):
        """Restore a checkpoint into an identically constructed agent.

        The one difference tolerated is what the critics are conditioned on;
        their input width differs, so they are left as they are (acting and
        evaluation read only the actor and the encoder).

        Raises:
            ValueError: On a different architecture or observation layout, or
                when this agent has the gate and the checkpoint none.
        """
        ckpt = self._torch_load_safe(path)
        self._check_net(path, ckpt, "load")
        self.actor.load_state_dict(ckpt["actor"])
        stored_ctc = bool(ckpt.get("critic_task_context", False))
        stored_cpd = int(ckpt.get("critic_privileged_dim", 0))
        if (stored_ctc == self.critic_task_context
                and stored_cpd == self.critic_privileged_dim):
            self.critic.load_state_dict(ckpt["critic"])
            self.critic_target.load_state_dict(ckpt["critic_target"])
        else:
            print(f"[info] {path}: its critics were conditioned differently "
                  f"(task={stored_ctc}, plant state={stored_cpd} ch); critics "
                  f"not loaded (acting and evaluation use only the actor and "
                  f"encoder)")
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"].to(device))
        if "ret_rms" in ckpt and self.ret_rms is not None:
            mean, var, count = ckpt["ret_rms"]
            self.ret_rms.mean = float(mean)
            self.ret_rms.var = float(var)
            self.ret_rms.count = float(min(count, self.ret_rms.max_count))
        if self.use_context and "encoder" in ckpt:
            self.encoder.load_state_dict(ckpt["encoder"])
        if self.aux is not None and "aux" in ckpt:
            mine = self.aux.state_dict()
            if all(k in ckpt["aux"] and ckpt["aux"][k].shape == v.shape
                   for k, v in mine.items()):
                self.aux.load_state_dict(ckpt["aux"])
            else:
                print(f"[info] {path}: its aux head predicts a different "
                      f"target width; aux head not loaded")
        if self.fault_head is not None and "fault_head" in ckpt:
            self.fault_head.load_state_dict(ckpt["fault_head"])
        if self.adapt_gate:
            if "base_actor" not in ckpt or "gate" not in ckpt:
                raise ValueError(
                    f"{path} holds no adaptation gate, but this agent acts "
                    f"through one. Retrain this arm (a context-free source "
                    f"is loaded with warm_start_from).")
            self.base_actor.load_state_dict(ckpt["base_actor"])
            self.gate.load_state_dict(ckpt["gate"])
            self._mark_base_stale()
            stored = ckpt.get("gate_sharpness")
            if stored is not None and float(stored) != self.gate_sharpness:
                print(f"[info] adopting checkpoint gate_sharpness={stored} "
                      f"(was {self.gate_sharpness})")
                self.gate_sharpness = float(stored)
        for key, have in (("use_safeguard", self.use_safeguard),
                          ("uses_apn", self.apn is not None),
                          ("apn_hard_project", self.apn_hard_project)):
            want = ckpt.get(key)
            if want is not None and bool(want) != bool(have):
                print(f"[!] {path}: checkpoint has {key}={want} but this agent "
                      f"was built with {key}={have}. The arm's safety stack "
                      f"does not match its weights.")
        self._mark_rollout_stale()
        self.clear_z()

    def warm_start_from(self, path, anchor_weight=0.0, anchor_fault_frac=1.0,
                        decay_to_source=False, anchor_fault_dims=None):
        """Start from another agent's trained policy and critics.

        Loads the actor, both critics and the target critic, the temperature,
        the reward scaler and the observation statistics of a checkpoint with
        the same ``net_config`` (a context-free agent's, whose encoder options
        are not compared). FiLM projection weights stay at zero, so this agent
        acts exactly as the source for every z until training grows them.
        With the gate, the loaded policy is also its frozen base (and the
        anchor).

        Args:
            path: Checkpoint written by ``save``.
            anchor_weight: Weight of the pull toward the loaded policy; 0
                disables it.
            anchor_fault_frac: Share of that weight kept after a fault.
            decay_to_source: Decay the actor's weights toward the loaded ones.
            anchor_fault_dims: Optional per-action-dimension share of the
                post-fault gap the anchor counts.

        Raises:
            ValueError: On a different network architecture or observation
                layout, a mismatched tensor other than a FiLM weight, or an
                anchor setting out of range.
        """
        if not 0.0 <= float(anchor_fault_frac) <= 1.0:
            raise ValueError(f"anchor_fault_frac must be in [0, 1], got "
                             f"{anchor_fault_frac}")
        dims = (np.ones(self.action_dim) if anchor_fault_dims is None
                else np.asarray(anchor_fault_dims, dtype=np.float32).ravel())
        if dims.size != self.action_dim or np.any((dims < 0.0) | (dims > 1.0)):
            raise ValueError(f"anchor_fault_dims needs {self.action_dim} "
                             f"values in [0, 1], got {anchor_fault_dims}")
        ckpt = self._torch_load_safe(path)
        self._check_net(path, ckpt, "warm-start from")
        for key, module in (("actor", self.actor), ("critic", self.critic),
                            ("critic_target", self.critic_target)):
            module.load_state_dict(_film_padded(
                ckpt[key], module.state_dict(), f"{path} [{key}]",
                self._z_dim))
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"].to(device))
        if "ret_rms" in ckpt and self.ret_rms is not None:
            mean, var, count = ckpt["ret_rms"]
            self.ret_rms.mean = float(mean)
            self.ret_rms.var = float(var)
            self.ret_rms.count = float(min(count, self.ret_rms.max_count))
        for key, have in (("use_safeguard", self.use_safeguard),
                          ("uses_apn", self.apn is not None)):
            want = ckpt.get(key)
            if want is not None and bool(want) != bool(have):
                print(f"[info] warm start from {path}: the source ran with "
                      f"{key}={want}, this agent runs with {key}={have}")
        self.set_anchor_weights(pre=float(anchor_weight))
        self.anchor_fault_frac = float(anchor_fault_frac)
        self.anchor_fault_dim_scale = torch.as_tensor(
            dims, dtype=torch.float32, device=device)
        self._actor_decay_ref = (
            {id(p): p.detach().clone() for p in self.actor.parameters()}
            if decay_to_source else None)
        # The source's per-reactor history stays as it was trained: the base
        # policy and the anchor read the same features, and this arm adapts
        # through z.
        self.history_frozen = True
        for p in self.actor.history.parameters():
            p.requires_grad_(False)
        if self.adapt_gate:
            self.base_actor.load_state_dict(self.actor.state_dict())
            self._mark_base_stale()
        self.anchor_actor = None
        if self.anchor_weight > 0.0:
            if self.adapt_gate:
                self.anchor_actor = self.base_actor
            else:
                self.anchor_actor = copy.deepcopy(self.actor)
                for p in self.anchor_actor.parameters():
                    p.requires_grad_(False)
        self._mark_rollout_stale()
        self.clear_z()
