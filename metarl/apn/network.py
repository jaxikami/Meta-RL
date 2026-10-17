"""Action Projection Network (APN) for the three-reactor plant.

The lean study's shield (``MetaRL_Lean/apn_lean.py``), widened to three
reactors and the storage tank. Two pieces:

  1. A **learned scalar field** ``h(s, a)``: the signed, normalized distance to
     the nearest constraint boundary (CBF-like), regressed offline on
     synthetic one-control-interval lookahead labels (``metarl.apn.data``,
     ``metarl.apn.train``) and then frozen. With concentrations masked it covers
     the inventory constraints a single control interval can keep: each
     reactor's overflow and underflow, and storage overflow. Storage
     UNDERFLOW is left to the policy: its only fix is a batch ready to unload,
     which is decided tens of hours earlier (see ``metarl.apn.data``).
  2. A **projection operator** (``project_to_safe``) that walks the actor's
     action up ``grad_a h`` until ``sigmoid(h) >= threshold``, staying inside
     ``[-1, 1]^12`` and the stage mask's active subspace, never moving a
     reactor's stage clock.

Neither guarantees feasibility: the walk has a bounded step budget. Callers
re-apply the stage mask afterwards, so precedence is
mask > projection > actor.
"""

import os

import numpy as np
import torch
import torch.nn as nn

# ─── Projection operator defaults (the lean study's) ─────────────────────────
PROJ_MAX_STEPS = 5
PROJ_LR = 0.4

# Margins are regressed in units of MARGIN_SCALE x normalized constraint
# distance, so the logistic head resolves small near-boundary margins.
MARGIN_SCALE = 10.0

# Actions already rated >= PROJ_TARGET_MARGIN are returned untouched; a
# corrected action is pushed to PROJ_REPAIR_MARGIN, past the regressor's
# near-boundary error. Thresholds on the regressor's output, not part of its
# loss, so changing them needs no retraining.
PROJ_TARGET_MARGIN = 0.005
PROJ_THRESHOLD = float(1.0 / (1.0 + np.exp(-MARGIN_SCALE * PROJ_TARGET_MARGIN)))
PROJ_REPAIR_MARGIN = 0.02
PROJ_REPAIR_THRESHOLD = float(
    1.0 / (1.0 + np.exp(-MARGIN_SCALE * PROJ_REPAIR_MARGIN)))
PROJ_RETREAT_ITERS = 12

# Shared by every arm and seed (the APN depends on the plant, not on a run).
APN_PATH = os.environ.get(
    "PLANT_APN_PATH", os.path.join("policy", "apn_plant.pth"))


class ActionProjectionNetwork(nn.Module):
    """Signed safety-margin regressor over (normalized state, squashed action).

    Encoder -> fusion -> residual block -> scalar head, as in the lean study,
    at twice its width for the 75 + 12 inputs. The head is zero-initialised,
    so training starts at margin 0 everywhere.

    Attributes:
        obs_dim: Width of the shield's input (``layout.SHIELD_OBS_DIM``).
        action_dim: Width of the action (12).
        inert_dims: Action dimensions the projection never moves (the stage
            clocks): they do not enter the mass balance.
    """

    def __init__(self, obs_dim, action_dim, latent_dim=128, encoder_dim=128,
                 inert_dims=(0,)):
        """Build the margin regressor.

        Args:
            obs_dim: Normalized observation width the shield reads.
            action_dim: Action width.
            latent_dim: Width of the fused latent and the residual block.
            encoder_dim: Width of the state-action encoder.
            inert_dims: Action dimensions frozen by the projection.
        """
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.inert_dims = tuple(int(d) for d in inert_dims)

        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.Mish(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(encoder_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Mish(),
        )
        self.res_block = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Mish(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.res_act = nn.Mish()
        self.margin_head = nn.Linear(latent_dim, 1)
        nn.init.zeros_(self.margin_head.weight)
        nn.init.zeros_(self.margin_head.bias)

    def forward(self, state_norm, action):
        """Regress the signed safety margin (positive means safe).

        Args:
            state_norm: Normalized observations, shape ``(..., obs_dim)``.
            action: Squashed actions in [-1, 1], shape ``(..., action_dim)``.

        Returns:
            Margin tensor of shape ``(...)``, in MARGIN_SCALE units.
        """
        x = torch.cat([state_norm, action], dim=-1)
        h = self.encoder(x)
        h = self.fusion(h)
        h = self.res_act(h + self.res_block(h))
        return self.margin_head(h).squeeze(-1)

    def classify(self, state_norm, action):
        """Map the margin to a safety probability in (0, 1)."""
        return torch.sigmoid(self.forward(state_norm, action))


def _inert(net, like):
    """Boolean mask over the action dims the projection must not move."""
    m = torch.zeros(like.shape[-1], dtype=torch.bool, device=like.device)
    for d in getattr(net, "inert_dims", (0,)):
        m[d] = True
    return m


def project_to_safe(net, state_norm, action, stage_mask=None,
                    max_steps=PROJ_MAX_STEPS, lr=PROJ_LR,
                    threshold=PROJ_THRESHOLD,
                    repair_threshold=PROJ_REPAIR_THRESHOLD,
                    retreat_iters=PROJ_RETREAT_ITERS):
    """Project actions onto the learned safe set by sign-gradient ascent.

    The lean study's operator, row by row. Guards: the network's
    ``inert_dims`` (every reactor's stage clock) and the stage mask's locked
    dimensions never move; gradient components pushing into an active
    ``[-1, 1]`` bound are zeroed; a stalled unsafe row drives every active
    flow down. A corrected row is walked to ``repair_threshold`` (hysteresis);
    one still short after ``max_steps`` is bisected toward the shut-valve
    action (unlocked flows at -1).

    Args:
        net: A frozen ``ActionProjectionNetwork``.
        state_norm: Normalized observations, shape ``(B, obs_dim)``.
        action: Squashed actions in [-1, 1], shape ``(B, action_dim)``.
        stage_mask: Optional binary mask, shape ``(B, action_dim)``.
        max_steps: Maximum ascent steps.
        lr: Base step size, scaled by severity.
        threshold: ``sigmoid(margin)`` below which an action is corrected.
        repair_threshold: ``sigmoid(margin)`` a corrected action is pushed to.
        retreat_iters: Bisection iterations for the retreat; 0 disables it.

    Returns:
        Tuple of (projected action detached, number of ascent steps taken).
    """
    a = action.detach().clone()
    s = state_norm.detach()
    stage_mask = (torch.ones_like(a) if stage_mask is None
                  else stage_mask.detach())
    inert = _inert(net, a)

    steps_taken = 0
    with torch.no_grad():
        admitted = net.classify(s, a) < threshold
    if not bool(admitted.any()):
        return a, 0
    target = max(float(threshold), float(repair_threshold))

    with torch.enable_grad():
        for _ in range(max_steps):
            a_var = a.clone().requires_grad_(True)
            margin = net(s, a_var)
            with torch.no_grad():
                p = torch.sigmoid(margin)
                active = admitted & (p < target)
            if not bool(active.any()):
                break

            grad = torch.autograd.grad(margin.sum(), a_var)[0]
            steps_taken += 1

            with torch.no_grad():
                grad = grad.clone()
                grad[..., inert] = 0.0
                grad = grad * stage_mask
                grad_original = grad.clone()

                at_lower = (a_var.data <= -0.9999) & (grad < 0)
                at_upper = (a_var.data >= 0.9999) & (grad > 0)
                grad[at_lower | at_upper] = 0.0
                grad_dir = grad.sign()

                # Anti-stall: drive every active flow down.
                stalled = (grad_original.sign() == 0).all(dim=-1) & active
                if bool(stalled.any()):
                    fallback = -1.0 * stage_mask[stalled]
                    fallback[..., inert] = 0.0
                    grad_dir[stalled] = fallback

                unsafe_ratio = torch.clamp(
                    (target - p) / max(target, 1e-6), min=0.0)
                step = (lr * (0.35 + 1.65 * unsafe_ratio)
                        * active.to(grad_dir.dtype)).unsqueeze(-1)
                a = torch.clamp(a_var.data + step * grad_dir, -1.0, 1.0)

    if retreat_iters > 0:
        a = _retreat(net, s, a, stage_mask, admitted, target, retreat_iters,
                     inert)
    return a.detach(), steps_taken


@torch.no_grad()
def _retreat(net, s, a, stage_mask, admitted, target, iters, inert):
    """Bisect still-unsafe rows toward the shut-valve action.

    Unlocked flows go to -1; locked dimensions and the inert clocks keep their
    value, so the mask guarantee and the clock freeze both survive.

    Returns:
        Actions with the retreat applied, shape ``(B, action_dim)``.
    """
    p = net.classify(s, a)
    short = admitted & (p < target)
    if not bool(short.any()):
        return a

    anchor = torch.where(stage_mask > 0, torch.full_like(a, -1.0), a)
    anchor[..., inert] = a[..., inert]
    p_anchor = net.classify(s, anchor)

    bis = short & (p_anchor >= target)
    if bool(bis.any()):
        lo = torch.zeros_like(p)
        hi = torch.ones_like(p)
        d = anchor - a
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            ok = net.classify(s, a + mid.unsqueeze(-1) * d) >= target
            hi = torch.where(ok, mid, hi)
            lo = torch.where(ok, lo, mid)
        a = torch.where(bis.unsqueeze(-1), a + hi.unsqueeze(-1) * d, a)

    worse = short & ~bis & (p_anchor > p)
    return torch.where(worse.unsqueeze(-1), anchor, a)


def plant_signature():
    """The plant constants a trained shield is only valid for.

    Returns:
        Dict of plain Python values; ``load_apn`` refuses a checkpoint whose
        stored signature differs.
    """
    from metarl.env.config import (get_reactor_config, S_MAX, S_MIN,
                                   S_OUT_RATE, INITIAL_LF, INITIAL_C5,
                                   MAX_FLOW_OUT_REAC)
    from metarl.env.layout import (OBS_LAYOUT_VERSION, SHIELD_OBS_DIM,
                                   ACTION_DIM)
    cfg = get_reactor_config()
    return {
        "Mmax": [float(v) for v in cfg["Mmax"]],
        "max_flow_LF": [round(float(v), 9) for v in cfg["max_flow_LF"]],
        "max_flow_C5": [round(float(v), 9) for v in cfg["max_flow_C5"]],
        "max_flow_out": round(float(MAX_FLOW_OUT_REAC), 9),
        "S_MAX": float(S_MAX), "S_MIN": float(S_MIN),
        "S_OUT_RATE": round(float(S_OUT_RATE), 9),
        "INITIAL_LF": float(INITIAL_LF), "INITIAL_C5": float(INITIAL_C5),
        "obs_layout": int(OBS_LAYOUT_VERSION),
        "shield_obs_dim": int(SHIELD_OBS_DIM), "action_dim": int(ACTION_DIM),
    }


def load_apn(path=None, obs_dim=None, action_dim=None, device=None,
             strict=True):
    """Load frozen APN weights, refusing one trained for another plant.

    Args:
        path: Checkpoint path; defaults to ``APN_PATH``.
        obs_dim: Expected shield input width (checked when given).
        action_dim: Expected action width (checked when given).
        device: Torch device; defaults to CUDA when available.
        strict: Raise when the checkpoint is missing, rather than returning
            None.

    Returns:
        A frozen ``ActionProjectionNetwork`` in eval mode, or None when the
        file is absent and ``strict`` is False.

    Raises:
        SystemExit: When the checkpoint is missing (strict) or was trained
            under a different observability setting, quality spec or plant.
    """
    p = APN_PATH if path is None else path
    dev = device if device is not None else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if not os.path.exists(p):
        msg = (f"[!] no action-projection network at {p!r}. "
               f"Train one first:  python -m metarl.apn.train")
        if strict:
            raise SystemExit(msg)
        print(msg)
        return None

    ckpt = torch.load(p, map_location=dev, weights_only=False)

    from metarl.env.config import (MASK_INTERNAL_CONCENTRATIONS,
                                   MIN_HARVEST_TITER, MIN_PRODUCT_TITER)
    retrain = (f"    Retrain it:  move {p} aside, then "
               f"python -m metarl.apn.train")
    if bool(ckpt.get("mask_internal_concentrations")) != bool(
            MASK_INTERNAL_CONCENTRATIONS):
        raise SystemExit(
            f"[!] observability mismatch for the APN at {p!r}: trained with "
            f"mask_internal_concentrations = "
            f"{ckpt.get('mask_internal_concentrations')}, configured "
            f"{MASK_INTERNAL_CONCENTRATIONS}.\n" + retrain)
    stored_sig = ckpt.get("plant_signature")
    if stored_sig != plant_signature():
        raise SystemExit(
            f"[!] the APN at {p!r} was trained for a different plant "
            f"(reactor sizes, storage tank, feedstock pools or observation "
            f"layout).\n" + retrain)
    if not MASK_INTERNAL_CONCENTRATIONS:
        want = {"min_product_titer": float(MIN_PRODUCT_TITER),
                "min_harvest_titer": float(MIN_HARVEST_TITER)}
        drift = {k: (ckpt.get(k), v) for k, v in want.items()
                 if float(ckpt.get(k, -1.0)) != v}
        if drift:
            raise SystemExit(
                f"[!] the APN at {p!r} was trained against different quality "
                f"specs ({drift}).\n" + retrain)

    net = ActionProjectionNetwork(
        ckpt["obs_dim"], ckpt["action_dim"],
        latent_dim=ckpt.get("latent_dim", 128),
        encoder_dim=ckpt.get("encoder_dim", 128),
        inert_dims=tuple(ckpt.get("inert_dims", (0,)))).to(dev)
    for want, have, what in ((obs_dim, net.obs_dim, "obs_dim"),
                             (action_dim, net.action_dim, "action_dim")):
        if want is not None and int(want) != int(have):
            raise SystemExit(f"[!] the APN at {p!r} has {what}={have}, the "
                             f"agent needs {want}.\n" + retrain)
    net.load_state_dict(ckpt["model"])
    net.eval()
    for prm in net.parameters():
        prm.requires_grad_(False)
    return net


def save_apn(net, path=None, meta=None):
    """Write APN weights plus the metadata ``load_apn`` checks.

    Args:
        net: The network to save.
        path: Destination; defaults to ``APN_PATH``.
        meta: Optional dict of extra fields (training diagnostics).

    Returns:
        The path written.
    """
    from metarl.env.config import (MASK_INTERNAL_CONCENTRATIONS,
                                   MIN_HARVEST_TITER, MIN_PRODUCT_TITER)
    p = APN_PATH if path is None else path
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    ckpt = {
        "model": net.state_dict(),
        "obs_dim": net.obs_dim,
        "action_dim": net.action_dim,
        "latent_dim": net.fusion[0].out_features,
        "encoder_dim": net.encoder[0].out_features,
        "inert_dims": list(net.inert_dims),
        "margin_scale": MARGIN_SCALE,
        "min_product_titer": float(MIN_PRODUCT_TITER),
        "min_harvest_titer": float(MIN_HARVEST_TITER),
        "mask_internal_concentrations": bool(MASK_INTERNAL_CONCENTRATIONS),
        "plant_signature": plant_signature(),
    }
    if meta:
        ckpt.update(meta)
    torch.save(ckpt, p)
    return p


__all__ = ["ActionProjectionNetwork", "project_to_safe", "load_apn", "save_apn",
           "plant_signature", "APN_PATH", "MARGIN_SCALE", "PROJ_MAX_STEPS",
           "PROJ_LR", "PROJ_THRESHOLD", "PROJ_TARGET_MARGIN",
           "PROJ_REPAIR_MARGIN", "PROJ_REPAIR_THRESHOLD", "PROJ_RETREAT_ITERS"]
