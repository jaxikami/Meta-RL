"""Multi-head Action Projection Network (APN): the safety layer of Safe PPO.

The APN scores a (state, action) pair with one signed margin per constraint
head (positive: safe by that much; negative: violated by that much, both as a
fraction of the limit) and moves an unsafe action back inside every head's
boundary.  It works in three stages:

    L1  estimate   The margin of every head.  Exact heads are the plant's own
                   mass-balance and titer arithmetic written in torch, plus a
                   residual identified online; learned heads are a per-reactor
                   network (KineticNet) trained on simulator rollouts of the
                   kinetics (metarl/apn/train.py).
    L2  correct    A closed-form minimum-norm step on the heads that are short
                   of their boundary, damped by priority so the more important
                   head wins a conflict, then a final pass on the containment
                   heads alone.  Drain valves wider than needed to empty
                   their reactor are trimmed first, so they stay a lever.  No optimizer and no iterations beyond a few
                   re-linearizations.
    L3  adapt      Buffers raised by the plant's own violation alarms
                   (AdaptiveBackoff) and the exact heads' residual fitted to
                   realized transitions by recursive least squares
                   (AdaptiveResidual), for faults the observation does not show.

Every margin is differentiable in the action, so the same heads give the actor
a safety gradient during the PPO update (safety_loss()).

Heads (constraint id in brackets; R = one per reactor, P = plant-wide):

    overflow       [1] R exact    M after the hold <= 95% Mmax
    underflow      [2] R exact    M after the hold >= 105% of the 10 kg dry floor (only a
                                  leak, seen by the residual, can get there: the plant
                                  never drains below 50 kg)
    storage_over   [3] P exact    S after the hold <= 95% S_max
    storage_under  [4] P exact    S after the hold >= 105% S_min
    titer_drain    [7] R exact    no discharge while the titer is below 102% of
                                  MIN_HARVEST_TITER
    purity         [8] R learned  titer when unload begins, if the action is held,
                                  >= 101% of MIN_PRODUCT_TITER (stages 0-2 only)
    heat           [9] P learned  plant CO2 evolution rate over the next
                                  HEAT_HORIZON_H hours, if every action is held,
                                  <= 95% MAX_HEAT_LOAD

Plant-wide heads are sums of per-reactor terms, and KineticNet sees one reactor
at a time, so the APN works for any number of reactors without retraining.
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os
from dataclasses import dataclass

# ─── Third-Party ──────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import MAX_HEAT_LOAD, MIN_HARVEST_TITER, MIN_PRODUCT_TITER, S_MAX, S_MIN
from metarl.apn.constants import (
    ACTION_DIM, DISCHARGE_FLOOR_KG, DRY_FLOOR_KG, F_OUT_KGH, FC5_NORM, FLF_NORM, HEAT_BUFFER,
    HEAT_TAU, HOLD_H, I_ETH, I_FC5, I_FLF, I_M, I_MMAX, I_S, I_STAGE, KINETIC_PATH, M_NORM,
    MASS_BUFFER, PURITY_BUFFER, REACTOR_FEATURES, S_OUT_KGH, STORAGE_BUFFER, TITER_BUFFER,
    TITER_NORM, UNDERFLOW_BUFFER
)
from metarl.apn.kinetic import KineticNet

# ─── Head Registry ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HeadSpec:
    """One constraint head.

    Attributes:
        name: Head name (key of head_margins()).
        group: Violation id the head guards; L3 backs it off on that
            id's alarms.
        per_reactor: One margin per reactor (else one for the plant).
        exact: Computed from the plant's own arithmetic (else KineticNet).
        priority: Weight in the correction step: the higher one wins a conflict.
        dims: Action dims the head may move (0 time, 1 LF, 2 C5, 3 discharge).
    """
    name: str
    group: int
    per_reactor: bool
    exact: bool
    priority: float
    dims: tuple


HEADS = (
    HeadSpec("overflow", 1, True, True, 100.0, (1, 2, 3)),
    HeadSpec("underflow", 2, True, True, 100.0, (1, 2, 3)),
    HeadSpec("storage_over", 3, False, True, 100.0, (3,)),
    HeadSpec("storage_under", 4, False, True, 3.0, (3,)),
    HeadSpec("titer_drain", 7, True, True, 30.0, (3,)),
    HeadSpec("purity", 8, True, False, 1.0, (0, 1, 2)),
    HeadSpec("heat", 9, False, False, 30.0, (0, 1, 2)),
)
# Physical containment: re-checked alone after the joint correction so a
# compromise with a softer head can never leave them violated.
CONTAINMENT_GROUPS = (1, 2, 3)
GROUPS = tuple(sorted({h.group for h in HEADS}))
# project() trims a valve this fraction below the opening that empties the
# reactor in one hold, where the drain responds to the valve again.
DRAIN_TRIM = 1e-3


def stage_mask(state):
    """SERL stage mask from the observation (FermentationEnv.get_action_mask).

    Args:
        state: Tensor [..., 28].

    Returns:
        Tensor [..., 4]: 1 where the dim is live in the reactor's stage.
    """
    st = state[..., I_STAGE:I_STAGE + 5]
    return torch.stack([1.0 - st[..., 4], st[..., 0] + st[..., 1], st[..., 1], st[..., 3]], dim=-1)


class Layout:
    """Column layout of the flattened margins for one reactor count.

    Attributes:
        specs: HeadSpec of every column.
        reactor: Reactor of every column (-1 for a plant-wide head).
        priority: Tensor [H] of column priorities.
        containment: Bool tensor [H], columns re-checked in the final pass.
        influence: Tensor [H, n * 4], the action entries each column may move.
    """

    def __init__(self, heads, n, device):
        specs, reactor, influence = [], [], []
        for h in heads:
            dims = torch.zeros(ACTION_DIM)
            dims[list(h.dims)] = 1.0
            if h.per_reactor:
                for i in range(n):
                    row = torch.zeros(n, ACTION_DIM)
                    row[i] = dims
                    specs.append(h)
                    reactor.append(i)
                    influence.append(row.flatten())
            else:
                specs.append(h)
                reactor.append(-1)
                influence.append(dims.repeat(n))
        self.specs = specs
        self.reactor = reactor
        self.priority = torch.tensor([h.priority for h in specs], device=device)
        self.containment = torch.tensor([h.group in CONTAINMENT_GROUPS for h in specs],
                                        device=device)
        self.influence = torch.stack(influence).to(device)
        self.groups = [h.group for h in specs]

# ─── L1 + L2: the APN ─────────────────────────────────────────────────────────

class MultiHeadAPN(nn.Module):
    """Margins of every constraint head, and the projection built on them.

    Attributes:
        kinetic: KineticNet, or None when the learned heads are off.
        heads: The active HeadSpecs (the learned ones only with a KineticNet).
    """

    def __init__(self, use_kinetic=True, hidden=256):
        super().__init__()
        self.kinetic = KineticNet(hidden) if use_kinetic else None
        self.heads = tuple(h for h in HEADS if h.exact or use_kinetic)
        self._layouts = {}

    def load_kinetic(self, path=KINETIC_PATH, map_location=None):
        """Load trained KineticNet weights; returns False when there are none."""
        if self.kinetic is None:
            return True
        if not os.path.exists(path):
            return False
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        self.kinetic.load_state_dict(ckpt["state_dict"])
        return True

    def layout(self, n, device):
        """The (cached) column layout for n reactors."""
        key = (n, str(device))
        if key not in self._layouts:
            self._layouts[key] = Layout(self.heads, n, device)
        return self._layouts[key]

    # ── Margins ───────────────────────────────────────────────────────────────

    def head_margins(self, state, action, offsets=None, residual=None):
        """Signed margin of every head.

        Args:
            state: Tensor [B, n, 28] (normalised observation).
            action: Tensor [B, n, 4] in [-1, 1].
            offsets: Optional dict group -> L3 offset (fraction of the limit),
                added to that group's static buffer.
            residual: Optional (alpha [B|1, n], beta [B|1, n], beta_s [B|1])
                from AdaptiveResidual: realized feed = (1 + alpha) x commanded,
                and beta x M (per reactor) and beta_s x S_max (storage) more
                mass per hold than the nominal balance.

        Returns:
            Dict head name -> margin, [B, n] for per-reactor heads and [B] for
            plant-wide ones.
        """
        off = offsets or {}
        mask = stage_mask(state)
        u = (action.clamp(-1.0, 1.0) + 1.0) * 0.5 * mask
        mmax = state[..., I_MMAX] * M_NORM
        m_now = state[..., I_M] * mmax
        f_in = (u[..., 1] * state[..., I_FLF] * FLF_NORM
                + u[..., 2] * state[..., I_FC5] * FC5_NORM) * 3600.0 * HOLD_H
        # The plant drains at most down to its discharge floor.
        drain = torch.minimum(u[..., 3] * F_OUT_KGH * HOLD_H,
                              (m_now - DISCHARGE_FLOOR_KG).clamp(min=0.0))
        dm = f_in - drain
        ds = drain.sum(-1) - S_OUT_KGH * HOLD_H
        if residual is not None:
            alpha, beta, beta_s = residual
            dm = dm + alpha * f_in + beta * m_now
            ds = ds + beta_s * S_MAX
        m_next = m_now + dm
        s_next = state[:, 0, I_S] * S_MAX + ds

        out = {
            "overflow": ((1.0 - MASS_BUFFER - off.get(1, 0.0)) * mmax - m_next) / mmax,
            "underflow": (m_next - DRY_FLOOR_KG * (1.0 + UNDERFLOW_BUFFER)) / mmax
                         - off.get(2, 0.0),
            "storage_over": ((1.0 - STORAGE_BUFFER - off.get(3, 0.0)) * S_MAX - s_next) / S_MAX,
            "storage_under": (s_next - S_MIN * (1.0 + STORAGE_BUFFER)) / S_MAX - off.get(4, 0.0),
        }
        # A discharge below the titer floor: the shortfall times how open the
        # valve is, so closing it is the (only) way back to zero.
        titer = state[..., I_ETH] * TITER_NORM
        spec = MIN_HARVEST_TITER * (1.0 + TITER_BUFFER + off.get(7, 0.0))
        out["titer_drain"] = -u[..., 3] * F.relu(spec - titer) / MIN_HARVEST_TITER

        if self.kinetic is not None:
            heat, titer_h = self.kinetic(state[..., :REACTOR_FEATURES], action)
            spec_p = MIN_PRODUCT_TITER * (1.0 + PURITY_BUFFER + off.get(8, 0.0))
            pre_harvest = state[..., I_STAGE:I_STAGE + 3].sum(-1) > 0.5
            out["purity"] = torch.where(pre_harvest, (titer_h - spec_p) / MIN_PRODUCT_TITER,
                                        torch.ones_like(titer_h))
            load = heat.sum(1) / MAX_HEAT_LOAD                       # [B, H]
            peak = HEAT_TAU * torch.logsumexp(load / HEAT_TAU, dim=-1)
            out["heat"] = (1.0 - HEAT_BUFFER - off.get(9, 0.0)) - peak
        return out

    def _flatten(self, heads):
        """head_margins() output as [B, H] in layout() column order."""
        cols = [heads[h.name] if h.per_reactor else heads[h.name].unsqueeze(1)
                for h in self.heads]
        return torch.cat(cols, dim=1)

    def margins(self, state, action, offsets=None, residual=None):
        """Margins flattened to [B, H] in layout() column order."""
        return self._flatten(self.head_margins(state, action, offsets, residual))

    def _margins_graph(self, state, action, offsets=None, residual=None):
        """head_margins() with its autograd graph in the action, for _jacobian().

        Returns:
            A tuple of the heads dict and the action leaf they depend on.
        """
        with torch.enable_grad():
            a = action.detach().requires_grad_(True)
            return self.head_margins(state, a, offsets, residual), a

    def _jacobian(self, heads, a):
        """Jacobian [B, H, n * 4] of _margins_graph() heads in their action leaf.

        Every per-reactor head depends on its own reactor's action only, so
        one backward pass per head gives all of its rows; the plant heads take
        one each.  Consumes the graph.
        """
        B, n, _ = a.shape
        eye = torch.eye(n, device=a.device)
        js = []
        with torch.enable_grad():
            for k, h in enumerate(self.heads):
                g = torch.autograd.grad(heads[h.name].sum(), a,
                                        retain_graph=k < len(self.heads) - 1,
                                        allow_unused=True)[0]
                if g is None:
                    g = torch.zeros_like(a)
                if h.per_reactor:
                    js.append((g.unsqueeze(1) * eye[None, :, :, None]).reshape(B, n, n * ACTION_DIM))
                else:
                    js.append(g.reshape(B, 1, n * ACTION_DIM))
        return torch.cat(js, 1).detach()

    # ── L2: Correction ────────────────────────────────────────────────────────

    @staticmethod
    def _correct(a, m, J, active, priority, eps, damping):
        """One minimum-norm step that brings every active margin up to eps.

        Solves  min ||delta||  s.t.  m_k + J_k delta = eps  (k active), with
        each equation softened by damping / priority: the k x k system is
        solved in closed form, so a lower-priority head yields in a conflict.
        Entries pinned at a bound and pushed further out are frozen.

        Args:
            a: Actions [B, n, 4].
            m: Margins [B, H].
            J: Jacobian [B, H, D] restricted to the movable entries.
            active: Bool [B, H].
            priority: Tensor [H].
            eps: Target margin.
            damping: Base damping.

        Returns:
            Corrected actions [B, n, 4] clamped to [-1, 1].
        """
        B, H, D = J.shape
        flat = a.reshape(B, 1, D)
        pinned = ((flat >= 1.0 - 1e-4) & (J > 0)) | ((flat <= -1.0 + 1e-4) & (J < 0))
        J = J.masked_fill(pinned, 0.0)
        # A head with no lever left cannot be helped by this step.
        act = (active & (J.pow(2).sum(-1) > 1e-10)).to(J.dtype)
        A = J * act.unsqueeze(-1)
        r = (eps - m).clamp(min=0.0) * act
        G = A @ A.transpose(1, 2) + torch.diag_embed(damping / priority + (1.0 - act))
        y = torch.linalg.solve(G, r.unsqueeze(-1))
        delta = (A.transpose(1, 2) @ y).squeeze(-1)
        return (a.reshape(B, D) + delta).clamp(-1.0, 1.0).reshape_as(a)

    @staticmethod
    def _trim_drain(state, a):
        """Close every unloading valve to just below the opening that empties its reactor.

        A reactor that empties within the hold drains the same mass at any
        wider opening, so the storage heads have no gradient in that valve
        and the correction could never close it.  Just below that opening
        the drain follows the valve again: the plant does the same, and the
        valve is a lever once more.

        Args:
            state: Tensor [B, n, 28].
            a: Actions [B, n, 4] in [-1, 1].

        Returns:
            Actions [B, n, 4] with the discharge dim of stage-3 reactors capped.
        """
        m_now = state[..., I_M] * state[..., I_MMAX] * M_NORM
        u_empty = (m_now - DISCHARGE_FLOOR_KG).clamp(min=0.0) / (F_OUT_KGH * HOLD_H)
        a_empty = (2.0 * u_empty * (1.0 - DRAIN_TRIM) - 1.0).clamp(-1.0, 1.0)
        unloading = stage_mask(state)[..., 3] > 0
        a = a.clone()
        a[..., 3] = torch.where(unloading, torch.minimum(a[..., 3], a_empty), a[..., 3])
        return a

    @torch.no_grad()
    def project(self, state, action, offsets=None, residual=None, iters=3,
                containment_iters=2, eps=1e-3, damping=1e-4):
        """L2: move an action inside every head's boundary.

        Args:
            state: Tensor [B, n, 28].
            action: Tensor [B, n, 4] in [-1, 1], stage-masked.
            offsets: L3 offsets (AdaptiveBackoff.offsets()).
            residual: L3 residual (AdaptiveResidual.tensors()).
            iters: Joint correction steps (re-linearized each time).
            containment_iters: Final steps on the containment heads alone.
            eps: Margin each corrected head is brought up to.
            damping: Base damping of the correction step.

        Returns:
            A tuple of the projected action [B, n, 4] and a stats dict with
            'noop' (bool), 'iters' (int) and 'groups' (set of constraint ids
            whose heads had to be corrected).
        """
        B, n, _ = action.shape
        lay = self.layout(n, action.device)
        a = action.detach().clone()
        m = self.margins(state, a, offsets, residual)
        stats = {"noop": True, "iters": 0, "groups": set()}
        if bool((m >= 0.0).all()):
            return a, stats
        stats["noop"] = False
        a = self._trim_drain(state, a)

        # Each pass: linearize only when some head is short, correct, then
        # re-check.  The re-check's forward pass keeps its graph, so the next
        # linearization needs only the backward passes.
        heads, leaf = self._margins_graph(state, a, offsets, residual)
        m = self._flatten(heads).detach()
        move = lay.influence.unsqueeze(0) * stage_mask(state).reshape(B, 1, n * ACTION_DIM)
        cont = lay.containment.unsqueeze(0)
        passes = [(None, damping)] * iters + [(cont, damping * 1e-2)] * containment_iters
        for k, (only, damp) in enumerate(passes):
            active = m < 0.0
            if only is not None:
                active = active & only
            if not bool(active.any()):
                if only is None:
                    continue    # joint passes done: go to the containment check
                break
            J = self._jacobian(heads, leaf)
            stats["iters"] += 1
            stats["groups"].update(lay.groups[c] for c in active.any(0).nonzero().flatten().tolist())
            a = self._correct(a, m, J * move, active, lay.priority, eps, damp)
            heads, leaf = self._margins_graph(state, a, offsets, residual)
            m = self._flatten(heads).detach()
        return a, stats

    # ── Safety Gradient ───────────────────────────────────────────────────────

    def safety_loss(self, state, action, sharpness=50.0, eps=1e-3):
        """Differentiable penalty on every head's shortfall (the actor's safety gradient).

        A smooth hinge, softplus(sharpness * (eps - m)) / sharpness, so safe
        margins contribute almost nothing and each violated head pulls the
        action along its own margin gradient.  Use with frozen APN weights:
        the gradient reaches the actor only through ``action``.

        Args:
            state: Tensor [B, n, 28].
            action: Tensor [B, n, 4], a differentiable function of the actor.

        Returns:
            Scalar loss.
        """
        m = self.margins(state, action)
        return (F.softplus(sharpness * (eps - m)) / sharpness).mean()
