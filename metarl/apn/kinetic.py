"""KineticNet: the per-reactor network behind the APN's learned heads."""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.apn.constants import (
    ACTION_DIM, HEAT_HORIZON_H, HEAT_REF, I_CREDIT, REACTOR_FEATURES, TITER_REF
)

class KineticNet(nn.Module):
    """Per-reactor surrogate of the kinetics the learned heads need.

    Given one reactor's features and its action, predicts what happens if the
    action is held (re-masked at every 5 h decision, feeds capped at 95% Mmax,
    as the safety layer would):

        heat   its CO2 evolution rate (kg/h) at each of the next
               HEAT_HORIZON_H hours.
        titer  its titer (g/L) when unload begins (meaningful in stages 0-2).

    It sees no other reactor: the plant heat load is the sum of the reactors'
    profiles, so the same weights serve any number of reactors.
    """

    def __init__(self, hidden=256, heat_steps=HEAT_HORIZON_H):
        super().__init__()
        self.heat_steps = heat_steps
        self.trunk = nn.Sequential(
            nn.Linear(REACTOR_FEATURES + ACTION_DIM, hidden), nn.LayerNorm(hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Mish(),
        )
        self.heat_head = nn.Linear(hidden, heat_steps)
        self.titer_head = nn.Linear(hidden, 1)

    def forward(self, features, action):
        """Predict the held-action heat profile and harvest titer.

        Args:
            features: Tensor [..., REACTOR_FEATURES] (state[..., :25]).
            action: Tensor [..., 4] in [-1, 1], stage-masked.

        Returns:
            A tuple of heat [..., heat_steps] (kg/h) and titer [...] (g/L).
        """
        # A turnaround_delay fault stretches the charge and cleanup past their
        # base credits (normalised credits > 1), a range the training data
        # never visits.  Clamp to it: a stage with more than its base time
        # left behaves alike over the heat horizon.
        credit = features[..., I_CREDIT:I_CREDIT + 1].clamp(max=1.0)
        features = torch.cat([features[..., :I_CREDIT], credit, features[..., I_CREDIT + 1:]], -1)
        h = self.trunk(torch.cat([features, action], dim=-1))
        heat = F.softplus(self.heat_head(h)) * HEAT_REF
        titer = F.softplus(self.titer_head(h)).squeeze(-1) * TITER_REF
        return heat, titer
