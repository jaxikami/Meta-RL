"""L3 of the APN: adaptation to faults the observation does not show.

    AdaptiveBackoff    raises a constraint's buffer on the plant's alarms, and
                       re-locates a capacity limit a fault has moved from the
                       alarm's reported depth
    AdaptiveResidual   recursive-least-squares fit of the mass-balance residual
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import MAX_HEAT_LOAD, S_MAX
from metarl.apn.apn import GROUPS
from metarl.apn.constants import (
    DISCHARGE_FLOOR_KG, F_OUT_KGH, FC5_NORM, FLF_NORM, HEAT_BUFFER, HOLD_H, I_FC5, I_FLF, I_M,
    I_MMAX, I_S, I_STAGE, M_NORM, S_OUT_KGH, STORAGE_BUFFER
)

# ─── L3: Adaptation ───────────────────────────────────────────────────────────

class AdaptiveBackoff:
    """Raises a constraint's buffer when the plant alarms on it (adaptive conformal style).

    For every constraint group g, after each control step:

        delta_g <- clip(delta_g + eta * (alarm_g - target_rate), 0, delta_max_g)

    where alarm_g is 1 if the plant reported a violation event of id g in that
    step.  delta_g is in the margin's units (fraction of the limit) and is
    reset every episode.

    A fault that moves a capacity the observation does not show (a smaller
    storage tank, a weaker cooling loop) would make that rule alarm at its
    target rate for the rest of the episode.  calibrate() instead reads where
    the limit now is from the alarm itself: the plant reports how far past the
    limit it went, so limit = measured level / (1 + depth).  The buffer that
    keeps the head's static margin below the new limit becomes a floor delta_g
    never relaxes below (a capacity once lost is assumed lost for the episode).
    An alarm caused by a model error instead of a moved limit locates the limit
    where it always was, and leaves the floor at 0.
    """

    DELTA_MAX = {1: 0.05, 2: 0.05, 3: 0.15, 4: 0.05, 7: 0.05, 8: 0.05, 9: 0.25}

    def __init__(self, eta=0.02, target_rate=0.05):
        self.eta = eta
        self.target_rate = target_rate
        self.reset()

    def reset(self):
        """Back to the static buffers."""
        self.delta = {g: 0.0 for g in GROUPS}
        self.floor = {g: 0.0 for g in GROUPS}

    def update(self, violation_counts):
        """Advance every group's offset with one control step's alarms.

        Args:
            violation_counts: info["violation_counts"] of the step (id -> events).
        """
        for g in GROUPS:
            alarm = 1.0 if violation_counts.get(g, 0) > 0 else 0.0
            self.delta[g] = float(np.clip(
                self.delta[g] + self.eta * (alarm - self.target_rate), self.floor[g],
                self.DELTA_MAX[g]))

    def calibrate(self, info, s_before, s_after):
        """Re-locate the storage and heat-removal limits from one step's alarms.

        Args:
            info: The step's info dict ('violation_counts', the alarms' depths
                'storage_overflow_violation' and 'heat_load_violation', and the
                measured 'heat_load_peak').
            s_before, s_after: Storage level (fraction of S_MAX) at the start
                and end of the hold.  The peak in between is at least the
                larger one, so the limit read from it is a lower bound.
        """
        counts = info.get("violation_counts", {})
        if counts.get(3, 0) > 0:
            cap = max(s_before, s_after) / (1.0 + info["storage_overflow_violation"])
            self._raise_floor(3, (1.0 - STORAGE_BUFFER) * (1.0 - cap))
        if counts.get(9, 0) > 0:
            cap = info["heat_load_peak"] / (1.0 + info["heat_load_violation"]) / MAX_HEAT_LOAD
            self._raise_floor(9, (1.0 - HEAT_BUFFER) * (1.0 - cap))

    def _raise_floor(self, g, value):
        """Raise group g's floor (and its offset) to ``value``."""
        self.floor[g] = float(np.clip(max(self.floor[g], value), 0.0, self.DELTA_MAX[g]))
        self.delta[g] = max(self.delta[g], self.floor[g])

    def offsets(self):
        """Current offsets, group -> fraction of the limit."""
        return dict(self.delta)


class AdaptiveResidual:
    """Recursive-least-squares identification of the exact heads' mass residual.

    Per reactor, the realized mass change over a hold minus the nominal one is
    regressed on (commanded feed, current mass):

        dM_real - dM_nominal = alpha * feed + beta * M

    so a pump delivering 30% more than commanded shows as alpha = 0.3 and a
    1%/h leak as beta = -0.05 per 5 h hold.  The storage residual (beta_s, in
    fractions of S_max per hold) absorbs a change in the downstream draw.
    Updates use only holds in which the reactor stayed in one stage (so no
    cleanup reset or unload cut-off broke the balance), with exponential
    forgetting so the estimate follows a fault that starts mid-episode.  It is
    linear in the action, so the exact heads stay exact and differentiable.
    """

    def __init__(self, n_reactors, forget=0.97, p0=10.0, alpha_lim=0.5,
                 beta_lim=0.1, beta_s_lim=0.05):
        self.n = n_reactors
        self.forget = forget
        self.p0 = p0
        self.alpha_lim = alpha_lim
        self.beta_lim = beta_lim
        self.beta_s_lim = beta_s_lim
        self.reset()

    def reset(self):
        """Forget everything (nominal plant)."""
        self.theta = np.zeros((self.n, 2))
        self.P = np.tile(np.eye(2) * self.p0, (self.n, 1, 1))
        self.beta_s = 0.0
        self.p_s = self.p0

    def tensors(self, device):
        """(alpha [1, n], beta [1, n], beta_s [1]) for MultiHeadAPN."""
        t = torch.as_tensor(self.theta, dtype=torch.float32, device=device)
        return (t[:, 0].unsqueeze(0), t[:, 1].unsqueeze(0),
                torch.tensor([self.beta_s], dtype=torch.float32, device=device))

    @staticmethod
    def _balance(obs, action):
        """Nominal per-hold feed, drain, mass and Mmax (kg) from one observation."""
        st = obs[:, I_STAGE:I_STAGE + 5]
        mask = np.stack([1.0 - st[:, 4], st[:, 0] + st[:, 1], st[:, 1], st[:, 3]], axis=1)
        u = (np.clip(action, -1.0, 1.0) + 1.0) * 0.5 * mask
        mmax = obs[:, I_MMAX] * M_NORM
        m_now = obs[:, I_M] * mmax
        feed = (u[:, 1] * obs[:, I_FLF] * FLF_NORM + u[:, 2] * obs[:, I_FC5] * FC5_NORM) * 3600.0 * HOLD_H
        drain = np.minimum(u[:, 3] * F_OUT_KGH * HOLD_H, np.maximum(m_now - DISCHARGE_FLOOR_KG, 0.0))
        return feed, drain, m_now, mmax

    def update(self, obs, action, next_obs):
        """Fit the residual to one realized hold.

        Args:
            obs: Observation the action was chosen on, [n, 28].
            action: Action sent to the plant, [n, 4].
            next_obs: Observation after the hold, [n, 28].
        """
        feed, drain, m_now, mmax = self._balance(obs, action)
        m_next = next_obs[:, I_M] * mmax
        stage = obs[:, I_STAGE:I_STAGE + 5].argmax(1)
        stage_next = next_obs[:, I_STAGE:I_STAGE + 5].argmax(1)
        steady = (stage == stage_next) & (stage != 4)
        err = (m_next - m_now - feed + drain) / mmax
        lam = self.forget
        for i in np.nonzero(steady)[0]:
            x = np.array([feed[i] / mmax[i], m_now[i] / mmax[i]])
            P = self.P[i]
            k = P @ x / (lam + x @ P @ x)
            self.theta[i] += k * (err[i] - x @ self.theta[i])
            self.P[i] = (P - np.outer(k, x @ P)) / lam
        self.theta[:, 0] = np.clip(self.theta[:, 0], -self.alpha_lim, self.alpha_lim)
        self.theta[:, 1] = np.clip(self.theta[:, 1], -self.beta_lim, self.beta_lim)

        # Storage: only holds in which every draining reactor kept draining
        # and the tank did not run dry.
        s_now = obs[0, I_S] * S_MAX
        s_next = next_obs[0, I_S] * S_MAX
        draining = drain > 0.0
        if np.all(stage_next[draining] == 3) and s_next > 1e-3 * S_MAX:
            err_s = (s_next - s_now - drain.sum() + S_OUT_KGH * HOLD_H) / S_MAX
            k = self.p_s / (lam + self.p_s)
            self.beta_s = float(np.clip(self.beta_s + k * (err_s - self.beta_s),
                                        -self.beta_s_lim, self.beta_s_lim))
            self.p_s = (self.p_s - k * self.p_s) / lam
