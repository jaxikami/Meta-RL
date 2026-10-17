"""Feasibility-gated plateau / divergence early stopping."""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np


class EarlyStopper:
    """Feasibility-gated plateau / divergence stopping (the lean study's).

    Until any check is feasible, progress is a lower violation mean or worst
    episode. From the first feasible check on, progress is a feasible check
    whose plant score beats the best -- the same comparison checkpoint
    selection uses, so every new best checkpoint resets patience. Counters
    are in checks, not iterations.
    """

    def __init__(self, min_checks=12, patience=12, min_rel_delta=1e-3,
                 rel_floor=1.0, max_viol=2.0, max_worst_viol=8.0,
                 diverge_factor=5.0, diverge_viol_floor=10.0,
                 diverge_rel_drop=0.5, diverge_patience=3, diverge_min_checks=6):
        self.min_checks = min_checks
        self.patience = patience
        self.min_rel_delta = min_rel_delta
        self.rel_floor = rel_floor
        self.max_viol = max_viol
        self.max_worst_viol = max_worst_viol
        self.diverge_factor = diverge_factor
        self.diverge_viol_floor = diverge_viol_floor
        self.diverge_rel_drop = diverge_rel_drop
        self.diverge_patience = diverge_patience
        self.diverge_min_checks = diverge_min_checks
        self.feasible_seen = False
        self.best_viol = float("inf")
        self.best_viol_worst = float("inf")
        self.min_viol = float("inf")
        self.best_reward = -float("inf")
        self.best_score = -float("inf")
        self.plateau_count = 0
        self.diverge_count = 0
        self.n_checks = 0

    def _tol(self, ref):
        return self.min_rel_delta * max(abs(ref), self.rel_floor)

    def patience_bar(self, width=10):
        """ASCII gauge of checks without progress, e.g. ``[###-------]3/12``."""
        filled = min(width, round(width * self.plateau_count / self.patience))
        return f"[{'#' * filled}{'-' * (width - filled)}]{self.plateau_count}/{self.patience}"

    def should_stop(self, it, reward, violations, viol_worst, score,
                    extra_ok=True):
        """Record one check and decide whether to stop; returns (stop, reason)."""
        self.n_checks += 1
        self.min_viol = min(self.min_viol, violations)
        feasible = (violations <= self.max_viol
                    and viol_worst <= self.max_worst_viol
                    and bool(extra_ok))
        if feasible and not self.feasible_seen:
            self.feasible_seen = True
        if self.feasible_seen:
            gained = feasible and (
                not np.isfinite(self.best_score)
                or score > self.best_score + self._tol(self.best_score))
            if gained:
                self.best_score = score
            if feasible and (not np.isfinite(self.best_reward)
                             or reward > self.best_reward):
                self.best_reward = reward
        else:
            mean_gained = (not np.isfinite(self.best_viol)
                           or violations < self.best_viol - self._tol(self.best_viol))
            worst_gained = (not np.isfinite(self.best_viol_worst)
                            or viol_worst < self.best_viol_worst
                            - self._tol(self.best_viol_worst))
            if mean_gained:
                self.best_viol = violations
            if worst_gained:
                self.best_viol_worst = viol_worst
            gained = mean_gained or worst_gained
        self.plateau_count = 0 if gained else self.plateau_count + 1

        causes = []
        if self.diverge_factor is not None:
            viol_thresh = max(self.diverge_factor * self.min_viol,
                              self.diverge_viol_floor)
            if violations > viol_thresh:
                causes.append(f"violations {violations:.2f} > {viol_thresh:.2f}")
            if self.feasible_seen and np.isfinite(self.best_reward):
                reward_thresh = self.best_reward - self.diverge_rel_drop * max(
                    abs(self.best_reward), self.rel_floor)
                if reward < reward_thresh:
                    causes.append(f"reward {reward:.2f} < {reward_thresh:.2f}")
        self.diverge_count = 0 if gained else (
            self.diverge_count + 1 if causes else 0)

        if (self.diverge_count >= self.diverge_patience
                and self.n_checks >= self.diverge_min_checks):
            return True, (f"DIVERGING at iter {it}: {', '.join(causes)} "
                          f"for {self.diverge_count} checks")
        if self.n_checks < self.min_checks:
            return False, ""
        if self.plateau_count >= self.patience:
            best = (f"best feasible plant score {self.best_score:.4g}"
                    if self.feasible_seen
                    else f"never feasible, best violations {self.best_viol:.2f}, "
                         f"best worst-episode {self.best_viol_worst:.2f}")
            return True, (f"plateaued at iter {it}: no progress for "
                          f"{self.plateau_count} checks ({best})")
        return False, ""
