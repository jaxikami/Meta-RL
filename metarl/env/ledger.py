"""Product ledger, lots and Lagrangian duals (``LedgerMixin``).

Lot release and rejection, the duals' updates, and the terminal accounting
(open lots, the mass-balance guard) of an episode's last step.
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np


class LedgerMixin:
    """Product ledger, lots and duals of ``PlantFermentationEnv``."""

    def _zero_lambdas(self):
        """Set every dual multiplier to zero."""
        n = self.n_reactors
        self.reactor_overflow_lambda = np.zeros(n)
        self.reactor_underflow_lambda = np.zeros(n)
        self.harvest_titer_lambda = np.zeros(n)
        self.product_purity_lambda = np.zeros(n)
        self.storage_overflow_lambda = 0.0
        self.storage_underflow_lambda = 0.0

    # ------------------------------------------------------- bookkeeping

    def _compute_total_tracked_mass(self):
        """Tracked in-system mass: reactors, storage and both feedstock pools.

        Mass leaves the system only by the storage tank's downstream draw, by
        leakage and at each batch's recharge, all through
        ``episode_mass_discharged``.

        Returns:
            Total tracked mass in kg.
        """
        return float(np.sum(self.M) + self.S_mass + self.total_lf + self.total_c5)

    @property
    def mean_product_titer(self):
        """Mass-weighted mean ethanol titer of everything discharged so far."""
        if self.product_mass <= 0.0:
            return 0.0
        return float(self.product_eth / self.product_mass)

    @property
    def lot_mean_titer(self):
        """Mass-weighted mean titer of each reactor's lot in progress (0 if empty)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.lot_mass > 0.0,
                            self.lot_eth / np.maximum(self.lot_mass, 1e-12), 0.0)

    def _close_lot(self, i):
        """Release or reject reactor ``i``'s lot in progress, then open an empty one.

        Returns:
            None when the lot is empty, else a dict with ``passed``,
            ``shortfall`` (relative, >= 0) and ``clawback`` (reward to
            withdraw, >= 0).
        """
        if self.lot_mass[i] <= 0.0:
            return None
        spec = self.min_product_titer
        shortfall = max(0.0, (spec - self.lot_eth[i] / self.lot_mass[i]) / spec)
        passed = shortfall <= 0.0
        clawback = 0.0
        if passed:
            self.on_spec_eth += float(self.lot_eth[i])
            self.reactor_on_spec_eth[i] += float(self.lot_eth[i])
            self.lots_passed += 1
            self.reactor_lots_passed[i] += 1
        else:
            self.lots_failed += 1
            self.reactor_lots_failed[i] += 1
            self.worst_lot_shortfall = max(self.worst_lot_shortfall, shortfall)
            clawback = (1.0 - self.OFF_SPEC_VALUE_FRAC) * float(self.lot_prod_reward[i])
        self.lot_mass[i] = self.lot_eth[i] = self.lot_prod_reward[i] = 0.0
        self._reset_lot_assay(i)
        return {"passed": passed, "shortfall": shortfall, "clawback": clawback}

    def _lot_purity_penalty(self, i, shortfall, severity=None):
        """Purity cost of a rejected lot of reactor ``i`` (None: the plant).

        Args:
            i: Reactor index whose dual prices it, or None for "the plant
                shipped nothing", priced on the mean of the reactors' duals.
            shortfall: Relative shortfall below MIN_PRODUCT_TITER.
            severity: Multiplier on PURITY_PENALTY_COEF; defaults to the
                shortfall.

        Returns:
            The penalty, >= 0.
        """
        sev = shortfall if severity is None else severity
        penalty = self.PURITY_PENALTY_COEF * sev
        lag = max(0.0, shortfall - self.lagrange_target_slack)
        idx = np.arange(self.n_reactors) if i is None else np.array([i])
        penalty += float(np.mean(self.product_purity_lambda[idx])) * lag
        self.product_purity_lambda[idx] = self._update_lambda(
            self.product_purity_lambda[idx], np.full(idx.size, max(lag, 1e-9)))
        return penalty

    @property
    def use_lagrangian(self):
        """Whether adaptive Lagrangian multipliers shape the reward.

        False pins every dual at zero, removing only the adaptive ``lambda x
        violation`` term; the fixed spike, soft walls, purity shaping and
        penalty, and the instability penalty all remain.
        """
        return self._use_lagrangian

    @use_lagrangian.setter
    def use_lagrangian(self, value):
        """Enable or disable Lagrangian shaping, zeroing the duals when off."""
        self._use_lagrangian = bool(value)
        self.lagrange_updates_enabled = bool(value)
        if not value:
            self._zero_lambdas()

    def _update_lambda(self, lam, violation):
        """Advance dual multipliers by one step (element-wise).

        Grows by ``lagrange_lr x violation`` while violated, decays
        exponentially while satisfied, clamped to [0, lagrange_max].

        Args:
            lam: Array of current multipliers.
            violation: Array of slack-adjusted violation magnitudes (>= 0).

        Returns:
            Updated array.
        """
        lam = np.asarray(lam, dtype=np.float64)
        if not self.lagrange_updates_enabled:
            return lam.copy()
        violation = np.asarray(violation, dtype=np.float64)
        out = np.where(violation > 0.0, lam + self.lagrange_lr * violation,
                       lam * (1.0 - self.lagrange_decay))
        return np.clip(out, 0.0, self.lagrange_max)

    def _add_event(self, info_out, cid, depth=None, reactor=None):
        """Add one terminal violation event of ``cid`` to an outer-step info."""
        info_out["violation_count"] = int(info_out["violation_count"]) + 1
        counts = dict(info_out.get("violation_counts", {}))
        counts[cid] = counts.get(cid, 0) + 1
        info_out["violation_counts"] = counts
        info_out["violation_ids"] = sorted(counts)
        info_out["violation_id"] = max(counts)
        info_out["hard_violation"] = info_out["constraint_violated"] = True
        if depth is not None:
            d_sum = dict(info_out.get("violation_depth_sum", {}))
            d_max = dict(info_out.get("violation_depth_max", {}))
            d_sum[cid] = d_sum.get(cid, 0.0) + depth
            d_max[cid] = max(d_max.get(cid, 0.0), depth)
            info_out["violation_depth_sum"] = d_sum
            info_out["violation_depth_max"] = d_max
        if reactor is not None:
            ev = np.array(info_out["reactor_violation_events"])
            ev[reactor] += 1
            info_out["reactor_violation_events"] = ev

    def _finalize_episode(self, total_reward, info_out):
        """Release every open lot, then apply the mass-balance guard.

        Args:
            total_reward: Reward accumulated over the terminating outer step.
            info_out: Outer-step info dict, mutated in place.

        Returns:
            The reward after terminal penalties.
        """
        # ── Constraint 4: every open lot is released or rejected ─────────────
        penalty, clawback = 0.0, 0.0
        for i in range(self.n_reactors):
            closed = self._close_lot(i)
            if closed is None:
                continue
            if closed["passed"]:
                self.product_purity_lambda[i] = self._update_lambda(
                    self.product_purity_lambda[i:i + 1], np.zeros(1))[0]
            else:
                clawback += closed["clawback"]
                penalty += self._lot_purity_penalty(i, closed["shortfall"])
                self._add_event(info_out, 4, depth=closed["shortfall"], reactor=i)
        shipped = self.product_mass > 0.0
        if not shipped:
            # Shipping nothing at all fails the spec, so doing nothing is not
            # "safe": one event for the plant, priced as a full miss.
            penalty += self._lot_purity_penalty(None, 0.0, severity=1.0)
            self._add_event(info_out, 4, depth=1.0)
        purity_ok = bool(shipped and self.lots_failed == 0)

        info_out["mean_titer_final"] = float(self.mean_product_titer)
        info_out["product_mass_final"] = float(self.product_mass)
        info_out["purity_shortfall"] = float(self.worst_lot_shortfall)
        info_out["purity_ok"] = purity_ok
        info_out["lots_passed"] = int(self.lots_passed)
        info_out["lots_failed"] = int(self.lots_failed)
        info_out["reactor_lots_passed"] = self.reactor_lots_passed.copy()
        info_out["reactor_lots_failed"] = self.reactor_lots_failed.copy()
        info_out["lot_mass"] = np.zeros(self.n_reactors)
        info_out["lot_titer"] = np.zeros(self.n_reactors)
        # The headline production number: ethanol in released lots only.
        info_out["on_spec_ethanol"] = float(self.on_spec_eth)
        info_out["reactor_on_spec_ethanol"] = self.reactor_on_spec_eth.copy()
        info_out["clawback"] = float(info_out.get("clawback", 0.0)) + clawback
        info_out["purity_penalty"] = float(penalty)
        total_reward -= penalty + clawback

        # ── Constraint 6: mass-conservation guard ────────────────────────────
        current_total = self._compute_total_tracked_mass()
        expected_total = self.episode_initial_total_mass - self.episode_mass_discharged
        error = current_total - expected_total
        tol = max(self.mass_balance_abs_tol,
                  self.mass_balance_rel_tol * max(self.episode_initial_total_mass, 1.0))
        ok = abs(error) <= tol
        info_out["mass_balance_error"] = float(error)
        info_out["mass_balance_tol"] = float(tol)
        info_out["mass_balance_ok"] = bool(ok)
        if not ok:
            self._add_event(info_out, 6)
            mb_penalty = self.MASS_BALANCE_PENALTY_COEF * abs(error) / max(tol, 1e-6)
            total_reward -= mb_penalty
            info_out["mass_balance_penalty"] = float(mb_penalty)
        return total_reward
