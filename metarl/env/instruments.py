"""The plant's instrument model (``InstrumentMixin``).

What the online gauges, the at-line analyzer and the off-gas meters report to
the policy; reward, constraints and metrics always read the true state.
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import hashlib

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import SENSOR_SEED


class InstrumentMixin:
    """Instrument model of ``PlantFermentationEnv``."""

    def _sensor_seed(self):
        """Derive this episode's sensor seed, honouring ``sensor_stream_key``."""
        if self.sensor_stream_key is None:
            k = int(self._sensor_counter)
            self._sensor_counter += 1
            return [int(SENSOR_SEED), 0, k]
        # blake2b, not hash(): PYTHONHASHSEED randomizes str hashing per
        # process.
        digest = hashlib.blake2b(repr(self.sensor_stream_key).encode("utf-8"),
                                 digest_size=8).digest()
        return [int(SENSOR_SEED), 1, int.from_bytes(digest, "big")]

    def _reset_sensors(self):
        """Re-arm the instrument model for a new episode."""
        self._sensor_rng = np.random.default_rng(self._sensor_seed())
        self._analyzer_pending = []
        self._analyzer_last_draw = 0
        if not self.sensor_model:
            self._analyzer_held = None
            return
        self._analyzer_held = self._analyzer_draw(self.get_state_norm())

    def _analyzer_draw(self, true_obs):
        """Corrupt the analyzer channels of ``true_obs`` with assay error.

        Args:
            true_obs: The exact normalized observation, shape (STATE_DIM,).

        Returns:
            Float32 array of the analyzer channels only, in
            ``self.analyzer_channels`` order.
        """
        vals = true_obs[self.analyzer_channels].astype(np.float64)
        scale = self.analyzer_noise_rsd * np.abs(vals) + self.analyzer_noise_floor
        noisy = vals + self._sensor_rng.normal(0.0, 1.0, size=vals.shape) * scale
        drawn = np.maximum(noisy, 0.0).astype(np.float32)
        drawn[self._lot_assay_pos[self.lot_mass <= 0.0]] = 0.0
        return drawn

    def _reset_lot_assay(self, i):
        """Void reactor ``i``'s lot-titer results still held or in the lab."""
        if self._analyzer_held is None:
            return
        pos = self._lot_assay_pos[i]
        self._analyzer_held[pos] = 0.0
        for _, vals in self._analyzer_pending:
            vals[pos] = 0.0

    def _sensor_observe(self, true_obs):
        """Map the exact state vector to what the plant's instruments report.

        Advances the analyzer clock, so call it exactly once per inner step.

        Args:
            true_obs: The exact normalized observation, shape (STATE_DIM,).

        Returns:
            The measured observation, float32 shape (STATE_DIM,). When
            ``sensor_model`` is False this is ``true_obs`` unchanged and no
            random numbers are drawn.
        """
        if not self.sensor_model:
            return true_obs

        t = int(self.time_step_count)
        if t - self._analyzer_last_draw >= self.analyzer_sample_period:
            self._analyzer_pending.append(
                (t + self.analyzer_report_lag, self._analyzer_draw(true_obs)))
            self._analyzer_last_draw = t
        if self._analyzer_pending:
            keep = []
            for avail_at, vals in self._analyzer_pending:
                if avail_at <= t:
                    self._analyzer_held = vals
                else:
                    keep.append((avail_at, vals))
            self._analyzer_pending = keep

        obs = np.array(true_obs, dtype=np.float32, copy=True)
        if self._analyzer_held is not None:
            obs[self.analyzer_channels] = self._analyzer_held

        vals = obs[self.online_channels].astype(np.float64)
        noisy = vals * (1.0 + self._sensor_rng.normal(
            0.0, self.online_noise_rsd, size=vals.shape))
        obs[self.online_channels] = np.maximum(noisy, 0.0).astype(np.float32)
        if self.observe_co2_rate:
            cer = obs[self.cer_channels].astype(np.float64) * (
                1.0 + self._sensor_rng.normal(0.0, self.cer_noise_rsd,
                                              size=self.n_reactors))
            obs[self.cer_channels] = np.maximum(cer, 0.0).astype(np.float32)
        obs[self.lot_titer_channels[self.lot_mass <= 0.0]] = 0.0
        if self.mask_internal_concentrations:
            obs[self.species_channels] = 0.0
        return obs
