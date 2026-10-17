"""Training logs and JSON output."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import json
import os

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.training.config import LOG_DIR


def _write_log(log, label):
    """Write one arm's training log to LOG_DIR."""
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, f"{label.lower()}_log.json"), "w") as f:
        json.dump(log, f, indent=2, default=_json_default)


def _json_default(o):
    """JSON encoder fallback for NumPy values."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)
