"""Constants the APN shares: observation layout, plant arithmetic and head buffers."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import (  # noqa: F401  (re-exported for the safety layer)
    C_MAX_LIMITS, CONTROL_ACTION_INTERVAL, DISCHARGE_FLOOR_KG, DRY_FLOOR_KG, FC5_NORM, FLF_NORM,
    M_NORM, MAX_FLOW_OUT_REAC, S_OUT_RATE, SPECIES_IDX
)

# ─── Observation Layout (envs.plant.FermentationPlant.get_state_norm) ─────────────
I_ETH = SPECIES_IDX['Eth']
I_M = 15
I_STAGE = 16            # one-hot over 16:21
I_CREDIT = 21           # stage credits left / the stage's base credits
I_MMAX, I_FLF, I_FC5 = 22, 23, 24
I_S = 25
# Per-reactor features KineticNet reads: concentrations, mass, stage, credit
# and the reactor's hardware limits (everything but the plant-wide fields).
REACTOR_FEATURES = 25
ACTION_DIM = 4

# Denormalisation constants of get_state_norm(): M_NORM, FLF_NORM and
# FC5_NORM (envs.config), and the titer's.
TITER_NORM = float(C_MAX_LIMITS[I_ETH])

# ─── Plant Constants ──────────────────────────────────────────────────────────
# An action is held for CONTROL_ACTION_INTERVAL one-hour inner steps.  The
# plant never drains a vessel below DISCHARGE_FLOOR_KG; DRY_FLOOR_KG is the
# underflow limit (violation id 2).
HOLD_H = float(CONTROL_ACTION_INTERVAL)
F_OUT_KGH = MAX_FLOW_OUT_REAC * 3600.0
S_OUT_KGH = S_OUT_RATE * 3600.0

# Static buffers inside each margin (L3 adds its offsets on top).
MASS_BUFFER = 0.05
STORAGE_BUFFER = 0.05
UNDERFLOW_BUFFER = 0.05
TITER_BUFFER = 0.02
PURITY_BUFFER = 0.01
HEAT_BUFFER = 0.05

# Learned heads: heat-load profile horizon (hours, held action) and the
# output scales KineticNet's softplus outputs are multiplied by.
HEAT_HORIZON_H = 10
HEAT_REF = 500.0       # kg CO2/h per reactor
TITER_REF = 100.0      # g/L
# Temperature of the smooth max over the heat profile, in fractions of the
# cap; it overestimates the peak by at most HEAT_TAU * log(HEAT_HORIZON_H).
HEAT_TAU = 0.005

KINETIC_PATH = os.path.join("policy", "apn_kinetic.pth")
