"""The sub-study's one arm, registered in ``metarl``'s ``ARMS``.

Registered at import, here and in every worker process (which re-imports the
``five_reactor`` entry point when it spawns), so ``get_arm`` resolves it
there too.
"""

# ─── Local ────────────────────────────────────────────────────────────────────
import five_reactor  # noqa: F401  (selects the layout before metarl loads)
from metarl.training.arms import ARMS, ArmSpec
from five_reactor.config import ARM

# meta's switches less the warm start: there is no saferl here, so no anchor,
# L2-SP decay, adaptation gate or no-regression selection either.
ARMS.setdefault(ARM, ArmSpec(
    ARM, ARM.upper(), multitask=True, use_context=True, lagrangian=True,
    action_mask=True, apn=True, apn_soft=False, warm_start=None,
    blurb="meta-RL from scratch on the five-reactor plant: duals + stage "
          "mask + action projection + latent task inference, a known "
          "downstream-draw change at mid-episode"))
SPEC = ARMS[ARM]
