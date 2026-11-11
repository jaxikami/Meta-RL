"""Five-reactor sub-study: meta-RL under a known mid-episode change.

Five vessels share the three-reactor plant's tank, cooling loop and feedstock
(``metarl.env.config``, ``PLANT_LAYOUT="five"``). At hour 300 of every
episode the tank's downstream draw steps to ``m`` x nominal, with ``m`` drawn
from a tight normal in training and a wide one in evaluation
(``five_reactor.config``). One arm, ``meta5``, trains from scratch.

Importing this package selects the five-reactor layout and this folder as
the output root before anything imports ``metarl``; worker processes
inherit both through the environment.
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os
import sys

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))

if "metarl.env.config" in sys.modules:
    _layout = sys.modules["metarl.env.config"].PLANT_LAYOUT
    if _layout != "five":
        raise ImportError(
            f"metarl was imported with PLANT_LAYOUT={_layout!r} before "
            f"five_reactor; import five_reactor first")
os.environ["PLANT_LAYOUT"] = "five"
os.environ.setdefault("PLANT_OUTPUT_ROOT", STUDY_DIR)
