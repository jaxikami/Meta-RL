"""The fault registry: kind codes, aliases and selection presets.

The fault kinds and what each does to the plant are Fermentation-PPO's
(``metarl.env.disturbances``, ported verbatim); this module only names them
for the meta-RL side: a code per kind (the fault-kind classifier's classes),
aliases, and the presets ``parse_active_disturbances`` accepts.
"""

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.disturbances import (VALID_DISTURBANCES, PLANT_DISTURBANCES,
                                     ADDITIVE_DISTURBANCES, FAULT_PATTERNS)

__all__ = ["VALID_DISTURBANCES", "PLANT_DISTURBANCES", "ADDITIVE_DISTURBANCES",
           "FAULT_PATTERNS", "DISTURBANCE_INDEX_MAP", "DISTURBANCE_ALIASES",
           "DISTURBANCE_KIND_INDEX", "N_DISTURBANCE_KINDS",
           "parse_active_disturbances"]

DISTURBANCE_INDEX_MAP = {i + 1: k for i, k in enumerate(VALID_DISTURBANCES)}

DISTURBANCE_ALIASES = {
    "kinetics_decay": "disturb_kinetics_decay",
    "disturb_toxin_spike": "toxin_spike",
    "disturb_cell_viability": "cell_viability",
    "disturb_contamination": "contamination",
    "disturb_downstream_rate": "downstream_rate",
}

# Each kind's code in DISTURBANCE_INDEX_MAP, as fault_kind() reports it (0 is
# "no fault has acted").
DISTURBANCE_KIND_INDEX = {k: i for i, k in DISTURBANCE_INDEX_MAP.items()}
N_DISTURBANCE_KINDS = len(DISTURBANCE_INDEX_MAP)


def parse_active_disturbances(selection):
    """Parse a disturbance selection into a tuple of disturbance kind strings.

    Accepted forms:
        - None or 'all': every kind in VALID_DISTURBANCES.
        - Integers, e.g. [1, 3] -> ("leak", "feed_conc"), 1-based per
          DISTURBANCE_INDEX_MAP; the whole list is read 0-based if it
          contains a 0.
        - Kind names or aliases.
        - Presets: 'flow', 'kinetics', 'biology' / 'biochemical', 'storage',
          'scheduling', 'cooling', 'none'.

    Args:
        selection: Ints, strings, or preset name.

    Returns:
        Tuple of disturbance kind strings, in selection order, deduplicated.

    Raises:
        ValueError: On an unknown index or kind name.
        TypeError: On an element that is neither int nor str.
    """
    if selection is None or selection == "all":
        return VALID_DISTURBANCES
    if selection == "none":
        return ()
    if selection == "flow":
        return ("leak", "feed_gain", "feed_conc")
    if selection == "storage":
        return ("downstream_rate", "storage_capacity")
    if selection == "scheduling":
        return ("turnaround_delay",)
    if selection == "cooling":
        return ("cooling_capacity",)
    if selection in ("kinetics", "disturb_kinetics_decay", "kinetics_decay"):
        return ("disturb_kinetics_decay",)
    if selection in ("biology", "biochemical"):
        return ("disturb_kinetics_decay", "toxin_spike", "cell_viability",
                "contamination")

    if isinstance(selection, (int, str)):
        selection = [selection]

    use_zero_based = any(isinstance(x, int) and x == 0 for x in selection)

    active = []
    for item in selection:
        if isinstance(item, int):
            idx = item if not use_zero_based else item + 1
            if idx in DISTURBANCE_INDEX_MAP:
                k = DISTURBANCE_INDEX_MAP[idx]
                if k not in active:
                    active.append(k)
            else:
                raise ValueError(
                    f"invalid disturbance index {item}; expected 1..{len(DISTURBANCE_INDEX_MAP)}")
        elif isinstance(item, str):
            k = DISTURBANCE_ALIASES.get(item, item)
            if k in VALID_DISTURBANCES:
                if k not in active:
                    active.append(k)
            else:
                raise ValueError(
                    f"unknown disturbance {item!r}; expected one of {VALID_DISTURBANCES}")
        else:
            raise TypeError(f"unsupported disturbance selector type: {type(item)}")
    return tuple(active)
