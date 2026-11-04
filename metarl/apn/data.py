"""Training data for the APN's learned heads (apn.KineticNet).

The exact heads (mass balance, titer floor) need no data: they are the plant's
own arithmetic.  The learned heads predict what one reactor does if its action
is held, so their data are (reactor state, action) pairs labelled by holding
the action in the vectorized plant (plant_sim.held_rollout):

    heat   the reactor's CO2 evolution rate at each of the next
           HEAT_HORIZON_H hours;
    titer  its titer when unload begins (states in stages 0-2 only).

States come from random piecewise-constant behaviour trajectories
(plant_sim.behaviour_states), so concentrations are ones the kinetics actually
produce, over the main plant's three reactor types plus some other hardware and
a share of kinetic faults.  Actions mix uniform samples with high/low feed
regimes; locked dims are set to the defaults select_action() leaves them at.
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os
import time

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.apn.constants import HEAT_HORIZON_H
from metarl.apn.plant_sim import behaviour_actions, behaviour_states, label, masked_action

DATA_PATH = os.path.join("policy", "apn_kinetic_data.npz")


def sample_actions(n, rng):
    """Raw actions for labelling: half uniform, half behaviour-like (feed regimes).

    Returns:
        Actions [n, 4] in [-1, 1].
    """
    a = rng.uniform(-1.0, 1.0, (n, 4))
    regime = rng.random(n) < 0.5
    a[regime] = behaviour_actions(int(regime.sum()), rng)
    return a


def make_dataset(obs, params, rng, workers=None, max_hours=400):
    """Label (state, action) pairs for KineticNet.

    Args:
        obs: Per-reactor observations [N, 25].
        params: Kinetic modifiers in force [N, 3].
        rng: NumPy Generator for the actions.
        workers: Labelling processes (default: all cores).
        max_hours: Cap on the wait for unload.

    Returns:
        Dict with 'obs' [N, 25], 'action' [N, 4] (stage-masked), 'heat'
        [N, HEAT_HORIZON_H] (kg/h) and 'titer' [N] (g/L, NaN when the reactor
        was not before unload or never reached it).
    """
    stage = obs[:, 16:21].argmax(1)
    action = masked_action(stage, sample_actions(len(obs), rng))
    heat, titer, _ = label(obs, params, action, HEAT_HORIZON_H, max_hours, workers)
    return {"obs": obs.astype(np.float32), "action": action,
            "heat": heat.astype(np.float32), "titer": titer.astype(np.float32)}


def generate(n_reactors=2048, n_samples=200000, seed=0, workers=None, path=DATA_PATH):
    """Generate and save the KineticNet dataset.

    Args:
        n_reactors: Behaviour trajectories (600 h each, a state every 5 h).
        n_samples: Labelled pairs drawn from their states.
        seed: Seed.
        workers: Processes (default: all cores).
        path: Output .npz.

    Returns:
        The dataset dict.
    """
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    obs, params = behaviour_states(n_reactors, seed=seed, workers=workers)
    print(f"[Data] {len(obs):,} behaviour states in {time.perf_counter() - t0:.0f} s")
    idx = rng.choice(len(obs), size=min(n_samples, len(obs)), replace=False)
    t0 = time.perf_counter()
    data = make_dataset(obs[idx], params[idx], rng, workers)
    pre = ~np.isnan(data["titer"])
    print(f"[Data] {len(idx):,} labelled pairs in {time.perf_counter() - t0:.0f} s "
          f"({pre.mean():.0%} with a harvest titer; "
          f"heat p50/p99 {np.percentile(data['heat'], 50):.0f}/{np.percentile(data['heat'], 99):.0f} kg/h, "
          f"titer p5/p50/p95 {np.nanpercentile(data['titer'], [5, 50, 95]).round(1)} g/L)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **data)
    return data


if __name__ == "__main__":
    generate()
