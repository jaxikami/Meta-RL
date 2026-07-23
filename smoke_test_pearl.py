"""Fast CPU smoke test for the uncertainty + PEARL additions.

Runs quickly on CPU and exercises every new/changed code path to catch import,
shape, and wiring errors WITHOUT doing any real training.  It does not check
learning quality — only that the pipeline executes end to end.

Run:
    python smoke_test_pearl.py

A passing run prints "ALL SMOKE CHECKS PASSED".
"""

import numpy as np
import torch

from env_config import (
    sample_task, nominal_task, build_param_vector, PARAM_DIM,
)
from env_safe import FermentationEnv
from agent_pearl import PEARLAgent, MultiTaskReplayBuffer


def _short_env():
    """Build a 3-reactor env with a short horizon for a fast smoke test."""
    env = FermentationEnv(n_reactors=3)
    env.max_steps = 20          # ~4 outer steps at control_action_interval=5
    return env


def check_task_sampling():
    """Task dicts have the right shapes and nominal reduces to the baseline."""
    nt = nominal_task()
    assert nt["params"].shape == (PARAM_DIM,), nt["params"].shape
    assert nt["C_C5"].shape == (15,) and nt["C_LF"].shape == (15,)
    rng = np.random.default_rng(0)
    for _ in range(20):
        t = sample_task(rng)
        assert t["params"].shape == (PARAM_DIM,)
        assert np.all(np.isfinite(t["params"]))
        assert np.all(t["params"][3:9] > 0.0)   # enzyme caps/affinities positive
        assert t["C_C5"].shape == (15,) and t["C_LF"].shape == (15,)
        assert t["context"].shape == (10,)
    # Nominal toxin factors are unity → identical to the un-randomized physics.
    assert np.allclose(nt["params"][9:12], 1.0)
    print("[OK] task sampling / nominal reduction")


def check_env_steps():
    """Env runs under nominal, randomized, and each disturbance scenario."""
    env = _short_env()

    # Nominal
    s = env.reset()
    assert s.shape == (env.n_reactors, env.stateDim), s.shape
    a = np.zeros((env.n_reactors, env.actionDim), dtype=np.float32)
    s, r, done, info = env.step(a)
    assert s.shape == (env.n_reactors, env.stateDim)
    assert np.asarray(r).shape == (env.n_reactors,)

    # Randomized tasks
    env.randomize_tasks = True
    env.seed_tasks(1)
    for _ in range(3):
        env.reset()
        for _ in range(3):
            env.step(a)
    env.randomize_tasks = False

    # Disturbance scenarios
    for kind, mag in [("inlet_step", 0.7), ("enzyme_drop", 0.5),
                      ("mass_drift", -0.02)]:
        env.clear_disturbances()
        env.schedule_disturbance(kind, onset_step=5, magnitude=mag)
        env.set_task(sample_task(np.random.default_rng(2)))
        env.reset()
        done = False
        steps = 0
        while not done and steps < 10:
            _, _, done, _ = env.step(a)
            steps += 1
    env.clear_disturbances()
    env.set_task(None)
    print("[OK] env steps under nominal / randomized / disturbances")


def check_agent(use_context):
    """Agent builds, selects safe actions, buffers transitions, and updates."""
    env = _short_env()
    agent = PEARLAgent(env.stateDim, env.actionDim, n_reactors=env.n_reactors,
                       use_context=use_context, use_safeguard=True)
    replay = MultiTaskReplayBuffer(env.stateDim, env.actionDim, env.n_reactors)

    # Collect several short episodes across two tasks so update() has data.
    # Each episode yields max_steps/control_action_interval outer transitions.
    tasks = [sample_task(np.random.default_rng(k)) for k in range(2)]
    for tid, task in enumerate(tasks):
        for _ in range(12):
            env.set_task(task)
            state = env.reset()
            agent.clear_z()
            done = False
            while not done:
                safe, raw = agent.select_action(state, deterministic=False)
                assert safe.shape == (env.n_reactors, env.actionDim), safe.shape
                assert np.all(safe >= -1.0001) and np.all(safe <= 1.0001)
                nxt, r, done, info = env.step(safe)
                r_sum = float(np.sum(r))
                replay.add(tid, state, safe, r_sum, nxt, done)
                agent.update_context(state, safe, r_sum, nxt)
                state = nxt

    # A handful of gradient updates should run without error.
    # (Each task holds ~12 * (max_steps/control_action_interval) transitions.)
    stats = None
    for _ in range(3):
        stats = agent.update(replay, meta_batch=2, batch_size=16, context_size=8)
    assert stats is not None, "update() returned None despite buffered data"
    for k in ("critic_loss", "actor_loss", "alpha"):
        assert np.isfinite(stats[k]), (k, stats[k])

    # Deterministic (evaluation) action path.
    state = env.reset()
    agent.clear_z()
    safe, _ = agent.select_action(state, deterministic=True)
    assert safe.shape == (env.n_reactors, env.actionDim)

    # Save / load round-trip.
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ckpt.pth")
        agent.save(p)
        agent.load(p)
    tag = "meta (use_context=True)" if use_context else "baseline (use_context=False)"
    print(f"[OK] agent {tag}: select/update/save/load")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    check_task_sampling()
    check_env_steps()
    check_agent(use_context=True)
    check_agent(use_context=False)
    print("\nALL SMOKE CHECKS PASSED")
