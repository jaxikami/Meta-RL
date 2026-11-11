"""Fast CPU smoke test of the five-reactor sub-study.

Checks, without real training:

  * the layout: five vessels of the three-reactor plant's hardware types, a
    (140,) observation and (20,) action,
    the 26-entry descriptor, the stage mask per row equal to the
    environment's, and the plant-wide hardware unchanged;
  * the change: the training and evaluation samplers have their SDs and
    clips; the step acts at hour 300 exactly, moves the descriptor's
    downstream entry, and the tank's draw becomes m x 1.5 t/h;
  * a few training iterations run end to end (collection, updates, a
    held-out check, selection, checkpoint, log and figure);
  * held-out checks in worker processes equal the in-process ones (so the
    workers resolve meta5);
  * the evaluation's branches equal from-scratch episodes, a frozen branch
    acts on the onset's latent throughout, and the summary and figures build;
  * the calibration recipe runs.

Outputs go to a temporary folder. Run from the repository root:
    python five_reactor/smoke_test.py      # prints "FIVE-REACTOR SMOKE PASSED"
"""

import os
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_TMP = tempfile.mkdtemp(prefix="five_reactor_smoke_")
os.environ["PLANT_OUTPUT_ROOT"] = _TMP
os.environ.setdefault("PLANT_DEVICE", "cpu")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import five_reactor  # noqa: E402,F401
import five_reactor.arms  # noqa: E402,F401  (registers meta5, here and in workers)
from metarl.env.config import (CTX_DOWNSTREAM, MAX_HEAT_LOAD, N_REACTORS,  # noqa: E402
                               S_MAX, S_OUT_RATE, TASK_CONTEXT_DIM,
                               get_main_reactor_config, nominal_task,
                               sample_task)
from metarl.env.layout import (ACTION_DIM, REACTOR_OBS_DIM, STATE_DIM,  # noqa: E402
                               apply_stage_mask_np, stage_mask_np)
from metarl.training.config import LOG_DIR, PLOT_DIR, POLICY_DIR  # noqa: E402
from metarl.training.rollouts import rollout_deterministic  # noqa: E402
from metarl.training.tasks import apply_fault_schedule, make_env  # noqa: E402


def check_layout():
    env = make_env()
    assert N_REACTORS == 5 and STATE_DIM == 140 and ACTION_DIM == 20
    assert TASK_CONTEXT_DIM == 2 + 4 * 5 + 4
    # Every vessel is one of the three-reactor plant's hardware types.
    main = get_main_reactor_config()
    types = set(zip(main['Mmax'], main['max_flow_LF'], main['max_flow_C5']))
    assert all(hw in types for hw in zip(env.Mmax, env.max_flow_LF, env.max_flow_C5))
    assert env.S_max == S_MAX and env.max_heat_load == MAX_HEAT_LOAD
    assert env.S_out_rate == S_OUT_RATE
    env.set_task(sample_task(np.random.default_rng(0)))
    obs = env.reset()
    assert obs.shape == (STATE_DIM,)
    rng = np.random.default_rng(1)
    for _ in range(40):
        m_env = env.get_action_mask(torch.as_tensor(obs)).numpy().reshape(-1)
        assert np.array_equal(m_env, stage_mask_np(obs)), "stage mask differs"
        obs, _, done, _ = env.step(rng.uniform(-1, 1, ACTION_DIM))
    assert obs.reshape(N_REACTORS, REACTOR_OBS_DIM).shape == (5, 28)
    print("  layout: five of the three-reactor plant's vessels, obs (140,), "
          "action (20,), plant-wide hardware unchanged")


def check_change():
    from five_reactor.change import change_onset, change_schedule, sample_magnitude
    from five_reactor.config import (CHANGE_EVAL_CLIP, CHANGE_EVAL_SD,
                                     CHANGE_MEAN, CHANGE_TRAIN_CLIP,
                                     CHANGE_TRAIN_SD)
    rng = np.random.default_rng(0)
    tr = np.array([sample_magnitude(rng, "train") for _ in range(20000)])
    wd = np.array([sample_magnitude(rng, "eval") for _ in range(20000)])
    assert abs(tr.mean() - CHANGE_MEAN) < 0.005 and abs(tr.std() - CHANGE_TRAIN_SD) < 0.005
    assert tr.min() >= CHANGE_TRAIN_CLIP[0] and tr.max() <= CHANGE_TRAIN_CLIP[1]
    assert abs(wd.std() - CHANGE_EVAL_SD) < 0.01
    assert wd.min() >= CHANGE_EVAL_CLIP[0] and wd.max() <= CHANGE_EVAL_CLIP[1]
    assert change_onset(600) == 300

    m = 1.37
    env = make_env()
    apply_fault_schedule(env, change_schedule(m))
    env.set_task(nominal_task())
    obs = env.reset()
    idle = np.tile([np.tanh(-0.34657359), -1.0, -1.0, -1.0], N_REACTORS)
    ctx, drop = {}, {}
    while env.time_step_count < 320:
        if env.time_step_count == 285:
            env.S_mass = 300000.0          # idle runs the tank dry by ~230 h
        t, s0 = env.time_step_count, env.S_mass
        ctx[t] = env.get_effective_context()[CTX_DOWNSTREAM]
        obs, _, _, _ = env.step(apply_stage_mask_np(obs, idle))
        drop[t] = (s0 - env.S_mass) / 5.0
    assert ctx[300] == 1.0 and ctx[305] == m, (ctx[300], ctx[305])
    assert np.isclose(drop[290], 1500.0) and np.isclose(drop[300], 1500.0 * m), (drop[290], drop[300])
    print(f"  change: train SD {tr.std():.3f}, eval SD {wd.std():.3f}; the draw "
          f"steps to {m} x 1.5 t/h at hour 300")


def _tiny_training():
    import five_reactor.train as T
    for k, v in dict(N_TRAIN_TASKS=1, WARMUP_EPISODES_PER_TASK=1,
                     META_COLLECT_EPISODES_PER_ITER=1, UPDATES_PER_ITER=1,
                     MIN_BUFFER=8, BATCH_SIZE=16, META_BATCH=1,
                     HELDOUT_EVAL_TASKS=1, HELDOUT_EVAL_EVERY=2,
                     COLLECT_IN_WORKERS=False).items():
        setattr(T, k, v)
    T.make_corner_tasks = lambda: []
    T._open_workers = lambda spec: None
    agent = T.train(iterations=2)
    assert os.path.exists(os.path.join(POLICY_DIR, "meta5.pth"))
    assert os.path.exists(os.path.join(POLICY_DIR, "meta5_best.pth"))
    assert os.path.exists(os.path.join(LOG_DIR, "meta5_log.json"))
    assert os.path.exists(os.path.join(PLOT_DIR, "training_curves.png"))
    assert not agent.adapt_gate and agent.use_context
    print("  training: 2 iterations, held-out check, checkpoint, log and figure")
    return agent


def check_heldout_pool(agent):
    from metarl.training.workers import RolloutPool
    from five_reactor.arms import SPEC
    from five_reactor.train import heldout_check
    env = make_env()
    tasks = [sample_task(np.random.default_rng(5))]
    a = heldout_check(env, agent, tasks)
    pool = RolloutPool(SPEC, 2)
    try:
        b = heldout_check(env, agent, tasks, pool)
    finally:
        pool.close()
    for k in ("violations", "reward", "on_spec_ethanol", "m_mae"):
        assert np.isclose(a[k], b[k], rtol=1e-6, atol=1e-9), (k, a[k], b[k])
    print(f"  held-out: worker pool equals in-process (score {a['score']:.3e}, "
          f"decode MAE {a['m_mae']:.3f})")


def check_evaluation(agent):
    import five_reactor.evaluate as E
    from five_reactor.change import change_schedule
    E.EVAL_SWEEP = (1.0, 1.6)
    env = make_env()
    task = sample_task(np.random.default_rng(9))
    draw = (1.31, 1.72)
    names = E.task_branches(draw)
    seen = {"fixed": [], "z_onset": None}
    real = agent.select_action

    def spy(state, deterministic=False):
        if agent.z_mode == "fixed":
            seen["fixed"].append(agent.z.detach().clone())
        return real(state, deterministic=deterministic)

    agent.select_action = spy
    try:
        rows = E.task_rollouts(env, agent, task, draw)
    finally:
        agent.select_action = real
    assert len(rows) == len(names)
    for (name, m, frozen), row in zip(names, rows):
        if frozen:
            continue
        ref = rollout_deterministic(env, agent, task, "mean",
                                    None if m == 1.0 else change_schedule(m))
        assert np.isclose(ref["score"], row["score"]) and ref["violations"] == row["violations"], name
    z0 = seen["fixed"][0]
    assert seen["fixed"] and all(np.array_equal(z.numpy(), z0.numpy()) for z in seen["fixed"])
    summary = E.summarize([rows, rows], [draw, draw])
    assert {"unchanged", "train", "wide", "sweep"} <= set(summary)
    assert summary["wide"]["infer"]["score"]["n"] == 2
    os.makedirs(LOG_DIR, exist_ok=True)
    import json
    from metarl.training.logs import _json_default
    with open(os.path.join(LOG_DIR, "eval.json"), "w") as f:
        json.dump({"branches": [list(b) for b in names], "draws": [draw, draw],
                   "rows": [rows, rows]}, f, default=_json_default)
    with open(os.path.join(LOG_DIR, "eval_summary.json"), "w") as f:
        json.dump(summary, f, default=_json_default)
    from five_reactor.plots import plot_eval
    plot_eval()
    for stem in ("change_response", "change_inference"):
        assert os.path.exists(os.path.join(PLOT_DIR, f"{stem}.png")), stem
    print(f"  evaluation: {len(rows)} branches from one prefix equal from-scratch "
          f"episodes; frozen z held for {len(seen['fixed'])} decisions; summary "
          f"and figures built")


def check_calibration_recipe():
    from five_reactor.calibrate import recipe, run_episode
    from five_reactor.change import change_schedule
    r = run_episode(lambda o: recipe(o, 0.75, 0.80, 0.85), None, change_schedule(1.3),
                    after=lambda o: recipe(o, 1.1, 0.80, 0.70))
    assert r["lots_passed"] > 0
    print(f"  calibration recipe: {r['lots_passed']} lots released, "
          f"{r['events']} events")


def main():
    t0 = time.time()
    print("=" * 70 + "\nFIVE-REACTOR SMOKE TEST\n" + "=" * 70)
    check_layout()
    check_change()
    check_calibration_recipe()
    agent = _tiny_training()
    check_heldout_pool(agent)
    check_evaluation(agent)
    print(f"\nFIVE-REACTOR SMOKE PASSED ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
