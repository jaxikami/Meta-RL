"""Fast CPU smoke test for the three-reactor meta-RL pipeline.

Exercises every code path -- shapes, wiring, constraints, storage, faults,
token networks, the GRU posterior, warm start, adaptation gate, shield,
branched and worker rollouts -- without real training, and asserts the design
properties that must not silently regress:

  * the environment IS Fermentation-PPO's: the ported modules equal its
    ``envs/`` sources but for their import lines, and Safe PPO's environment
    steps bit-identically to the original under every fault kind (skipped when ../Fermentation-PPO is absent);
  * the observation layout: each token is its reactor's observation row; the
    NumPy and Torch stage masks agree, and equal the environment's per row;
  * pump gains and leaks are drawn per reactor; the nominal task leaves the
    plant exactly as Fermentation-PPO builds it;
  * a fixed-rate staggered recipe runs the nominal plant cleanly, scores
    positive and outranks doing nothing (which runs the tank dry);
  * mass is conserved exactly under leaks, storage draw and recharges;
  * storage overflow (3) and underflow (4) fire, and the downstream_rate
    fault changes the tank's draw;
  * each batch is its own lot: a fault that rejects one reactor's lots leaves
    the other reactors' released lots counted, and the reward keeps exactly
    the value of the released product;
  * a fault scheduled on one reactor changes only that reactor; plant-wide
    kinds refuse a reactor; the training-only fault flag latches;
  * the per-reactor violation events add up to the environment's counts;
  * every fault kind hurts a scripted policy on at least one axis;
  * the longest episode fits the recurrent encoder's training window;
  * evaluation and held-out magnitudes lie strictly inside the meta-training
    ranges;
  * the online GRU posterior equals the training posterior; a warm-started
    agent acts exactly as its saferl source for every z; behind a shut
    adaptation gate it executes the source, behind an open one its actor;
  * the projection never moves a stage clock or a locked dimension;
  * branched held-out rollouts equal from-scratch ones, and worker-process
    rollouts match in-process ones.

Run from the repository root:
    python tests/smoke_test.py           # prints "ALL SMOKE CHECKS PASSED"
"""

import os
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# A CPU check: the agent's training device too.
os.environ.setdefault("PLANT_DEVICE", "cpu")
# Fermentation-PPO's checkout, for the parity check.
PPO_ROOT = os.environ.get("FERMENTATION_PPO_ROOT",
                          os.path.join(os.path.dirname(_ROOT), "Fermentation-PPO"))

import numpy as np
import torch

from metarl.env.config import (N_REACTORS, nominal_task, sample_task,
                               standardize_context, TASK_CONTEXT_DIM,
                               C_C5_BASE, C_LF_BASE, S_MAX, S_OUT_RATE,
                               S_INITIAL, R_MASS, R_STAGE, CTX_GAIN, CTX_LEAK,
                               CTX_KIN, CTX_DOWNSTREAM, CTX_TURNAROUND,
                               CTX_STORAGE_CAP, CTX_COOLING_CAP,
                               REACTOR_FLOW_SCALE, REACTOR_OBS_DIM,
                               PRODUCT_MASS_SCALE, TITER_SCALE,
                               get_main_reactor_config)
from metarl.env.kinetics import integrate_rk4
from metarl.env.layout import (STATE_DIM, ACTION_DIM, SHIELD_OBS_DIM,
                               STORAGE_CHANNEL, PRIVILEGED_STATE_DIM,
                               PRIV_LOT_MASS, PRIV_LOT_TITER,
                               DEFAULT_MASKED_ACTION, TIME_ACTION_DIMS,
                               stage_mask_np, apply_stage_mask_np,
                               reactor_channel, reactor_action)
from metarl.env.faults import (VALID_DISTURBANCES, DISTURBANCE_KIND_INDEX,
                               PLANT_DISTURBANCES)
from metarl.env.plant import DISCHARGE_CUTOFF, FermentationPlant
from metarl.env.adapter import PlantFermentationEnv
from metarl.agent.common import device, LATENT_DIM
from metarl.agent.networks import token_index
from metarl.agent.replay import MultiTaskReplayBuffer
from metarl.agent.pearl import PEARLAgent
from metarl.apn.network import project_to_safe
from metarl.apn.data import (safety_margin, safety_margins, sample_states,
                             sample_actions, shut_action,
                             assert_plant_constants_match)
from metarl.training.config import (CONTEXT_SIZE, META_AUX_STATE_TARGETS,
                                    META_DISTURB_KINDS,
                                    META_DISTURB_MAG_RANGES,
                                    META_FAULT_PATTERNS, EVAL_DISTURBANCES,
                                    FAULT_PATTERNS, HELDOUT_DISTURBANCES,
                                    META_ANCHOR_FAULT_DIMS, CONSTRAINT_NAMES)
from metarl.training.arms import _build_agent
from metarl.training.early_stopping import EarlyStopper
from metarl.training.tasks import (make_env, make_task_pool, resolve_reactor,
                                   maybe_schedule_disturbance,
                                   pattern_schedule, apply_fault_schedule)
from metarl.training.evaluate import eval_fault_scenarios
from metarl.training.rollouts import (collect_rollout, rollout_deterministic,
                                      _start_episode, _advance,
                                      _episode_metrics, branched_rollouts,
                                      scenario_rollouts,
                                      collect_episode_record,
                                      ingest_episode_record)
from metarl.training.heldout import (_regressions,
                                     blended_heldout_score, violation_price)
from metarl.training.workers import RolloutPool
from metarl.env.config import PROD_REWARD_PER_ETH, VIOLATION_SPIKE
from metarl.apn.train import new_apn

CTX = CONTEXT_SIZE
T1 = -0.34657359          # clock action of exactly 1.0x
# Fault kinds that cost the scripted policies nothing, by design of the
# kinetics (MetaRL_Lean/README: "a row where a tie is expected").
TIE_EXPECTED_KINDS = ("cell_viability",)
PORTED_MODULES = ("kinetics", "disturbances", "plant", "safe")


def _clock(x):
    """Clock multiplier -> action value."""
    return 2.0 * (x - 0.5) / 1.5 - 1.0


def _stage(obs, i):
    return int(np.argmax(obs[reactor_channel(i, R_STAGE):
                             reactor_channel(i, R_STAGE) + 5]))


def _pol_idle(obs):
    """No feed, no discharge, unit clocks."""
    return np.tile([T1, -1.0, -1.0, -1.0], N_REACTORS)


def _pol_deliberate(obs, target=0.80, s_cap=0.85, clocks=(1.0, 0.8, 0.6)):
    """Staggered scheduler: clocks 1.0 / 0.8 / 0.6x through charge and
    co-feed, feed toward a mass target scaled by each pump's size, drain only
    while the tank has room."""
    a = []
    S = obs[STORAGE_CHANNEL]
    for i in range(N_REACTORS):
        m = obs[reactor_channel(i, R_MASS)]
        st = _stage(obs, i)
        hr = np.clip((target - m) / 0.15, 0.0, 1.0)
        frac = hr / REACTOR_FLOW_SCALE[i] if REACTOR_FLOW_SCALE[i] > 1 else hr
        f = 2.0 * frac - 1.0
        out = 1.0 if (st == 3 and S < s_cap) else -1.0
        c = _clock(clocks[i]) if st in (0, 1) else T1
        a += [c, f, f, out]
    return np.array(a)


def _pol_open_loop(obs, clocks=(1.0, 0.8, 0.6), s_cap=0.85):
    """The scheduler's recipe at FIXED feed rates, sized so each vessel
    reaches ~80% of Mmax on the nominal plant: blind to the level, so a pump
    fault reaches the inventory constraints. The plant's clean reference."""
    a = []
    S = obs[STORAGE_CHANNEL]
    for i in range(N_REACTORS):
        st = _stage(obs, i)
        u = min(1.0, 0.784 * clocks[i] / REACTOR_FLOW_SCALE[i])
        out = 1.0 if (st == 3 and S < s_cap) else -1.0
        c = _clock(clocks[i]) if st in (0, 1) else T1
        a += [c, 2.0 * u - 1.0, 2.0 * u - 1.0, out]
    return np.array(a)


def _pol_full_throttle(obs):
    """Every pump and valve fully open, unit clocks (the stage mask applies)."""
    return np.tile([T1, 1.0, 1.0, 1.0], N_REACTORS)


def _run(env, policy, task=None, faults=(), record=None):
    """Roll out a scripted policy through the stage mask.

    Returns:
        Tuple (total_reward, violation_counts, last_info, extras) where extras
        sums prod / clawback, keeps the per-reactor events and any
        ``record(env, info)`` outputs.
    """
    env.clear_disturbances()
    for f in faults:
        env.schedule_disturbance(*f[:3], reactor=f[3] if len(f) > 3 else None)
    env.set_task(task if task is not None else nominal_task())
    obs = env.reset()
    total, counts, done, info = 0.0, {}, False, {}
    extras = {"prod": 0.0, "clawback": 0.0, "steps": 0, "rec": [],
              "reactor_events": np.zeros(N_REACTORS, dtype=np.int64)}
    while not done:
        obs, r, done, info = env.step(apply_stage_mask_np(obs, policy(obs)))
        total += r
        extras["prod"] += info["prod"]
        extras["clawback"] += info["clawback"]
        extras["steps"] += 1
        extras["reactor_events"] += info["reactor_violation_events"]
        for k, v in info["violation_counts"].items():
            counts[k] = counts.get(k, 0) + v
        if record is not None:
            extras["rec"].append(record(env, info))
    return total, counts, info, extras


# ─────────────────────────────────────────────────────────────── environment

def _ppo_modules():
    """Fermentation-PPO's environment modules, or None without its checkout."""
    if not os.path.isdir(os.path.join(PPO_ROOT, "envs")):
        return None
    if PPO_ROOT not in sys.path:
        sys.path.insert(0, PPO_ROOT)
    import importlib
    return {m: importlib.import_module(f"envs.{m}")
            for m in ("config", "safe", "disturbances")}


def check_parity_with_fermentation_ppo():
    """The environment is Fermentation-PPO's, source and behaviour."""
    ppo = _ppo_modules()
    if ppo is None:
        print(f"  parity with Fermentation-PPO SKIPPED (no checkout at {PPO_ROOT})")
        return
    here = os.path.join(_ROOT, "metarl", "env")

    def body(path):
        """Source with either project's module paths neutralised."""
        with open(path, encoding="utf-8") as f:
            text = f.read().replace("\r\n", "\n")
        for pkg in ("metarl.env", "envs"):
            text = text.replace(f"from {pkg}.", "from PKG.")
            text = text.replace(f"{pkg.replace('.', '/')}/", "PKG/")
        return text

    for m in PORTED_MODULES:
        a = body(os.path.join(PPO_ROOT, "envs", f"{m}.py"))
        b = body(os.path.join(here, f"{m}.py"))
        assert a == b, f"metarl/env/{m}.py differs from Fermentation-PPO's envs/{m}.py"
    import metarl.env.config as cfg
    ref = ppo["config"]
    for name in dir(ref):
        if name.startswith("_") or name in ("np",):
            continue
        va, vb = getattr(ref, name), getattr(cfg, name)
        if callable(va):
            va, vb = va(), vb()
        if isinstance(va, dict):
            assert va.keys() == vb.keys() and all(
                np.array_equal(va[k], vb[k]) for k in va), name
        else:
            assert np.array_equal(np.asarray(va), np.asarray(vb)), name
    hw = ref.get_main_reactor_config()
    worst = 0
    for ref_cls, cls in ((ppo["safe"].FermentationEnv, PlantFermentationEnv),):
        for label in ppo["disturbances"].EVAL_DISTURBANCES:
            pattern = ("step", "recurrent")[len(label) % 2]
            ref_env = ref_cls(n_reactors=3, Mmax=hw['Mmax'],
                              max_flow_LF=hw['max_flow_LF'],
                              max_flow_C5=hw['max_flow_C5'])
            env = cls()
            sched = ppo["disturbances"].eval_schedule(label, pattern, index=1)
            ppo["disturbances"].apply_schedule(ref_env, sched)
            apply_fault_schedule(env, sched)
            env.set_task(nominal_task())
            assert np.array_equal(ref_env.reset().reshape(-1), env.reset())
            rng = np.random.default_rng(len(label))
            done = False
            while not done:
                a = rng.uniform(-1.0, 1.0, ACTION_DIM)
                s1, r1, d1, i1 = ref_env.step(a.copy())
                obs, r, done, info = env.step(a.copy())
                assert d1 == done and np.array_equal(s1.reshape(-1), obs), label
                assert np.sum(r1) == r, (label, np.sum(r1), r)
                assert i1["violation_counts"] == info["violation_counts"], label
                assert i1["violation_events"] == info["violation_events"], label
                local = sum(v for k, v in info["violation_counts"].items()
                            if k in (1, 2, 7, 8))
                worst = max(worst, abs(local - int(
                    info["reactor_violation_events"].sum())))
    assert worst <= 1, worst   # 8: one plant-wide event when nothing shipped
    print(f"  parity with Fermentation-PPO OK ({len(PORTED_MODULES)} modules "
          f"equal its sources but for imports; the environment steps "
          f"bit-identically under all {len(ppo['disturbances'].EVAL_DISTURBANCES)}"
          f" evaluation faults; per-reactor events add up)")


def check_layout_and_masks():
    """Observation layout, token index and stage masks."""
    assert STATE_DIM == 84 and ACTION_DIM == 12 and SHIELD_OBS_DIM == STATE_DIM
    assert REACTOR_OBS_DIM == PlantFermentationEnv().get_state_norm().shape[1]
    tok = token_index().numpy()
    assert tok.shape == (N_REACTORS, REACTOR_OBS_DIM)
    for i in range(N_REACTORS):
        assert list(tok[i]) == list(range(i * REACTOR_OBS_DIM,
                                          (i + 1) * REACTOR_OBS_DIM))
    env = PlantFermentationEnv(max_steps=400)
    env.set_task(nominal_task())
    obs = env.reset()
    seen = set()
    done = False
    while not done:
        m_np = stage_mask_np(obs)
        m_t = PlantFermentationEnv.get_action_mask(
            torch.as_tensor(obs, dtype=torch.float32)).numpy()
        m_row = FermentationPlant.get_action_mask(torch.as_tensor(
            env.get_state_norm(), dtype=torch.float32)).numpy().reshape(-1)
        assert np.allclose(m_np, m_t) and np.allclose(m_t, m_row), (m_np, m_t)
        assert np.array_equal(obs, env.get_state_norm().reshape(-1))
        seen.update(env.current_stage.tolist())
        obs, _, done, _ = env.step(apply_stage_mask_np(obs, _pol_deliberate(obs)))
    assert {0, 1, 2, 3} <= seen, seen
    assert np.allclose(DEFAULT_MASKED_ACTION[list(TIME_ACTION_DIMS)], np.tanh(T1))
    print(f"  layout and masks OK (obs {STATE_DIM} = {N_REACTORS} rows of "
          f"{REACTOR_OBS_DIM}, tokens {tok.shape}, stages seen {sorted(seen)})")


def check_task_sampling():
    """Per-reactor gains / leaks, context layout, the nominal plant untouched."""
    nom = nominal_task()
    assert nom["context"].shape == (TASK_CONTEXT_DIM,)
    rng = np.random.default_rng(0)
    gains = []
    for _ in range(400):
        t = sample_task(rng)
        assert t["flow"]["feed_gain"].shape == (N_REACTORS,)
        gains.append(t["flow"]["feed_gain"])
        assert np.all(np.isfinite(t["context"]))
    gains = np.asarray(gains)
    corr = np.corrcoef(gains.T)
    assert np.all(np.abs(corr[np.triu_indices(N_REACTORS, 1)]) < 0.2), corr
    for _ in range(200):
        t = sample_task(rng, ood=True)
        assert np.all(t["flow"]["leak_frac"] >= 0.0)
    z = standardize_context(nom["context"])
    assert z.shape == (TASK_CONTEXT_DIM,) and np.all(np.abs(z) < 5)
    # The nominal task is the plant as built: base feeds, unit gains, no leak.
    env = PlantFermentationEnv()
    env.set_task(sample_task(rng))
    env.reset()
    assert not np.array_equal(env.C_LF, np.tile(C_LF_BASE, (N_REACTORS, 1)))
    env.set_task(nominal_task())
    env.reset()
    assert np.array_equal(env.C_LF, np.tile(C_LF_BASE, (N_REACTORS, 1)))
    assert np.array_equal(env.C_C5, np.tile(C_C5_BASE, (N_REACTORS, 1)))
    assert np.array_equal(env.feed_gain, np.ones(N_REACTORS))
    assert np.array_equal(env._task_leak, np.zeros(N_REACTORS))
    assert np.array_equal(env.get_effective_context(), nom["context"])
    print("  task sampling OK (gains independent per reactor; OOD leaks "
          "non-negative; the nominal task is the plant as built)")


def check_kinetic_rows():
    """A different kinetic row per reactor reaches only that reactor."""
    env = PlantFermentationEnv()
    tiled = np.ascontiguousarray(env.params)
    c = np.full((N_REACTORS, 15), 5.0)
    m = np.full(N_REACTORS, 1e5)
    z3 = np.zeros(N_REACTORS)
    slow = tiled.copy()
    slow[1, 0:3] *= 0.2
    a = integrate_rk4(c, m, z3, z3, np.zeros((N_REACTORS, 15)), tiled, 120.0, 30)[0]
    b = integrate_rk4(c, m, z3, z3, np.zeros((N_REACTORS, 15)), slow, 120.0, 30)[0]
    assert np.array_equal(a[[0, 2]], b[[0, 2]]) and not np.allclose(a[1], b[1])
    print("  kinetic rows OK (per-reactor parameter rows act on their reactor only)")


def check_reference_policy_and_reward_ordering():
    """The fixed-rate recipe is clean and outranks doing nothing."""
    env = PlantFermentationEnv()
    r_good, c_good, info, ex = _run(env, _pol_open_loop)
    assert not c_good, f"the reference policy is not clean: {c_good}"
    assert r_good > 0.0, r_good
    assert info["lots_passed"] >= 4 and info["lots_failed"] == 0, info
    r_idle, c_idle, info_i, _ = _run(env, _pol_idle)
    assert r_good > r_idle, (r_good, r_idle)
    assert c_idle.get(4, 0) > 100, "idling must run the storage tank dry"
    assert c_idle.get(8, 0) == 1, "shipping nothing must fail purity once"
    # Lockstep clocks overflow the shared tank: the storage couples reactors.
    r_lock, c_lock, _, _ = _run(env, lambda o: _pol_deliberate(
        o, s_cap=1.1, clocks=(1.0, 1.0, 1.0)))
    assert c_lock.get(3, 0) > 0, "unstaggered unloads never overflowed the tank"
    print(f"  reference policy OK (clean, {info['lots_passed']} lots released, "
          f"reward {r_good:.2f} > idle {r_idle:.1f}; idle runs the tank dry "
          f"for {c_idle[4]} h; lockstep unloads overflow it for {c_lock[3]} h)")
    return r_good, info


def check_mass_conservation():
    """Mass is conserved exactly under leaks, storage draw and recharges."""
    worst = 0.0
    cases = [(), (("leak", 100, 0.05, 1),), (("leak", 50, 1.0, None),),
             (("leak", 200, 0.25, 2), ("downstream_rate", 100, 1.6)),
             (("downstream_rate", 10, 0.2),)]
    for env in (PlantFermentationEnv(),):
        for faults in cases:
            task = sample_task(np.random.default_rng(len(faults)))
            _, counts, info, _ = _run(env, _pol_full_throttle, task=task,
                                      faults=faults)
            assert info["mass_balance_ok"], (faults, info["mass_balance_error"])
            assert 6 not in counts
            assert np.all(env.M >= env.DRY_FLOOR - 1e-9)
            assert env.S_mass >= 0.0
            worst = max(worst, abs(info["mass_balance_error"]))
    print(f"  mass conservation OK (worst residual {worst:.2e} kg over leaks, "
          f"task leaks, tank draw and recharges)")


def check_storage():
    """The tank drains at S_OUT_RATE, and downstream_rate scales the draw."""
    env = PlantFermentationEnv()
    env.set_task(nominal_task())
    env.clear_disturbances()
    obs = env.reset()
    s0 = env.S_mass
    for _ in range(4):
        obs, _, _, info = env.step(apply_stage_mask_np(obs, _pol_idle(obs)))
    assert np.isclose(s0 - env.S_mass, 20 * S_OUT_RATE * 3600.0), (s0, env.S_mass)
    env.clear_disturbances()
    env.schedule_disturbance("downstream_rate", 0, 0.5)
    obs = env.reset()
    for _ in range(4):
        obs, _, _, _ = env.step(apply_stage_mask_np(obs, _pol_idle(obs)))
    assert np.isclose(S_INITIAL - env.S_mass, 10 * S_OUT_RATE * 3600.0)
    assert env.fault_kind() == DISTURBANCE_KIND_INDEX["downstream_rate"]
    assert np.isclose(env.get_effective_context()[CTX_DOWNSTREAM], 0.5)
    # Overflow at the wall: a full tank and an open valve in unload.
    env.clear_disturbances()
    obs = env.reset()
    env.S_mass = S_MAX - 1000.0
    env.current_stage[0] = 3
    env.M[0] = 150000.0
    env.C[0, 10] = 90.0
    _, _, _, info = env._step_internal(np.tile([T1, -1, -1, 1.0], N_REACTORS))
    assert 3 in info["violation_counts"], info["violation_counts"]
    print("  storage OK (draw at S_OUT_RATE, halved by downstream_rate 0.5; "
          "overflow fires at the wall)")


def check_lots_per_reactor():
    """A fault that rejects one reactor's lots leaves the others' counted."""
    env = PlantFermentationEnv()
    _, counts, info, ex = _run(env, _pol_open_loop,
                               faults=(("disturb_kinetics_decay", 5, 0.15, 2),))
    failed = info["reactor_lots_failed"]
    passed = info["reactor_lots_passed"]
    assert failed[2] >= 1 and failed[0] == 0 and failed[1] == 0, (passed, failed)
    assert info["reactor_on_spec_ethanol"][0] > 0 and info["reactor_on_spec_ethanol"][1] > 0
    assert np.isclose(np.sum(info["reactor_on_spec_ethanol"]), info["on_spec_ethanol"])
    assert counts.get(8, 0) == int(np.sum(failed))
    kept = ex["prod"] - ex["clawback"]
    want = PROD_REWARD_PER_ETH * info["on_spec_ethanol"]
    assert np.isclose(kept, want, rtol=1e-9), (kept, want)
    print(f"  lots per reactor OK (reactor 2 slowed: lots passed {passed.tolist()}"
          f", failed {failed.tolist()}; production reward kept = released "
          f"product exactly)")


def check_fault_targeting():
    """A fault on one reactor changes only that reactor; plant kinds refuse one."""
    env = PlantFermentationEnv(max_steps=30)
    rng = np.random.default_rng(3)
    for kind, mag in (("leak", 0.05), ("feed_gain", 1.4),
                      ("disturb_kinetics_decay", 0.3), ("toxin_spike", 5.0),
                      ("cell_viability", 0.4), ("contamination", 0.004)):
        for reactor in (1, None):
            env.clear_disturbances()
            env.schedule_disturbance(kind, 3, mag, reactor=reactor)
            env.set_task(nominal_task())
            obs = env.reset()
            assert not env.fault_active()
            for _ in range(2):
                obs, _, _, _ = env.step(apply_stage_mask_np(obs, _pol_deliberate(obs)))
            assert env.fault_active() and env.fault_kind() == DISTURBANCE_KIND_INDEX[kind]
            ctx = env.get_effective_context()
            moved = {
                "leak": ctx[CTX_LEAK:CTX_LEAK + 3] > 0,
                "feed_gain": ctx[CTX_GAIN:CTX_GAIN + 3] != 1.0,
                "disturb_kinetics_decay": ctx[CTX_KIN:CTX_KIN + 3] != 1.0,
                "toxin_spike": ctx[CTX_KIN:CTX_KIN + 3] != 1.0,
                "cell_viability": ctx[CTX_KIN:CTX_KIN + 3] != 1.0,
                "contamination": env._current_contam_rate > 0,
            }[kind]
            want = (np.arange(N_REACTORS) == 1) if reactor == 1 else np.ones(N_REACTORS, bool)
            assert np.array_equal(moved, want), (kind, reactor, moved)
    # A pump fault acts on the task's gain.
    task = sample_task(rng)
    env.clear_disturbances()
    env.schedule_disturbance("feed_gain", 3, 1.2, reactor=0)
    env.set_task(task)
    obs = env.reset()
    for _ in range(2):
        obs, _, _, _ = env.step(apply_stage_mask_np(obs, _pol_deliberate(obs)))
    g = task["flow"]["feed_gain"]
    assert np.isclose(env.feed_gain[0], 1.2 * g[0]) and np.allclose(env.feed_gain[1:], g[1:])
    for kind in PLANT_DISTURBANCES:
        try:
            env.schedule_disturbance(kind, 3, 0.5, reactor=0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kind} accepted a reactor")
    try:
        env.schedule_disturbance("mass_drift", 3, 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown kind was accepted")
    # A neutral magnitude never raises the flag.
    env.clear_disturbances()
    env.set_task(nominal_task())
    env.schedule_disturbance("feed_gain", 3, 1.0, reactor=0)
    obs = env.reset()
    for _ in range(3):
        obs, _, _, _ = env.step(apply_stage_mask_np(obs, _pol_deliberate(obs)))
    assert not env.fault_active()
    assert set(VALID_DISTURBANCES) == set(META_DISTURB_KINDS)
    # Meta-training draws: reactors spread over all three and plant-wide.
    reactors, patterns = set(), set()
    for _ in range(200):
        d = maybe_schedule_disturbance(env, prob=1.0, rng=rng,
                                       patterns=META_FAULT_PATTERNS)
        patterns.add(d["pattern"])
        for k, o, _, r, end in d["schedule"]:
            assert (r is None) if k in PLANT_DISTURBANCES else (r is None or 0 <= r < 3)
            assert 1 <= o < env.max_steps and (end is None or end > o)
            reactors.add(r)
    assert reactors == {0, 1, 2, None}, reactors
    assert patterns == set(META_FAULT_PATTERNS), patterns
    env.clear_disturbances()
    print("  fault targeting OK (one reactor or all; a pump fault scales the "
          "task's gain; plant kinds refuse a reactor; flag latches, not on a "
          "neutral magnitude)")


class _DrainProbe(PlantFermentationEnv):
    """Records the titer of every drained reactor at every inner step."""

    def _step_internal(self, action, is_first=True):
        out = super()._step_internal(action, is_first)
        a = out[3]["actual_a_reshaped"]
        drained = (a[:, 3] > 0.0) & (self._pre_M > 50.0)
        self.drained_titers.extend(self._pre_titer[drained].tolist())
        return out

    def reset(self):
        self.drained_titers = []
        return super().reset()


def check_discharge_cutoff_and_batch_boundary():
    """A sub-cutoff command shuts the valve; a recharged vessel is never drained."""
    env = PlantFermentationEnv()
    env.set_task(nominal_task())
    env.clear_disturbances()
    obs = env.reset()
    env.current_stage[0] = 3
    env.M[0] = 100000.0
    act = np.tile([T1, -1, -1, -1.0], N_REACTORS)
    act[3] = 2.0 * (0.5 * DISCHARGE_CUTOFF) - 1.0
    _, _, _, info = env._step_internal(act)
    assert info["actual_a_reshaped"][0, 3] == 0.0, "a sub-cutoff command drained"
    for clock in (0.9, 1.1, 2.0):
        env = _DrainProbe()
        pol = lambda o: np.tile([_clock(clock), 1.0, 1.0, 1.0], N_REACTORS)  # noqa: E731
        _run(env, pol)
        assert env.drained_titers and min(env.drained_titers) > 0.0, (
            f"clock {clock}x drained a recharged vessel")
    print("  discharge cutoff and batch boundary OK")


def check_episode_length_bound():
    """The longest episode fits the encoder's training window."""
    worst = 0
    for clock in (2.0, 1.7, 1.0):
        env = PlantFermentationEnv()
        pol = lambda o: np.tile([_clock(clock), 1.0, 1.0, 1.0], N_REACTORS)  # noqa: E731
        _, _, _, ex = _run(env, pol)
        worst = max(worst, ex["steps"])
    assert worst == env.max_steps // env.control_action_interval, worst
    assert worst < CTX, (worst, CTX)
    print(f"  episode length OK (longest {worst} control steps < window {CTX})")


def check_privileged_state_and_heat_load():
    """Privileged state, exact and reproducible observation, the heat load."""
    env = PlantFermentationEnv()
    env.set_task(sample_task(np.random.default_rng(1)))
    obs = env.reset()
    for _ in range(40):
        obs, _, _, _ = env.step(apply_stage_mask_np(obs, _pol_open_loop(obs)))
    assert np.array_equal(obs, env.get_state_norm().reshape(-1))
    p = env.privileged_state()
    assert p.shape == (PRIVILEGED_STATE_DIM,) and np.all(np.isfinite(p))
    assert np.allclose(p[:STATE_DIM], obs)
    assert np.allclose(p[PRIV_LOT_MASS:PRIV_LOT_MASS + N_REACTORS],
                       env.lot_mass / PRODUCT_MASS_SCALE)
    assert np.allclose(p[PRIV_LOT_TITER:PRIV_LOT_TITER + N_REACTORS],
                       env.lot_mean_titer / TITER_SCALE)
    assert np.any(env.lot_mass > 0.0), "no lot in progress: test is void"
    assert np.isclose(p[-1], env.time_step_count / env.max_steps)

    def trace(faults=()):
        _, _, _, ex = _run(PlantFermentationEnv(), _pol_open_loop, faults=faults,
                           record=lambda e, i: (e.observe(), e.heat_load()))
        return (np.stack([o for o, _ in ex["rec"]]),
                np.stack([h for _, h in ex["rec"]]))

    o1, base = trace()
    o2, _ = trace()
    assert np.array_equal(o1, o2), "the plant is not deterministic"
    # A kinetics fault drops ITS reactor's heat load only.
    _, hit = trace((("disturb_kinetics_decay", 100, 0.5, 0),))
    k = 100 // 5 + 1
    assert np.array_equal(base[:k - 1], hit[:k - 1])
    assert hit[k, 0] < 0.9 * base[k, 0], (hit[k], base[k])
    assert np.allclose(hit[k, 1:], base[k, 1:])
    print("  privileged state and heat load OK (exact, reproducible "
          "observation; lot channels in the privileged state; a kinetics "
          f"fault drops only its reactor's heat load, to {hit[k, 0] / base[k, 0]:.0%})")


def check_fault_coverage(ref_info):
    """Every fault kind hurts a scripted policy on at least one axis.

    As in the lean study: the constraint axis is probed with the open-loop
    recipe (a pump fault only overflows a policy blind to the level) and the
    closed-loop scheduler, the production axis with both.
    """
    base_eth = float(ref_info["on_spec_ethanol"])
    # The closed-loop scheduler is not clean on the nominal plant (its last
    # lot is still short at the episode's end): its violations count above
    # that baseline.
    _, base_c, info_c, _ = _run(PlantFermentationEnv(), _pol_deliberate)
    base_eth_c = float(info_c["on_spec_ethanol"])
    probes = {"leak": (0.05, None), "feed_gain": (1.4, None),
              "feed_conc": (0.3, None), "disturb_kinetics_decay": (0.3, None),
              "toxin_spike": (6.5, None), "cell_viability": (0.3, None),
              "contamination": (0.004, None), "downstream_rate": (0.3, None),
              "turnaround_delay": (5.0, None), "storage_capacity": (0.75, None),
              "cooling_capacity": (0.7, None)}
    assert set(probes) == set(VALID_DISTURBANCES)
    print("    kind                     open-loop viol    closed-loop viol  "
          "on-spec (open / closed)")
    for kind, (mag, reactor) in probes.items():
        fault = ((kind, 150, mag, reactor),)
        _, c_o, info_o, _ = _run(PlantFermentationEnv(), _pol_open_loop,
                                 faults=fault)
        _, c_c, info, _ = _run(PlantFermentationEnv(), _pol_deliberate,
                               faults=fault)
        loss_o = 1.0 - float(info_o["on_spec_ethanol"]) / base_eth
        loss_c = 1.0 - float(info["on_spec_ethanol"]) / base_eth_c
        print(f"    {kind:<24} {str(c_o):<17} {str(c_c):<17} "
              f"{-loss_o:+.0%} / {-loss_c:+.0%}")
        if kind in TIE_EXPECTED_KINDS:
            # The lean study's finding: the yeast regrows within the batch, so
            # a kill costs nothing at any magnitude. Kept as a tie row.
            continue
        worse_c = any(v > base_c.get(k, 0) for k, v in c_c.items())
        assert c_o or worse_c or max(loss_o, loss_c) > 0.10, (
            f"{kind}@{mag} neither violated anything nor cost >10% of "
            f"on-spec production")
    print("  fault coverage OK (every kind hurts on at least one axis)")


def check_magnitudes_in_support():
    """Evaluation, held-out and latent-gate magnitudes lie inside training ranges."""
    for d in EVAL_DISTURBANCES.values():
        lo, hi = META_DISTURB_MAG_RANGES[d["kind"]]
        assert lo < d["magnitude"] < hi, d
    for kind, mag, _ in HELDOUT_DISTURBANCES:
        lo, hi = META_DISTURB_MAG_RANGES[kind]
        assert lo < mag < hi, (kind, mag)
    print("  magnitudes in support OK")


def _apply_until(env, t_end):
    """Fire scheduled faults step by step up to ``t_end``, without biology."""
    for t in range(t_end):
        env.time_step_count = t
        env._apply_disturbances()


def check_fault_patterns():
    """Every kind has four patterns: step, spike, gradual and recurrent."""
    env = PlantFermentationEnv()
    env.set_task(nominal_task())
    grid = eval_fault_scenarios(env.max_steps // 2, env.max_steps)
    assert set(grid) == {f"{k}_{p}" for k in EVAL_DISTURBANCES
                         for p in FAULT_PATTERNS}, sorted(grid)
    for name, sched in grid.items():
        assert sched, name
        apply_fault_schedule(env, sched, index=1)
    # Evaluation schedules are Fermentation-PPO's.
    ppo = _ppo_modules()
    if ppo is not None:
        for label in EVAL_DISTURBANCES:
            for pattern in FAULT_PATTERNS:
                for index in range(N_REACTORS):
                    want = ppo["disturbances"].eval_schedule(label, pattern, index)
                    got = [dict(d, reactor=resolve_reactor(d["kind"], d["reactor"], index))
                           for d in grid[f"{label}_{pattern}"]]
                    assert got == want, (label, pattern, index)
    # spike: on, then the task's value again (multiplicative, and additive),
    # and the privileged fault flag with it.
    for kind, mag, probe in (("feed_gain", 1.3, lambda e: e.feed_gain[0]),
                             ("leak", 0.01, lambda e: e._current_leak_rate[0])):
        env.reset()
        apply_fault_schedule(env, pattern_schedule(kind, mag, "spike", 10, 0,
                                                   spike_h=5))
        base = probe(env)
        _apply_until(env, 12)
        on = probe(env)
        assert env.fault_active() and env.fault_kind() == DISTURBANCE_KIND_INDEX[kind]
        _apply_until(env, 20)
        assert on != base and np.isclose(probe(env), base), (kind, base, on)
        assert not env.fault_active() and env.fault_kind() == 0, kind
    # gradual ends at full strength; a ramped kill multiplies to one kill.
    env.reset()
    apply_fault_schedule(env, pattern_schedule("feed_gain", 1.3, "gradual", 10,
                                               0, gradual_h=50))
    _apply_until(env, 30)
    mid = env.feed_gain[0]
    _apply_until(env, 70)
    assert 1.0 < mid < 1.3 and np.isclose(env.feed_gain[0], 1.3), mid
    env.reset()
    c0 = env.C[1, 9]
    apply_fault_schedule(env, pattern_schedule("cell_viability", 0.5,
                                               "gradual", 10, 1, gradual_h=50))
    _apply_until(env, 70)
    assert np.isclose(env.C[1, 9], c0 * 0.5), (c0, env.C[1, 9])
    # toxin: a ramp's acute pulses add up to one step's; recurrence pulses
    # anew (a magnitude whose two pulses stay below the broth cap).
    pulses = {}
    for pattern in ("step", "gradual", "recurrent"):
        env.reset()
        f0 = env.C[2, 6]
        apply_fault_schedule(env, pattern_schedule(
            "toxin_spike", 2.0, pattern, 10, 2, gradual_h=50,
            windows=((10, 30), (40, 60))))
        _apply_until(env, 70)
        pulses[pattern] = env.C[2, 6] - f0
    assert np.isclose(pulses["gradual"], pulses["step"]), pulses
    assert np.isclose(pulses["recurrent"], 2 * pulses["step"]), pulses
    # recurrent: off between windows, on again after.
    env.reset()
    apply_fault_schedule(env, pattern_schedule("downstream_rate", 0.5,
                                               "recurrent", 0, None,
                                               windows=((10, 20), (30, 40))))
    seen = []
    for t in range(45):
        env.time_step_count = t
        env._apply_disturbances()
        seen.append(env._downstream_mult)
    assert seen[15] == 0.5 and seen[25] == 1.0 and seen[35] == 0.5, seen
    env.clear_disturbances()
    print(f"  fault patterns OK ({len(grid)} evaluation scenarios = "
          f"{len(EVAL_DISTURBANCES)} kinds x {', '.join(FAULT_PATTERNS)}"
          + (", Fermentation-PPO's schedules" if ppo is not None else "")
          + "; spikes clear, ramps reach full strength, toxin pulses telescope)")


def check_plant_wide_faults():
    """turnaround_delay, storage_capacity and cooling_capacity act on the
    stage times, the storage wall and the heat-load cap (9)."""
    env = PlantFermentationEnv()
    env.set_task(nominal_task())
    # turnaround_delay: a cleanup that starts after onset takes 3x as long,
    # and so does the charge that follows it.
    env.clear_disturbances()
    env.schedule_disturbance("turnaround_delay", 0, 3.0)
    env.reset()
    env.current_stage[0] = 3
    env.M[0] = 50.0
    env.stage_credits[0] = 0.5
    env._step_internal(np.tile([T1, -1, -1, -1.0], N_REACTORS))
    assert env.current_stage[0] == 4
    assert np.isclose(env.stage_credits[0], 3.0 * env.base_credits[4]), env.stage_credits[0]
    env.stage_credits[0] = 0.5
    env._step_internal(np.tile([T1, -1, -1, -1.0], N_REACTORS))
    assert env.current_stage[0] == 0
    assert np.isclose(env.stage_credits[0], 3.0 * env.base_credits[0]), env.stage_credits[0]
    assert np.isclose(env.get_effective_context()[CTX_TURNAROUND], 3.0)
    assert env.n_batches_completed == 1 and env.reactor_batches_completed[0] == 1
    # storage_capacity: the overflow wall moves to 90% of S_MAX.
    for mag, want in ((1.0, False), (0.9, True)):
        env.clear_disturbances()
        env.schedule_disturbance("storage_capacity", 0, mag)
        env.reset()
        env.S_mass = 0.95 * S_MAX
        _, _, _, info = env._step_internal(np.tile([T1, -1, -1, -1.0], N_REACTORS))
        assert (3 in info["violation_counts"]) == want, (mag, info["violation_counts"])
    assert np.isclose(env.get_effective_context()[CTX_STORAGE_CAP], 0.9)
    # cooling_capacity: the heat-load cap moves, nothing observed does.
    loads = {}
    for mag in (1.0, 0.05):
        env.clear_disturbances()
        env.schedule_disturbance("cooling_capacity", 20, mag)
        obs = env.reset()
        counts = {}
        for _ in range(8):
            obs, _, _, info = env.step(apply_stage_mask_np(obs, _pol_open_loop(obs)))
            for k, v in info["violation_counts"].items():
                counts[k] = counts.get(k, 0) + v
        loads[mag] = (counts.get(9, 0), obs.copy(), env.disturbance_probe())
    assert loads[1.0][0] == 0 and loads[0.05][0] > 0, loads
    assert np.array_equal(loads[1.0][1], loads[0.05][1])
    assert np.isclose(loads[0.05][2]["cooling_capacity"], 0.05 * env.max_heat_load)
    assert np.isclose(env.get_effective_context()[CTX_COOLING_CAP], 0.05)
    env.clear_disturbances()
    print(f"  plant-wide faults OK (turnaround_delay stretches cleanup and "
          f"charge; storage_capacity moves the overflow wall; cooling_capacity "
          f"moves the heat-load cap: {loads[0.05][0]} heat-load events, same "
          f"observation)")


def check_constraint_ids_and_lagrangian():
    """The constraint names are the environment's ids; the Lagrangian switch."""
    assert set(CONSTRAINT_NAMES) == set(range(1, 10))
    assert CONSTRAINT_NAMES[3] == "storage_overflow" and CONSTRAINT_NAMES[9] == "heat_load"
    env = make_env(lagrangian=False)
    assert isinstance(env, PlantFermentationEnv) and env.fixed_violation_penalties
    assert not env.lagrange_updates_enabled and not env.use_lagrangian
    _, counts, _, _ = _run(env, _pol_full_throttle)
    assert counts and env.reactor_overflow_lambda.max() == 0.0
    env = make_env(lagrangian=True)
    assert env.use_lagrangian and env.reset_lambdas_per_episode
    _, _, _, _ = _run(env, _pol_full_throttle)
    assert env.reactor_overflow_lambda.max() > 0.0
    # The reward rates the selection prices a violation with are the
    # environment's (locals of its step, so read from the source).
    with open(os.path.join(_ROOT, "metarl", "env", "safe.py"), encoding="utf-8") as f:
        src = f.read()
    assert f"BASE_SPIKE = {VIOLATION_SPIKE} if self.fixed_violation_penalties" in src
    assert "reward_coef = 0.5" in src and "(total_harvested_eth/1e7)" in src
    assert np.isclose(PROD_REWARD_PER_ETH, 0.5 / 1e7)
    print("  constraint ids and Lagrangian switch OK (ids 1-9 are the "
          "environment's; lagrangian=False is the fixed-penalty case study; "
          "selection prices a violation "
          f"at {violation_price():.1e} ethanol, the environment's rate)")


# ───────────────────────────────────────────────────────────────── agent

def _ctx_agent(**kw):
    """A context agent configured as the meta arm (gate optional)."""
    return PEARLAgent(STATE_DIM, ACTION_DIM, aux_mode="task",
                      critic_task_context=True,
                      critic_privileged_dim=PRIVILEGED_STATE_DIM,
                      aux_state_targets=META_AUX_STATE_TARGETS, **kw)


def check_agent_shapes_and_modes():
    """Action shapes, stage mask, z modes, token mixing and update paths."""
    env = PlantFermentationEnv(max_steps=40)
    agent = _ctx_agent()
    obs = env.reset()
    a, raw = agent.select_action(obs, deterministic=True)
    assert a.shape == (ACTION_DIM,) and raw.shape == (ACTION_DIM,)
    m = stage_mask_np(obs)
    assert np.allclose(a[m == 0], DEFAULT_MASKED_ACTION[m == 0], atol=1e-6)
    for mode in ("zero", "mean", "sample", "fixed", "hold"):
        agent.z_mode = mode
        agent.clear_z()
        agent.update_context(obs, a, 0.1, obs)
        assert agent.z.shape == (LATENT_DIM,)
        if mode == "zero":
            assert torch.all(agent.z == 0)
    agent.z_mode = "mean"
    # At init the cross-reactor attention is the identity, so a reactor's
    # action reads its own history only; once it has weight, the others'.
    x = torch.rand(4, STATE_DIM, device=device)
    y = x.clone()
    y[:, reactor_channel(2, R_MASS)] += 0.5
    zz = torch.zeros(4, LATENT_DIM, device=device)
    act = lambda o: agent.actor(agent.actor.single_step_features(o), zz)[0]  # noqa: E731
    with torch.no_grad():
        d0 = (act(x) - act(y)).abs()
        assert d0[:, :8].max() < 1e-6 and d0[:, 8:].max() > 0
        agent.actor.history.attn.out_proj.weight.normal_(0.0, 0.2)
        d1 = (act(x) - act(y)).abs()
        assert d1[:, :8].max() > 1e-6, "the attention does not couple reactors"
        # The policy is recurrent: the same reading after another history
        # is another action.
        h = agent.actor.initial_state(4, device)
        f_a, _ = agent.actor.history_step(x, agent.actor.history_step(y, h)[1])
        f_b, _ = agent.actor.history_step(x, agent.actor.history_step(x, h)[1])
        assert (agent.actor(f_a, zz)[0] - agent.actor(f_b, zz)[0]).abs().max() > 1e-6
    print("  agent shapes OK (12 actions, mask applied, five z modes; "
          "reactors couple through attention over their histories; the "
          "policy reads its history)")


def check_history_windows():
    """Training-window history features equal the online ones, across episodes."""
    torch.manual_seed(1)
    env = PlantFermentationEnv(max_steps=60)
    agent = PEARLAgent(STATE_DIM, ACTION_DIM, use_context=False)
    with torch.no_grad():
        agent.actor.history.attn.out_proj.weight.normal_(0.0, 0.2)
    agent._mark_rollout_stale()
    replay = MultiTaskReplayBuffer(STATE_DIM, ACTION_DIM)
    online, online_next = [], []
    for _ in range(3):
        obs = env.reset()
        agent.clear_z()
        h = agent.actor.initial_state(1, device)
        done = False
        while not done:
            a, _ = agent.select_action(obs, deterministic=False)
            nxt, r, done, _ = env.step(a)
            replay.add(0, obs, a, r, nxt, done)
            with torch.no_grad():
                f, h = agent.actor.history_step(
                    torch.as_tensor(obs, dtype=torch.float32,
                                    device=device).unsqueeze(0), h)
                f2, _ = agent.actor.history_step(
                    torch.as_tensor(nxt, dtype=torch.float32,
                                    device=device).unsqueeze(0), h)
            online.append(f[0].cpu())
            online_next.append(f2[0].cpu())
            obs = nxt
    n = replay.size(0)
    L = 8
    starts = np.arange(0, n - L + 1, 3)
    batch, rows, lo, hi = replay.sample_windows(0, None, 20, chunk_len=L,
                                                starts=starts)
    o, o2 = batch[0].float(), batch[3].float()
    with torch.no_grad():
        f, f2 = agent._history_features(rows, lo, hi, o, o2)
    idx = (starts[:, None] + np.arange(L)[None, :]).reshape(-1)
    want = torch.stack([online[i] for i in idx])
    want2 = torch.stack([online_next[i] for i in idx])
    assert torch.allclose(f.cpu(), want, atol=1e-4), "history window != online"
    assert torch.allclose(f2.cpu(), want2, atol=1e-4), "next-state history != online"
    print("  history windows OK (training features = online GRU + attention "
          "at every step, next state included, across episode boundaries)")


def check_recurrent_encoder():
    """The training posterior equals the online GRU state across episodes."""
    env = PlantFermentationEnv(max_steps=60)
    window = 20
    agent = PEARLAgent(STATE_DIM, ACTION_DIM)
    agent.z_mode = "mean"
    replay = MultiTaskReplayBuffer(STATE_DIM, ACTION_DIM)
    online = []
    for _ in range(3):
        obs = env.reset()
        agent.clear_z()
        done = False
        while not done:
            a = np.random.uniform(-1, 1, ACTION_DIM)
            online.append(agent.z_mu.clone())
            nobs, r, done, _ = env.step(a)
            replay.add(0, obs, a, r, nobs, done)
            agent.update_context(obs, a, r, nobs)
            obs = nobs
    n = replay.size(0)
    steps = replay._tasks[0].episode_steps(np.arange(n))
    assert steps.max() + 1 < window
    _, rows, lo, hi = replay.sample_windows(0, n, window, chunk_len=n, starts=[0])
    with torch.no_grad():
        mu, sig = agent._window_posterior(rows, lo, hi)
    want = torch.stack(online).to(mu.device)
    assert torch.allclose(mu, want, atol=1e-4), (mu - want).abs().max()
    starts = torch.as_tensor(np.flatnonzero(steps == 0), device=mu.device)
    assert torch.all(mu[starts] == 0) and torch.all(sig[starts] == 1)
    assert not torch.allclose(mu, torch.zeros_like(mu)), "the GRU is inert"
    _, rows_s, lo_s, hi_s = replay.sample_windows(0, n, 4, chunk_len=n, starts=[0])
    try:
        agent._window_posterior(rows_s, lo_s, hi_s)
    except ValueError:
        pass
    else:
        raise AssertionError("a window shorter than an episode was accepted")
    # Meta-arm training: privileged replay, task-aware critics, titer targets.
    tagent = _ctx_agent()
    assert tagent.aux.net[-1].out_features == TASK_CONTEXT_DIM + N_REACTORS
    preplay = MultiTaskReplayBuffer(STATE_DIM, ACTION_DIM,
                                    priv_dim=PRIVILEGED_STATE_DIM)
    rng = np.random.default_rng(4)
    for _ in range(3):
        collect_rollout(env, tagent, preplay, 0, sample_task(rng),
                          disturb=True, rng=rng, patterns=META_FAULT_PATTERNS)
    stats = tagent.update(preplay, meta_batch=1, batch_size=16, context_size=window)
    for k in ("critic_loss", "actor_loss", "kl", "aux_loss"):
        assert np.isfinite(stats[k]), (k, stats[k])
    # A context-free arm (saferl) stores no privileged state and still updates.
    sagent = PEARLAgent(STATE_DIM, ACTION_DIM, use_context=False)
    sreplay = MultiTaskReplayBuffer(STATE_DIM, ACTION_DIM)
    collect_rollout(env, sagent, sreplay, 0, nominal_task())
    sstats = sagent.update(sreplay, meta_batch=1, batch_size=8, min_buffer=8)
    assert np.isfinite(sstats["critic_loss"]), sstats
    print("  recurrent encoder OK (training posterior = online GRU state; "
          "short windows refused; meta-arm update finite)")


def _saved_source(d):
    """A 'trained' saferl-like source checkpoint with nonzero FiLM weights.

    Its critics read the task and privileged state like every arm's, so the
    warm start copies their FiLM weights except the latent's columns.
    """
    src = PEARLAgent(STATE_DIM, ACTION_DIM, use_context=False,
                     critic_task_context=True,
                     critic_privileged_dim=PRIVILEGED_STATE_DIM)
    src.observe_obs(np.random.default_rng(0).random((64, STATE_DIM)))
    with torch.no_grad():
        for net in (src.actor, src.critic, src.critic_target):
            for name, prm in net.named_parameters():
                if ".films." in f".{name}":
                    prm.normal_(0.0, 0.1)
            for name, prm in net.named_parameters():
                if name.endswith("out.weight") or name.endswith("ff2.weight"):
                    prm.normal_(0.0, 0.05)
    path = os.path.join(d, "src.pth")
    src.save(path)
    return src, path


def check_warm_start_and_gate():
    """Warm start acts as the source for every z; the gate blends base / actor."""
    torch.manual_seed(3)
    env = PlantFermentationEnv(max_steps=60)
    with tempfile.TemporaryDirectory() as d:
        src, path = _saved_source(d)
        agent = _ctx_agent(adapt_gate=True)
        agent.warm_start_from(path, anchor_weight=1.0, decay_to_source=True,
                              anchor_fault_dims=META_ANCHOR_FAULT_DIMS)
        agent.freeze_obs_norm()
        assert agent.anchor_actor is agent.base_actor
        obs = torch.rand(16, STATE_DIM, device=device)
        act = torch.rand(16, ACTION_DIM, device=device) * 2 - 1
        z0 = torch.zeros(16, src._z_dim, device=device)
        priv = torch.rand(16, PRIVILEGED_STATE_DIM, device=device)
        ctx = torch.randn(16, TASK_CONTEXT_DIM, device=device)
        with torch.no_grad():
            f_src = src.actor.single_step_features(obs)
            f = agent.actor.single_step_features(obs)
            assert torch.allclose(f, f_src, atol=1e-6), "warm history differs"
            assert agent.history_frozen and not any(
                p.requires_grad for p in agent.actor.history.parameters())
            m_src, _ = src.actor(f_src, z0)
            q_src = src.critic(f_src, act, src._critic_cond(z0, ctx, priv))
            for _ in range(3):
                z = torch.randn(16, agent._z_dim, device=device) * 2
                m, _ = agent.actor(f, z)
                assert torch.allclose(m, m_src, atol=1e-5), "warm actor reads z"
                cond = agent._critic_cond(z, ctx, priv)
                for q, q_ref in zip(agent.critic(f, act, cond), q_src):
                    assert torch.allclose(q, q_ref, atol=1e-4), "warm critic differs"
            g = agent._gate_open(agent.gate, torch.randn(64, agent._z_dim) * 3)
            assert float(g.max()) < 1e-3, "the gate starts open"
        # An actor that reads z; shut gate -> source, open -> actor.
        with torch.no_grad():
            for f in agent.actor.trunk.films:
                f.proj.weight.normal_(0.0, 0.3)
        agent._mark_rollout_stale()
        o = env.reset()
        od = torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
        zf = np.full(agent._z_dim, 2.0, dtype=np.float32)
        agent.z_mode = "fixed"
        agent.set_fixed_z(zf)
        agent.clear_z()
        with torch.no_grad():
            fd = src.actor.single_step_features(od)
            base_a = torch.tanh(src.actor(
                fd, torch.zeros(1, src._z_dim))[0]).numpy()[0]
            actor_a = torch.tanh(agent.actor(
                fd, torch.as_tensor(zf).unsqueeze(0))[0]).numpy()[0]
        assert not np.allclose(base_a, actor_a, atol=1e-3)
        raw = {}
        snap = agent.posterior_state()
        for opening in (0.0, 0.25, 1.0):
            agent.set_posterior_state(snap)       # each from a fresh history
            agent.gate_override = opening
            raw[opening] = agent.select_action(o, deterministic=True)[1]
        agent.gate_override = None
        assert np.allclose(raw[0.0], base_a, atol=1e-5)
        assert np.allclose(raw[1.0], actor_a, atol=1e-5)
        assert np.allclose(raw[0.25], base_a + 0.25 * (actor_a - base_a), atol=1e-5)
        # Held actor, then the anchored and gated update.
        replay = MultiTaskReplayBuffer(STATE_DIM, ACTION_DIM,
                                       priv_dim=PRIVILEGED_STATE_DIM)
        rng = np.random.default_rng(0)
        for _ in range(4):
            collect_rollout(env, agent, replay, 0, sample_task(rng),
                              disturb=True, rng=rng)
        a0 = [p.detach().clone() for p in agent.actor.parameters()]
        st = agent.update(replay, meta_batch=1, batch_size=16, context_size=CTX,
                          train_actor=False)
        assert all(torch.equal(p0, p) for p0, p in zip(a0, agent.actor.parameters()))
        st = agent.update(replay, meta_batch=1, batch_size=16, context_size=CTX)
        assert np.isfinite(st["actor_loss"]) and np.isfinite(st["gate_loss"])
        # Open training gate: a training rollout acts through the actor on
        # every step, faulted or not (the detector itself is still shut),
        # leaves no floor behind, and the update trains the fault-kind head;
        # a deterministic rollout acts through the learned gate.
        teach = _ctx_agent(adapt_gate=True, train_gate_open=True,
                           fault_cls_weight=1.0, encoder_update_every=3)
        teach.warm_start_from(path, anchor_weight=1.0)
        seen = []
        act_fn = teach.select_action

        def spy(state, deterministic=False):
            out = act_fn(state, deterministic=deterministic)
            seen.append((deterministic, env.fault_kind() > 0, teach.last_gate))
            return out

        teach.select_action = spy
        treplay = MultiTaskReplayBuffer(STATE_DIM, ACTION_DIM,
                                        priv_dim=PRIVILEGED_STATE_DIM)
        for _ in range(2):
            collect_rollout(env, teach, treplay, 0, sample_task(rng),
                            schedule=[("leak", 20, 0.01, 0)])
        assert teach.gate_floor is None, "training gate floor left set"
        train_steps = [(f, g) for det, f, g in seen if not det]
        assert any(f for f, _ in train_steps) and any(not f for f, _ in train_steps)
        assert all(g == 1.0 for _, g in train_steps), "training gate not open"
        seen.clear()
        rollout_deterministic(env, teach, sample_task(rng),
                              disturbance=[("leak", 20, 0.01, 0)])
        assert seen and all(g < 1e-3 for _, _, g in seen),             "a deterministic rollout acted through the training gate"
        del teach.select_action
        # The encoder steps on every third update only.
        e0 = [p.detach().clone() for p in teach.encoder.parameters()]
        for _ in range(2):
            st = teach.update(treplay, meta_batch=1, batch_size=16,
                              context_size=CTX)
        assert all(torch.equal(p0, p)
                   for p0, p in zip(e0, teach.encoder.parameters())),             "the encoder stepped off its interval"
        st = teach.update(treplay, meta_batch=1, batch_size=16, context_size=CTX)
        assert not all(torch.equal(p0, p)
                       for p0, p in zip(e0, teach.encoder.parameters()))
        assert np.isfinite(st["actor_loss"]) and st["fault_cls_loss"] > 0.0
        tp = os.path.join(d, "open_gate.pth")
        teach.save(tp)
        t2 = _ctx_agent(adapt_gate=True, train_gate_open=True,
                        fault_cls_weight=1.0)
        t2.load(tp)
        for t_a, t_b in zip(teach.fault_head.state_dict().values(),
                            t2.fault_head.state_dict().values()):
            assert torch.equal(t_a.cpu(), t_b.cpu()), "fault head"
        # Checkpoints and rollout weights carry the base and the gate.
        p = os.path.join(d, "gated.pth")
        agent.save(p)
        other = _ctx_agent(adapt_gate=True)
        other.load(p)
        worker = _ctx_agent(adapt_gate=True)
        worker.load_rollout_weights(agent.rollout_weights())
        for ag in (other, worker):
            for net in ("actor", "base_actor", "gate"):
                for t_a, t_b in zip(getattr(agent, net).state_dict().values(),
                                    getattr(ag, net).state_dict().values()):
                    assert torch.equal(t_a.cpu(), t_b.cpu()), net
        # A different architecture is refused.
        small = PEARLAgent(STATE_DIM, ACTION_DIM, critic_hidden=64)
        try:
            small.warm_start_from(path)
        except ValueError:
            pass
        else:
            raise AssertionError("warm start across architectures accepted")
    print("  warm start and adaptation gate OK (source policy and critics for "
          "every z; shut gate = source, open = actor; held actor; training "
          "rollouts act through the actor, deterministic ones through the "
          "gate; fault-kind head; encoder step interval; gated checkpoints "
          "and rollout weights)")


def check_shield():
    """The projection never moves a clock or a locked dimension; label sanity."""
    torch.manual_seed(0)
    net = new_apn()
    with torch.no_grad():
        net.margin_head.weight.normal_(0.0, 0.5)
        net.margin_head.bias.fill_(-0.5)
    rng = np.random.default_rng(0)
    o = torch.from_numpy(sample_states(512, rng))
    a = torch.from_numpy(sample_actions(512, rng))
    mask = PlantFermentationEnv.get_action_mask(o)
    proj, steps = project_to_safe(net, o, a, stage_mask=mask)
    frozen = 1.0 - mask
    frozen[:, list(TIME_ACTION_DIMS)] = 1.0
    assert float(((proj - a).abs() * frozen).max()) == 0.0
    assert steps > 0, "the untrained shield admitted nothing; test is void"
    env = PlantFermentationEnv()
    assert_plant_constants_match(env)
    # Synthetic states have the environment's structure: its capacity
    # channels, and the plant-wide channels repeated in every row.
    rows = sample_states(64, rng).reshape(64, N_REACTORS, REACTOR_OBS_DIM)
    real = env.reset().reshape(N_REACTORS, REACTOR_OBS_DIM)
    assert np.allclose(rows[:, :, 22:25], real[None, :, 22:25])
    assert np.allclose(rows[:, :, 25:], rows[:, :1, 25:])
    # A nearly full tank (below the 5% back-off) with a reactor draining hard
    # is unsafe; shutting is safe.
    env.S_mass = 0.93 * S_MAX
    env.current_stage[0] = 3
    env.M[0] = 150000.0
    s = torch.as_tensor(env.observe()[None, :SHIELD_OBS_DIM])
    drain = torch.as_tensor(np.tile([0.0, -1, -1, -1.0], N_REACTORS)[None],
                            dtype=torch.float32)
    drain[0, reactor_action(0, 3)] = 1.0
    mm = safety_margins(s, drain)
    assert float(mm[0, -1]) < 0.0 and float(safety_margin(s, shut_action(1))) > 0.0
    assert mm.shape[1] == 2 * N_REACTORS + 1
    print(f"  shield OK (clocks and locked dims never move; synthetic states "
          f"shaped like the plant's; storage-overflow margin: drain "
          f"{float(mm[0, -1]):+.3f}, shut safe)")


def check_branched_rollouts_and_pool():
    """Branched held-out rollouts equal from-scratch ones; workers match."""
    from metarl.apn.network import load_apn
    env = make_env()
    try:
        usable = load_apn(strict=False, device=torch.device("cpu")) is not None
    except SystemExit:          # a shield trained for another plant
        usable = False
    agent = (_build_agent(env, "saferl") if usable
             else PEARLAgent(STATE_DIM, ACTION_DIM, use_context=False))
    tasks = make_task_pool(2, seed=5)
    faults = (("feed_gain", 1.4, "rotate"), ("downstream_rate", 0.6, None))
    onset = env.max_steps // 2
    for i, t in enumerate(tasks):
        rows = branched_rollouts(env, agent, t, faults, onset, index=i)
        for k, f in enumerate(faults):
            r = resolve_reactor(f[0], f[2], i)
            sched = {"kind": f[0], "onset": onset, "magnitude": f[1], "reactor": r}
            ag_mode = agent.z_mode
            agent.z_mode = "mean"
            ep = _start_episode(env, agent, t, sched)
            _advance(env, agent, ep)
            scratch = _episode_metrics(env, agent, ep)
            agent.z_mode = ag_mode
            assert scratch["violations"] == rows[k + 1]["violations"]
            assert np.isclose(scratch["on_spec_ethanol"],
                              rows[k + 1]["on_spec_ethanol"])
            assert np.isclose(scratch["reward"], rows[k + 1]["reward"])
    # Evaluation scenarios (every pattern, so both first onsets) branched
    # from one undisturbed prefix equal from-scratch episodes exactly,
    # shield counters included.
    scen = {n: d for n, d in eval_fault_scenarios(onset, env.max_steps).items()
            if n.startswith("leak_")}
    names = [None] + sorted(scen)
    rows = scenario_rollouts(env, agent, tasks[1], [scen.get(n) for n in names],
                             index=1)
    for n, row in zip(names, rows):
        scratch = rollout_deterministic(env, agent, tasks[1],
                                        disturbance=scen.get(n), index=1)
        for k in ("reward", "violations", "on_spec_ethanol", "proj",
                  "viol_depth_sum"):
            assert scratch[k] == row[k], (n, k, scratch[k], row[k])
        assert np.array_equal(scratch["act_traj"], row["act_traj"]), n
    # The held-out score that selection and early stopping read.
    ho = blended_heldout_score(env, agent, tasks)
    assert np.isfinite(ho["score"]) and set(ho["panel"]) and np.isclose(
        ho["undisturbed_score"], ho["undisturbed_on_spec_ethanol"]
        - violation_price() * ho["undisturbed_violations"])
    if not usable:
        print("  branched rollouts OK (no APN checkpoint for this plant: "
              "worker pool skipped)")
        return
    t0 = time.time()
    pool = RolloutPool("saferl", 2)
    sched = [("feed_gain", 50, 1.3, 1)]
    try:
        par = pool.run(agent, tasks, faults, onset, "mean")
        far = pool.collect_async(agent, [(0, tasks[0], sched, 123)])[0].result()
    finally:
        pool.close()
    ser = [branched_rollouts(env, agent, t, faults, onset, index=i)
           for i, t in enumerate(tasks)]
    for a_rows, b_rows in zip(par, ser):
        for a_, b_ in zip(a_rows, b_rows):
            assert a_["violations"] == b_["violations"]
            assert np.isclose(a_["on_spec_ethanol"], b_["on_spec_ethanol"])
    # A training episode collected in a worker is the in-process one (both
    # on one torch thread), and lands in the replay the same way. (Last: an
    # in-process episode moves this agent's normalisers.)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        near = collect_episode_record(env, agent, 0, tasks[0], sched, 123)
    finally:
        torch.set_num_threads(threads)
    assert far["steps"] == near["steps"], (far["steps"], near["steps"])
    assert np.isclose(far["reward"], near["reward"]), (far["reward"], near["reward"])
    rep = MultiTaskReplayBuffer(STATE_DIM, ACTION_DIM)
    ingest_episode_record(agent, rep, far)
    assert rep.size(0) == far["steps"] == len(far["transitions"])
    print(f"  branched rollouts and worker pool OK ({time.time() - t0:.0f}s; "
          f"a worker-collected training episode equals the in-process one)")


def check_selection_logic():
    """NO_REGRESS reads per-task gaps and tails; the stopper's gate works."""
    ref = {"undisturbed_viol_per_task": [0.0] * 10,
           "undisturbed_eth_per_task": [1e7] * 10,
           "undisturbed_violations": 0.0, "undisturbed_on_spec_ethanol": 1e7,
           "panel": {}}
    one_task = dict(ref, undisturbed_eth_per_task=[1e7] * 9 + [0.5e7])
    assert _regressions(one_task, ref) == []
    two_tasks = dict(ref, undisturbed_eth_per_task=[1e7] * 8 + [0.5e7] * 2)
    assert _regressions(two_tasks, ref) == ["undisturbed:eth"]
    tail = dict(ref, undisturbed_viol_per_task=[0.0] * 9 + [40.0])
    assert "undisturbed:tail" in _regressions(tail, ref)
    s = EarlyStopper(min_checks=1, patience=2, max_viol=3, max_worst_viol=12)
    assert not s.should_stop(1, 0.0, 10.0, 20.0, 0.0)[0]
    assert not s.should_stop(2, 0.0, 1.0, 5.0, 1.0)[0] and s.feasible_seen
    print("  selection logic OK")


def main():
    t0 = time.time()
    print("=" * 70 + "\nPLANT SMOKE TEST\n" + "=" * 70)
    check_parity_with_fermentation_ppo()
    check_layout_and_masks()
    check_task_sampling()
    check_kinetic_rows()
    _, ref_info = check_reference_policy_and_reward_ordering()
    check_mass_conservation()
    check_storage()
    check_lots_per_reactor()
    check_fault_targeting()
    check_discharge_cutoff_and_batch_boundary()
    check_episode_length_bound()
    check_privileged_state_and_heat_load()
    check_fault_coverage(ref_info)
    check_magnitudes_in_support()
    check_fault_patterns()
    check_plant_wide_faults()
    check_constraint_ids_and_lagrangian()
    check_agent_shapes_and_modes()
    check_history_windows()
    check_recurrent_encoder()
    check_warm_start_and_gate()
    check_shield()
    check_selection_logic()
    check_branched_rollouts_and_pool()
    print(f"\nALL SMOKE CHECKS PASSED ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
