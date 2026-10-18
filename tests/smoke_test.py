"""Fast CPU smoke test for the three-reactor meta-RL pipeline.

Exercises every code path -- shapes, wiring, constraints, storage, faults,
token networks, the GRU posterior, warm start, adaptation gate, shield,
branched and worker rollouts -- without real training, and asserts the design
properties that must not silently regress:

  * the observation layout: the shield reads a prefix that excludes every CO2
    rate, each token holds its reactor's block, its CO2 rate and the plant
    channels; the NumPy and Torch stage masks agree;
  * the chemistry is pinned on every task and in every reactor; pump gains and
    leaks are drawn per reactor;
  * the allocation-free RK4 (one parameter row per reactor) matches the
    reference integrator, and the CO2 kernel matches the reference rates;
  * a scripted staggered scheduler runs the nominal plant cleanly, scores
    positive and outranks doing nothing (which runs the tank dry);
  * mass is conserved exactly under leaks, storage draw and recharges;
  * storage overflow (7) and underflow (8) fire, and the downstream_rate
    fault changes the tank's draw;
  * each batch is its own lot: a fault that rejects one reactor's lots leaves
    the other reactors' released lots counted, and the reward keeps exactly
    the value of the released product;
  * a fault scheduled on one reactor changes only that reactor; plant-wide
    kinds refuse a reactor; the training-only fault flag latches;
  * every fault kind hurts the scripted policy on at least one axis;
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

import numpy as np
import torch

from metarl.env.config import (N_REACTORS, nominal_task, sample_task,
                               standardize_context, TASK_CONTEXT_DIM,
                               NOMINAL_PARAMS, C_MAX_LIMITS, C_C5_BASE,
                               C_LF_BASE, S_MAX, S_MIN, S_OUT_RATE, S_INITIAL,
                               R_MASS, R_STAGE, CTX_GAIN, CTX_LEAK, CTX_KIN,
                               CTX_DOWNSTREAM, CTX_TURNAROUND, CTX_STORAGE_CAP,
                               REACTOR_FLOW_SCALE,
                               MIN_PRODUCT_TITER)
from metarl.env.core import (integrate_rk4, integrate_rk4_reference,
                             calculate_rates_numba, co2_formation_rates)
from metarl.env.layout import (STATE_DIM, ACTION_DIM, SHIELD_OBS_DIM,
                               CER_START, GLOBAL_START, STORAGE_CHANNEL,
                               PRIVILEGED_STATE_DIM, DEFAULT_MASKED_ACTION,
                               TIME_ACTION_DIMS, ETHANOL_CHANNELS,
                               stage_mask_np, apply_stage_mask_np,
                               reactor_channel, reactor_action)
from metarl.env.faults import (VALID_DISTURBANCES, DISTURBANCE_KIND_INDEX,
                               PLANT_DISTURBANCES)
from metarl.env.plant import PlantFermentationEnv, DISCHARGE_CUTOFF
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
                                    META_ANCHOR_FAULT_DIMS)
from metarl.training.arms import _build_agent
from metarl.training.early_stopping import EarlyStopper
from metarl.training.tasks import (make_env, make_task_pool, resolve_reactor,
                                   maybe_schedule_disturbance,
                                   pattern_schedule, apply_fault_schedule)
from metarl.training.evaluate import eval_fault_scenarios
from metarl.training.rollouts import (collect_rollout, sensor_stream_key,
                                      _start_episode, _advance,
                                      _episode_metrics, branched_rollouts,
                                      collect_episode_record,
                                      ingest_episode_record)
from metarl.training.heldout import _regressions, RolloutPool
from metarl.apn.train import new_apn

CTX = CONTEXT_SIZE
T1 = -0.34657359          # clock action of exactly 1.0x
# Fault kinds that cost the scripted policies nothing, by design of the
# kinetics (MetaRL_Lean/README: "a row where a tie is expected").
TIE_EXPECTED_KINDS = ("cell_viability",)


def _clock(x):
    """Clock multiplier -> action value."""
    return 2.0 * (x - 0.5) / 1.5 - 1.0


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
        st = int(np.argmax(obs[reactor_channel(i, R_STAGE):
                               reactor_channel(i, R_STAGE) + 5]))
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
    fault reaches the inventory constraints."""
    a = []
    S = obs[STORAGE_CHANNEL]
    for i in range(N_REACTORS):
        st = int(np.argmax(obs[reactor_channel(i, R_STAGE):
                               reactor_channel(i, R_STAGE) + 5]))
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
        sums prod / clawback and keeps any ``record(env, info)`` outputs.
    """
    env.clear_disturbances()
    for f in faults:
        env.schedule_disturbance(*f[:3], reactor=f[3] if len(f) > 3 else None)
    env.set_task(task if task is not None else nominal_task())
    obs = env.reset()
    total, counts, done, info = 0.0, {}, False, {}
    extras = {"prod": 0.0, "clawback": 0.0, "steps": 0, "rec": []}
    while not done:
        obs, r, done, info = env.step(apply_stage_mask_np(obs, policy(obs)))
        total += r
        extras["prod"] += info["prod"]
        extras["clawback"] += info["clawback"]
        extras["steps"] += 1
        for k, v in info["violation_counts"].items():
            counts[k] = counts.get(k, 0) + v
        if record is not None:
            extras["rec"].append(record(env, info))
    return total, counts, info, extras


# ─────────────────────────────────────────────────────────────── environment

def check_layout_and_masks():
    """Observation layout, shield prefix, token index and stage masks."""
    assert STATE_DIM == 78 and ACTION_DIM == 12 and SHIELD_OBS_DIM == 75
    assert CER_START == SHIELD_OBS_DIM, "the CO2 rates must follow the shield prefix"
    tok = token_index().numpy()
    assert tok.shape == (N_REACTORS, 28)
    for i in range(N_REACTORS):
        block = set(range(i * 24, (i + 1) * 24))
        assert block <= set(tok[i]) and CER_START + i in tok[i]
        assert set(range(GLOBAL_START, GLOBAL_START + 3)) <= set(tok[i])
    env = PlantFermentationEnv(max_steps=400, sensor_model=False)
    env.set_task(nominal_task())
    obs = env.reset()
    seen = set()
    done = False
    while not done:
        m_np = stage_mask_np(obs)
        m_t = PlantFermentationEnv.get_action_mask(
            torch.as_tensor(obs, dtype=torch.float32)).numpy()
        assert np.allclose(m_np, m_t), (m_np, m_t)
        seen.update(env.current_stage.tolist())
        obs, _, done, _ = env.step(apply_stage_mask_np(obs, _pol_deliberate(obs)))
    assert {0, 1, 2, 3} <= seen, seen
    assert np.allclose(DEFAULT_MASKED_ACTION[list(TIME_ACTION_DIMS)], np.tanh(T1))
    print(f"  layout and masks OK (obs {STATE_DIM}, shield prefix "
          f"{SHIELD_OBS_DIM}, tokens {tok.shape}, stages seen {sorted(seen)})")


def check_task_sampling():
    """Chemistry pinned, per-reactor gains / leaks, context layout."""
    nom = nominal_task()
    assert nom["context"].shape == (TASK_CONTEXT_DIM,)
    rng = np.random.default_rng(0)
    gains = []
    for _ in range(400):
        t = sample_task(rng)
        assert np.array_equal(t["params"], NOMINAL_PARAMS)
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
    print("  task sampling OK (chemistry pinned on 400 draws; gains "
          "independent per reactor; OOD leaks non-negative)")


def check_fast_integrator():
    """The per-reactor RK4 matches the reference; the CO2 kernel matches too."""
    rng = np.random.default_rng(7)
    params = np.array(NOMINAL_PARAMS, dtype=np.float64)
    tiled = np.ascontiguousarray(np.tile(params, (N_REACTORS, 1)))
    worst = 0.0
    for _ in range(30):
        c = (rng.uniform(0.0, 0.8, size=(N_REACTORS, 15))
             * C_MAX_LIMITS.astype(np.float64))
        m = rng.uniform(1000.0, 220000.0, size=N_REACTORS)
        f_lf = rng.uniform(0.0, 4 * 2487.0 / 3600.0, size=N_REACTORS)
        f_c5 = rng.uniform(0.0, 4 * 628.0 / 3600.0, size=N_REACTORS)
        cin = ((f_c5[:, None] * C_C5_BASE + f_lf[:, None] * C_LF_BASE)
               / np.maximum(f_lf + f_c5, 1e-12)[:, None])
        F_in = f_lf + f_c5
        F_out = rng.uniform(0.0, 5000.0 / 3600.0, size=N_REACTORS)
        fast = integrate_rk4(c, m, F_in, F_out, cin, tiled, 120.0, 30)
        ref = integrate_rk4_reference(c, m, F_in, F_out, cin, params, 120.0, 30)
        assert fast[2] == ref[2]
        assert np.allclose(fast[0], ref[0], rtol=1e-9, atol=1e-12)
        assert np.allclose(fast[1], ref[1], rtol=1e-12)
        worst = max(worst, float(np.max(np.abs(fast[0] - ref[0]))
                                 / max(float(np.max(np.abs(ref[0]))), 1e-12)))
        c_safe = np.maximum(c, 1e-8)
        co2 = co2_formation_rates(c_safe, tiled)
        co2_ref = calculate_rates_numba(c_safe, params)[:, 11]
        assert np.allclose(co2, co2_ref, rtol=1e-9, atol=1e-15)
    # A different row per reactor reaches only that reactor.
    c = np.full((N_REACTORS, 15), 5.0)
    m = np.full(N_REACTORS, 1e5)
    z3 = np.zeros(N_REACTORS)
    slow = tiled.copy()
    slow[1, 0:3] *= 0.2
    a = integrate_rk4(c, m, z3, z3, np.zeros((N_REACTORS, 15)), tiled, 120.0, 30)[0]
    b = integrate_rk4(c, m, z3, z3, np.zeros((N_REACTORS, 15)), slow, 120.0, 30)[0]
    assert np.array_equal(a[[0, 2]], b[[0, 2]]) and not np.allclose(a[1], b[1])
    print(f"  fast RK4 matches reference OK (worst relative diff {worst:.1e}); "
          f"per-reactor parameter rows act on their reactor only")


def check_reference_policy_and_reward_ordering():
    """The scripted scheduler is clean and outranks doing nothing."""
    env = PlantFermentationEnv(sensor_model=False)
    r_good, c_good, info, ex = _run(env, _pol_deliberate)
    assert not c_good, f"the reference policy is not clean: {c_good}"
    assert r_good > 0.0, r_good
    assert info["lots_passed"] >= 4 and info["lots_failed"] == 0, info
    r_idle, c_idle, info_i, _ = _run(env, _pol_idle)
    assert r_good > r_idle, (r_good, r_idle)
    assert c_idle.get(8, 0) > 100, "idling must run the storage tank dry"
    assert c_idle.get(4, 0) == 1, "shipping nothing must fail purity once"
    # Lockstep clocks overflow the shared tank: the storage couples reactors.
    r_lock, c_lock, _, _ = _run(env, lambda o: _pol_deliberate(
        o, s_cap=1.1, clocks=(1.0, 1.0, 1.0)))
    assert c_lock.get(7, 0) > 0, "unstaggered unloads never overflowed the tank"
    print(f"  reference policy OK (clean, {info['lots_passed']} lots released, "
          f"reward {r_good:.1f} > idle {r_idle:.1f}; idle runs the tank dry "
          f"for {c_idle[8]} h; lockstep unloads overflow it for {c_lock[7]} h)")
    return r_good, info


def check_mass_conservation():
    """Mass is conserved exactly under leaks, storage draw and recharges."""
    env = PlantFermentationEnv(sensor_model=False)
    worst = 0.0
    cases = [(), (("leak", 100, 0.05, 1),), (("leak", 50, 1.0, None),),
             (("leak", 200, 0.25, 2), ("downstream_rate", 100, 1.6)),
             (("downstream_rate", 10, 0.2),)]
    for faults in cases:
        _, counts, info, _ = _run(env, _pol_full_throttle, faults=faults)
        assert info["mass_balance_ok"], (faults, info["mass_balance_error"])
        assert 6 not in counts
        assert np.all(env.M >= env.DRY_FLOOR - 1e-9)
        assert env.S_mass >= 0.0
        worst = max(worst, abs(info["mass_balance_error"]))
    print(f"  mass conservation OK (worst residual {worst:.2e} kg over leaks, "
          f"tank draw and recharges)")


def check_storage():
    """The tank drains at S_OUT_RATE, and downstream_rate scales the draw."""
    env = PlantFermentationEnv(sensor_model=False)
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
    assert 7 in info["violation_counts"], info["violation_counts"]
    print("  storage OK (draw at S_OUT_RATE, halved by downstream_rate 0.5; "
          "overflow fires at the wall)")


def check_lots_per_reactor():
    """A fault that rejects one reactor's lots leaves the others' counted."""
    env = PlantFermentationEnv(sensor_model=False)
    _, counts, info, ex = _run(env, _pol_deliberate,
                               faults=(("disturb_kinetics_decay", 5, 0.15, 2),))
    failed = info["reactor_lots_failed"]
    passed = info["reactor_lots_passed"]
    assert failed[2] >= 1 and failed[0] == 0 and failed[1] == 0, (passed, failed)
    assert info["reactor_on_spec_ethanol"][0] > 0 and info["reactor_on_spec_ethanol"][1] > 0
    assert counts.get(4, 0) == int(np.sum(failed))
    kept = ex["prod"] - ex["clawback"]
    want = env.PROD_REWARD_COEF * info["on_spec_ethanol"] / env.PROD_REWARD_SCALE
    assert np.isclose(kept, want, rtol=1e-9), (kept, want)
    print(f"  lots per reactor OK (reactor 2 slowed: lots passed {passed.tolist()}"
          f", failed {failed.tolist()}; production reward kept = released "
          f"product exactly)")


def check_fault_targeting():
    """A fault on one reactor changes only that reactor; plant kinds refuse one."""
    env = PlantFermentationEnv(max_steps=30, sensor_model=False)
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
    print("  fault targeting OK (one reactor or all; plant kinds refuse a "
          "reactor; flag latches, not on a neutral magnitude)")


def check_discharge_cutoff_and_batch_boundary():
    """A sub-cutoff command shuts the valve; a recharged vessel is never drained."""
    env = PlantFermentationEnv(sensor_model=False)
    env.set_task(nominal_task())
    env.clear_disturbances()
    obs = env.reset()
    env.current_stage[0] = 3
    env.M[0] = 100000.0
    act = np.tile([T1, -1, -1, -1.0], N_REACTORS)
    act[3] = 2.0 * (0.5 * DISCHARGE_CUTOFF) - 1.0
    _, _, _, info = env._step_internal(act)
    assert not info["draining"][0], "a sub-cutoff command drained the vessel"
    for clock in (0.9, 1.1, 2.0):
        env = PlantFermentationEnv(sensor_model=False)
        pol = lambda o: np.tile([_clock(clock), 1.0, 1.0, 1.0], N_REACTORS)  # noqa: E731
        _, counts, info, ex = _run(env, pol, record=lambda e, i: min(
            i["drain_titers"], default=1.0))
        assert min(ex["rec"]) > 0.0, f"clock {clock}x drained a recharged vessel"
    print("  discharge cutoff and batch boundary OK")


def check_episode_length_bound():
    """The longest episode fits the encoder's training window."""
    worst = 0
    for clock in (2.0, 1.7, 1.0):
        env = PlantFermentationEnv(sensor_model=False)
        pol = lambda o: np.tile([_clock(clock), 1.0, 1.0, 1.0], N_REACTORS)  # noqa: E731
        _, _, _, ex = _run(env, pol)
        worst = max(worst, ex["steps"])
    assert worst < CTX, (worst, CTX)
    print(f"  episode length OK (longest {worst} control steps < window {CTX})")


def check_privileged_state_and_sensors():
    """Privileged state, sensor model, stream keys and the CO2 tail."""
    env = PlantFermentationEnv()
    env.set_task(sample_task(np.random.default_rng(1)))
    obs = env.reset()
    for _ in range(20):
        obs, _, _, _ = env.step(apply_stage_mask_np(obs, _pol_deliberate(obs)))
    assert np.all(obs[env.species_channels] == 0.0), "masked species leaked"
    p = env.privileged_state()
    assert p.shape == (PRIVILEGED_STATE_DIM,) and np.all(np.isfinite(p))
    assert np.any(p[env.species_channels] > 0.0)
    assert np.isclose(p[-1], env.time_step_count / env.max_steps)
    exact = PlantFermentationEnv(sensor_model=False)
    exact.set_task(nominal_task())
    o = exact.reset()
    assert np.array_equal(o, exact.get_state_norm())

    def stream(key):
        e = PlantFermentationEnv()
        e.set_task(nominal_task())
        e.sensor_stream_key = key
        out = [e.reset()]
        for _ in range(5):
            out.append(e.step(apply_stage_mask_np(out[-1], _pol_deliberate(out[-1])))[0])
        return np.stack(out)

    assert np.array_equal(stream(("a", 1)), stream(("a", 1)))
    assert not np.array_equal(stream(("a", 1)), stream(("a", 2)))
    off = PlantFermentationEnv(observe_co2_rate=False)
    off.set_task(nominal_task())
    o = off.reset()
    for _ in range(20):
        o, _, _, _ = off.step(apply_stage_mask_np(o, _pol_deliberate(o)))
    assert np.all(o[CER_START:] == 0.0)
    # A kinetics fault on a fermenting reactor drops ITS rate only.
    def cer_after(faults):
        e = PlantFermentationEnv(sensor_model=False)
        _, _, _, ex = _run(e, _pol_deliberate, faults=faults,
                           record=lambda e_, i: e_.cer.copy())
        return np.stack(ex["rec"])
    base = cer_after(())
    hit = cer_after((("disturb_kinetics_decay", 100, 0.5, 0),))
    k = 100 // 5 + 1
    assert np.array_equal(base[:k - 1], hit[:k - 1])
    assert hit[k, 0] < 0.9 * base[k, 0], (hit[k], base[k])
    assert np.allclose(hit[k, 1:], base[k, 1:])
    print("  privileged state, sensors and CO2 tail OK (stream keys "
          "reproducible; a kinetics fault drops only its reactor's rate "
          f"to {hit[k, 0] / base[k, 0]:.0%})")


def check_fault_coverage(ref_info):
    """Every fault kind hurts a scripted policy on at least one axis.

    As in the lean study: the constraint axis is probed with the open-loop
    recipe (a pump fault only overflows a policy blind to the level), the
    production axis with the closed-loop scheduler.
    """
    env = PlantFermentationEnv(sensor_model=False)
    _, c_open, _, _ = _run(env, _pol_open_loop)
    assert not c_open, f"the open-loop recipe is not clean: {c_open}"
    base_eth = float(ref_info["on_spec_ethanol"])
    probes = {"leak": (0.05, None), "feed_gain": (1.4, None),
              "feed_conc": (0.3, None), "disturb_kinetics_decay": (0.3, None),
              "toxin_spike": (6.5, None), "cell_viability": (0.3, None),
              "contamination": (0.004, None), "downstream_rate": (0.3, None),
              "turnaround_delay": (5.0, None), "storage_capacity": (0.75, None)}
    print("    kind                     open-loop viol    closed-loop viol  "
          "closed-loop on-spec")
    for kind, (mag, reactor) in probes.items():
        fault = ((kind, 150, mag, reactor),)
        _, c_o, _, _ = _run(PlantFermentationEnv(sensor_model=False),
                            _pol_open_loop, faults=fault)
        _, c_c, info, _ = _run(PlantFermentationEnv(sensor_model=False),
                               _pol_deliberate, faults=fault)
        loss = 1.0 - float(info["on_spec_ethanol"]) / base_eth
        print(f"    {kind:<24} {str(c_o):<17} {str(c_c):<17} {-loss:+.0%}")
        if kind in TIE_EXPECTED_KINDS:
            # The lean study's finding: the yeast regrows within the batch, so
            # a kill costs nothing at any magnitude. Kept as a tie row.
            continue
        assert c_o or c_c or loss > 0.10, (
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
    env = PlantFermentationEnv(sensor_model=False)
    env.set_task(nominal_task())
    grid = eval_fault_scenarios(env.max_steps // 2, env.max_steps)
    assert set(grid) == {f"{k}_{p}" for k in EVAL_DISTURBANCES
                         for p in FAULT_PATTERNS}, sorted(grid)
    for name, sched in grid.items():
        assert sched, name
        apply_fault_schedule(env, sched, index=1)
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
          f"{len(EVAL_DISTURBANCES)} kinds x {', '.join(FAULT_PATTERNS)}; "
          f"spikes clear, ramps reach full strength, toxin pulses telescope)")


def check_scheduling_and_storage_faults():
    """Turnaround conflicts (9) are counted; turnaround_delay and
    storage_capacity act on the stage times and the overflow wall."""
    env = PlantFermentationEnv(sensor_model=False)
    env.set_task(nominal_task())
    env.clear_disturbances()
    # Every vessel starts the episode charging: exempt.
    obs = env.reset()
    _, _, _, info = env.step(apply_stage_mask_np(obs, _pol_idle(obs)))
    assert 9 not in info["violation_counts"], info["violation_counts"]
    # Two vessels in turnaround after a batch: one conflict event per hour.
    env.reset()
    env.reactor_batches_completed[:] = 1
    env.current_stage[:] = (4, 0, 2)
    env.stage_credits[:] = (50.0, 50.0, 50.0)
    _, _, _, info = env._step_internal(np.tile([T1, -1, -1, -1.0], N_REACTORS))
    assert info["violation_counts"].get(9) == 1, info["violation_counts"]
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
    # storage_capacity: the overflow wall moves to 90% of S_MAX.
    for mag, want in ((1.0, False), (0.9, True)):
        env.clear_disturbances()
        env.schedule_disturbance("storage_capacity", 0, mag)
        env.reset()
        env.S_mass = 0.95 * S_MAX
        _, _, _, info = env._step_internal(np.tile([T1, -1, -1, -1.0], N_REACTORS))
        assert (7 in info["violation_counts"]) == want, (mag, info["violation_counts"])
    assert np.isclose(env.get_effective_context()[CTX_STORAGE_CAP], 0.9)
    env.clear_disturbances()
    print("  scheduling and storage faults OK (first charge exempt; two "
          "vessels in turnaround conflict; turnaround_delay stretches "
          "cleanup and charge; storage_capacity moves the overflow wall)")


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
    # At init every mixer block is the identity, so a reactor's action reads
    # its own token only; once the blocks have weight, it reads the others.
    x = torch.rand(4, STATE_DIM)
    y = x.clone()
    y[:, reactor_channel(2, R_MASS)] += 0.5
    zz = torch.zeros(4, LATENT_DIM)
    with torch.no_grad():
        d0 = (agent.actor(x, zz)[0] - agent.actor(y, zz)[0]).abs()
        assert d0[:, :8].max() < 1e-6 and d0[:, 8:].max() > 0
        for blk in agent.actor.mixer.blocks:
            blk.out.weight.normal_(0.0, 0.2)
        d1 = (agent.actor(x, zz)[0] - agent.actor(y, zz)[0]).abs()
        assert d1[:, :8].max() > 1e-6, "the mixer does not couple reactors"
    print("  agent shapes OK (12 actions, mask applied, five z modes; "
          "reactors couple through the attention blocks)")


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
    """A 'trained' saferl-like source checkpoint with nonzero FiLM weights."""
    src = PEARLAgent(STATE_DIM, ACTION_DIM, use_context=False)
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
        z0 = torch.zeros(16, 1, device=device)
        priv = torch.rand(16, PRIVILEGED_STATE_DIM, device=device)
        ctx = torch.randn(16, TASK_CONTEXT_DIM, device=device)
        with torch.no_grad():
            m_src, _ = src.actor(obs, z0)
            q_src = src.critic(obs, act, z0)
            for _ in range(3):
                z = torch.randn(16, agent._z_dim, device=device) * 2
                m, _ = agent.actor(obs, z)
                assert torch.allclose(m, m_src, atol=1e-5), "warm actor reads z"
                cond = agent._critic_cond(z, ctx, priv)
                for q, q_ref in zip(agent.critic(obs, act, cond), q_src):
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
            base_a = torch.tanh(src.actor(od, torch.zeros(1, 1))[0]).numpy()[0]
            actor_a = torch.tanh(agent.actor(
                od, torch.as_tensor(zf).unsqueeze(0))[0]).numpy()[0]
        assert not np.allclose(base_a, actor_a, atol=1e-3)
        raw = {}
        for opening in (0.0, 0.25, 1.0):
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
          "every z; shut gate = source, open = actor; held actor; gated "
          "checkpoints and rollout weights)")


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
    env = PlantFermentationEnv(sensor_model=False)
    assert_plant_constants_match(env)
    # A nearly full tank (below the 5% back-off) with a reactor draining hard
    # is unsafe; shutting is safe.
    env.reset()
    env.S_mass = 0.93 * S_MAX
    env.current_stage[0] = 3
    env.M[0] = 150000.0
    s = torch.as_tensor(env.get_state_norm()[None, :SHIELD_OBS_DIM])
    drain = torch.as_tensor(np.tile([0.0, -1, -1, -1.0], N_REACTORS)[None],
                            dtype=torch.float32)
    drain[0, reactor_action(0, 3)] = 1.0
    mm = safety_margins(s, drain)
    assert float(mm[0, -1]) < 0.0 and float(safety_margin(s, shut_action(1))) > 0.0
    assert mm.shape[1] == 2 * N_REACTORS + 1
    print(f"  shield OK (clocks and locked dims never move; storage-overflow "
          f"margin: drain {float(mm[0, -1]):+.3f}, shut safe)")


def check_branched_rollouts_and_pool():
    """Branched held-out rollouts equal from-scratch ones; workers match."""
    from metarl.apn.network import APN_PATH
    env = make_env()
    agent = (_build_agent(env, "saferl") if os.path.exists(APN_PATH)
             else PEARLAgent(STATE_DIM, ACTION_DIM, use_context=False))
    tasks = make_task_pool(2, seed=5)
    faults = (("feed_gain", 1.4, "rotate"), ("downstream_rate", 0.6, None))
    onset = env.max_steps // 2
    for i, t in enumerate(tasks):
        rows = branched_rollouts(env, agent, t, faults, onset, index=i)
        for k, f in enumerate(faults):
            r = resolve_reactor(f[0], f[2], i)
            sched = {"kind": f[0], "onset": onset, "magnitude": f[1], "reactor": r}
            # The branch shares the undisturbed run's noise stream.
            prev = env.sensor_stream_key
            ag_mode = agent.z_mode
            agent.z_mode = "mean"
            ep = _start_episode(env, agent, t, sched, sensor_stream_key(t))
            _advance(env, agent, ep)
            scratch = _episode_metrics(env, agent, ep)
            agent.z_mode = ag_mode
            env.sensor_stream_key = prev
            assert scratch["violations"] == rows[k + 1]["violations"]
            assert np.isclose(scratch["on_spec_ethanol"],
                              rows[k + 1]["on_spec_ethanol"])
            assert np.isclose(scratch["reward"], rows[k + 1]["reward"])
    if isinstance(agent, PEARLAgent) and agent.apn is None:
        print("  branched rollouts OK (no APN checkpoint: worker pool skipped)")
        return
    t0 = time.time()
    pool = RolloutPool("saferl", 2, sensor_model=env.sensor_model)
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
    check_layout_and_masks()
    check_task_sampling()
    check_fast_integrator()
    _, ref_info = check_reference_policy_and_reward_ordering()
    check_mass_conservation()
    check_storage()
    check_lots_per_reactor()
    check_fault_targeting()
    check_discharge_cutoff_and_batch_boundary()
    check_episode_length_bound()
    check_privileged_state_and_sensors()
    check_fault_coverage(ref_info)
    check_magnitudes_in_support()
    check_fault_patterns()
    check_scheduling_and_storage_faults()
    check_agent_shapes_and_modes()
    check_recurrent_encoder()
    check_warm_start_and_gate()
    check_shield()
    check_selection_logic()
    check_branched_rollouts_and_pool()
    print(f"\nALL SMOKE CHECKS PASSED ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
