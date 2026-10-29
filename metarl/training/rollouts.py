"""Episodes: stochastic training rollouts and deterministic (branched) ones."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import copy

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import standardize_context, N_REACTORS
from metarl.env.layout import ACTION_DIM, ACTION_DIM_PER_REACTOR
from metarl.training.config import CONSTRAINT_NAMES, DEPTH_CONSTRAINTS
from metarl.training.tasks import (as_fault_schedule, apply_fault_schedule,
                                   maybe_schedule_disturbance)


def collect_rollout(env, agent, replay, task_id, task, add_to_replay=True,
                    disturb=False, rng=None, strat=None, patterns=None,
                    schedule=None):
    """Run one stochastic training episode; returns (reward, violations, steps).

    ``schedule`` (any ``as_fault_schedule`` form, [] for none) installs a
    fault schedule drawn elsewhere; otherwise ``disturb`` draws one here.
    """
    if schedule is not None:
        apply_fault_schedule(env, schedule)
    elif disturb:
        maybe_schedule_disturbance(env, rng=rng, task_id=task_id, strat=strat,
                                   patterns=patterns)
    else:
        env.clear_disturbances()
    env.set_task(task)
    state = env.reset()
    agent.z_mode = "sample"
    agent.clear_z()

    total_r, total_v, steps = 0.0, 0, 0
    step_rewards, acted_on, led_to = [], [], []
    store_priv = (add_to_replay and replay is not None
                  and getattr(replay, "priv_dim", 0) > 0)
    priv = env.privileged_state() if store_priv else None
    done = False
    # Training rollouts of train_gate_open act through the actor everywhere.
    if getattr(agent, "train_gate_open", False):
        agent.gate_floor = 1.0
    try:
        while not done:
            acted_on.append(state)
            # Privileged: whether (and which) fault had acted before this
            # action.
            fault = env.fault_kind()
            safe_a, _ = agent.select_action(state, deterministic=False)
            next_state, r, done, info = env.step(safe_a)
            if add_to_replay and replay is not None:
                # The EFFECTIVE context after this step, clipped (a fired leak
                # is many prior SDs away).
                ctx_true = np.clip(
                    standardize_context(env.get_effective_context()), -6.0, 6.0)
                next_priv = env.privileged_state() if store_priv else None
                replay.add(task_id, state, safe_a, r, next_state, done,
                           ctx_true, fault=fault, priv=priv,
                           next_priv=next_priv)
                priv = next_priv
            agent.update_context(state, safe_a, r, next_state)
            led_to.append(next_state)
            state = next_state
            total_r += r
            total_v += int(info["violation_events"])
            step_rewards.append(r)
            steps += 1
    finally:
        agent.gate_floor = None
    agent.observe_rewards(step_rewards)
    agent.observe_obs(np.stack(acted_on), next_observations=np.stack(led_to))
    return total_r, total_v, steps


class TransitionRecorder:
    """Stands in for a replay buffer: keeps ``add``'s arguments in order."""

    def __init__(self, priv_dim=0):
        self.priv_dim = int(priv_dim)
        self.items = []

    def add(self, *args, **kwargs):
        self.items.append((args, kwargs))


def collect_episode_record(env, agent, task_id, task, schedule, seed):
    """One training episode, recorded for another process's replay buffer.

    Seeds this process's NumPy and torch generators from ``seed``, so the
    episode is a function of its arguments.

    Returns:
        Dict with reward / violations / steps, the recorded ``transitions``
        (``replay.add`` arguments), the visited ``obs`` / ``next_obs`` and
        ``rewards`` (for the normalisers), and the shield's raw counters.
    """
    np.random.seed(int(seed) % (2 ** 32))
    torch.manual_seed(int(seed))
    agent.proj_stats(reset=True)
    rec = TransitionRecorder(priv_dim=getattr(agent, "critic_privileged_dim", 0))
    r, v, steps = collect_rollout(env, agent, rec, task_id, task,
                                  schedule=list(schedule))
    obs = np.stack([a[1] for a, _ in rec.items])
    next_obs = np.stack([a[4] for a, _ in rec.items])
    proj = agent.proj_counts()
    agent.proj_stats(reset=True)
    return {"reward": r, "violations": v, "steps": steps,
            "transitions": rec.items, "obs": obs, "next_obs": next_obs,
            "rewards": [float(a[3]) for a, _ in rec.items], "proj": proj}


def ingest_episode_record(agent, replay, rec):
    """Add a ``collect_episode_record`` result to ``replay`` and ``agent``.

    Exactly what ``collect_rollout`` does in-process after its episode: the
    transitions, then the reward and observation normalisers, then the
    shield's counters.
    """
    for args, kwargs in rec["transitions"]:
        replay.add(*args, **kwargs)
    agent.observe_rewards(rec["rewards"])
    agent.observe_obs(rec["obs"], next_observations=rec["next_obs"])
    agent.add_proj_counts(rec["proj"])
    return rec["reward"], rec["violations"], rec["steps"]


CONSTRAINT_IDS = tuple(CONSTRAINT_NAMES)


def _start_episode(env, agent, task, disturbance, index=0):
    """Install a task and its faults, reset the plant and the posterior."""
    agent.gate_floor = None
    apply_fault_schedule(env, disturbance, index)
    env.set_task(task)
    state = env.reset()
    agent.clear_z()
    return {"state": state, "done": False, "info": {}, "total_r": 0.0,
            # The shield's counters at the start, so the episode's own are
            # the difference at its end.
            "proj0": np.asarray(agent.proj_counts()),
            "total_v": 0, "acts": [], "r_traj": [], "v_traj": [],
            "act_traj": [], "probe_traj": [], "storage_traj": [],
            "gate_traj": [], "stage_traj": [],
            "viol_by_constraint": {cid: 0 for cid in CONSTRAINT_IDS},
            "viol_by_reactor": np.zeros(N_REACTORS, dtype=np.int64),
            "viol_depth_sum": {cid: 0.0 for cid in DEPTH_CONSTRAINTS},
            "viol_depth_max": {cid: 0.0 for cid in DEPTH_CONSTRAINTS}}


def _advance(env, agent, ep, stop_at=None):
    """Step a deterministic episode, accumulating into ``ep``.

    Args:
        env: The environment.
        agent: The agent.
        ep: Accumulator from ``_start_episode``; updated in place.
        stop_at: Optional inner step: stops at the last control-step boundary
            from which the next step cannot reach it.
    """
    while not ep["done"]:
        if (stop_at is not None
                and env.time_step_count + env.control_action_interval > stop_at):
            return
        state = ep["state"]
        ep["stage_traj"].append(env.current_stage.copy())
        safe_a, _ = agent.select_action(state, deterministic=True)
        ep["gate_traj"].append(agent.last_gate)
        next_state, r, done, info = env.step(safe_a)
        agent.update_context(state, safe_a, r, next_state)
        ep["state"], ep["done"], ep["info"] = next_state, done, info
        ep["total_r"] += r
        ep["total_v"] += int(info["violation_events"])
        for cid, n in info.get("violation_counts", {}).items():
            if int(cid) in ep["viol_by_constraint"]:
                ep["viol_by_constraint"][int(cid)] += int(n)
        for cid, d in info.get("violation_depth_sum", {}).items():
            if int(cid) in ep["viol_depth_sum"]:
                ep["viol_depth_sum"][int(cid)] += float(d)
        for cid, d in info.get("violation_depth_max", {}).items():
            if int(cid) in ep["viol_depth_max"]:
                ep["viol_depth_max"][int(cid)] = max(
                    ep["viol_depth_max"][int(cid)], float(d))
        ep["viol_by_reactor"] += np.asarray(info["reactor_violation_events"])
        act = np.asarray(info["actual_action"], dtype=float)
        ep["acts"].append(act.reshape(N_REACTORS, ACTION_DIM_PER_REACTOR)[:, 1:4])
        ep["act_traj"].append(act)
        ep["probe_traj"].append(env.disturbance_probe())
        ep["storage_traj"].append(float(env.S_mass))
        ep["r_traj"].append(r)
        ep["v_traj"].append(float(info["violation_events"] > 0))


def _episode_metrics(env, agent, ep):
    """Summarise a finished deterministic episode."""
    info, acts = ep["info"], ep["acts"]
    probe = {}
    if ep["probe_traj"]:
        for k in ep["probe_traj"][0]:
            probe[k] = np.asarray([p[k] for p in ep["probe_traj"]], dtype=float)
    return {
        "reward": ep["total_r"],
        "violations": ep["total_v"],
        "viol_by_constraint": ep["viol_by_constraint"],
        "viol_by_reactor": ep["viol_by_reactor"].astype(float).tolist(),
        "viol_depth_sum": ep["viol_depth_sum"],
        "viol_depth_max": ep["viol_depth_max"],
        "feasible": 1.0 if env.time_step_count >= env.max_steps else 0.0,
        "ethanol": float(env.product_eth),
        "on_spec_ethanol": float(info.get("on_spec_ethanol", 0.0)),
        "reactor_on_spec_ethanol": [float(v) for v in info.get(
            "reactor_on_spec_ethanol", np.zeros(N_REACTORS))],
        "product_mass": float(env.product_mass),
        "mean_titer": float(env.mean_product_titer),
        "purity_ok": 1.0 if info.get("purity_ok", False) else 0.0,
        "lots_passed": int(info.get("lots_passed", 0)),
        "lots_failed": int(info.get("lots_failed", 0)),
        "batches": int(env.n_batches_completed),
        "reactor_batches": env.reactor_batches_completed.astype(int).tolist(),
        "feed_used": float((env.initial_lf - env.total_lf)
                           + (env.initial_c5 - env.total_c5)),
        "storage_min": float(env.storage_min_seen),
        "storage_max": float(env.storage_max_seen),
        # (N, 3): mean executed LF, C5 and discharge per reactor.
        "act_mean": (np.mean(np.stack(acts), axis=0) if acts
                     else np.full((N_REACTORS, 3), np.nan)),
        "reward_traj": np.asarray(ep["r_traj"]),
        "viol_traj": np.asarray(ep["v_traj"]),
        # (steps, 12): executed action in [0, 1] per control step.
        "act_traj": (np.stack(ep["act_traj"]) if ep["act_traj"]
                     else np.zeros((0, ACTION_DIM))),
        "probe_traj": probe,
        "storage_traj": np.asarray(ep["storage_traj"], dtype=float),
        "gate_traj": np.asarray(ep["gate_traj"], dtype=float),
        # (steps, N): every reactor's recipe stage at each decision.
        "stage_traj": (np.stack(ep["stage_traj"]) if ep["stage_traj"]
                       else np.zeros((0, N_REACTORS), dtype=int)),
        "z_mu": agent.z_mu.detach().cpu().numpy().copy(),
        "z_sigma_sq": agent.z_sigma_sq.detach().cpu().numpy().copy(),
        # The shield's (calls, no-ops, steps, moved) in this episode.
        "proj": tuple(int(c) for c in
                      np.asarray(agent.proj_counts()) - ep["proj0"]),
    }


def rollout_deterministic(env, agent, task, z_mode="mean", disturbance=None,
                          index=0):
    """Run one deterministic episode and return production and safety metrics."""
    prev_mode = agent.z_mode
    try:
        agent.z_mode = z_mode
        ep = _start_episode(env, agent, task, disturbance, index)
        _advance(env, agent, ep)
        return _episode_metrics(env, agent, ep)
    finally:
        agent.z_mode = prev_mode


def scenario_rollouts(env, agent, task, schedules, z_mode="mean", index=0):
    """Deterministic rollouts of one task under several fault schedules.

    Every schedule is undisturbed up to its first onset, so the undisturbed
    episode is simulated once, snapshotted just before each distinct first
    onset, and each faulted run continues from its snapshot: every row is
    exactly the episode a from-scratch ``rollout_deterministic`` produces,
    shield counters included.

    Args:
        schedules: Sequence of fault schedules (any ``as_fault_schedule``
            form; None or [] is undisturbed). "rotate" reactors resolve with
            ``index``.
        index: The task's index in its pool.

    Returns:
        List of episode-metric dicts, one per schedule, in order.
    """
    scheds = [as_fault_schedule(s, index) for s in schedules]
    firsts = [min(d["onset"] for d in s) if s else None for s in scheds]
    prev_mode = agent.z_mode
    try:
        agent.z_mode = z_mode
        ep = _start_episode(env, agent, task, None)
        snaps = {}
        for t in sorted({f for f in firsts if f is not None}):
            _advance(env, agent, ep, stop_at=t)
            snaps[t] = (copy.deepcopy(env), agent.posterior_state(),
                        copy.deepcopy(ep), np.asarray(agent.proj_counts()))
        _advance(env, agent, ep)
        trunk = _episode_metrics(env, agent, ep)
        rows = []
        for sched, first in zip(scheds, firsts):
            if first is None:
                rows.append(trunk)
                continue
            snap_env, snap_z, snap_ep, snap_proj = snaps[first]
            b_env, b_ep = copy.deepcopy(snap_env), copy.deepcopy(snap_ep)
            agent.set_posterior_state(snap_z)
            # The prefix's shield counts belong to this episode too.
            b_ep["proj0"] = (b_ep["proj0"] + np.asarray(agent.proj_counts())
                             - snap_proj)
            for d in sched:
                b_env.schedule_disturbance(d["kind"], d["onset"],
                                           d["magnitude"],
                                           reactor=d["reactor"],
                                           end_step=d["end"])
            _advance(b_env, agent, b_ep)
            rows.append(_episode_metrics(b_env, agent, b_ep))
        return rows
    finally:
        agent.z_mode = prev_mode


def branched_rollouts(env, agent, task, faults, onset, z_mode="mean", index=0):
    """Deterministic rollouts of one task: undisturbed, then under each fault.

    Every fault steps at ``onset`` (``scenario_rollouts``).

    Args:
        faults: Sequence of (kind, magnitude[, reactor]); "rotate" reactors
            resolve with ``index``.
        index: The task's index in its pool.

    Returns:
        List of episode-metric dicts: undisturbed first, then one per fault.
    """
    schedules = [None] + [[(f[0], onset, f[1], f[2] if len(f) > 2 else None)]
                          for f in faults]
    return scenario_rollouts(env, agent, task, schedules, z_mode, index=index)
