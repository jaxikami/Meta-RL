"""Held-out checks: scoring, no-regression and the parallel worker pool.

The blended held-out score that checkpoint selection and early stopping read,
the no-regression comparison against a warm-start source, the per-fault-kind
anchor adaptation, and the worker processes that run the checks.
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import SENSOR_MODEL_ENABLED
from metarl.env.faults import (DISTURBANCE_ALIASES, DISTURBANCE_KIND_INDEX,
                               DISTURBANCE_INDEX_MAP)
from metarl.training.config import (SELECT_MAX_WORST_VIOL, SELECT_VIOL_WEIGHT,
                                    GATE_FAULT_KINDS, HELDOUT_DISTURBANCES,
                                    HELDOUT_DISTURB_WEIGHT, META_ANCHOR_UP,
                                    META_ANCHOR_DOWN, META_ANCHOR_MIN,
                                    META_ANCHOR_MAX, NO_REGRESS_ETH_REL,
                                    NO_REGRESS_VIOL_ABS, NO_REGRESS_VIOL_REL,
                                    NO_REGRESS_TAIL_REL,
                                    NO_REGRESS_ETH_REL_CLEAN,
                                    NO_REGRESS_CLEAN_VIOL)
from metarl.training.arms import ARMS, get_arm, _build_agent
from metarl.training.tasks import make_env
from metarl.training.rollouts import branched_rollouts
from metarl.training.metrics import (latent_separability, gate_opening,
                                     _finite_mean)


def violation_price(env):
    """Ethanol one violation event is worth at the reward's own exchange rate."""
    return (SELECT_VIOL_WEIGHT * float(env.BASE_SPIKE)
            * float(env.PROD_REWARD_SCALE) / float(env.PROD_REWARD_COEF))


def _score_rows(rows, viol_price=0.0):
    """Aggregate one held-out rollout per task into a held-out score."""
    sep = latent_separability([r["z_mu"] for r in rows],
                              [r["z_sigma_sq"] for r in rows])
    per_task_viol = [float(r["violations"]) for r in rows]
    per_task_eth = [float(r["on_spec_ethanol"]) for r in rows]
    violations = float(np.mean(per_task_viol))
    on_spec = float(np.mean(per_task_eth))
    gates = [np.asarray(r.get("gate_traj", ()), dtype=float) for r in rows]
    gate_late = _finite_mean([gate_opening(g, len(g) // 2) for g in gates])
    return {
        "gate_late": gate_late,
        "violations": violations,
        "viol_worst": float(max(per_task_viol)) if per_task_viol else 0.0,
        "viol_per_task": per_task_viol,
        "eth_per_task": per_task_eth,
        "reward": float(np.mean([r["reward"] for r in rows])),
        "on_spec_ethanol": on_spec,
        "score": on_spec - float(viol_price) * violations,
        "purity_rate": float(np.mean([r["purity_ok"] for r in rows])),
        "z_spread": sep["spread"],
        "z_post_std": sep["post_std"],
        "z_snr": sep["snr"],
    }


def branch_name(kind, magnitude):
    """A held-out panel branch's key, e.g. ``"feed_gain@1.4"``."""
    return f"{kind}@{magnitude:g}"


def blended_heldout_score(env, agent, tasks, z_mode="mean", disturbed=True,
                          pool=None, gate_skip=()):
    """Held-out score blending the undisturbed set with HELDOUT_DISTURBANCES.

    The lean study's blend (see its docstring): ``violations``, ``reward``,
    ``on_spec_ethanol`` and ``score`` are the blend, ``viol_worst`` the worst
    single episode anywhere, ``gate_violations`` / ``gate_viol_worst`` what
    the feasibility gate reads, ``panel`` each branch's own numbers.
    """
    faults = tuple(HELDOUT_DISTURBANCES) if disturbed else ()
    onset = env.max_steps // 2
    price = violation_price(env)
    if pool is not None:
        per_task = pool.run(agent, tasks, faults, onset, z_mode)
    else:
        per_task = [branched_rollouts(env, agent, t, faults, onset, z_mode,
                                      index=i)
                    for i, t in enumerate(tasks)]
    base = _score_rows([rows[0] for rows in per_task], price)
    out = dict(base)
    for k in ("violations", "reward", "on_spec_ethanol", "score",
              "viol_worst", "viol_per_task", "eth_per_task", "gate_late"):
        out[f"undisturbed_{k}"] = base[k]
    if not disturbed:
        for k in ("violations", "reward", "on_spec_ethanol", "score",
                  "viol_worst"):
            out[f"disturbed_{k}"] = base[k]
        out["gate_violations"] = base["violations"]
        out["gate_viol_worst"] = base["viol_worst"]
        out["panel"] = {}
        return out
    panel = [_score_rows([rows[k + 1] for rows in per_task], price)
             for k in range(len(faults))]
    names = [branch_name(f[0], f[1]) for f in faults]
    gated = [base] + [p for f, name, p in zip(faults, names, panel)
                      if f[0] in GATE_FAULT_KINDS and name not in gate_skip]
    w = HELDOUT_DISTURB_WEIGHT
    for k in ("violations", "reward", "on_spec_ethanol", "score"):
        d = float(np.mean([p[k] for p in panel]))
        out[f"disturbed_{k}"] = d
        out[k] = (1.0 - w) * base[k] + w * d
    d_worst = float(max(p["viol_worst"] for p in panel))
    out["disturbed_viol_worst"] = d_worst
    out["viol_worst"] = max(base["viol_worst"], d_worst)
    out["gate_violations"] = max(g["violations"] for g in gated)
    out["gate_viol_worst"] = max(g["viol_worst"] for g in gated)
    out["panel"] = {name: {k: p[k] for k in (
                        "violations", "viol_worst", "on_spec_ethanol", "score",
                        "viol_per_task", "eth_per_task", "gate_late")}
                    for name, p in zip(names, panel)}
    return out


def _gap_without_worst_task(gaps):
    """Mean per-task gap with the largest one left out."""
    g = np.sort(np.asarray(gaps, dtype=float))
    return float(g[:-1].mean()) if g.size > 1 else float(g.sum())


def _branch_fault_code(cause):
    """The fault code a NO_REGRESS cause or branch name belongs to (0: undisturbed)."""
    branch = cause.rsplit(":", 1)[0] if ":" in cause else cause
    kind = branch.split("@", 1)[0]
    kind = DISTURBANCE_ALIASES.get(kind, kind)
    return int(DISTURBANCE_KIND_INDEX.get(kind, 0))


def _anchor_drift(regress, loaded_misses):
    """The NO_REGRESS causes of one check that move the anchor weights.

    Drift is a regression the loaded policy did not have at the first check;
    ``:tail`` causes reject a check but never ratchet its kind's anchor.
    """
    return [c for c in regress
            if c not in loaded_misses
            and not c.endswith(":tail")
            and not (c.endswith(":eth")
                     and ({c[:-4] + ":viol", c[:-4] + ":tail"}
                          & set(loaded_misses)))]


def _adapt_anchor(agent, drift):
    """Move the anchor weights after a held-out check, per fault kind."""
    def moved(w, hit):
        return float(np.clip(w * (META_ANCHOR_UP if hit else META_ANCHOR_DOWN),
                             META_ANCHOR_MIN, META_ANCHOR_MAX))
    hit = {_branch_fault_code(c) for c in drift}
    kinds = agent.anchor_kind_weights_np()
    agent.set_anchor_weights(
        pre=moved(agent.anchor_weight, 0 in hit),
        kinds=[moved(float(w), k in hit) for k, w in enumerate(kinds)])


def _anchor_kind_log(agent):
    """The per-kind post-fault anchor weights, keyed by fault kind name."""
    w = agent.anchor_kind_weights_np()
    return {name: float(w[code]) for code, name in DISTURBANCE_INDEX_MAP.items()}


def _regressions(ho, ref):
    """Held-out branches on which a check is worse than a reference check.

    Task by task, with the NO_REGRESS_* tolerances (``viol`` / ``eth``
    worst task left out, ``tail`` any single task); see the lean study.

    Returns:
        List of ``"branch:axis"`` strings, empty when nothing regresses.
    """
    def branches(s):
        yield ("undisturbed",
               s.get("undisturbed_viol_per_task", [s["undisturbed_violations"]]),
               s.get("undisturbed_eth_per_task",
                     [s["undisturbed_on_spec_ethanol"]]))
        for name, p in s.get("panel", {}).items():
            yield (name, p.get("viol_per_task", [p["violations"]]),
                   p.get("eth_per_task", [p["on_spec_ethanol"]]))

    have = {name: (v, e) for name, v, e in branches(ho)}
    out = []
    for name, v_ref, e_ref in branches(ref):
        if name not in have:
            continue
        v, e = (np.asarray(x, dtype=float) for x in have[name])
        v_ref, e_ref = (np.asarray(x, dtype=float) for x in (v_ref, e_ref))
        if v.shape != v_ref.shape or e.shape != e_ref.shape:
            raise ValueError(f"NO_REGRESS: branch {name} was scored on "
                             f"{v.size} tasks here and {v_ref.size} in the "
                             f"reference")
        tol_v = max(NO_REGRESS_VIOL_ABS, NO_REGRESS_VIOL_REL * v_ref.mean())
        if _gap_without_worst_task(v - v_ref) > tol_v:
            out.append(f"{name}:viol")
        tail = np.maximum(SELECT_MAX_WORST_VIOL, NO_REGRESS_TAIL_REL * v_ref)
        if np.any(v - v_ref > tail):
            out.append(f"{name}:tail")
        tol_e = (NO_REGRESS_ETH_REL_CLEAN
                 if v_ref.mean() <= NO_REGRESS_CLEAN_VIOL
                 else NO_REGRESS_ETH_REL)
        if _gap_without_worst_task(e_ref - e) > tol_e * e_ref.mean():
            out.append(f"{name}:eth")
    return out



_WORKER_AGENTS = {}


def _rollout_worker_init():
    """Worker setup: one torch thread each, since the workers run in parallel."""
    torch.set_num_threads(1)


def _rollout_job(job):
    """Run ``branched_rollouts`` for one task inside a worker process."""
    arm, sensor_model, weights, task, faults, onset, z_mode, index = job
    key = (arm, sensor_model)
    if key not in _WORKER_AGENTS:
        env = make_env(lagrangian=ARMS[arm].lagrangian,
                       sensor_model=sensor_model)
        _WORKER_AGENTS[key] = (env, _build_agent(env, arm))
    env, agent = _WORKER_AGENTS[key]
    agent.load_rollout_weights(weights)
    return branched_rollouts(env, agent, task, faults, onset, z_mode,
                             index=index)


class RolloutPool:
    """Worker processes running one arm's held-out rollouts, one task per job.

    CPU-only (``PLANT_DEVICE=cpu`` is set for the pool's lifetime). Each
    worker builds its own environment and agent once and receives the current
    rollout weights with every job, then runs the same code as the in-process
    path (the duals reset every episode, so no env carries state across
    episodes).
    """

    def __init__(self, arm, workers, sensor_model=None):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        self.arm = get_arm(arm).name
        self.sensor_model = bool(SENSOR_MODEL_ENABLED if sensor_model is None
                                 else sensor_model)
        self._prev_device = os.environ.get("PLANT_DEVICE")
        os.environ["PLANT_DEVICE"] = "cpu"
        self._ex = ProcessPoolExecutor(
            max_workers=int(workers), mp_context=mp.get_context("spawn"),
            initializer=_rollout_worker_init)

    def run(self, agent, tasks, faults, onset, z_mode):
        """Run ``branched_rollouts`` for every task with ``agent``'s weights."""
        weights = agent.rollout_weights()
        jobs = [(self.arm, self.sensor_model, weights, t, tuple(faults),
                 onset, z_mode, i) for i, t in enumerate(tasks)]
        return list(self._ex.map(_rollout_job, jobs))

    def close(self):
        """Shut the workers down and restore PLANT_DEVICE."""
        self._ex.shutdown(wait=True)
        if self._prev_device is None:
            os.environ.pop("PLANT_DEVICE", None)
        else:
            os.environ["PLANT_DEVICE"] = self._prev_device
