"""Worker processes for rollouts: held-out checks, training episodes, evaluation.

Each worker builds an environment and agent per arm once, receives the
current rollout weights with every job, and runs the same code as the
in-process path (the duals reset every episode, so no environment carries
state across episodes).
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os

# ─── Third-Party ──────────────────────────────────────────────────────────────
import torch

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.training.arms import ARMS, get_arm, _build_agent
from metarl.training.tasks import make_arm_env
from metarl.training.rollouts import (scenario_rollouts, branched_rollouts,
                                      collect_episode_record)

# Per-episode trajectories only a traced evaluation scenario reads; dropped
# from the others before they cross the process boundary.
TRACE_FIELDS = ("act_traj", "probe_traj", "storage_traj")

_WORKER_AGENTS = {}


def _rollout_worker_init():
    """Worker setup: one torch thread each, since the workers run in parallel."""
    torch.set_num_threads(1)


def _worker_agent(arm, weights):
    """This worker's environment and agent for ``arm``, with ``weights`` installed."""
    if arm not in _WORKER_AGENTS:
        env = make_arm_env(ARMS[arm])
        _WORKER_AGENTS[arm] = (env, _build_agent(env, arm))
    env, agent = _WORKER_AGENTS[arm]
    agent.load_rollout_weights(weights)
    return env, agent


def _collect_job(job):
    """Run one stochastic training episode inside a worker process."""
    arm, weights, task_id, task, schedule, seed = job
    env, agent = _worker_agent(arm, weights)
    return collect_episode_record(env, agent, task_id, task, schedule, seed)


def _rollout_job(job):
    """Run ``branched_rollouts`` for one task inside a worker process."""
    arm, weights, task, faults, onset, z_mode, index = job
    env, agent = _worker_agent(arm, weights)
    return branched_rollouts(env, agent, task, faults, onset, z_mode,
                             index=index)


def _scenario_job(job):
    """Run ``scenario_rollouts`` for one task inside a worker process."""
    arm, weights, task, schedules, z_mode, index, traced = job
    env, agent = _worker_agent(arm, weights)
    rows = scenario_rollouts(env, agent, task, schedules, z_mode, index=index)
    return [r if keep else {k: v for k, v in r.items() if k not in TRACE_FIELDS}
            for r, keep in zip(rows, traced)]


class RolloutPool:
    """Worker processes running rollouts: held-out checks and evaluation (one
    task per job) and stochastic training episodes (one episode per job).

    CPU-only (``PLANT_DEVICE=cpu`` is set for the pool's lifetime). ``arm``
    is the default for every call; evaluation passes each arm explicitly.
    """

    def __init__(self, arm, workers):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        self.arm = None if arm is None else get_arm(arm).name
        self._prev_device = os.environ.get("PLANT_DEVICE")
        os.environ["PLANT_DEVICE"] = "cpu"
        self._ex = ProcessPoolExecutor(
            max_workers=int(workers), mp_context=mp.get_context("spawn"),
            initializer=_rollout_worker_init)

    def _arm(self, arm):
        name = self.arm if arm is None else get_arm(arm).name
        if name is None:
            raise ValueError("this pool has no default arm: pass one")
        return name

    def run(self, agent, tasks, faults, onset, z_mode, arm=None):
        """Run ``branched_rollouts`` for every task with ``agent``'s weights."""
        weights = agent.rollout_weights()
        jobs = [(self._arm(arm), weights, t, tuple(faults),
                 onset, z_mode, i) for i, t in enumerate(tasks)]
        return list(self._ex.map(_rollout_job, jobs))

    def scenarios_async(self, agent, tasks, schedules, z_mode, traced=None,
                        arm=None):
        """Start ``scenario_rollouts`` of every task with ``agent``'s weights.

        Args:
            schedules: Fault schedules, the same for every task.
            traced: Optional booleans, one per schedule: keep that row's
                ``TRACE_FIELDS``. Default: keep every row's.

        Returns:
            Futures of per-task row lists, in task order.
        """
        weights = agent.rollout_weights()
        traced = (True,) * len(schedules) if traced is None else tuple(traced)
        return [self._ex.submit(_scenario_job,
                                (self._arm(arm), weights, t, list(schedules),
                                 z_mode, i, traced))
                for i, t in enumerate(tasks)]

    def collect_async(self, agent, jobs, arm=None):
        """Start training episodes with ``agent``'s current weights.

        Args:
            agent: The training agent (its weights are copied now).
            jobs: (task_id, task, schedule, seed) tuples.

        Returns:
            Futures of ``collect_episode_record`` results, in job order.
        """
        weights = agent.rollout_weights()
        return [self._ex.submit(_collect_job,
                                (self._arm(arm), weights) + tuple(j))
                for j in jobs]

    def close(self):
        """Shut the workers down and restore PLANT_DEVICE."""
        self._ex.shutdown(wait=True)
        if self._prev_device is None:
            os.environ.pop("PLANT_DEVICE", None)
        else:
            os.environ["PLANT_DEVICE"] = self._prev_device
