"""Vectorized single-reactor plant for labelling the APN's learned heads.

FermentationEnv's per-reactor physics (stage credits and transitions, cleanup
reset, discharge cut-off and interlock, feed and outflow limits, the RK4
kinetics) for many independent reactors at once, with no storage tank, shared
feedstock or reward: the learned heads (apn.KineticNet) only need what one
reactor does under its own held action.  check_against_env() verifies that it
reproduces FermentationEnv step for step.

Rollouts are piecewise constant like the agent's: an action is decided every
DECISION_H hours, masked by the reactor's stage at that moment (locked dims at
the defaults select_action() leaves them at) and, like the safety layer, its
feeds are cut so the reactor ends the hold at most at FEED_CAP x Mmax.
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os
from concurrent.futures import ProcessPoolExecutor

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import (
    BASE_CREDITS, C_C5_BASE, C_LF_BASE, C_MAX_LIMITS, CHARGE_MASS_KG, CONTROL_ACTION_INTERVAL,
    DISCHARGE_FLOOR_KG, FC5_NORM, FLF_NORM, HEAT_LOAD_SCALE, INIT_CELL_CONC, M_NORM,
    MAX_FLOW_OUT_REAC, N_INNER_STEPS, SPECIES_IDX, TIME_STEP_DURATION, UNLOAD_COMPLETE_KG,
    get_main_reactor_config
)
from metarl.env.kinetics import _ETA, _PH_G_FACTOR, _PH_X_FACTOR, calculate_rates_numba, integrate_rk4
from metarl.env.plant import DISCHARGE_CUTOFF

N_SUBSTEPS = N_INNER_STEPS
DECISION_H = CONTROL_ACTION_INTERVAL
FEED_CAP = 0.95                # the overflow head's wall
INIT_CELL = INIT_CELL_CONC
I_ETH, I_CO2, I_CELL = SPECIES_IDX['Eth'], SPECIES_IDX['CO2'], SPECIES_IDX['Cell']
NOMINAL_PARAMS = np.array([_ETA, _PH_G_FACTOR, _PH_X_FACTOR])
# Default of a stage-masked time dim (select_action's tanh(-0.34657359)).
MASKED_TIME = np.tanh(-0.34657359)


class Plant:
    """State of many independent reactors (all arrays share the first dim N).

    Attributes:
        C: Concentrations [N, 15].
        M: Mass [N] (kg).
        stage: Stage [N] (int).
        credits: Stage credits [N] (h).
        Mmax, FLF, FC5: Hardware limits [N] (kg, kg/s, kg/s).
        params: Kinetic modifiers [N, 3].
    """

    def __init__(self, C, M, stage, credits, Mmax, FLF, FC5, params=None):
        self.C = np.ascontiguousarray(C, dtype=np.float64)
        self.M = np.array(M, dtype=np.float64)
        self.stage = np.array(stage, dtype=np.int64)
        self.credits = np.array(credits, dtype=np.float64)
        self.Mmax = np.array(Mmax, dtype=np.float64)
        self.FLF = np.array(FLF, dtype=np.float64)
        self.FC5 = np.array(FC5, dtype=np.float64)
        n = self.M.size
        self.params = (np.tile(NOMINAL_PARAMS, (n, 1)) if params is None
                       else np.ascontiguousarray(params, dtype=np.float64))

    @classmethod
    def fresh(cls, Mmax, FLF, FC5, params=None):
        """Reactors as FermentationEnv.reset() leaves them."""
        n = len(Mmax)
        C = np.zeros((n, 15))
        C[:, I_CELL] = INIT_CELL
        return cls(C, np.full(n, CHARGE_MASS_KG), np.zeros(n, int), np.full(n, BASE_CREDITS[0]),
                   Mmax, FLF, FC5, params)

    @classmethod
    def from_obs(cls, obs, params=None):
        """Rebuild reactors from normalized observations [N, >=25] (exact on the nominal plant)."""
        obs = np.asarray(obs, dtype=np.float64)
        stage = obs[:, 16:21].argmax(1)
        Mmax = obs[:, 22] * M_NORM
        return cls(obs[:, :15] * C_MAX_LIMITS, obs[:, 15] * Mmax, stage,
                   obs[:, 21] * BASE_CREDITS[stage], Mmax, obs[:, 23] * FLF_NORM,
                   obs[:, 24] * FC5_NORM, params)

    def subset(self, idx):
        """A copy of reactors ``idx``."""
        return Plant(self.C[idx], self.M[idx], self.stage[idx], self.credits[idx], self.Mmax[idx],
                     self.FLF[idx], self.FC5[idx], self.params[idx])

    def obs(self):
        """Per-reactor observation features [N, 25] as get_state_norm() builds them."""
        return np.hstack([
            self.C / C_MAX_LIMITS, (self.M / self.Mmax)[:, None], np.eye(5)[self.stage],
            (self.credits / BASE_CREDITS[self.stage])[:, None], (self.Mmax / M_NORM)[:, None],
            (self.FLF / FLF_NORM)[:, None], (self.FC5 / FC5_NORM)[:, None],
        ]).astype(np.float32)

    def heat_load(self):
        """CO2 evolution rate per reactor (kg/h), as FermentationPlant.heat_load()."""
        return calculate_rates_numba(self.C, self.params)[:, I_CO2] * self.M * HEAT_LOAD_SCALE

    # ── Dynamics ──────────────────────────────────────────────────────────────

    def step(self, u):
        """One 1-hour inner step of FermentationEnv._step_internal, per reactor.

        Args:
            u: Commands in [0, 1], [N, 4] ((action + 1) / 2, as the env maps it).
        """
        u = u.copy()
        mult = u[:, 0] * 1.5 + 0.5
        u[u[:, 3] < DISCHARGE_CUTOFF, 3] = 0.0
        self.credits -= mult
        due = self.credits <= 0
        reset = due & (self.stage == 4)
        hold = due & (self.stage == 3) & (self.M >= UNLOAD_COMPLETE_KG)
        adv = due & ~reset & ~hold
        self.credits[hold] = 0.0
        self.stage[adv] += 1
        self.credits[adv] = BASE_CREDITS[self.stage[adv]]
        if np.any(reset):
            self.stage[reset] = 0
            self.M[reset] = CHARGE_MASS_KG
            self.C[reset] = 0.0
            self.C[reset, I_CELL] = INIT_CELL
            self.credits[reset] = BASE_CREDITS[0]
        u[self.stage != 3, 3] = 0.0

        f_lf = u[:, 1] * self.FLF
        f_c5 = u[:, 2] * self.FC5
        f_out = np.minimum(u[:, 3] * MAX_FLOW_OUT_REAC,
                           np.maximum(0.0, (self.M - DISCHARGE_FLOOR_KG) / TIME_STEP_DURATION))
        f_in = f_lf + f_c5
        # Mass discharged into storage this step (kg).
        self.last_out = f_out * TIME_STEP_DURATION
        c_in = np.zeros_like(self.C)
        fed = f_in > 0
        c_in[fed] = (f_c5[fed, None] * C_C5_BASE + f_lf[fed, None] * C_LF_BASE) / f_in[fed, None]
        self.C, self.M, _ = integrate_rk4(self.C, self.M, f_in, f_out, c_in, self.params,
                                          TIME_STEP_DURATION / N_SUBSTEPS, N_SUBSTEPS)

    def decide(self, action):
        """Mask and feed-cap a held action at a decision point.

        Args:
            action: Raw actions in [-1, 1], [N, 4].

        Returns:
            Commands in [0, 1], [N, 4], for the next DECISION_H inner steps.
        """
        u = (masked_action(self.stage, action).astype(np.float64) + 1.0) / 2.0
        hold_s = TIME_STEP_DURATION * DECISION_H
        feed = (u[:, 1] * self.FLF + u[:, 2] * self.FC5) * hold_s
        out = np.minimum(u[:, 3] * MAX_FLOW_OUT_REAC * hold_s, np.maximum(self.M - DISCHARGE_FLOOR_KG, 0.0))
        room = np.maximum(FEED_CAP * self.Mmax - self.M + out, 0.0)
        scale = np.where(feed > room, room / np.maximum(feed, 1e-9), 1.0)
        u[:, 1] *= scale
        u[:, 2] *= scale
        return u


def masked_action(stage, action):
    """Stage-mask raw actions as select_action() does (locked dims at their defaults).

    Args:
        stage: Stages [N].
        action: Raw actions [N, 4] in [-1, 1].

    Returns:
        Masked actions [N, 4] (float32).
    """
    live = np.stack([stage != 4, stage <= 1, stage == 1, stage == 3], axis=1)
    out = np.where(live, np.clip(action, -1.0, 1.0), -1.0)
    out[:, 0] = np.where(live[:, 0], out[:, 0], MASKED_TIME)
    return out.astype(np.float32)


# ─── Held-Action Rollouts ─────────────────────────────────────────────────────

def held_rollout(plant, action, heat_hours, max_hours=400):
    """Hold ``action`` (re-decided every DECISION_H hours) and record the learned heads' targets.

    Args:
        plant: Plant (modified in place).
        action: Raw actions [N, 4] in [-1, 1].
        heat_hours: Length of the heat-load profile to record.
        max_hours: Cap on the wait for unload.

    Returns:
        A tuple of heat [N, heat_hours] (kg/h at the end of each hour), the
        harvest titer [N] (titer standing when unload begins, i.e. what the
        first drain draws; NaN for reactors already in stage 3 or 4 or that
        never got there) and reached [N] (bool).
    """
    n = plant.M.size
    heat = np.zeros((n, heat_hours))
    titer = np.full(n, np.nan)
    pending = plant.stage <= 2
    reached = np.zeros(n, dtype=bool)
    active = np.arange(n)
    sub = plant
    u = None
    for t in range(max_hours):
        if t >= heat_hours:
            keep = pending[active]
            if not keep.any():
                break
            if not keep.all():
                sub = sub.subset(np.nonzero(keep)[0])
                u = u[keep]
                active = active[keep]
        if t % DECISION_H == 0:
            u = sub.decide(action[active])
        before = sub.stage.copy()
        titer_before = sub.C[:, I_ETH].copy()
        sub.step(u)
        entered = (before == 2) & (sub.stage == 3) & pending[active]
        if entered.any():
            idx = active[entered]
            titer[idx] = titer_before[entered]
            pending[idx] = False
            reached[idx] = True
        if t < heat_hours:
            heat[active, t] = sub.heat_load()
    return heat, titer, reached


def _label_chunk(job):
    """Worker: held_rollout on one chunk (module-level so processes can pickle it)."""
    obs, params, action, heat_hours, max_hours = job
    return held_rollout(Plant.from_obs(obs, params), action, heat_hours, max_hours)


def label(obs, params, action, heat_hours, max_hours=400, workers=None, chunk=4096):
    """held_rollout() for many (state, action) pairs, split over processes.

    Args:
        obs: Per-reactor observations [N, >=25].
        params: Kinetic modifiers [N, 3] (None: nominal).
        action: Raw actions [N, 4].
        heat_hours: Heat-profile length.
        max_hours: Cap on the wait for unload.
        workers: Processes (default: all cores; 1 runs in this process).
        chunk: Samples per job.

    Returns:
        heat [N, heat_hours], titer [N], reached [N], as held_rollout().
    """
    n = len(obs)
    if params is None:
        params = np.tile(NOMINAL_PARAMS, (n, 1))
    jobs = [(obs[i:i + chunk], params[i:i + chunk], action[i:i + chunk], heat_hours, max_hours)
            for i in range(0, n, chunk)]
    workers = workers or os.cpu_count() or 1
    if workers <= 1 or len(jobs) == 1:
        parts = [_label_chunk(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            parts = list(pool.map(_label_chunk, jobs))
    return tuple(np.concatenate(p) for p in zip(*parts))


# ─── Behaviour Trajectories (state coverage) ──────────────────────────────────

def sample_hardware(n, rng, wild_frac=0.2):
    """Reactor hardware: the main plant's three types, plus some outside them.

    Args:
        n: Number of reactors.
        rng: NumPy Generator.
        wild_frac: Share with log-uniform flow capacity (0.4-5x the base pump)
            and 0.6-1.4x the base volume, so the learned heads generalize to
            other plants (the scaling study).

    Returns:
        Mmax, FLF, FC5 arrays [n].
    """
    cfg = get_main_reactor_config()
    kind = rng.integers(0, 3, n)
    Mmax = cfg['Mmax'][kind].copy()
    FLF = cfg['max_flow_LF'][kind].copy()
    FC5 = cfg['max_flow_C5'][kind].copy()
    wild = rng.random(n) < wild_frac
    k = int(wild.sum())
    mult = np.exp(rng.uniform(np.log(0.4), np.log(5.0), k))
    FLF[wild] = 2487 / 3600 * mult
    FC5[wild] = 628 / 3600 * mult
    Mmax[wild] = 220000.0 * rng.uniform(0.6, 1.4, k)
    return Mmax, FLF, FC5


def behaviour_actions(n, rng):
    """Random raw actions spanning slow/fast schedules and low/high feeds.

    Returns:
        Actions [n, 4] in [-1, 1].
    """
    a = np.empty((n, 4))
    a[:, 0] = rng.uniform(-1.0, 1.0, n)
    mode = rng.random(n)
    feed = np.where(mode < 0.4, rng.uniform(-1.0, 1.0, n),
                    np.where(mode < 0.7, rng.uniform(0.3, 1.0, n), rng.uniform(-1.0, -0.3, n)))
    a[:, 1] = feed
    a[:, 2] = np.where(rng.random(n) < 0.5, feed, rng.uniform(-1.0, 1.0, n))
    a[:, 3] = np.where(rng.random(n) < 0.8, rng.uniform(0.0, 1.0, n), -1.0)
    return a


def _trajectory_chunk(job):
    """Worker: run behaviour trajectories and record the state at every decision."""
    n, hours, seed, fault_frac = job
    rng = np.random.default_rng(seed)
    params = np.tile(NOMINAL_PARAMS, (n, 1))
    Mmax, FLF, FC5 = sample_hardware(n, rng)
    plant = Plant.fresh(Mmax, FLF, FC5, params)
    # Kinetic faults the observation shows only through their effect: slower
    # hydrolysis / uptake from a random hour, and a one-off loss of viable cells.
    faulty = rng.random(n) < fault_frac
    onset = rng.integers(0, hours, n)
    scale = rng.uniform(0.2, 1.0, n)
    kill = faulty & (rng.random(n) < 0.3)
    obs, prm = [], []
    action = behaviour_actions(n, rng)
    u = None
    for t in range(hours):
        hit = faulty & (onset == t)
        if hit.any():
            plant.params[hit] = NOMINAL_PARAMS * scale[hit, None]
            plant.C[hit & kill, I_CELL] *= 0.5
        if t % DECISION_H == 0:
            switch = rng.random(n) < 0.35
            action[switch] = behaviour_actions(int(switch.sum()), rng)
            obs.append(plant.obs())
            prm.append(plant.params.copy())
            u = plant.decide(action)
        plant.step(u)
    return np.concatenate(obs), np.concatenate(prm)


def behaviour_states(n_reactors, hours=600, seed=0, fault_frac=0.2, workers=None, chunk=256):
    """Reactor states visited by random piecewise-constant behaviour.

    Args:
        n_reactors: Independent reactors to run (each for ``hours``).
        hours: Episode length.
        seed: Base seed.
        fault_frac: Share of reactors given a kinetic fault.
        workers: Processes (default: all cores).
        chunk: Reactors per job.

    Returns:
        A tuple of observations [N, 25] (one per reactor and decision) and
        the kinetic modifiers in force [N, 3].
    """
    jobs = [(min(chunk, n_reactors - i), hours, seed * 100003 + i, fault_frac)
            for i in range(0, n_reactors, chunk)]
    workers = workers or os.cpu_count() or 1
    if workers <= 1 or len(jobs) == 1:
        parts = [_trajectory_chunk(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            parts = list(pool.map(_trajectory_chunk, jobs))
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


# ─── Consistency Check ────────────────────────────────────────────────────────

def check_against_env(hours=600, seed=0):
    """Run FermentationEnv and Plant side by side on the same commands.

    Returns:
        Max absolute differences over the episode: dict with 'C', 'M',
        'stage', 'credits' and 'heat'.
    """
    from metarl.env.safe import FermentationEnv
    cfg = get_main_reactor_config()
    env = FermentationEnv(3, cfg['Mmax'], cfg['max_flow_LF'], cfg['max_flow_C5'])
    env.initial_lf = env.initial_c5 = 1e12   # unlimited feedstock, as Plant assumes
    env.reset()
    plant = Plant.fresh(cfg['Mmax'], cfg['max_flow_LF'], cfg['max_flow_C5'])
    rng = np.random.default_rng(seed)
    diff = {"C": 0.0, "M": 0.0, "stage": 0, "credits": 0.0, "heat": 0.0}
    action = behaviour_actions(3, rng)
    for t in range(hours):
        if t % DECISION_H == 0:
            action = behaviour_actions(3, rng)
            u = plant.decide(action)
        env._step_internal(u.flatten() * 2.0 - 1.0)
        plant.step(u)
        diff["C"] = max(diff["C"], float(np.abs(env.C - plant.C).max()))
        diff["M"] = max(diff["M"], float(np.abs(env.M - plant.M).max()))
        diff["stage"] = max(diff["stage"], int(np.abs(env.current_stage - plant.stage).max()))
        diff["credits"] = max(diff["credits"], float(np.abs(env.stage_credits - plant.credits).max()))
        diff["heat"] = max(diff["heat"], float(np.abs(env.heat_load() - plant.heat_load()).max()))
    return diff


if __name__ == "__main__":
    print(check_against_env())
