"""Training meta5: metarl's meta arm, from scratch, under the known change.

metarl's loop (``metarl.training.train._train``) less everything the warm
start brings (actor warm-up, anchor, gate, no-regression reference), and
with two changes: every training episode carries the change, its magnitude
drawn from the training distribution, and the held-out checks branch each
held-out task at the onset into HELDOUT_MAGNITUDES, all inside that
distribution. The unchanged branch (m = 1, far outside it) is logged, never
selected on.
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch

try:
    import tqdm
    _trange = tqdm.trange
except ImportError:                                    # pragma: no cover
    _trange = range

# ─── Local ────────────────────────────────────────────────────────────────────
import five_reactor  # noqa: F401  (selects the layout before metarl loads)
from metarl.env.config import (CTX_DOWNSTREAM, TASK_CONTEXT_CENTER,
                               TASK_CONTEXT_SCALE, sample_task)
from metarl.agent.replay import MultiTaskReplayBuffer
from metarl.training.arms import _build_agent, arm_paths
from metarl.training.config import (APN_REFINE_EVERY, APN_REFINE_EPISODES,
                                    BATCH_SIZE, CONTEXT_SIZE,
                                    COLLECT_IN_WORKERS,
                                    EARLY_STOP_DIVERGE_FACTOR,
                                    EARLY_STOP_DIVERGE_MIN_CHECKS,
                                    EARLY_STOP_DIVERGE_PATIENCE,
                                    EARLY_STOP_DIVERGE_REL_DROP,
                                    EARLY_STOP_DIVERGE_VIOL_FLOOR,
                                    EARLY_STOP_ENABLED, EARLY_STOP_MIN_ITERS,
                                    EARLY_STOP_MIN_REL_DELTA,
                                    EARLY_STOP_REL_FLOOR, HELDOUT_EVAL_EVERY,
                                    HELDOUT_EVAL_TASKS, HELDOUT_TASK_SEED,
                                    HELDOUT_Z_MODE, KL_ANNEAL_ITERS,
                                    META_BATCH, META_COLLECT_EPISODES_PER_ITER,
                                    META_ITERATIONS, META_KL_LAMBDA, MIN_BUFFER,
                                    N_TRAIN_TASKS, POLICY_DIR, SEED,
                                    UPDATES_PER_ITER, WARMUP_EPISODES_PER_TASK)
from metarl.training.early_stopping import EarlyStopper
from metarl.training.heldout import _score_rows, violation_price
from metarl.training.logs import _write_log
from metarl.training.metrics import latent_separability
from metarl.training.rollouts import branched_rollouts
from metarl.training.tasks import (make_arm_env, make_corner_tasks,
                                   make_task_pool, seed_everything)
from metarl.training.train import _Collector, _open_workers, _print_header
from five_reactor.arms import SPEC
from five_reactor.change import change_onset, change_schedule, sample_magnitude
from five_reactor.config import (CHANGE_KIND, CHANGE_MEAN, CHANGE_TRAIN_SD,
                                 EARLY_STOP_PATIENCE, HELDOUT_MAGNITUDES,
                                 SELECT_MAX_VIOL, SELECT_MAX_WORST_VIOL)

# Update diagnostics logged every iteration (NaN when an update has none).
UPDATE_STATS = ("critic_loss", "actor_loss", "kl", "aux_loss",
                "fault_cls_loss", "map_loss", "safety_loss", "alpha", "z_norm")
HELDOUT_STATS = ("violations", "viol_worst", "reward", "on_spec_ethanol",
                 "score", "purity_rate", "z_snr", "m_mae",
                 "unchanged_violations", "unchanged_on_spec_ethanol")


def decode_magnitude(agent, z_mu):
    """The aux head's reading of the downstream draw (x nominal) from a posterior mean.

    The aux head regresses the standardised effective task descriptor, whose
    ``downstream_rate`` entry is the change's magnitude once it has acted.
    """
    if agent.aux is None:
        return float("nan")
    dev = next(agent.aux.parameters()).device
    z = torch.as_tensor(np.asarray(z_mu, dtype=np.float32), device=dev)
    with torch.no_grad():
        y = agent.aux(z.reshape(1, -1))[0, CTX_DOWNSTREAM].item()
    return float(y * TASK_CONTEXT_SCALE[CTX_DOWNSTREAM]
                 + TASK_CONTEXT_CENTER[CTX_DOWNSTREAM])


class _ChangeCollector(_Collector):
    """metarl's collector, every episode carrying the known change."""

    def _draw(self, tid):
        task = self.draw_task(tid)
        m = sample_magnitude(self.rng, "train")
        return (tid, task, change_schedule(m, int(self.env.max_steps)),
                int(self.job_rng.integers(2 ** 31)))


def heldout_check(env, agent, tasks, pool=None):
    """Held-out score over HELDOUT_MAGNITUDES, each branched at the onset.

    Returns:
        Dict: the branch means of violations, reward, on-spec ethanol, score
        and purity; the worst episode on any branch; the posterior SNR
        across all branches; the decoded draw's mean absolute error after
        the change (``m_mae``); the unchanged branch's violations and
        on-spec ethanol (diagnostic only); and each branch's own numbers.
    """
    faults = tuple((CHANGE_KIND, m, None) for m in HELDOUT_MAGNITUDES)
    onset = change_onset(env.max_steps)
    if pool is not None:
        per_task = pool.run(agent, tasks, faults, onset, HELDOUT_Z_MODE)
    else:
        per_task = [branched_rollouts(env, agent, t, faults, onset,
                                      HELDOUT_Z_MODE, index=i)
                    for i, t in enumerate(tasks)]
    price = violation_price()
    branches = [[rows[k + 1] for rows in per_task]
                for k in range(len(faults))]
    scored = [_score_rows(b, price) for b in branches]
    pooled = [r for b in branches for r in b]
    sep = latent_separability([r["z_mu"] for r in pooled],
                              [r["z_sigma_sq"] for r in pooled])
    err = [abs(decode_magnitude(agent, r["z_mu"]) - m)
           for m, b in zip(HELDOUT_MAGNITUDES, branches) for r in b]
    unchanged = _score_rows([rows[0] for rows in per_task], price)
    out = {k: float(np.mean([s[k] for s in scored]))
           for k in ("violations", "reward", "on_spec_ethanol", "score",
                     "purity_rate")}
    out.update(viol_worst=float(max(s["viol_worst"] for s in scored)),
               z_snr=sep["snr"], m_mae=float(np.mean(err)),
               unchanged_violations=unchanged["violations"],
               unchanged_on_spec_ethanol=unchanged["on_spec_ethanol"],
               panel={f"m={m:g}": {k: s[k] for k in (
                   "violations", "viol_worst", "on_spec_ethanol", "score")}
                      for m, s in zip(HELDOUT_MAGNITUDES, scored)})
    return out


class Selection:
    """The best held-out check: highest score among the feasible ones
    (mean and worst-episode violations within SELECT_*), or the fewest
    violations before any check was feasible."""

    def __init__(self):
        self.feasible, self.score, self.viol, self.iteration = (
            False, -float("inf"), float("inf"), None)

    def consider(self, ho, it):
        ok = (ho["violations"] <= SELECT_MAX_VIOL
              and ho["viol_worst"] <= SELECT_MAX_WORST_VIOL)
        better = ((ok and (not self.feasible or ho["score"] > self.score))
                  or (not ok and not self.feasible
                      and ho["violations"] < self.viol))
        if better:
            self.feasible = self.feasible or ok
            self.score, self.viol, self.iteration = (ho["score"],
                                                     ho["violations"], it)
        return better


def _early_stopper():
    if not EARLY_STOP_ENABLED:
        return None
    return EarlyStopper(
        min_checks=EARLY_STOP_MIN_ITERS, patience=EARLY_STOP_PATIENCE,
        min_rel_delta=EARLY_STOP_MIN_REL_DELTA, rel_floor=EARLY_STOP_REL_FLOOR,
        max_viol=SELECT_MAX_VIOL, max_worst_viol=SELECT_MAX_WORST_VIOL,
        diverge_factor=EARLY_STOP_DIVERGE_FACTOR,
        diverge_viol_floor=EARLY_STOP_DIVERGE_VIOL_FLOOR,
        diverge_rel_drop=EARLY_STOP_DIVERGE_REL_DROP,
        diverge_patience=EARLY_STOP_DIVERGE_PATIENCE,
        diverge_min_checks=EARLY_STOP_DIVERGE_MIN_CHECKS)


def _new_log():
    log = {k: [] for k in ("iter", "avg_reward", "avg_violations",
                           *UPDATE_STATS, "heldout_iter", "heldout_panel",
                           *(f"heldout_{k}" for k in HELDOUT_STATS))}
    log.update(arm=SPEC.name, config=SPEC.summary(),
               change={"kind": CHANGE_KIND, "mean": CHANGE_MEAN,
                       "train_sd": CHANGE_TRAIN_SD,
                       "heldout": list(HELDOUT_MAGNITUDES)})
    return log


def train(iterations=None):
    """Train meta5; returns the trained agent.

    Args:
        iterations: Override META_ITERATIONS (the smoke test runs a few).
    """
    spec = SPEC
    n_iters = META_ITERATIONS if iterations is None else int(iterations)
    seed_everything(SEED)
    env = make_arm_env(spec)
    agent = _build_agent(env, spec, train_apn_if_missing=True)
    agent.enable_refiner(APN_REFINE_EVERY, APN_REFINE_EPISODES, seed=SEED)
    replay = MultiTaskReplayBuffer(env.stateDim, env.actionDim,
                                   priv_dim=agent.critic_privileged_dim)
    task_rng = np.random.default_rng(SEED + 2)
    heldout = (make_task_pool(HELDOUT_EVAL_TASKS, seed=HELDOUT_TASK_SEED)
               + make_corner_tasks())

    _print_header(spec, env, agent, N_TRAIN_TASKS,
                  META_COLLECT_EPISODES_PER_ITER)
    print(f"  known change: {CHANGE_KIND} x m at hour {change_onset(env.max_steps)}"
          f", m ~ N({CHANGE_MEAN}, {CHANGE_TRAIN_SD}) in training; held-out "
          f"m = {', '.join(f'{m:g}' for m in HELDOUT_MAGNITUDES)}; from "
          f"scratch, no gate")
    log = _new_log()
    workers = _open_workers(spec)
    collector = _ChangeCollector(env, agent, replay, spec,
                                 lambda tid: sample_task(task_rng), None,
                                 np.random.default_rng(SEED + 1),
                                 workers if COLLECT_IN_WORKERS else None)
    collector.finish(collector.start(
        [t for t in range(N_TRAIN_TASKS) for _ in range(WARMUP_EPISODES_PER_TASK)]))
    stopper = _early_stopper()
    best = Selection()
    final_path, best_path = arm_paths(spec)
    reward_hist, viol_hist = [], []
    ho = {"violations": float("nan"), "viol_worst": float("nan"),
          "m_mae": float("nan")}

    pbar = _trange(1, n_iters + 1)
    for it in pbar:
        agent.kl_lambda = META_KL_LAMBDA * min(1.0, it / KL_ANNEAL_ITERS)
        started = collector.start([int(np.random.randint(N_TRAIN_TASKS))
                                   for _ in range(META_COLLECT_EPISODES_PER_ITER)])
        if started[1] is None:
            for r, v in collector.finish(started):
                reward_hist.append(r)
                viol_hist.append(v)
        stats = None
        for u in range(UPDATES_PER_ITER):
            stats = agent.update(replay, meta_batch=META_BATCH,
                                 batch_size=BATCH_SIZE,
                                 context_size=CONTEXT_SIZE,
                                 min_buffer=MIN_BUFFER,
                                 want_stats=(u == UPDATES_PER_ITER - 1))
        if started[1] is not None:
            for r, v in collector.finish(started):
                reward_hist.append(r)
                viol_hist.append(v)
        avg_r = float(np.mean(reward_hist[-50:]))
        avg_v = float(np.mean(viol_hist[-50:]))

        stop, reason = False, ""
        if it == 1 or it % HELDOUT_EVAL_EVERY == 0:
            try:
                ho = heldout_check(env, agent, heldout, workers)
            except Exception as exc:
                if workers is None:
                    raise
                print(f"\n[held-out] worker pool failed ({exc!r}); "
                      f"continuing in-process")
                workers.close()
                workers = collector.pool = None
                ho = heldout_check(env, agent, heldout)
            log["heldout_iter"].append(it)
            log["heldout_panel"].append(ho["panel"])
            for k in HELDOUT_STATS:
                log[f"heldout_{k}"].append(ho[k])
            if best.consider(ho, it):
                os.makedirs(POLICY_DIR, exist_ok=True)
                agent.save(best_path)
            if stopper is not None:
                stop, reason = stopper.should_stop(
                    it, ho["reward"], ho["violations"], ho["viol_worst"],
                    ho["score"])

        if stats is not None:
            log["iter"].append(it)
            log["avg_reward"].append(avg_r)
            log["avg_violations"].append(avg_v)
            for k in UPDATE_STATS:
                log[k].append(stats.get(k, float("nan")))
            if hasattr(pbar, "set_postfix"):
                post = {"Rwd": f"{avg_r:.1f}", "Viol": f"{avg_v:.1f}",
                        "hoV": f"{ho['violations']:.1f}",
                        "hoW": f"{ho['viol_worst']:.0f}",
                        "mErr": f"{ho['m_mae']:.3f}"}
                if stopper is not None:
                    post["Pat"] = stopper.patience_bar()
                pbar.set_postfix(post)
        if it % 100 == 0:
            os.makedirs(POLICY_DIR, exist_ok=True)
            agent.save(final_path)
            _write_log(log, spec.label)
        if stop:
            print(f"\n[early stop] {reason}")
            break

    if workers is not None:
        workers.close()
    os.makedirs(POLICY_DIR, exist_ok=True)
    agent.save(final_path)
    log["selected_iter"] = best.iteration
    log["selected_feasible"] = best.feasible
    _write_log(log, spec.label)
    print(f"[{spec.label}] selected iter {best.iteration} (held-out score "
          f"{best.score:.4e}, {best.viol:.2f} violations"
          + ("" if best.feasible else "; NEVER FEASIBLE: selected on "
             "violations alone, calibrate SELECT_MAX_*") + ")")
    from five_reactor.plots import plot_training
    plot_training()
    return agent
