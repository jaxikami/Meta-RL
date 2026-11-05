"""Training one arm: collection, updates, held-out checks, selection."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os
from dataclasses import dataclass, field

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch

try:
    import tqdm
    _trange = tqdm.trange
except ImportError:                                    # pragma: no cover
    _trange = range

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import nominal_task, sample_task, N_REACTORS
from metarl.agent.common import LATENT_DIM, device
from metarl.agent.replay import MultiTaskReplayBuffer
from metarl.training.config import (APN_REFINE_EVERY, APN_REFINE_EPISODES,
                                    SEED, N_TRAIN_TASKS, META_ITERATIONS,
                                    META_HOLDOUT_KINDS,
                                    COLLECT_EPISODES_PER_ITER,
                                    META_COLLECT_EPISODES_PER_ITER,
                                    UPDATES_PER_ITER, META_BATCH, BATCH_SIZE,
                                    MIN_BUFFER, CONTEXT_SIZE,
                                    WARMUP_EPISODES_PER_TASK,
                                    HELDOUT_EVAL_EVERY, HELDOUT_EVAL_TASKS,
                                    HELDOUT_TASK_SEED, HELDOUT_WORKERS,
                                    HELDOUT_Z_MODE, USE_FILM, AUX_MODE,
                                    NET_ACTOR_HIDDEN, NET_CRITIC_HIDDEN,
                                    NET_CRITIC_TOKEN_HIDDEN, NET_MIXER_LAYERS,
                                    NET_MIXER_HEADS, NET_DEPTH,
                                    NET_ENCODER_HIDDEN, META_FAULT_PATTERNS,
                                    Z_INFORM_MIN_SNR, POLICY_DIR,
                                    EARLY_STOP_ENABLED, EARLY_STOP_MIN_ITERS,
                                    EARLY_STOP_PATIENCE,
                                    EARLY_STOP_MIN_REL_DELTA,
                                    EARLY_STOP_REL_FLOOR, SELECT_MAX_VIOL,
                                    SELECT_MAX_WORST_VIOL, GATE_FAULT_KINDS,
                                    EARLY_STOP_DIVERGE_FACTOR,
                                    EARLY_STOP_DIVERGE_VIOL_FLOOR,
                                    EARLY_STOP_DIVERGE_REL_DROP,
                                    EARLY_STOP_DIVERGE_PATIENCE,
                                    EARLY_STOP_DIVERGE_MIN_CHECKS,
                                    META_KL_LAMBDA, KL_ANNEAL_ITERS,
                                    META_ACTOR_WARMUP_ITERS,
                                    HELDOUT_WARMUP_EVAL_EVERY,
                                    META_ANCHOR_WEIGHT, META_ANCHOR_FAULT_FRAC,
                                    META_ANCHOR_FAULT_DIMS, GATE_LOSS_WEIGHT,
                                    GATE_SHARPNESS, HOLD_PENALTY_WEIGHT,
                                    HOLD_MARGIN, HOLD_GATE_TARGET,
                                    FINETUNE_FROM, COLLECT_IN_WORKERS)
from metarl.training.arms import (arm_paths, get_arm, _build_agent,
                                  write_holdout)
from metarl.training.early_stopping import EarlyStopper
from metarl.training.tasks import (seed_everything, make_arm_env, make_task_pool,
                                   corner_risk, make_corner_tasks,
                                   MagnitudeStratifier,
                                   maybe_schedule_disturbance)
from metarl.training.rollouts import collect_rollout, ingest_episode_record
from metarl.training.heldout import (blended_heldout_score, _anchor_drift,
                                     _adapt_anchor, _anchor_kind_log,
                                     _regressions)
from metarl.training.workers import RolloutPool
from metarl.training.logs import _write_log
from metarl.training.plots import plot_training_curves


# Update diagnostics logged every iteration, with the value logged when an
# update does not report one (None: always reported).
_UPDATE_STATS = (("critic_loss", None), ("actor_loss", None), ("kl", None),
                 ("aux_loss", None), ("fault_cls_loss", 0.0),
                 ("map_loss", 0.0), ("safety_loss", 0.0),
                 ("anchor_loss", 0.0),
                 ("anchor_gap_nominal", float("nan")),
                 ("anchor_gap_fault", float("nan")),
                 ("gate_loss", float("nan")), ("gate_nominal", float("nan")),
                 ("hold_loss", float("nan")), ("stall_frac", float("nan")),
                 ("gate_fault", float("nan")), ("alpha", None),
                 ("z_norm", None))


def _log_update(log, it, avg_r, avg_v, stats, pstats, gated):
    """Append one iteration's training diagnostics to ``log``."""
    log["iter"].append(it)
    log["avg_reward"].append(avg_r)
    log["avg_violations"].append(avg_v)
    for k, default in _UPDATE_STATS:
        v = stats[k] if default is None else stats.get(k, default)
        log[k].append(v if gated or k != "gate_loss" else float("nan"))
    for k in ("proj_noop_frac", "proj_mean_steps"):
        log[k].append(pstats[k])


class _Collector:
    """One arm's training episodes: drawn here, run in a worker pool (during
    the updates) or in-process, added to the replay.

    Jobs are drawn in the in-process order (task first, then the fault), so
    the path that runs an episode does not change what is drawn. A pool that
    fails is dropped and the episodes run in-process.
    """

    def __init__(self, env, agent, replay, spec, draw_task, strat, rng, pool):
        self.env, self.agent, self.replay = env, agent, replay
        self.multitask = spec.multitask
        self.patterns = META_FAULT_PATTERNS if spec.multitask else None
        self.draw_task, self.strat, self.rng = draw_task, strat, rng
        self.job_rng = np.random.default_rng(SEED + 3)
        self.pool = pool

    def _draw(self, tid):
        """One episode's (task id, task, fault schedule, seed)."""
        task = self.draw_task(tid)
        d = (maybe_schedule_disturbance(self.env, rng=self.rng, task_id=tid,
                                        strat=self.strat,
                                        patterns=self.patterns)
             if self.multitask else None)
        self.env.clear_disturbances()
        return (tid, task, [] if d is None else list(d["schedule"]),
                int(self.job_rng.integers(2 ** 31)))

    def start(self, tids):
        """Start one episode per task id; returns (jobs, futures or None)."""
        jobs = [self._draw(t) for t in tids]
        if self.pool is not None:
            try:
                return jobs, self.pool.collect_async(self.agent, jobs)
            except Exception as exc:                      # pragma: no cover
                print(f"\n[collect] worker pool failed ({exc!r}); "
                      f"collecting in-process")
                self.pool = None
        return jobs, None

    def finish(self, started):
        """Add ``start``'s episodes to the replay; (reward, viol) each."""
        jobs, futures = started
        out = []
        for k, (tid, task, sched, _) in enumerate(jobs):
            res = None
            if futures is not None:
                try:
                    res = ingest_episode_record(self.agent, self.replay,
                                                futures[k].result())
                except Exception as exc:                  # pragma: no cover
                    print(f"\n[collect] worker episode failed ({exc!r}); "
                          f"collecting in-process")
                    self.pool = None
            if res is None:
                res = collect_rollout(self.env, self.agent, self.replay, tid,
                                      task, schedule=sched)
            out.append(res[:2])
        return out


LOG_KEYS = ("iter", "avg_reward", "avg_violations", "critic_loss",
            "actor_loss", "kl", "aux_loss", "fault_cls_loss", "map_loss",
            "safety_loss", "anchor_loss",
            "alpha", "anchor_gap_nominal", "anchor_gap_fault",
            "gate_loss", "gate_nominal", "gate_fault", "hold_loss", "stall_frac",
            "z_norm", "proj_noop_frac", "proj_mean_steps",
            "heldout_iter", "heldout_violations", "heldout_reward",
            "heldout_on_spec_ethanol", "heldout_purity_rate",
            "heldout_z_snr", "heldout_score",
            "heldout_undisturbed_violations", "heldout_undisturbed_reward",
            "heldout_undisturbed_on_spec_ethanol",
            "heldout_undisturbed_score",
            "heldout_disturbed_violations", "heldout_disturbed_reward",
            "heldout_disturbed_on_spec_ethanol", "heldout_disturbed_score",
            "heldout_viol_worst", "heldout_undisturbed_viol_worst",
            "heldout_disturbed_viol_worst",
            "heldout_gate_violations", "heldout_gate_viol_worst",
            "heldout_panel", "heldout_undisturbed_per_task",
            "heldout_regressions", "heldout_anchor_weight",
            "heldout_anchor_kind_weights", "heldout_gate_late")

# Held-out results logged at every check, as "heldout_<key>".
_HELDOUT_STATS = ("violations", "reward", "on_spec_ethanol", "score",
                  "purity_rate", "z_snr", "undisturbed_violations",
                  "undisturbed_reward", "undisturbed_on_spec_ethanol",
                  "undisturbed_score", "disturbed_violations",
                  "disturbed_reward", "disturbed_on_spec_ethanol",
                  "disturbed_score", "viol_worst",
                  "undisturbed_viol_worst", "disturbed_viol_worst",
                  "gate_violations", "gate_viol_worst")


@dataclass
class _Selection:
    """The best held-out check so far, and how many checks each gate rejected."""
    viol: float = float("inf")
    eth: float = -float("inf")
    iteration: "int | None" = None
    score: float = float("nan")
    panel_viol: float = float("nan")
    worst: float = float("inf")
    feasible: bool = False
    tail_rejects: int = 0
    regress_rejects: int = 0
    regress_causes: dict = field(default_factory=dict)

    def consider(self, ho, regress, it):
        """Judge one held-out check; True when it is the new best.

        A check is feasible when its gated mean and worst-episode violations
        pass and it regresses nowhere. The best is the highest-scoring
        feasible check, or, before any check was feasible, the one with the
        fewest gated violations.
        """
        gate_v, gate_w = ho["gate_violations"], ho["gate_viol_worst"]
        mean_ok = gate_v <= SELECT_MAX_VIOL
        tail_ok = gate_w <= SELECT_MAX_WORST_VIOL
        feasible = mean_ok and tail_ok and not regress
        if mean_ok and not tail_ok:
            self.tail_rejects += 1
        if mean_ok and tail_ok and regress:
            self.regress_rejects += 1
            for cause in regress:
                self.regress_causes[cause] = self.regress_causes.get(cause, 0) + 1
        if feasible:
            better = (not self.feasible) or ho["score"] > self.score
        else:
            better = (not self.feasible) and gate_v < self.viol
        if better:
            self.feasible = self.feasible or feasible
            self.viol, self.eth, self.score, self.iteration = (
                gate_v, ho["on_spec_ethanol"], ho["score"], it)
            self.worst = gate_w
            self.panel_viol = ho["disturbed_violations"]
        return better


def _print_header(spec, env, agent, n_tasks, n_collect):
    """The arm, its networks and the plant, before training."""
    use_context = spec.use_context
    pc = agent.param_counts()
    dev_name = (f"{device} ({torch.cuda.get_device_name(device)})"
                if device.type == "cuda" else str(device))
    print(f"\n{'=' * 78}\nTRAIN {spec.label}  ({spec.blurb})\n"
          f"  {spec.summary()}  "
          f"tasks={'fresh per episode' if spec.multitask else n_tasks}"
          f"  film={USE_FILM}  aux={AUX_MODE if use_context else 'n/a'}  "
          f"device={dev_name}\n"
          f"  net: {N_REACTORS} reactor tokens, {NET_MIXER_LAYERS} attention "
          f"blocks x {NET_MIXER_HEADS} heads; actor token/trunk "
          f"{NET_ACTOR_HIDDEN} x{NET_DEPTH} ({pc['actor'] / 1e6:.2f}M); critics "
          f"token {NET_CRITIC_TOKEN_HIDDEN}, trunk {NET_DEPTH}x"
          f"{NET_CRITIC_HIDDEN} ({pc['critic'] / 1e6:.2f}M for the pair)"
          + (f"; GRU {NET_ENCODER_HIDDEN} ({pc['encoder'] / 1e6:.2f}M), "
             f"latent {LATENT_DIM}" if use_context else "") + "\n"
          + (f"  posterior: GRU over the episode (training window "
             f"{CONTEXT_SIZE}); aux also regresses privileged channels "
             f"{[c for c, _, _ in agent.aux_state_targets]}; critics read z + "
             f"the true task + the true plant state and clock "
             f"({agent.critic_privileged_dim} ch) (privileged, training "
             f"only)\n" if use_context else "")
          + f"{'=' * 78}")
    print(f"  plant: Fermentation-PPO's {type(env).__mro__[2].__name__}, "
          f"observed exactly; control interval "
          f"{env.control_action_interval} h")


def _warm_start(spec, agent, n_collect):
    """Load the warm-start source and set the adaptation options.

    Returns:
        Tuple (source checkpoint path or None, actor warm-up iterations).
    """
    if not spec.warm_start:
        return None, 0
    label = spec.label
    actor_warmup = max(1, META_ACTOR_WARMUP_ITERS)
    src_final, src_best = arm_paths(spec.warm_start)
    src = next((p for p in (src_best, src_final) if os.path.exists(p)), None)
    if src is None:
        raise SystemExit(
            f"[{label}] warm start needs a trained {spec.warm_start} "
            f"checkpoint ({src_best} or {src_final}). Train "
            f"{spec.warm_start} first ('all' trains it before "
            f"{spec.name}), or set warm_start=None in "
            f"ARMS['{spec.name}'].")
    try:
        agent.warm_start_from(src, anchor_weight=META_ANCHOR_WEIGHT,
                              anchor_fault_frac=META_ANCHOR_FAULT_FRAC,
                              decay_to_source=True,
                              anchor_fault_dims=META_ANCHOR_FAULT_DIMS)
    except ValueError as exc:
        raise SystemExit(
            f"[{label}] {exc} Run `python -m metarl "
            f"train_{spec.warm_start}` first, or `all`.") from None
    agent.freeze_obs_norm()
    if FINETUNE_FROM and spec.use_context:
        agent.load(FINETUNE_FROM)
        actor_warmup = 0
        print(f"  fine-tuning from {FINETUNE_FROM} (no actor warm-up)")
    if HOLD_PENALTY_WEIGHT > 0.0 and spec.use_context:
        agent.hold_penalty_weight = HOLD_PENALTY_WEIGHT
        agent.hold_margin = HOLD_MARGIN
        agent.hold_gate_target = HOLD_GATE_TARGET and agent.adapt_gate
        print(f"  hold rule: penalty {HOLD_PENALTY_WEIGHT:g}, margin "
              f"{HOLD_MARGIN:g}, gate target {agent.hold_gate_target}")
    print(f"  warm start: {src}; actor held for {actor_warmup} "
          f"iteration(s), anchor weight {META_ANCHOR_WEIGHT:g} "
          f"(x{META_ANCHOR_FAULT_FRAC:g} after a fault), adapted per fault "
          f"kind; no-regression selection on")
    print(f"  adapting actor: weight decay toward {spec.warm_start}; "
          f"exploration noise "
          + ("white" if agent.explore_noise_beta == 0.0 else
             f"colored (1/f^{agent.explore_noise_beta:g})"))
    if agent.adapt_gate:
        print(f"  adaptation gate: {spec.warm_start}'s policy acts until "
              f"a detector on z (fitted to the privileged fault flag, "
              f"weight {GATE_LOSS_WEIGHT:g}) opens it; sharpness "
              f"{GATE_SHARPNESS:g}"
              + ("; training rollouts act through the actor everywhere "
                 "(gate open)" if agent.train_gate_open else ""))
    if agent.fault_head is not None:
        print(f"  fault-kind classifier on z: class-balanced, weight "
              f"{agent.fault_cls_weight:g} (privileged target, training "
              f"only)")
    if spec.use_context:
        print(f"  {n_collect} episodes per iteration; encoder steps on "
              f"1 update in {agent.encoder_update_every}")
    return src, actor_warmup


def _noregress_reference(spec, env, src, heldout):
    """The warm-start source's held-out score, as it runs, for no-regression."""
    src_spec = get_arm(spec.warm_start)
    ref_agent = _build_agent(env, src_spec)
    ref_agent.load(src)
    print(f"  scoring the no-regression reference ({src_spec.name}, "
          f"apn={'on' if src_spec.apn else 'off'}) ...")
    ref = blended_heldout_score(env, ref_agent, heldout, z_mode=HELDOUT_Z_MODE,
                                disturbed=spec.multitask)
    print(f"  reference: undisturbed {ref['undisturbed_violations']:.2f}"
          f" viol / {ref['undisturbed_on_spec_ethanol']:.3e} on-spec "
          f"eth; panel {ref['disturbed_violations']:.2f} viol / "
          f"{ref['disturbed_on_spec_ethanol']:.3e} on-spec eth; "
          f"plant score {ref['score']:.4e}")
    return ref


def _gate_skip(spec, ref_score):
    """Panel checks the source itself fails the absolute gate on.

    Those are judged against the source by NO_REGRESS instead.
    """
    skip = set()
    if ref_score is None:
        return skip
    for name, p in ref_score["panel"].items():
        if (name.split("@")[0] in GATE_FAULT_KINDS
                and (p["violations"] > SELECT_MAX_VIOL
                     or p["viol_worst"] > SELECT_MAX_WORST_VIOL)):
            skip.add(name)
    if skip:
        print(f"  {spec.warm_start} fails the absolute gate on "
              + ", ".join(f"{n} ({ref_score['panel'][n]['violations']:.1f}"
                          f" mean, {ref_score['panel'][n]['viol_worst']:.0f}"
                          f" worst)" for n in sorted(skip))
              + "; judged there against it by NO_REGRESS instead")
    return skip


def _new_log(spec, gate_skip, src, ref_score):
    """An empty training log (``LOG_KEYS``) with the run's settings."""
    log = {k: [] for k in LOG_KEYS}
    log["arm"] = spec.name
    log["config"] = spec.summary()
    log["gate_skipped"] = sorted(gate_skip)
    if ref_score is not None:
        log["noregress_reference"] = {
            "source": src, "score": ref_score["score"],
            "undisturbed": {
                k: ref_score[f"undisturbed_{k}"]
                for k in ("violations", "on_spec_ethanol", "score",
                          "viol_per_task", "eth_per_task")},
            "panel": ref_score["panel"]}
    return log


def _open_workers(spec):
    """The held-out (and collection) worker pool, or None to run in-process."""
    if HELDOUT_WORKERS <= 1:
        return None
    try:
        workers = RolloutPool(spec, HELDOUT_WORKERS)
    except Exception as exc:                              # pragma: no cover
        print(f"  held-out checks: worker pool unavailable ({exc!r}); "
              f"running in-process")
        return None
    print(f"  held-out checks: {HELDOUT_WORKERS} worker processes"
          + ("; training episodes collected there too, during the "
             "updates" if COLLECT_IN_WORKERS else ""))
    return workers


def _early_stopper():
    """The early stopper, or None when ``EARLY_STOP_ENABLED`` is off."""
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


def _log_heldout(log, ho, it):
    """Append one held-out check's results to ``log`` (the regression-free part)."""
    for k in _HELDOUT_STATS:
        log[f"heldout_{k}"].append(ho[k])
    log["heldout_undisturbed_per_task"].append(
        {"viol": ho["undisturbed_viol_per_task"],
         "eth": ho["undisturbed_eth_per_task"]})
    log["heldout_iter"].append(it)


def _postfix(spec, agent, stats, pstats, avg_r, avg_v, ho, regress, stopper):
    """The progress bar's fields."""
    post = {"Rwd": f"{avg_r:.1f}", "Viol": f"{avg_v:.1f}",
            "hoV": f"{ho['gate_violations']:.1f}",
            "hoW": f"{ho['gate_viol_worst']:.0f}"}
    if os.environ.get("PLANT_VERBOSE_PBAR", "0") == "1":
        post.update({"QL": f"{stats['critic_loss']:.1f}",
                     "a": f"{stats['alpha']:.2f}"})
        if spec.use_context:
            post.update({"snr": f"{ho['z_snr']:.2f}",
                         "KL": f"{stats['kl']:.3f}",
                         "aux": f"{stats['aux_loss']:.3f}"})
        if spec.apn:
            post.update({"Pok": f"{pstats['proj_noop_frac']:.2f}"})
    if spec.warm_start and agent.adapt_gate:
        post.update({"gate": f"{stats['gate_nominal']:.2f}/"
                             f"{stats['gate_fault']:.2f}"})
    if spec.warm_start:
        post.update({"reg": str(len(regress))})
    if stopper is not None:
        post.update({"Pat": stopper.patience_bar()})
    return post


def _train(arm):
    """Train one arm; every switch comes from its ``ArmSpec``.

    Returns:
        The trained agent.
    """
    spec = get_arm(arm)
    label = spec.label
    n_collect = (META_COLLECT_EPISODES_PER_ITER if spec.multitask
                 else COLLECT_EPISODES_PER_ITER)
    seed_everything(SEED)
    env = make_arm_env(spec)
    agent = _build_agent(env, spec, train_apn_if_missing=True)
    agent.enable_refiner(APN_REFINE_EVERY, APN_REFINE_EPISODES, seed=SEED)
    replay = MultiTaskReplayBuffer(env.stateDim, env.actionDim,
                                   priv_dim=agent.critic_privileged_dim)
    rng = np.random.default_rng(SEED + 1)
    if spec.multitask:
        write_holdout()
        if META_HOLDOUT_KINDS:
            print(f"  held out of training (tier 3): "
                  f"{', '.join(META_HOLDOUT_KINDS)}")

    pool = (make_task_pool(N_TRAIN_TASKS, seed=0) if spec.multitask
            else [nominal_task()])
    task_rng = np.random.default_rng(SEED + 2)

    def draw_task(tid):
        """The task for one training episode stored in bucket ``tid``.

        Multitask arms draw a fresh prior task; the buckets only shard the
        replay.
        """
        return sample_task(task_rng) if spec.multitask else pool[tid]

    strat = MagnitudeStratifier()
    corner = make_corner_tasks()
    heldout = make_task_pool(HELDOUT_EVAL_TASKS, seed=HELDOUT_TASK_SEED) + corner
    if corner:
        print(f"  held-out pool: {HELDOUT_EVAL_TASKS} random + {len(corner)} "
              f"corner (risk " + ", ".join(f"{corner_risk(t):.2f}" for t in corner)
              + ")")

    _print_header(spec, env, agent, len(pool), n_collect)
    src, actor_warmup = _warm_start(spec, agent, n_collect)
    no_regress = bool(spec.warm_start)
    ref_score = (_noregress_reference(spec, env, src, heldout) if no_regress
                 else None)
    gate_skip = _gate_skip(spec, ref_score)
    log = _new_log(spec, gate_skip, src, ref_score)

    workers = _open_workers(spec)
    collector = _Collector(env, agent, replay, spec, draw_task, strat, rng,
                           workers if COLLECT_IN_WORKERS else None)
    collector.finish(collector.start(
        [tid for tid in range(len(pool))
         for _ in range(WARMUP_EPISODES_PER_TASK)]))
    stopper = _early_stopper()

    best = _Selection()
    regress = []
    loaded_misses = None
    reward_hist, viol_hist = [], []
    ho = {"violations": float("nan"), "z_snr": float("nan"),
          "gate_violations": float("nan"), "gate_viol_worst": float("nan")}
    final_path, best_path = arm_paths(spec)

    pbar = _trange(1, META_ITERATIONS + 1)
    for it in pbar:
        if spec.use_context:
            agent.kl_lambda = META_KL_LAMBDA * (1.0 if FINETUNE_FROM else min(1.0, it / KL_ANNEAL_ITERS))

        # This iteration's episodes run with the weights from before its
        # updates (in the workers, concurrently with them) and enter the
        # replay after them.
        started = collector.start([int(np.random.randint(len(pool)))
                                   for _ in range(n_collect)])
        if started[1] is None:
            for r, v in collector.finish(started):
                reward_hist.append(r)
                viol_hist.append(v)

        stats = None
        for u in range(UPDATES_PER_ITER):
            stats = agent.update(
                replay, meta_batch=META_BATCH if spec.multitask else 1,
                batch_size=BATCH_SIZE, context_size=CONTEXT_SIZE,
                min_buffer=MIN_BUFFER,
                want_stats=(u == UPDATES_PER_ITER - 1),
                train_actor=it > actor_warmup)
        if started[1] is not None:
            for r, v in collector.finish(started):
                reward_hist.append(r)
                viol_hist.append(v)

        avg_r = float(np.mean(reward_hist[-50:]))
        avg_v = float(np.mean(viol_hist[-50:]))

        fresh = False
        stop, reason = False, ""
        held = it <= actor_warmup
        every = HELDOUT_WARMUP_EVAL_EVERY if held else HELDOUT_EVAL_EVERY
        if it == 1 or it == actor_warmup or it % every == 0:
            try:
                ho = blended_heldout_score(env, agent, heldout,
                                           z_mode=HELDOUT_Z_MODE,
                                           disturbed=spec.multitask,
                                           pool=workers, gate_skip=gate_skip)
            except Exception as exc:
                if workers is None:
                    raise
                print(f"\n[held-out] worker pool failed ({exc!r}); "
                      f"continuing in-process")
                workers.close()
                workers = None
                collector.pool = None
                ho = blended_heldout_score(env, agent, heldout,
                                           z_mode=HELDOUT_Z_MODE,
                                           disturbed=spec.multitask,
                                           gate_skip=gate_skip)
            fresh = True
            _log_heldout(log, ho, it)
            regress = []
            if no_regress:
                regress = _regressions(ho, ref_score)
                if loaded_misses is None:
                    loaded_misses = set(regress)
                    log["noregress_loaded_misses"] = list(regress)
                    if regress:
                        print(f"\n[{label}] the loaded {spec.warm_start} "
                              f"policy, run as {spec.name} runs, falls short "
                              f"of the reference on: {', '.join(regress)}")
                elif agent.anchor_actor is not None and it > actor_warmup:
                    _adapt_anchor(agent, _anchor_drift(regress, loaded_misses))
            log["heldout_panel"].append(ho.get("panel", {}))
            log["heldout_regressions"].append(list(regress))
            log["heldout_anchor_weight"].append(agent.anchor_weight)
            log["heldout_anchor_kind_weights"].append(_anchor_kind_log(agent))
            log["heldout_gate_late"].append(
                {"undisturbed": ho["undisturbed_gate_late"],
                 **{n: p["gate_late"] for n, p in ho.get("panel", {}).items()}})
            if best.consider(ho, regress, it):
                os.makedirs(POLICY_DIR, exist_ok=True)
                agent.save(best_path)

        if stopper is not None and fresh and (it == 1 or not held):
            stop, reason = stopper.should_stop(
                it, ho["reward"], ho["gate_violations"], ho["gate_viol_worst"],
                ho["score"], extra_ok=not regress)

        if stats is not None:
            pstats = agent.proj_stats() if spec.apn else {
                "proj_noop_frac": float("nan"), "proj_mean_steps": float("nan")}
            _log_update(log, it, avg_r, avg_v, stats, pstats,
                        agent.adapt_gate)
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(_postfix(spec, agent, stats, pstats, avg_r,
                                          avg_v, ho, regress, stopper))

        if it % 100 == 0:
            os.makedirs(POLICY_DIR, exist_ok=True)
            agent.save(final_path)
            _write_log(log, label)

        if stop:
            print(f"\n[early stop] {reason}")
            break

    if workers is not None:
        workers.close()
    os.makedirs(POLICY_DIR, exist_ok=True)
    agent.save(final_path)
    _write_log(log, label)
    plot_training_curves()
    _print_summary(spec, agent, best, ref_score, gate_skip, log, ho)
    return agent


def _print_summary(spec, agent, best, ref_score, gate_skip, log, ho):
    """What selection picked, what each gate rejected, and warnings."""
    label, no_regress = spec.label, bool(spec.warm_start)
    if spec.multitask:
        gated = [k + (f" (vs {spec.warm_start})"
                      if any(n.startswith(k + "@") for n in gate_skip) else "")
                 for k in GATE_FAULT_KINDS]
        gate_src = "undisturbed + " + "/".join(gated)
    else:
        gate_src = "undisturbed"
    print(f"[{label}] best held-out ({gate_src} gate): violations={best.viol:.2f} "
          f"worst-episode={best.worst:.1f} on-spec ethanol={best.eth:.3e} "
          f"plant score={best.score:.4e}"
          + (f" panel violations={best.panel_viol:.2f}" if spec.multitask else "")
          + f" at iter {best.iteration}")
    if ref_score is not None:
        print(f"[{label}] {spec.warm_start} as it runs, on the same checks: "
              f"plant score={ref_score['score']:.4e} panel violations="
              f"{ref_score['disturbed_violations']:.2f}")
    print(f"[{label}] tail gate (SELECT_MAX_WORST_VIOL={SELECT_MAX_WORST_VIOL}) "
          f"rejected {best.tail_rejects} check(s) the mean band would have "
          f"admitted")
    if no_regress:
        top = sorted(best.regress_causes.items(), key=lambda kv: -kv[1])[:4]
        print(f"[{label}] no-regression gate rejected {best.regress_rejects} "
              f"check(s) the absolute gate admitted"
              + (": most often " + ", ".join(f"{k} x{n}" for k, n in top)
                 if top else "")
              + f"; final anchor weight {agent.anchor_weight:.3g} pre-fault")
        if best.feasible and best.iteration == 1:
            print(f"[{label}] WARNING: nothing improved on the loaded "
                  f"{spec.warm_start} policy without regressing somewhere; "
                  f"the selected checkpoint is that policy and ADAPT will "
                  f"read exactly 0.")
    if not best.feasible:
        print(f"[{label}] WARNING: no checkpoint was ever feasible (mean <= "
              f"{SELECT_MAX_VIOL} and worst <= {SELECT_MAX_WORST_VIOL}"
              + (f", and no regression vs {spec.warm_start}" if no_regress else "")
              + "); selection fell back to minimum-mean-violations. Treat this "
                "arm's checkpoint as selected on safety alone. If the plant's "
                "nominal optimum itself violates more than the gate allows, "
                "calibrate SELECT_MAX_VIOL / SELECT_MAX_WORST_VIOL.")
    if spec.use_context and np.isfinite(ho.get("z_snr", float("nan"))):
        print(f"[{label}] final latent separability SNR = {ho['z_snr']:.4f} "
              f"(gate wants > {Z_INFORM_MIN_SNR})")
    if agent.adapt_gate and best.iteration in log["heldout_iter"]:
        g = log["heldout_gate_late"][log["heldout_iter"].index(best.iteration)]
        print(f"[{label}] adaptation gate at the selected check (iter "
              f"{best.iteration}), mean opening over the second half: "
              + ", ".join(f"{k} {v:.2f}" for k, v in g.items()))
