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
from agent_pearl import (
    PEARLAgent, MultiTaskReplayBuffer, RunningMeanStd, product_of_gaussians,
    device,
)


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


def check_reward_scale_window():
    """RunningMeanStd stays a BOUNDED-window estimator and tracks a scale shift.

    Regression test for the primary divergence mechanism: with an unbounded
    count, the reward scale froze at its early-training value and every gradient
    in the run shrank with it.
    """
    rms = RunningMeanStd(max_count=1000)
    for _ in range(50):
        rms.update(np.random.randn(500) * 1.0)
    assert rms.count <= 1000 + 1e-3, rms.count
    std_small = rms.std
    assert 0.5 < std_small < 2.0, std_small
    # Now the reward distribution widens 50x (what happens when a policy
    # degrades and the Lagrangian penalties take over). The estimator must
    # FOLLOW it, not stay pinned to the old scale.
    for _ in range(50):
        rms.update(np.random.randn(500) * 50.0)
    assert rms.std > 10.0 * std_small, (std_small, rms.std)
    # Plain floats only (torch.load's weights_only unpickler rejects numpy
    # scalars inside a checkpoint).
    for v in (rms.mean, rms.var, rms.count):
        assert type(v) is float, (v, type(v))
    print("[OK] reward-scale estimator: bounded window + tracks scale shifts")


def check_posterior_consistency():
    """The online windowed posterior equals the batch posterior on the same set.

    The actor is trained on z drawn from ContextEncoder.forward over a context
    minibatch and deployed on z drawn from update_context's incremental
    accumulator. If those two disagree, the deployed z is off-distribution
    regardless of encoder quality — so assert they agree numerically.
    """
    env = _short_env()
    window = 8
    agent = PEARLAgent(env.stateDim, env.actionDim, n_reactors=env.n_reactors,
                       use_context=True, use_safeguard=True,
                       posterior_window=window)
    agent.z_mode = "mean"
    agent.clear_z()
    # Zero transitions => exactly the N(0, I) prior.
    assert torch.allclose(agent.z_mu, torch.zeros_like(agent.z_mu)), agent.z_mu
    assert torch.allclose(agent.z_sigma_sq, torch.ones_like(agent.z_sigma_sq))

    # Feed 3 * window transitions; only the LAST `window` may influence the
    # posterior.
    state = env.reset()
    trans = []
    for _ in range(3 * window):
        safe, _ = agent.select_action(state, deterministic=False)
        nxt, r, done, _ = env.step(safe)
        r_sum = float(np.sum(r))
        trans.append((state, safe, r_sum, nxt))
        agent.update_context(state, safe, r_sum, nxt)
        state = env.reset() if done else nxt

    flat = np.stack([
        agent._flatten_transition(o, a, rr / agent.ret_rms.std, o2)
        for (o, a, rr, o2) in trans[-window:]
    ])
    with torch.no_grad():
        mu_b, sig_b = agent.encoder(
            torch.as_tensor(flat, dtype=torch.float32, device=device))
    assert torch.allclose(agent.z_mu, mu_b, atol=1e-4), (agent.z_mu, mu_b)
    assert torch.allclose(agent.z_sigma_sq, sig_b, atol=1e-4)
    # Prior included => posterior can never be wider than the prior.
    assert torch.all(agent.z_sigma_sq <= 1.0 + 1e-6), agent.z_sigma_sq

    # Prior-inclusion is what makes the ZERO-factor case fall out of the formula
    # instead of needing a special case: with the prior, an empty context gives
    # sigma^2 == 1 (the prior itself); without it, 1/0 == inf.
    _, s_empty = product_of_gaussians(torch.zeros(0, 4), torch.ones(0, 4),
                                      include_prior=True)
    assert torch.allclose(s_empty, torch.ones_like(s_empty)), s_empty
    _, s_empty_np = product_of_gaussians(torch.zeros(0, 4), torch.ones(0, 4),
                                        include_prior=False)
    assert torch.all(torch.isinf(s_empty_np)), s_empty_np
    # One unit factor: 1/(1+1) with the prior, 1/1 without.
    _, s_one = product_of_gaussians(torch.zeros(1, 4), torch.ones(1, 4),
                                    include_prior=True)
    assert torch.allclose(s_one, torch.full_like(s_one, 0.5)), s_one
    print(f"[OK] online windowed posterior == batch posterior (window={window})")


def check_z_modes():
    """'mean' makes the meta policy genuinely deterministic; 'hold' freezes z.

    Under the old always-resample behaviour the meta agent's
    ``deterministic=True`` rollout was NOT deterministic, while the baseline's
    (z == 0) was — a confound in every meta-vs-baseline eval so far.
    """
    env = _short_env()
    agent = PEARLAgent(env.stateDim, env.actionDim, n_reactors=env.n_reactors,
                       use_context=True, use_safeguard=True,
                       posterior_window=8)

    def _traj(z_mode):
        """Run a few deterministic steps, returning (z, z_mu) per step."""
        agent.z_mode = z_mode
        agent.z_hold_after = 2
        env.set_task(nominal_task())
        state = env.reset()
        agent.clear_z()
        zs, mus = [], []
        for _ in range(6):
            a, _ = agent.select_action(state, deterministic=True)
            nxt, r, done, _ = env.step(a)
            agent.update_context(state, a, float(np.sum(r)), nxt)
            zs.append(agent.z.detach().clone())
            mus.append(agent.z_mu.detach().clone())
            state = env.reset() if done else nxt
        return zs, mus

    # 'mean': z IS the posterior mean at every step -> no sampling noise enters
    # the policy, so select_action(deterministic=True) is truly deterministic.
    zs, mus = _traj("mean")
    for zi, mi in zip(zs, mus):
        assert torch.allclose(zi, mi), "z_mode='mean' must use the posterior mean"

    # 'hold': z_hold_after=2 => steps 1,2 resample, then z is frozen even though
    # the posterior keeps updating underneath.
    zs, mus = _traj("hold")
    for zi in zs[2:]:
        assert torch.allclose(zi, zs[2]), "z_mode='hold' must freeze z"

    zs, _ = _traj("sample")
    assert not all(torch.allclose(zi, zs[0]) for zi in zs), \
        "z_mode='sample' should keep redrawing z (the old behaviour)"

    # 'zero': the ablation arm. z must be identically zero at every step even
    # though the posterior underneath is non-trivial, so the meta agent's latent
    # matches the baseline's exactly and the two differ only in training.
    zs, mus = _traj("zero")
    for zi in zs:
        assert torch.allclose(zi, torch.zeros_like(zi)), \
            "z_mode='zero' must ablate the latent entirely"
    assert not all(torch.allclose(mi, torch.zeros_like(mi)) for mi in mus), \
        "the posterior mean should still be tracked while z is ablated"

    # The baseline's z is pinned at zero in every mode.
    base = PEARLAgent(env.stateDim, env.actionDim, n_reactors=env.n_reactors,
                      use_context=False, use_safeguard=True)
    for m in ("sample", "mean", "hold"):
        base.z_mode = m
        base.clear_z()
        assert torch.allclose(base.z, torch.zeros_like(base.z))
    print("[OK] z modes: mean deterministic / hold frozen / sample redraws")


def check_action_masking():
    """Policy samples that reach the critic are stage-masked; logp excludes them.

    The replay buffer stores the MASKED executed action, so querying the critic
    with an unmasked a_pi (as the actor loss used to) evaluates it off the data
    distribution on every structurally locked dimension.
    """
    env = _short_env()
    agent = PEARLAgent(env.stateDim, env.actionDim, n_reactors=env.n_reactors,
                       use_context=True, use_safeguard=True)
    state = env.reset()
    obs = torch.as_tensor(np.stack([state, state]), dtype=torch.float32,
                          device=device)
    z = torch.zeros(2, agent._z_dim, device=device)

    mask = agent._stage_mask(obs)
    assert mask.shape == (2, env.n_reactors, env.actionDim), mask.shape
    assert torch.all((mask == 0) | (mask == 1))
    # Fresh reset => every reactor is in stage 0, where the mask is exactly
    # [time=free, LF=free, C5=locked, outflow=locked]. Asserted exactly rather
    # than as "at least one dim is locked", which is unfalsifiable here (the
    # stage one-hot can never make both C5 and outflow free at once).
    expected_mask = torch.tensor([1.0, 1.0, 0.0, 0.0],
                                 device=device).expand_as(mask)
    assert torch.allclose(mask, expected_mask), mask[0, 0]

    a, logp, n_active = agent._safe_sample(obs, z)
    assert a.shape == (2, env.n_reactors, env.actionDim), a.shape
    assert logp.shape == (2,), logp.shape
    assert n_active.shape == (2,), n_active.shape
    assert torch.allclose(n_active, mask.sum(dim=(-1, -2)))
    # Locked dims sit exactly at the SERL default, matching what the env
    # executed and what the buffer therefore stores.
    locked = (mask == 0)
    expected = agent._default_a.expand_as(a)[locked]
    assert torch.allclose(a[locked], expected, atol=1e-6)
    # Log-prob masking, with the sampling noise removed (deterministic=True
    # fixes u = mean) so this compares the masking itself, not two draws:
    #   all-ones mask must reproduce the unmasked log-prob exactly, and the real
    #   mask must change it (locked dims contribute 0 instead of a large,
    #   meaningless constant that would otherwise drive alpha and the TD target).
    _, logp_none = agent.actor.sample(obs, z, deterministic=True, mask=None)
    _, logp_ones = agent.actor.sample(obs, z, deterministic=True,
                                      mask=torch.ones_like(mask))
    _, logp_mask = agent.actor.sample(obs, z, deterministic=True, mask=mask)
    assert torch.allclose(logp_none, logp_ones, atol=1e-5), \
        "an all-ones mask must be a no-op on logp"
    assert not torch.allclose(logp_none, logp_mask), \
        "the stage mask must remove locked dims from the entropy accounting"

    # With the safeguard off, nothing is masked and n_active is the full count.
    plain = PEARLAgent(env.stateDim, env.actionDim, n_reactors=env.n_reactors,
                       use_context=True, use_safeguard=False)
    _, _, n_all = plain._safe_sample(obs, z)
    assert torch.allclose(
        n_all, torch.full((2,), float(env.actionDim * env.n_reactors),
                          device=device))
    print("[OK] action masking: critic inputs masked, entropy excludes locked dims")


def check_alpha_floor():
    """A configured alpha floor is respected by the temperature update.

    Regression test for the entropy collapse: alpha decayed 0.976 -> 0.055 with
    no rebound over 125 iterations, and the held-out optimum sat where alpha was
    still ~0.55.
    """
    env = _short_env()
    floor = 0.15
    agent = PEARLAgent(env.stateDim, env.actionDim, n_reactors=env.n_reactors,
                       use_context=False, use_safeguard=True, alpha_min=floor)
    replay = MultiTaskReplayBuffer(env.stateDim, env.actionDim, env.n_reactors)
    task = nominal_task()
    for _ in range(12):
        env.set_task(task)
        state = env.reset()
        agent.clear_z()
        done = False
        while not done:
            safe, _ = agent.select_action(state, deterministic=False)
            nxt, r, done, _ = env.step(safe)
            replay.add(0, state, safe, float(np.sum(r)), nxt, done)
            state = nxt

    # Drive log_alpha hard downward, then confirm the clamp holds it at the floor.
    with torch.no_grad():
        agent.log_alpha.fill_(-20.0)
    stats = agent.update(replay, meta_batch=1, batch_size=16, context_size=8)
    assert stats is not None
    assert agent.alpha.item() >= floor - 1e-6, agent.alpha.item()

    # With no floor, alpha may fall to the bare numerical bound.
    plain = PEARLAgent(env.stateDim, env.actionDim, n_reactors=env.n_reactors,
                       use_context=False, use_safeguard=True, alpha_min=None)
    with torch.no_grad():
        plain.log_alpha.fill_(-20.0)
    plain.update(replay, meta_batch=1, batch_size=16, context_size=8)
    assert plain.alpha.item() < floor, plain.alpha.item()

    # Per-dim entropy target overrides the total.
    tuned = PEARLAgent(env.stateDim, env.actionDim, n_reactors=env.n_reactors,
                       use_context=False, target_entropy_per_dim=-0.5)
    assert abs(tuned._entropy_per_dim + 0.5) < 1e-9, tuned._entropy_per_dim
    assert abs(tuned.target_entropy
               + 0.5 * env.actionDim * env.n_reactors) < 1e-9
    print(f"[OK] alpha floor respected ({floor}) and per-dim entropy target wired")


def check_early_stopper():
    """The stopper distinguishes divergence from a plateau.

    The old version reported "plateaued" after 700 iterations of monotone
    worsening (violations 7.9 -> 262), which is how the collapse stayed hidden.
    """
    from train_pearl_single_run import EarlyStopper

    st = EarlyStopper(min_checks=100, patience=100, min_delta=1.0, target=None,
                      diverge_factor=5.0, diverge_patience=3,
                      diverge_min_checks=6)
    stop, reason = False, ""
    for i, v in enumerate([300.0, 100.0, 8.0, 40.0, 120.0, 260.0, 270.0]):
        stop, reason = st.should_stop(i + 1, v)
        if stop:
            break
    assert stop and "DIVERGING" in reason, (stop, reason)

    # A genuine plateau must NOT be reported as divergence.
    st2 = EarlyStopper(min_checks=3, patience=3, min_delta=1.0, target=None,
                       diverge_factor=5.0, diverge_patience=3,
                       diverge_min_checks=6)
    for i in range(10):
        stop, reason = st2.should_stop(i + 1, 10.0)
        if stop:
            break
    assert stop and "plateaued" in reason, (stop, reason)
    print("[OK] early stopper: divergence vs plateau are distinguished")


def check_meta_disturbance_injection():
    """Meta-training disturbance injection fires at the configured rate.

    Regression test for the root cause of meta's disturb_mass_drift collapse:
    collect_rollout() only called env.set_task(), never
    env.schedule_disturbance(), so the encoder was trained exclusively on STATIC
    task draws and then evaluated on mid-episode regime changes it had never had
    a chance to represent.

    Asserts four separable things:
      1. rate  — p=1.0 always disturbs, p=0.0 NEVER disturbs, p=0.5 lands near
                 0.5 (a broken injector that silently no-ops would still pass a
                 "runs without error" check).
      2. range — every drawn onset/magnitude is inside the configured support,
                 and all three kinds are reachable.
      3. eval containment + NON-degeneracy — the eval literals lie strictly
                 inside the training ranges. Containment means eval is in-support;
                 strictness is what stops this from being memorisation of the
                 exact eval configuration, which would make a good mass_drift
                 number meaningless.
      4. wiring — collect_rollout(disturb=True) actually reaches the injector,
                 and disturb=False leaves the episode clean.
    """
    import train_pearl_single_run as T

    env = _short_env()
    rng = np.random.default_rng(0)

    # 1a. p=0.0 must never schedule anything, and must report None.
    for _ in range(200):
        got = T.maybe_schedule_disturbance(env, prob=0.0, rng=rng)
        assert got is None, got
        assert env._disturbances == [], env._disturbances

    # 1b. p=1.0 must schedule exactly one disturbance every time.
    kinds_seen = set()
    for _ in range(200):
        got = T.maybe_schedule_disturbance(env, prob=1.0, rng=rng)
        assert got is not None
        assert len(env._disturbances) == 1, env._disturbances
        # clear_disturbances() must run FIRST -- otherwise repeated episodes
        # accumulate schedules and later episodes get multiple disturbances.
        d = env._disturbances[0]
        assert d["kind"] == got["kind"] and d["onset"] == got["onset"]
        kinds_seen.add(got["kind"])

        # 2. Drawn values inside the configured support.
        lo, hi = T.META_DISTURB_MAG_RANGES[got["kind"]]
        assert lo <= got["magnitude"] <= hi, got
        f_lo, f_hi = T.META_DISTURB_ONSET_FRAC
        assert 1 <= got["onset"] <= env.max_steps - 1, got
        assert (int(f_lo * env.max_steps) - 1 <= got["onset"]
                <= int(round(f_hi * env.max_steps)) + 1), got
    assert kinds_seen == set(T.META_DISTURB_KINDS), kinds_seen

    # 1c. Empirical rate at p=0.5 over 2000 draws (3-sigma is ~0.034).
    p = 0.5
    n = 2000
    hits = sum(T.maybe_schedule_disturbance(env, prob=p, rng=rng) is not None
               for _ in range(n))
    rate = hits / n
    assert abs(rate - p) < 0.05, f"injection rate {rate:.3f} != {p}"

    # 3. Eval literals (see evaluate()'s scenarios) sit STRICTLY inside the
    # training ranges: in-support, but never the training distribution itself.
    eval_mags = {"inlet_step": 0.7, "enzyme_drop": 0.5, "mass_drift": -0.005}
    for kind, mag in eval_mags.items():
        lo, hi = T.META_DISTURB_MAG_RANGES[kind]
        assert lo < mag < hi, (kind, lo, mag, hi)
        assert hi - lo > 1e-6, (kind, lo, hi)   # not collapsed onto the literal
    f_lo, f_hi = T.META_DISTURB_ONSET_FRAC
    assert f_lo < 0.5 < f_hi, (f_lo, f_hi)      # eval onset == max_steps // 2
    assert set(eval_mags) == set(T.META_DISTURB_KINDS), \
        "every eval disturbance kind must be trainable"

    # 4. Wiring: collect_rollout must reach the injector when disturb=True and
    # must leave the env clean when it does not.
    agent = PEARLAgent(env.stateDim, env.actionDim, n_reactors=env.n_reactors,
                       use_context=True, use_safeguard=True)
    replay = MultiTaskReplayBuffer(env.stateDim, env.actionDim, env.n_reactors)
    task = sample_task(np.random.default_rng(3))
    prob0 = T.META_DISTURB_PROB
    try:
        T.META_DISTURB_PROB = 1.0
        T.collect_rollout(env, agent, replay, 0, task, disturb=True)
        assert len(env._disturbances) == 1, \
            "collect_rollout(disturb=True) did not schedule a disturbance"
        T.collect_rollout(env, agent, replay, 0, task, disturb=False)
        assert env._disturbances == [], \
            "collect_rollout(disturb=False) must leave the episode clean"
        # A 0%-probability config never disturbs even on the disturb=True path.
        T.META_DISTURB_PROB = 0.0
        for _ in range(3):
            T.collect_rollout(env, agent, replay, 0, task, disturb=True)
            assert env._disturbances == [], \
                "META_DISTURB_PROB=0 must never disturb an episode"
    finally:
        T.META_DISTURB_PROB = prob0
    env.clear_disturbances()
    env.set_task(None)
    print(f"[OK] meta disturbance injection: rate/ranges/wiring "
          f"(measured {rate:.3f} at p={p}, kinds={sorted(kinds_seen)})")


def check_severity_sweep_levels():
    """The severity sweep is anchored on eval, monotone, and extrapolates.

    Pure config invariants (no rollouts), guarding the three properties that make
    the sweep interpretable at all:
      1. level 0 of every kind IS the eval literal, so each sweep contains the
         current eval result as its first row and can be cross-checked against it;
      2. levels are strictly monotone in severity, so "first level that breaks"
         is a well-defined break POINT rather than an arbitrary hit;
      3. the last level lies outside META_DISTURB_MAG_RANGES, so the sweep
         actually probes past meta's training support — otherwise a "no break"
         result would only say the agents handle what they were trained on.
    """
    import train_pearl_single_run as T

    eval_mags = {"inlet_step": 0.7, "enzyme_drop": 0.5, "mass_drift": -0.005}
    assert set(T.SEVERITY_SWEEP_LEVELS) == set(T.META_DISTURB_KINDS), \
        (set(T.SEVERITY_SWEEP_LEVELS), set(T.META_DISTURB_KINDS))
    for kind, levels in T.SEVERITY_SWEEP_LEVELS.items():
        assert len(levels) >= 2, (kind, levels)
        # 1. anchored on the eval literal.
        assert abs(levels[0] - eval_mags[kind]) < 1e-12, (kind, levels[0])
        # 2. strictly monotone. For ALL THREE kinds a smaller number is more
        # severe (feeds/enzyme params are multiplied by magnitude, so smaller =
        # bigger cut; mass_drift is a signed per-step rate, so more negative =
        # faster shrinkage), hence one uniform direction check.
        assert all(b < a for a, b in zip(levels, levels[1:])), (kind, levels)
        # 3. escalates past meta's training support.
        lo, hi = T.META_DISTURB_MAG_RANGES[kind]
        assert not (lo <= levels[-1] <= hi), \
            f"{kind}: worst level {levels[-1]} is inside training support " \
            f"[{lo}, {hi}] — the sweep never leaves the trained regime"
    for arm in T.SEVERITY_SWEEP_AGENTS:
        assert arm in ("baseline", "meta", "meta_noz"), arm
    assert 0.0 < T.SEVERITY_BREAK_ETHANOL_FRAC < 1.0, T.SEVERITY_BREAK_ETHANOL_FRAC
    print("[OK] severity sweep levels: eval-anchored, monotone, extrapolating")


def check_observe_rewards():
    """observe_rewards moves the scale; update() must leave it untouched."""
    env = _short_env()
    agent = PEARLAgent(env.stateDim, env.actionDim, n_reactors=env.n_reactors,
                       use_context=False, use_safeguard=True)
    replay = MultiTaskReplayBuffer(env.stateDim, env.actionDim, env.n_reactors)
    task = nominal_task()
    for _ in range(12):
        env.set_task(task)
        state = env.reset()
        agent.clear_z()
        done = False
        while not done:
            safe, _ = agent.select_action(state, deterministic=False)
            nxt, r, done, _ = env.step(safe)
            replay.add(0, state, safe, float(np.sum(r)), nxt, done)
            state = nxt

    agent.observe_rewards([-100.0] * 500)
    before = (agent.ret_rms.count, agent.ret_rms.std)
    for _ in range(3):
        agent.update(replay, meta_batch=1, batch_size=16, context_size=8)
    after = (agent.ret_rms.count, agent.ret_rms.std)
    assert before == after, (before, after)
    print("[OK] reward scale updated only from collected data, not from update()")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    check_task_sampling()
    check_env_steps()
    check_reward_scale_window()
    check_posterior_consistency()
    check_z_modes()
    check_action_masking()
    check_observe_rewards()
    check_meta_disturbance_injection()
    check_severity_sweep_levels()
    check_alpha_floor()
    check_early_stopper()
    check_agent(use_context=True)
    check_agent(use_context=False)
    print("\nALL SMOKE CHECKS PASSED")
