"""The latent gate: does the meta arm's z carry task information it acts on?"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import json
import os

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.adapter import CONSEQUENCE_ONLY_FAULTS
from metarl.training.config import (SEED, Z_INFORM_MIN_SNR,
                                    Z_INFORM_MIN_ACTION_DELTA,
                                    Z_INFORM_MIN_PRIOR_SENS,
                                    Z_INFORM_MIN_METRIC_SPREAD,
                                    Z_INFORM_EPISODES, Z_INFORM_PROBE_STATES,
                                    Z_INFORM_DISTURBANCES, LOG_DIR)
from metarl.training.arms import (ARMS, BEST_META_PATH, META_PATH,
                                  _build_agent, _load_preferring_best)
from metarl.training.tasks import seed_everything, make_arm_env, make_task_pool
from metarl.training.rollouts import _start_episode, _advance, _episode_metrics
from metarl.training.evaluate import EVAL_ID_SEED
from metarl.training.metrics import (latent_separability, gate_opening,
                                     _finite_mean, _paired_delta,
                                     _fmt_delta_sci)
from metarl.training.logs import _json_default


def _z_rollout(env, agent, task, disturbance=None, z_mode="mean", fixed_z=None,
               keep_states=False, index=0):
    """Roll out once, optionally deploying a caller-supplied latent.

    Returns:
        The episode's metrics (``_episode_metrics``) plus ``states`` and
        ``times`` (each acted-on observation and its inner step; empty
        without ``keep_states``) and ``gate_times`` (every decision's step).
    """
    prev = agent.z_mode
    try:
        agent.z_mode = z_mode
        ep = _start_episode(env, agent, task, disturbance, index,
                            keep_states=keep_states)
        if z_mode == "fixed" and fixed_z is not None:
            agent.set_fixed_z(fixed_z)
        _advance(env, agent, ep)
        out = _episode_metrics(env, agent, ep)
    finally:
        agent.z_mode = prev
    times = np.asarray(ep["t_traj"], dtype=int)
    out.update(states=ep["states"] or [],
               times=times.tolist() if keep_states else [],
               gate_times=times,
               gate_labels=np.asarray(ep["label_traj"], dtype=bool))
    return out


def _probe_states(states, times, split, after, n):
    """Up to ``n`` observations from one side of an inner step, evenly spaced."""
    idx = [i for i, t in enumerate(times) if (t > split) == bool(after)]
    if len(idx) > n:
        idx = [idx[int(k)] for k in np.linspace(0, len(idx) - 1, n)]
    return [states[i] for i in idx]


def _actions_under_z(agent, states, z_vec, gate=None):
    """Re-score fixed observations under a fixed latent (optionally forcing the gate)."""
    if not states:
        return np.zeros((0, agent.action_dim)), np.zeros((0, agent.action_dim))
    # The policy is recurrent: every state is scored from the same snapshot
    # of its history state, which is restored afterwards.
    snap, prev_gate = agent.posterior_state(), agent.gate_override
    z_t = torch.as_tensor(np.asarray(z_vec, dtype=np.float32),
                          device=agent.z.device).reshape(-1)
    try:
        agent.gate_override = gate
        safe, raw = [], []
        for s in states:
            agent.set_posterior_state(snap)
            agent.z = z_t
            a, r = agent.select_action(s, deterministic=True)
            safe.append(a)
            raw.append(r)
        return np.stack(safe), np.stack(raw)
    finally:
        agent.set_posterior_state(snap)
        agent.gate_override = prev_gate


def actor_z_sensitivity(agent, task, n_states=Z_INFORM_PROBE_STATES, seed=0):
    """d|a|/d|z|: how far the actor's action moves per unit change of z.

    Scores ``n_states`` observations of one undisturbed episode (on a fresh
    environment, so no evaluated arm's duals move) under two prior draws of
    z with the gate forced open. Encoder separability alone passed a meta
    arm whose FiLM weights were exactly 0, since z was separable and unread.
    """
    env = make_arm_env(ARMS["meta"])
    m = _z_rollout(env, agent, task, None, z_mode="mean", keep_states=True)
    states = _probe_states(m["states"], m["times"], -1, True, n_states)
    rng = np.random.default_rng(seed)
    sens = []
    for s in states:
        z1 = rng.standard_normal(agent._z_dim).astype(np.float32)
        z2 = rng.standard_normal(agent._z_dim).astype(np.float32)
        a1, _ = _actions_under_z(agent, [s], z1, gate=1.0)
        a2, _ = _actions_under_z(agent, [s], z2, gate=1.0)
        sens.append(float(np.mean(np.abs(a1 - a2)))
                    / max(float(np.mean(np.abs(z1 - z2))), 1e-12))
    return float(np.mean(sens)) if sens else float("nan")


def z_informativeness(episodes=None):
    """Gate: does z carry task information the actor actually acts on?

    The lean study's five probes per scenario (separability, swapped-latent
    action change, prior-scale sensitivity, wrong / zeroed latent cost,
    metric spread), with probes 2 and 3 on observations after the onset.
    Faults that rotate over reactors resolve per task index.

    Returns:
        Dict of per-scenario probe results, also written to
        ``z_informativeness.json``.
    """
    seed_everything(SEED)
    n_ep = Z_INFORM_EPISODES if episodes is None else int(episodes)
    if n_ep < 2:
        raise SystemExit("z_informativeness needs at least 2 tasks")
    env = make_arm_env(ARMS["meta"])
    agent = _build_agent(env, "meta")
    _load_preferring_best(agent, BEST_META_PATH, META_PATH, "meta", strict=True)
    tasks = make_task_pool(n_ep, seed=EVAL_ID_SEED, ood=False)
    onset = env.max_steps // 2

    scenarios = {"undisturbed": None}
    for kind, mag, reactor in Z_INFORM_DISTURBANCES:
        scenarios[f"{kind}@{mag:g}"] = {"kind": kind, "onset": onset,
                                        "magnitude": mag, "reactor": reactor}
    out = {}
    swap = [(i + 1) % n_ep for i in range(n_ep)]
    print(f"z_informativeness: {n_ep} tasks x {len(scenarios)} scenarios")
    for sname, dist in scenarios.items():
        matched = [_z_rollout(env, agent, t, dist, z_mode="mean",
                              keep_states=True, index=i)
                   for i, t in enumerate(tasks)]
        z_mus = np.stack([m["z_mu"] for m in matched])
        sep = latent_separability(z_mus, [m["z_sigma_sq"] for m in matched])
        snr = sep["snr"]
        gate_pre = _finite_mean([
            gate_opening(m["gate_traj"][m["gate_times"] <= onset])
            for m in matched])
        # A consequence-only fault is invisible until its first alarm, so its
        # gate is read from then on.
        hidden = dist is not None and dist["kind"] in CONSEQUENCE_ONLY_FAULTS
        gate_post = _finite_mean([
            gate_opening(m["gate_traj"][m["gate_labels"] if hidden
                                        else m["gate_times"] > onset])
            for m in matched])
        gated = agent.adapt_gate

        def action_probes(after):
            d_safe, d_raw, d_zero, dz, sens = [], [], [], [], []
            rng = np.random.default_rng(0)
            for i, m in enumerate(matched):
                states = _probe_states(m["states"], m["times"], onset, after,
                                       Z_INFORM_PROBE_STATES)
                z1 = rng.standard_normal(agent._z_dim).astype(np.float32)
                z2 = rng.standard_normal(agent._z_dim).astype(np.float32)
                if not states:
                    continue
                a_own, r_own = _actions_under_z(agent, states, z_mus[i])
                a_oth, r_oth = _actions_under_z(agent, states, z_mus[swap[i]])
                a_zero, _ = _actions_under_z(agent, states,
                                             np.zeros_like(z_mus[i]))
                d_safe.append(float(np.mean(np.abs(a_own - a_oth))))
                d_raw.append(float(np.mean(np.abs(r_own - r_oth))))
                d_zero.append(float(np.mean(np.abs(a_own - a_zero))))
                dz.append(float(np.mean(np.abs(z_mus[i] - z_mus[swap[i]]))))
                a1, _ = _actions_under_z(agent, states, z1, gate=1.0)
                a2, _ = _actions_under_z(agent, states, z2, gate=1.0)
                denom = float(np.mean(np.abs(z1 - z2)))
                if denom > 0:
                    sens.append(float(np.mean(np.abs(a1 - a2))) / denom)

            def mean(xs):
                return float(np.mean(xs)) if xs else float("nan")
            return {"safe": mean(d_safe), "raw": mean(d_raw),
                    "zero": mean(d_zero), "dz": mean(dz),
                    "prior_sens": mean(sens)}

        post, pre = action_probes(True), action_probes(False)
        act_d_safe = post["safe"]
        prior_sens = post["prior_sens"]
        swapped = [_z_rollout(env, agent, tasks[i], dist, z_mode="fixed",
                              fixed_z=z_mus[swap[i]], index=i)
                   for i in range(n_ep)]
        zeroed = [_z_rollout(env, agent, t, dist, z_mode="zero", index=i)
                  for i, t in enumerate(tasks)]
        v_m = np.array([m["violations"] for m in matched], dtype=float)
        v_s = np.array([m["violations"] for m in swapped], dtype=float)
        v_z = np.array([m["violations"] for m in zeroed], dtype=float)
        e_m = np.array([m["on_spec_ethanol"] for m in matched], dtype=float)
        e_s = np.array([m["on_spec_ethanol"] for m in swapped], dtype=float)
        e_z = np.array([m["on_spec_ethanol"] for m in zeroed], dtype=float)
        swap_v = _paired_delta(v_s, v_m)
        zero_v = _paired_delta(v_z, v_m)
        swap_e = _paired_delta(e_m, e_s)
        zero_e = _paired_delta(e_m, e_z)
        v_spread = float(np.std(v_m))
        e_rel_spread = float(np.std(e_m) / max(abs(np.mean(e_m)), 1e-12))
        metric_ok = (v_spread > 0.5) or (e_rel_spread > Z_INFORM_MIN_METRIC_SPREAD)

        enc_ok = snr > Z_INFORM_MIN_SNR
        act_ok = prior_sens > Z_INFORM_MIN_PRIOR_SENS
        sens_ok = act_d_safe > Z_INFORM_MIN_ACTION_DELTA
        real = ((swap_v["significant"] and not swap_v["negligible"])
                or (swap_e["significant"] and not swap_e["negligible"]))
        if not enc_ok and not act_ok:
            verdict = "FAIL(encoder AND actor)"
        elif not enc_ok:
            verdict = "FAIL(encoder only)"
        elif not act_ok:
            verdict = "FAIL(actor ignores z)"
        elif not sens_ok:
            verdict = "FAIL(sensitivity)"
        elif not metric_ok:
            verdict = "INCONCLUSIVE(metric saturated)"
        elif not real:
            verdict = "INCONCLUSIVE(no measurable consequence)"
        else:
            verdict = "PASS"
        if gated and not gate_post >= 0.5:
            if dist is None:
                verdict = "BASE(gate shut: saferl acts, as designed undisturbed)"
            elif verdict == "PASS":
                verdict = (f"PASS(partial detection: gate open "
                           f"{gate_post:.0%} after onset)")
            else:
                verdict = (f"FAIL(gate: fault not detected, saferl acts; "
                           f"the cascade read {verdict})")
        print(f"  {sname:<26} SNR {snr:>5.2f}  gate(pre/post) "
              f"{gate_pre:.2f}/{gate_post:.2f}  d|a|/d|z| {prior_sens:.1e}  "
              f"wrong-z eth {_fmt_delta_sci(swap_e)}  -> {verdict}")
        out[sname] = {
            "n_tasks": n_ep, "disturbance": dist,
            "z_snr": snr, "z_snr_per_dim": sep["snr_per_dim"],
            "z_snr_mean_ratio": sep["mean_ratio"],
            "z_spread": sep["spread"], "z_post_std": sep["post_std"],
            "z_abs_mean": float(np.mean(np.abs(z_mus))),
            "z_mu_per_task": z_mus.tolist(),
            "gate_open_pre_onset": gate_pre,
            "gate_open_post_onset": gate_post,
            "gate_post_from_first_alarm": bool(hidden),
            "probe_split_step": int(onset),
            "action_delta_swapped_safe": post["safe"],
            "action_delta_swapped_raw": post["raw"],
            "action_delta_zero_safe": post["zero"],
            "swap_latent_distance": post["dz"],
            "action_sens_prior_scale": prior_sens,
            "action_delta_swapped_safe_pre_onset": pre["safe"],
            "action_sens_prior_scale_pre_onset": pre["prior_sens"],
            "viol_sd_across_tasks": v_spread,
            "on_spec_ethanol_rel_sd_across_tasks": e_rel_spread,
            "viol_matched": v_m.tolist(), "viol_swapped": v_s.tolist(),
            "viol_zeroed": v_z.tolist(),
            "on_spec_ethanol_matched": e_m.tolist(),
            "swap_cost_viol": swap_v, "zero_cost_viol": zero_v,
            "swap_cost_ethanol": swap_e, "zero_cost_ethanol": zero_e,
            "verdict": verdict,
        }
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, "z_informativeness.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=_json_default)
    print(f"\nwrote {path}")
    print("If the verdict is not PASS (or BASE undisturbed), ADAPT cannot be "
          "attributed to task inference, and a null ADAPT must not be "
          "reported as a negative result for meta-RL.")
    return out
