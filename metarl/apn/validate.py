"""Validation of the multi-head APN against the real plant.

Every check uses FermentationEnv or the vectorized plant (plant_sim), never the
APN's own predictions, as ground truth:

    1. Exact heads     the mass and storage balances they predict for a hold
                       against FermentationEnv stepping the same action.
    2. Learned heads   KineticNet on fresh (state, action) pairs from new
                       behaviour trajectories (a different seed from training).
    3. Projection      aggressive actions on visited plant states, projected and
                       then executed: violation rates per constraint with and
                       without the projection.  Containment and heat run the
                       real env (two holds, heat over the head's horizon);
                       purity holds each reactor's projected action to
                       harvest in the vectorized plant.
    4. Safety gradient a step down the safety loss raises violated margins.
    5. Latency         projection time per control step (CPU, one state).

Usage:
    python -m metarl.apn.validate
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os
import time

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.apn.data import make_dataset
from metarl.apn.apn import MultiHeadAPN
from metarl.apn.constants import HEAT_HORIZON_H, MASS_BUFFER, STORAGE_BUFFER
from metarl.env.config import MAX_HEAT_LOAD, MIN_PRODUCT_TITER, S_MAX, get_main_reactor_config
from metarl.env.safe import FermentationEnv
from metarl.apn.validation_plots import plot_validation
from metarl.apn.plant_sim import DECISION_H, Plant, behaviour_actions, behaviour_states, held_rollout, masked_action
from metarl.apn.train import evaluate, to_tensors

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def _make_env():
    cfg = get_main_reactor_config()
    env = FermentationEnv(3, cfg['Mmax'], cfg['max_flow_LF'], cfg['max_flow_C5'])
    env.lagrange_updates_enabled = False
    return env


def plant_states(n_episodes, seed, apn=None):
    """Plant snapshots visited by random behaviour on the 3-reactor plant.

    With ``apn``, every other episode projects the behaviour's actions, so the
    snapshots cover both an uncontrolled plant (often already past a wall) and
    the plant a safe agent operates.

    Returns:
        List of (observation [3, 28], internal-state dict) pairs, one per decision.
    """
    rng = np.random.default_rng(seed)
    env = _make_env()
    out = []
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            out.append((s.copy(), env._get_full_internal_state()))
            a = masked_action(env.current_stage, behaviour_actions(3, rng))
            if apn is not None and ep % 2:
                a = apn.project(torch.tensor(s)[None], torch.tensor(a)[None])[0][0].numpy()
            s, _, done, _ = env.step(a)
    return out


def check_exact_heads(apn, snaps, rng):
    """1. Predicted mass and storage after a hold vs FermentationEnv."""
    env = _make_env()
    err_m, err_s = [], []
    for obs, internal in snaps:
        env._set_full_internal_state(internal)
        a = masked_action(env.current_stage, behaviour_actions(3, rng))
        with torch.no_grad():
            h = apn.head_margins(torch.tensor(obs)[None], torch.tensor(a)[None])
        mmax = env.Mmax
        m_pred = ((1.0 - MASS_BUFFER) - h["overflow"][0].numpy()) * mmax
        s_pred = ((1.0 - STORAGE_BUFFER) - float(h["storage_over"][0])) * S_MAX
        stage0 = env.current_stage.copy()
        env.step(a)
        steady = stage0 == env.current_stage
        err_m.extend((np.abs(m_pred - env.M) / mmax)[steady & (stage0 != 4)])
        if np.all(steady):
            err_s.append(abs(s_pred - env.S_mass) / S_MAX)
    err_m, err_s = np.array(err_m), np.array(err_s)
    print(f"[1] Exact heads, {len(err_m):,} steady reactor holds: |M error| p50 {np.median(err_m):.2e}, "
          f"p99 {np.percentile(err_m, 99):.2e}, max {err_m.max():.2e} (fraction of Mmax); storage "
          f"p99 {np.percentile(err_s, 99):.2e}, max {err_s.max():.2e} (fraction of S_max)")
    metrics = {"n": len(err_m), "limit": 1e-3,
               "mass": [float(np.median(err_m)), float(np.percentile(err_m, 99)), float(err_m.max())],
               "storage": [float(np.median(err_s)), float(np.percentile(err_s, 99)), float(err_s.max())]}
    return metrics["mass"][1] < metrics["limit"], metrics


def check_learned_heads(apn, seed, workers):
    """2. KineticNet on fresh trajectories (independent seed)."""
    rng = np.random.default_rng(seed)
    obs, params = behaviour_states(256, seed=seed, workers=workers)
    idx = rng.choice(len(obs), size=min(20000, len(obs)), replace=False)
    data = make_dataset(obs[idx], params[idx], rng, workers)
    m = evaluate(apn.kinetic, *to_tensors(data, "cpu"))
    print(f"[2] Learned heads, {len(idx):,} fresh pairs: heat MAE {m['heat_mae']:.1f} kg/h "
          f"(R2 {m['heat_r2']:.4f}), harvest titer MAE {m['titer_mae']:.2f} g/L, purity sign "
          f"{m['purity_sign_acc']:.4f} (within 5 g/L of spec {m['purity_sign_acc_near_spec']:.3f})")
    metrics = {"n": len(idx), "heat_r2": (float(m["heat_r2"]), 0.98),
               "purity_sign_acc": (float(m["purity_sign_acc"]), 0.97),
               "purity_sign_acc_near_spec": float(m["purity_sign_acc_near_spec"]),
               "heat_mae": float(m["heat_mae"]), "titer_mae": float(m["titer_mae"])}
    return m["heat_r2"] > 0.98 and m["purity_sign_acc"] > 0.97, metrics


def _execute(env, internal, action):
    """Two holds of ``action`` (re-masked at the second decision) from a snapshot.

    Returns:
        (set of violation ids in the first hold, peak plant heat load over both holds).
    """
    env._set_full_internal_state(internal)
    env.clear_disturbances()
    _, _, _, info = env.step(action)
    ids = set(info["violation_counts"])
    peak = info["heat_load_peak"]
    for _ in range(HEAT_HORIZON_H // DECISION_H - 1):
        _, _, _, info = env.step(masked_action(env.current_stage, action))
        peak = max(peak, info["heat_load_peak"])
    return ids, peak


def check_projection(apn, snaps, rng):
    """3. What the projection certifies, checked by executing the action.

    For every head group the projection either brings the margins to >= 0
    (certified) or cannot (the state is already past a wall no action can undo
    within a hold, e.g. a tank fuller than its buffer with nothing to drain it).
    The certificate is what matters: among certified holds, how often does the
    real plant still violate?  Raw actions are executed too, for context.
    """
    env = _make_env()
    groups = {1: "reactor overflow", 3: "storage overflow", 4: "storage underflow",
              9: "heat cap (10 h)"}
    stat = {g: {"raw": 0, "cert": 0, "cert_viol": 0} for g in groups}
    purity_m, purity_t = [], []
    for obs, internal in snaps:
        env._set_full_internal_state(internal)
        stage = env.current_stage.copy()
        raw = masked_action(stage, behaviour_actions(3, rng))
        s_t = torch.tensor(obs)[None]
        proj, _ = apn.project(s_t, torch.tensor(raw)[None])
        with torch.no_grad():
            h = apn.head_margins(s_t, proj)
        proj = proj[0].numpy()
        # Mass moves monotonically within a hold, so a certified end state is
        # safe all the way unless the hold starts past the limit already (a
        # reactor that is being drained back below it still alarms first).
        past = {1: bool(np.any(env.M > env.Mmax)), 3: env.S_mass > env.S_max}
        cert = {1: bool((h["overflow"] >= 0).all()) and not past[1],
                3: bool(h["storage_over"] >= 0) and not past[3],
                4: bool(h["storage_under"] >= 0), 9: bool(h["heat"] >= 0)}
        for g in past:
            stat[g]["past"] = stat[g].get("past", 0) + past[g]
        ids_raw, peak_raw = _execute(env, internal, raw)
        ids, peak = _execute(env, internal, proj)
        for g in groups:
            hit_raw = (peak_raw > MAX_HEAT_LOAD) if g == 9 else (g in ids_raw)
            hit = (peak > MAX_HEAT_LOAD) if g == 9 else (g in ids)
            stat[g]["raw"] += hit_raw
            if cert[g]:
                stat[g]["cert"] += 1
                stat[g]["cert_viol"] += hit
        pre = stage <= 2
        if pre.any():
            plant = Plant.from_obs(obs[pre])
            _, titer, reached = held_rollout(plant, proj[pre].astype(np.float64), 1)
            purity_m.extend(h["purity"][0].numpy()[pre][reached])
            purity_t.extend(titer[reached])

    n = len(snaps)
    print(f"[3] Projection, {n:,} plant states with aggressive actions:")
    ok = True
    metrics = {"n": n, "groups": {}}
    for g, name in groups.items():
        s = stat[g]
        rate = s["cert_viol"] / max(s["cert"], 1)
        limit = 0.0 if g in (1, 3) else 0.01
        past = f" | already past the limit {s['past'] / n:5.1%}" if "past" in s else ""
        print(f"    {name:<18} raw actions violate {s['raw'] / n:6.1%} | certified "
              f"{s['cert'] / n:6.1%} of states | violated when certified {rate:6.2%}{past}")
        ok &= rate <= limit
        metrics["groups"][name] = {"raw": s["raw"] / n, "certified": s["cert"] / n,
                                   "past": s.get("past", 0) / n, "violated": rate, "limit": limit}
    m, t = np.array(purity_m), np.array(purity_t)
    cert = m >= 0.0
    false_safe = float(((t < MIN_PRODUCT_TITER) & cert).sum() / max(cert.sum(), 1))
    print(f"    {'purity (held)':<18} certified {cert.mean():6.1%} of pre-harvest batches | "
          f"off-spec when certified {false_safe:6.2%} ({cert.sum():,} batches)")
    ok &= false_safe <= 0.03
    metrics["groups"]["purity (held)"] = {"raw": None, "certified": float(cert.mean()), "past": 0.0,
                                          "violated": false_safe, "limit": 0.03}
    return ok, metrics


def check_safety_gradient(apn, snaps, rng, lr=0.05):
    """4. One step down the safety loss raises the violated margins."""
    obs = torch.tensor(np.stack([o for o, _ in snaps]))
    stage = obs[..., 16:21].argmax(-1).numpy()
    raw = np.stack([masked_action(st, behaviour_actions(3, rng)) for st in stage])
    a = torch.tensor(raw, requires_grad=True)
    m0 = apn.margins(obs, a).detach()
    loss = apn.safety_loss(obs, a)
    g, = torch.autograd.grad(loss, a)
    with torch.no_grad():
        a1 = (a - lr * g / (g.norm(dim=(1, 2), keepdim=True) + 1e-12)).clamp(-1, 1)
        m1 = apn.margins(obs, a1)
    viol = m0 < 0
    gain = (m1 - m0)[viol]
    print(f"[4] Safety gradient: one normalized step raises {float((gain > 0).float().mean()):.1%} of "
          f"{int(viol.sum()):,} violated margins (mean shortfall {float(m0[viol].mean()):.4f} -> "
          f"{float(m1[viol].mean()):.4f})")
    metrics = {"n": int(viol.sum()), "raised": float((gain > 0).float().mean()),
               "before": float(m0[viol].mean()), "after": float(m1[viol].mean())}
    return metrics["after"] > metrics["before"], metrics


def check_latency(apn, snaps, rng, n=300):
    """5. Projection time per control step (one state, CPU)."""
    times = []
    for obs, internal in snaps[:n]:
        stage = obs[:, 16:21].argmax(1)
        a = torch.tensor(masked_action(stage, behaviour_actions(3, rng)))[None]
        s = torch.tensor(obs)[None]
        t0 = time.perf_counter()
        apn.project(s, a)
        times.append(time.perf_counter() - t0)
    t = np.array(times[5:]) * 1e3
    print(f"[5] Projection latency (CPU, 3 reactors): p50 {np.median(t):.1f} ms, "
          f"p99 {np.percentile(t, 99):.1f} ms, max {t.max():.1f} ms")
    return True, {"ms": [float(np.median(t)), float(np.percentile(t, 99)), float(t.max())]}


def run_validation(apn=None, n_episodes=6, seed=12345, workers=None):
    """Run every check; returns True when all pass.

    Args:
        apn: MultiHeadAPN on CPU (loaded from policy/ when None).
        n_episodes: Behaviour episodes on the 3-reactor plant for checks 1, 3-5.
        seed: Seed (different from the training data's).
        workers: Processes for check 2's labelling.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if apn is None:
        apn = MultiHeadAPN()
        if not apn.load_kinetic(map_location="cpu"):
            raise FileNotFoundError("policy/apn_kinetic.pth not found; run python -m metarl.apn.train first.")
    apn = apn.cpu().eval()
    for p in apn.parameters():
        p.requires_grad_(False)
    snaps = plant_states(n_episodes, seed, apn)
    results = {
        "exact heads": check_exact_heads(apn, snaps, rng),
        "learned heads": check_learned_heads(apn, seed, workers),
        "projection": check_projection(apn, snaps, rng),
        "safety gradient": check_safety_gradient(apn, snaps, rng),
        "latency": check_latency(apn, snaps, rng),
    }
    print("Summary: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, (ok, _) in results.items()))
    print(f"Plot saved to {plot_validation(results)}")
    return all(ok for ok, _ in results.values())


if __name__ == "__main__":
    run_validation()
