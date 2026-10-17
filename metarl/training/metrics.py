"""Latent separability, gate openings and paired statistics."""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import N_REACTORS
from metarl.training.config import _T95, PRACTICAL_MIN_REL_EFFECT


def latent_separability(z_mus, z_sigma_sq):
    """How far apart different tasks' posteriors sit, in posterior widths.

    Joint SNR = sqrt(sum_d SNR_d^2), SNR_d = (SD across tasks of the mean) /
    (RMS posterior SD); ``mean_ratio`` is the older statistic.
    """
    mus = np.asarray(z_mus, dtype=np.float64)
    var = np.asarray(z_sigma_sq, dtype=np.float64)
    spread = mus.std(axis=0)
    post_rms = np.sqrt(var.mean(axis=0))
    per_dim = spread / np.maximum(post_rms, 1e-12)
    post_std = float(np.mean(np.sqrt(var)))
    return {
        "snr": float(np.sqrt(np.sum(per_dim ** 2))),
        "snr_per_dim": per_dim.tolist(),
        "mean_ratio": float(np.mean(spread) / max(post_std, 1e-12)),
        "spread": float(np.mean(spread)),
        "post_std": post_std,
        "spread_per_dim": spread.tolist(),
        "post_std_per_dim": post_rms.tolist(),
    }


def gate_opening(gate_traj, start=0, stop=None):
    """Mean adaptation-gate opening over decisions ``[start, stop)`` (NaN if none)."""
    g = np.asarray(gate_traj, dtype=float)[start:stop]
    g = g[np.isfinite(g)]
    return float(g.mean()) if g.size else float("nan")


def gate_opening_in_stage(gate_traj, stage_traj, stage, reactor=None, start=0):
    """Mean gate opening over decisions where a reactor was in one stage.

    Args:
        gate_traj: Per-decision openings.
        stage_traj: Per-decision stages, shape (steps, N).
        stage: Stage index.
        reactor: The reactor whose stage counts, or None for any reactor.
        start: First decision index considered.

    Returns:
        Float, NaN when there is no gate or no such decision.
    """
    g = np.asarray(gate_traj, dtype=float)
    s = np.asarray(stage_traj, dtype=int).reshape(-1, N_REACTORS)
    n = min(g.size, s.shape[0])
    g, s = g[start:n], s[start:n]
    hit = (s == stage).any(axis=1) if reactor is None else (s[:, reactor] == stage)
    g = g[hit & np.isfinite(g)]
    return float(g.mean()) if g.size else float("nan")


def episodes_violated(res):
    """(episodes with any violation, episodes) for one arm on one scenario."""
    if "episodes_violated" in res:
        return int(res["episodes_violated"]), int(res["episodes"])
    v = np.asarray(res.get("viol_per_ep", []), dtype=float)
    return int(np.sum(v > 0)), int(v.size)


def _finite_mean(xs):
    """Mean of the finite values, NaN when there are none."""
    xs = [float(x) for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def _t95(df):
    """Two-sided 95% t critical value (NaN when df < 1)."""
    return float("nan") if df < 1 else _T95.get(int(df), 1.960)


def _paired_delta(a, b, ref=None):
    """Paired mean difference of a - b with a 95% t interval."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    nan = float("nan")
    if a.size == 0 or a.size != b.size:
        return {"mean": nan, "se": nan, "ci95": nan, "n": 0,
                "significant": False, "paired": False, "rel": nan,
                "negligible": False}
    d = a - b
    mean = float(np.mean(d))
    n = int(d.size)
    scale = abs(float(np.mean(b))) if ref is None else abs(float(ref))
    rel = mean / scale if scale > 0 else nan
    if n < 2:
        return {"mean": mean, "se": nan, "ci95": nan, "n": n,
                "significant": False, "paired": True, "rel": rel,
                "negligible": False}
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    ci95 = float(_t95(n - 1) * se)
    sig = bool(abs(mean) > ci95) if se > 0.0 else bool(mean != 0.0)
    negligible = bool(sig and np.isfinite(rel)
                      and abs(rel) < PRACTICAL_MIN_REL_EFFECT)
    return {"mean": mean, "se": se, "ci95": ci95, "n": n, "significant": sig,
            "paired": True, "rel": rel, "negligible": negligible}


def _verdict_sign(d):
    """+1 / -1 for a significant, non-negligible gap in that direction, else 0."""
    if not d["significant"] or d["negligible"] or not np.isfinite(d["mean"]):
        return 0
    return 1 if d["mean"] > 0 else -1


def _fmt_delta(d):
    """Render a paired delta for a table."""
    if not np.isfinite(d["mean"]):
        return "n/a"
    if not np.isfinite(d["ci95"]):
        return f"{d['mean']:+.2f}"
    flag = (f"tiny({d['rel'] * 100:+.2f}%)" if d["negligible"]
            else "SIG" if d["significant"] else "ns")
    return f"{d['mean']:+.2f}+-{d['ci95']:.2f} {flag}"


def _fmt_delta_sci(d):
    """Render a paired delta whose magnitude is large (e.g. ethanol mass)."""
    if not np.isfinite(d["mean"]):
        return "n/a"
    rel = f"{d['rel'] * 100:+.1f}%" if np.isfinite(d["rel"]) else "n/a"
    flag = ("tiny" if d["negligible"] else "SIG" if d["significant"] else "ns")
    return f"{d['mean']:+.2e} ({rel}) {flag}"
