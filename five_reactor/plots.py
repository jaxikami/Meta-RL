"""Figures of the five-reactor sub-study (under its results/plot/).

    training_curves.png   training reward and violations; held-out score,
                          violations and the decoded draw's error
    change_response.png   Pi, burden B and the post-change feed and discharge
                          against m (the sweep), adaptive vs frozen z, the
                          training support shaded and the wide draws' density
    change_inference.png  decoded draw against the true m on the wide draws,
                          and the decoding error against hours since onset
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import json
import os

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
import five_reactor  # noqa: F401  (selects the layout before metarl loads)
from metarl.training.plots import (PLOT_SURFACE, _moving_average, _pyplot,
                                   _savefig, _style_ax)
from five_reactor.config import (ARM, CHANGE_MEAN, CHANGE_TRAIN_SD, LOG_DIR,
                                 SUPPORT_SD)

ADAPTIVE = "#2a78d6"
FROZEN = "#b4b3ad"
SUPPORT = "#e9eef8"


def _load(name):
    path = os.path.join(LOG_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _shade_support(ax):
    lo = CHANGE_MEAN - SUPPORT_SD * CHANGE_TRAIN_SD
    hi = CHANGE_MEAN + SUPPORT_SD * CHANGE_TRAIN_SD
    ax.axvspan(lo, hi, color=SUPPORT, zorder=0,
               label=f"training support (±{SUPPORT_SD:g} SD)")
    ax.axvline(1.0, color="#999", lw=0.8, ls=":")


def plot_training():
    """training_curves.png from meta5's training log."""
    log = _load(f"{ARM}_log.json")
    plt = _pyplot("training curves")
    if log is None or plt is None or not log["iter"]:
        return
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    fig.patch.set_facecolor(PLOT_SURFACE)
    w = 25
    panels = (
        (log["iter"], _moving_average(log["avg_reward"], w), "training reward"),
        (log["iter"], _moving_average(log["avg_violations"], w),
         "training violation events"),
        (log["heldout_iter"], log["heldout_score"], "held-out score (ethanol)"),
        (log["heldout_iter"], log["heldout_m_mae"],
         "held-out |decoded m - m| at episode end"),
    )
    for ax, (x, y, title) in zip(axes, panels):
        _style_ax(ax)
        ax.plot(x, y, color=ADAPTIVE, lw=1.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("iteration")
    if log.get("selected_iter"):
        for ax in axes[2:]:
            ax.axvline(log["selected_iter"], color="#d6532a", lw=1, ls="--")
    fig.tight_layout()
    _savefig(fig, "training_curves")


def plot_eval():
    """change_response.png and change_inference.png from the evaluation."""
    summary, ev = _load("eval_summary.json"), _load("eval.json")
    plt = _pyplot("evaluation figures")
    if summary is None or plt is None:
        return
    sweep = summary["sweep"]
    xs = np.array([float(x) for x in sweep])
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))
    fig.patch.set_facecolor(PLOT_SURFACE)
    panels = (("score", 1e7, "net production Π (×1e7, higher is better)"),
              ("burden", 1.0, "violation burden B (lower is better)"),
              ("post_feed", 1.0, "mean LF feed after the change"),
              ("post_discharge", 1.0, "mean discharge after the change"))
    for ax, (k, scale, title) in zip(axes, panels):
        _style_ax(ax)
        _shade_support(ax)
        a = np.array([sweep[x][k]["mean"] for x in sweep]) / scale
        ci = np.array([sweep[x][k]["ci95"] for x in sweep]) / scale
        ax.plot(xs, a, color=ADAPTIVE, lw=2, marker="o", ms=3.5,
                label="meta5 (z inferred)")
        ax.fill_between(xs, a - ci, a + ci, color=ADAPTIVE, alpha=0.15)
        if k in sweep[next(iter(sweep))].get("frozen", {}):
            f = np.array([sweep[x]["frozen"][k]["mean"] for x in sweep]) / scale
            fci = np.array([sweep[x]["frozen"][k]["ci95"] for x in sweep]) / scale
            ax.plot(xs, f, color=FROZEN, lw=2, marker="s", ms=3.5,
                    label="same policy, z frozen at onset")
            ax.fill_between(xs, f - fci, f + fci, color=FROZEN, alpha=0.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("downstream draw after the change, m (× nominal)")
    if ev is not None:
        wide = [d[1] for d in ev["draws"]]
        tw = axes[0].twinx()
        tw.hist(wide, bins=20, color="#d6532a", alpha=0.18, zorder=0)
        tw.set_yticks([])
        tw.set_ylabel("wide evaluation draws", fontsize=8, color="#d6532a")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("meta5 against the change's magnitude (paired sweep on every "
                 "evaluation task, 95% CIs)", fontsize=11)
    fig.tight_layout()
    _savefig(fig, "change_response")

    if ev is None:
        return
    names = [b[0] for b in ev["branches"]]
    frozen = [b[2] for b in ev["branches"]]
    k = next(j for j, (n, f) in enumerate(zip(names, frozen))
             if n == "wide" and not f)
    rows = [r[k] for r in ev["rows"]]
    fig, (ax_s, ax_t) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor(PLOT_SURFACE)
    _style_ax(ax_s)
    _shade_support(ax_s)
    m = np.array([r["m"] for r in rows])
    dec = np.array([r["decoded"][-1] for r in rows])
    inside = np.array([r["in_support"] for r in rows])
    ax_s.scatter(m[inside], dec[inside], s=14, color=ADAPTIVE, label="in support")
    ax_s.scatter(m[~inside], dec[~inside], s=14, color="#d6532a",
                 label="outside support")
    lim = (min(m.min(), dec.min()) - 0.05, max(m.max(), dec.max()) + 0.05)
    ax_s.plot(lim, lim, color="#666", lw=0.8)
    ax_s.set_xlabel("true m")
    ax_s.set_ylabel("aux head's decoded m at episode end")
    ax_s.set_title("inference of the change (wide draws)", fontsize=10)
    ax_s.legend(fontsize=8)
    _style_ax(ax_t)
    hours = summary["decode_after_h"]
    xt = np.arange(len(hours))
    for key, col, lab in (("wide_in_support", ADAPTIVE, "in support"),
                          ("wide_outside_support", "#d6532a", "outside support")):
        if key in summary:
            y = [d["mean"] for d in summary[key]["decode_mae"]]
            ax_t.plot(xt, y, color=col, marker="o", lw=2, label=lab)
    ax_t.set_xticks(xt, [f"+{h} h" if h != "end" else "end" for h in hours])
    ax_t.set_ylabel("mean |decoded m - m|")
    ax_t.set_title("decoding error after the onset", fontsize=10)
    ax_t.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, "change_inference")


def plot_all():
    plot_training()
    plot_eval()
