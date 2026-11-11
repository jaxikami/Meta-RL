"""Figures: arm comparisons, disturbance responses and training curves."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import json
import os

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import (MIN_PRODUCT_TITER, N_REACTORS, S_MAX, S_MIN,
                               REACTOR_FLOW_SCALE)
from metarl.training.config import (CONSTRAINT_NAMES,
                                    Z_INFORM_MIN_SNR, LOG_DIR, PLOT_DIR,
                                    TITER_PANEL_FLOOR_PAD, DISTURB_PROBES)
from metarl.training.arms import ARMS, ARM_ORDER
from metarl.training.metrics import (episodes_violated, _paired_delta,
                                     _verdict_sign)

ARM_COLORS = {"saferl": "#1baf7a", "meta": "#2a78d6"}
PLOT_INK, PLOT_INK2, PLOT_GRID = "#0b0b0b", "#52514e", "#d8d7d2"
PLOT_SURFACE = "#fcfcfb"
PLOT_NEUTRAL = "#b4b3ad"
CONSTRAINT_COLORS = {
    "overflow": "#2a78d6",
    "underflow": "#eb6834",
    "harvest_titer": "#1baf7a",
    "product_purity": "#eda100",
    "instability": "#e87ba4",
    "mass_balance": "#008300",
    "storage_overflow": "#6a4fb3",
    "storage_underflow": "#8c6d1f",
    "heat_load": "#0f8b8d",
}
BURDEN_MIX_MIN_LABEL_PCT = 6.0
# Burden axis turns symlog when the largest bar is this many times the
# smallest non-zero one, so a quiet scenario stays readable next to a bad one.
BURDEN_SYMLOG_RATIO = 30.0
ACTUATORS = (
    ("stage-time multiplier (x)", lambda a: a * 1.5 + 0.5),
    ("LF feed (frac. of max)", lambda a: a),
    ("C5 feed (frac. of max)", lambda a: a),
    ("discharge (frac. of max)", lambda a: a),
)


def _pyplot(what=""):
    """matplotlib.pyplot on the Agg backend, or None if unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:                             # pragma: no cover
        print(f"[plot] {what + ' ' if what else ''}skipped: {exc}")
        return None


def _savefig(fig, stem):
    """Write ``fig`` to PLOT_DIR/<stem>.png and close it."""
    import matplotlib.pyplot as plt
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig.savefig(os.path.join(PLOT_DIR, f"{stem}.png"), dpi=150)
    plt.close(fig)


def scenario_label(name):
    """A scenario's tick label."""
    return name.replace("disturb_", "D ").replace("_", " ")


def _style_ax(ax):
    """Shared axis styling."""
    ax.set_facecolor(PLOT_SURFACE)
    ax.grid(axis="y", alpha=0.5, color=PLOT_GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PLOT_GRID)
    ax.tick_params(colors=PLOT_INK2, labelsize=9)


def _plot_distributions(results, names, labels, arms):
    """Per-episode on-spec ethanol per arm and scenario (box plus points)."""
    import matplotlib.pyplot as plt
    cols = min(len(names), 6)
    rows = -(-len(names) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4.1 * cols, 5.0 * rows),
                            squeeze=False, sharey=True)
    fig.patch.set_facecolor(PLOT_SURFACE)
    for ax in axs.flat[len(names):]:
        ax.set_visible(False)
    for ax, n, lab in zip(axs.flat, names, labels):
        _style_ax(ax)
        data, present = [], []
        for arm in arms:
            v = np.asarray(results[n][arm].get("on_spec_ethanol_per_ep", []),
                           dtype=float)
            if v.size:
                data.append(v / 1e7)
                present.append(arm)
        if not data:
            continue
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        medianprops={"color": PLOT_INK, "lw": 1.8},
                        whiskerprops={"color": PLOT_INK2, "lw": 1.1},
                        capprops={"color": PLOT_INK2, "lw": 1.1},
                        flierprops={"marker": "o", "ms": 3,
                                    "markerfacecolor": PLOT_INK2,
                                    "markeredgecolor": "none", "alpha": 0.5})
        for patch, arm in zip(bp["boxes"], present):
            patch.set_facecolor(ARM_COLORS[arm])
            patch.set_alpha(0.35)
            patch.set_edgecolor(ARM_COLORS[arm])
            patch.set_linewidth(1.8)
        for i, (v, arm) in enumerate(zip(data, present), start=1):
            jitter = (np.arange(v.size) % 5 - 2) * 0.035
            ax.plot(np.full(v.size, i) + jitter, v, "o", ms=3.5,
                    color=ARM_COLORS[arm], alpha=0.75, markeredgecolor="none")
        ax.set_xticks(range(1, len(present) + 1))
        ax.set_xticklabels(present, rotation=20, ha="right", fontsize=9,
                           color=PLOT_INK2)
        ax.set_title(lab, fontsize=10.5, color=PLOT_INK, fontweight="bold")
    for row in axs:
        row[0].set_ylabel("On-spec ethanol per episode  (x10^7)",
                          fontsize=9.5, color=PLOT_INK2)
    fig.suptitle("Per-episode production distributions", fontweight="bold",
                 fontsize=12.5, color=PLOT_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _savefig(fig, "distributions")


def _plot_adapt(results, names, labels):
    """ADAPT (meta vs saferl) as paired deltas per scenario, three panels."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    def _per_ep(r, kind):
        key = {"burden": "burden_per_ep", "eth": "on_spec_ethanol_per_ep",
               "score": "score_per_ep"}[kind]
        return np.asarray(r.get(key, []), dtype=float)

    panels = (
        ("burden", "Violation burden avoided per episode (exceedance area "
                   "past the limits, limit-hours)", "saferl - meta  (B)",
         lambda v: f"{v:+.1f}"),
        ("eth", "On-spec ethanol gained", "meta - saferl  (% of saferl)",
         lambda v: f"{v:+.1f}%"),
        ("score", "Net production gained (on-spec ethanol less violations "
                  "and wear)", "meta - saferl  (% of saferl on-spec)",
         lambda v: f"{v:+.1f}%"),
    )
    x = np.arange(len(names))
    fig, axs = plt.subplots(len(panels), 1, sharex=True,
                            figsize=(max(9.0, 0.9 * len(names)),
                                     3.5 * len(panels)))
    fig.patch.set_facecolor(PLOT_SURFACE)
    for ax, (kind, title, ylab, fmt) in zip(axs, panels):
        _style_ax(ax)
        vals, cis, cols = [], [], []
        for n in names:
            s = _per_ep(results[n]["saferl"], kind)
            m = _per_ep(results[n]["meta"], kind)
            if s.size == 0 or s.size != m.size:
                vals.append(np.nan)
                cis.append(0.0)
                cols.append(PLOT_NEUTRAL)
                continue
            if kind in ("eth", "score"):
                # Net production can sit near zero: both read as a share
                # of saferl's gross on-spec output.
                scale = abs(float(results[n]["saferl"]["on_spec_ethanol"]))
                d = _paired_delta(m, s, ref=scale)
                k = 100.0 / scale if scale > 0 else np.nan
            else:
                d = _paired_delta(s, m, ref=float(np.mean(s)))
                k = 1.0
            v, ci = d["mean"] * k, d["ci95"] * k
            vals.append(v)
            cis.append(ci if np.isfinite(ci) else 0.0)
            if d["significant"] and not d["negligible"] and v != 0:
                cols.append(ARM_COLORS["meta" if v > 0 else "saferl"])
            else:
                cols.append(PLOT_NEUTRAL)
        vals, cis = np.asarray(vals), np.asarray(cis)
        ax.bar(x, np.nan_to_num(vals), 0.6, color=cols, yerr=cis, capsize=3,
               ecolor=PLOT_INK2, error_kw={"lw": 1.1})
        ax.axhline(0.0, color=PLOT_INK2, lw=1.0, zorder=1)
        for xi, v, ci in zip(x, vals, cis):
            if not np.isfinite(v):
                txt, y, dy, va = "n/a", 0.0, 3, "bottom"
            else:
                txt = "0" if v == 0 else fmt(v)
                up = v >= 0
                y = v + ci if up else v - ci
                dy, va = (3, "bottom") if up else (-3, "top")
            ax.annotate(txt, (xi, y), xytext=(0, dy),
                        textcoords="offset points", ha="center", va=va,
                        fontsize=7.2, color=PLOT_INK2)
        ax.set_title(title, fontsize=10.5, color=PLOT_INK, fontweight="bold", pad=8)
        ax.set_ylabel(ylab, fontsize=9.5, color=PLOT_INK2)
        ax.margins(y=0.22)
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(labels, rotation=25, ha="right", fontsize=9,
                            color=PLOT_INK2)
    handles = [
        Patch(color=ARM_COLORS["meta"], label="meta better (95% CI excludes 0)"),
        Patch(color=ARM_COLORS["saferl"], label="saferl better (95% CI excludes 0)"),
        Patch(color=PLOT_NEUTRAL, label="no significant or practical difference"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=9, labelcolor=PLOT_INK2, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("ADAPT -- meta vs saferl, paired over the same episodes "
                 "(up = meta better)", fontweight="bold", fontsize=12, color=PLOT_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    _savefig(fig, "compare_adapt")


def _ink_on(fill):
    """Ink or white, whichever reads on a bar of colour ``fill``."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(fill)
    return "#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else PLOT_INK


def _plot_burden(results, names, labels, arms):
    """Violation burden per arm and scenario, and what it is made of.

    Left: mean burden per episode (+1 std), labelled with the episodes that
    violated anything. Right: each arm's burden over every scenario, split
    by constraint. Replaces the event-count, episodes-violated, mix and
    severity figures.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, (ax, ax_mix) = plt.subplots(
        1, 2, figsize=(max(11.0, 0.9 * len(names) + 5.0), 6.6),
        gridspec_kw={"width_ratios": [max(len(names), 4), 3.2]})
    fig.patch.set_facecolor(PLOT_SURFACE)
    _style_ax(ax)
    x = np.arange(len(names))
    w = 0.8 / max(len(arms), 1)
    seen_vals = []
    for i, arm in enumerate(arms):
        vals = np.asarray([results[n][arm]["burden"] for n in names])
        err = np.asarray([results[n][arm]["burden_std"] for n in names])
        pos = x + (i - (len(arms) - 1) / 2) * w
        ax.bar(pos, vals, w, color=ARM_COLORS[arm], label=arm,
               yerr=np.vstack([np.zeros_like(err), err]), capsize=3,
               ecolor=PLOT_INK2, error_kw={"lw": 1.1})
        for xi, v, e, n in zip(pos, vals, err, names):
            k, eps = episodes_violated(results[n][arm])
            ax.annotate(f"{v:.1f}  ({k}/{eps})", (xi, v + e), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.8, color=PLOT_INK2, rotation=90)
        seen_vals.extend(vals.tolist())
    pos_vals = [v for v in seen_vals if v > 0]
    if pos_vals and max(pos_vals) > BURDEN_SYMLOG_RATIO * min(pos_vals):
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylabel("Violation burden B per episode  (symlog)", fontsize=9.5,
                      color=PLOT_INK2)
    else:
        ax.set_ylabel("Violation burden B per episode", fontsize=9.5,
                      color=PLOT_INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9,
                       color=PLOT_INK2)
    ax.margins(y=0.25)
    ax.set_title("Burden per scenario (lower is better; label: B  (episodes "
                 "violated / episodes))", fontsize=10.5, color=PLOT_INK,
                 fontweight="bold", pad=8)
    ax.legend(frameon=False, fontsize=9, labelcolor=PLOT_INK2, loc="upper left")

    # Composition over all scenarios, as a share of each arm's burden.
    _style_ax(ax_mix)
    ax_mix.grid(axis="y", alpha=0.0)
    ax_mix.grid(axis="x", alpha=0.5, color=PLOT_GRID, lw=0.8)
    order = list(CONSTRAINT_NAMES.values())
    parts = {arm: np.asarray([sum(float(results[n][arm]["burden_by_constraint"]
                                        .get(c, 0.0)) for n in names)
                              for c in order]) for arm in arms}
    seen = [c for j, c in enumerate(order)
            if any(parts[a][j] > 0 for a in arms)]
    for yi, arm in enumerate(arms):
        total = parts[arm].sum()
        if total <= 0:
            ax_mix.annotate("no violations", (0.0, yi), xytext=(4, 0),
                            textcoords="offset points", va="center",
                            fontsize=8.5, color=PLOT_INK2)
            continue
        left = 0.0
        for c in seen:
            pct = 100.0 * parts[arm][order.index(c)] / total
            if pct <= 0:
                continue
            ax_mix.barh(yi, pct, 0.6, left=left, color=CONSTRAINT_COLORS[c],
                        edgecolor=PLOT_SURFACE, linewidth=1.5)
            if pct >= BURDEN_MIX_MIN_LABEL_PCT:
                ax_mix.text(left + pct / 2, yi, f"{pct:.0f}%", ha="center",
                            va="center", fontsize=7.5,
                            color=_ink_on(CONSTRAINT_COLORS[c]))
            left += pct
    ax_mix.set_yticks(range(len(arms)))
    ax_mix.set_yticklabels(arms, fontsize=9.5, color=PLOT_INK2)
    ax_mix.invert_yaxis()
    ax_mix.set_xlim(0, 100)
    ax_mix.set_xlabel("Share of burden over all scenarios (%)", fontsize=9.5,
                      color=PLOT_INK2)
    ax_mix.set_title("What the burden is", fontsize=10.5, color=PLOT_INK,
                     fontweight="bold", pad=8)
    if seen:
        ax_mix.legend(handles=[Patch(color=CONSTRAINT_COLORS[c],
                                     label=c.replace("_", " ")) for c in seen],
                      loc="upper center", bbox_to_anchor=(0.5, -0.14),
                      ncol=2, frameon=False, fontsize=8, labelcolor=PLOT_INK2)
    return fig


def _plot_score(results, names, labels, arms):
    """Net production per arm and scenario, and averaged over scenarios.

    ``*`` marks the arm whose per-episode net production is significantly
    higher than every other arm's on that scenario (paired).
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(9.0, 0.9 * (len(names) + 1)), 6.4))
    fig.patch.set_facecolor(PLOT_SURFACE)
    _style_ax(ax)
    x = np.append(np.arange(len(names)), len(names) + 0.6)
    w = 0.8 / max(len(arms), 1)
    for i, arm in enumerate(arms):
        vals = np.asarray([results[n][arm]["score"] for n in names]) / 1e7
        err = np.asarray([results[n][arm]["score_std"] for n in names]) / 1e7
        vals, err = np.append(vals, vals.mean()), np.append(err, 0.0)
        pos = x + (i - (len(arms) - 1) / 2) * w
        ax.bar(pos, vals, w, color=ARM_COLORS[arm], label=arm, yerr=err,
               capsize=3, ecolor=PLOT_INK2, error_kw={"lw": 1.1})
        for j, (xi, v, e) in enumerate(zip(pos, vals, err)):
            best = j < len(names) and all(
                _verdict_sign(_paired_delta(
                    results[names[j]][arm]["score_per_ep"],
                    results[names[j]][o]["score_per_ep"],
                    ref=results[names[j]][o]["on_spec_ethanol"])) > 0
                for o in arms if o != arm)
            up = v >= 0
            ax.annotate(f"{v:.2f}" + ("*" if best else ""),
                        (xi, v + e if up else v - e),
                        xytext=(0, 3 if up else -3),
                        textcoords="offset points", ha="center",
                        va="bottom" if up else "top", fontsize=6.8,
                        color=PLOT_INK2, rotation=90)
    ax.axhline(0.0, color=PLOT_INK2, lw=1.0, zorder=1)
    ax.axvline(len(names) - 0.2, color=PLOT_GRID, lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels + ["mean over\nscenarios"], rotation=25,
                       ha="right", fontsize=9, color=PLOT_INK2)
    ax.set_ylabel("Net production Pi per episode  (x10^7 ethanol)",
                  fontsize=9.5, color=PLOT_INK2)
    ax.margins(y=0.2)
    ax.set_title("NET PRODUCTION Pi (higher is better; * = significantly "
                 "best)",
                 fontsize=10.5, color=PLOT_INK, fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=PLOT_INK2)
    return fig


def scenario_probe(name):
    """The DISTURB_PROBES key traced on scenario ``name``, or None.

    A kind's stepped scenario is traced: "<label>_extreme" in ``evaluate``
    (a step at EVAL_ONSET), "<label>_step" in the ablation's grid.
    """
    label, _, pattern = name.rpartition("_")
    return (label if pattern in ("step", "extreme") and label in DISTURB_PROBES
            else None)


def _shared_episode(res, arms):
    """The one episode on which every arm does comparatively well."""
    ranks = []
    for arm in arms:
        keys = list(zip(res[arm]["viol_per_ep"],
                        [-e for e in res[arm]["on_spec_ethanol_per_ep"]]))
        ranks.append([sum(k2 < k for k2 in keys) for k in keys])
    ranks = np.asarray(ranks)
    return int(min(range(ranks.shape[1]),
                   key=lambda i: (ranks[:, i].max(), ranks[:, i].sum())))


def _plot_disturbance_response(results, names, arms):
    """The faulted reactor's answering actuator (solid) vs the faulted quantity."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    dist = [n for n in names if scenario_probe(n)
            and all("traces" in results[n][a] for a in arms)]
    if not dist:
        print("[plot] disturbance response skipped: no per-episode traces")
        return
    fig, axs = plt.subplots(2, len(dist), figsize=(4.4 * len(dist), 8.6),
                            squeeze=False, sharex="col",
                            gridspec_kw={"height_ratios": [3, 1.3]})
    fig.patch.set_facecolor(PLOT_SURFACE)
    for c, n in enumerate(dist):
        ax, ax_s = axs[0][c], axs[1][c]
        _, qlabel, a_idx = DISTURB_PROBES[scenario_probe(n)]
        alabel, fn = ACTUATORS[a_idx]
        ep = _shared_episode(results[n], arms)
        tr0 = results[n][arms[0]]["traces"]
        reactor = tr0["reactor"][ep]
        handles = []
        for arm in arms:
            tr = results[n][arm]["traces"]
            act = np.asarray(tr["act"][ep], dtype=float)
            v = results[n][arm]["viol_per_ep"][ep]
            e = results[n][arm]["on_spec_ethanol_per_ep"][ep]
            ax.plot(fn(act), color=ARM_COLORS[arm], lw=1.6, zorder=3)
            ax_s.plot(np.asarray(tr["storage"][ep]) / S_MAX, color=ARM_COLORS[arm],
                      lw=1.3)
            handles.append(Line2D([0], [0], color=ARM_COLORS[arm], lw=1.6,
                                  label=f"{arm}: {v:.0f} viol, "
                                        f"{e / 1e6:.1f}M on-spec"))
        traces = {arm: np.asarray(results[n][arm]["traces"]["quantity"][ep],
                                  dtype=float) for arm in arms}
        ref = traces[arms[0]]
        shared = all(t.shape == ref.shape and np.allclose(t, ref, rtol=1e-4,
                                                          equal_nan=True)
                     for t in traces.values())
        ax2 = ax.twinx()
        if shared:
            ax2.plot(ref, color=PLOT_INK, ls=":", lw=1.9)
            handles.append(Line2D([0], [0], color=PLOT_INK, ls=":", lw=1.9,
                                  label=f"{qlabel} (right)"))
        else:
            for arm, t in traces.items():
                ax2.plot(t, color=ARM_COLORS[arm], ls=":", lw=1.5)
                handles.append(Line2D([0], [0], color=ARM_COLORS[arm], ls=":",
                                      lw=1.5, label=f"{arm} {qlabel} (right)"))
        ax2.set_ylabel(qlabel, fontsize=8.5, color=PLOT_INK2)
        ax2.tick_params(colors=PLOT_INK2, labelsize=8)
        for side in ("top", "left", "bottom"):
            ax2.spines[side].set_visible(False)
        ax2.spines["right"].set_color(PLOT_GRID)
        onset = results[n][arms[0]].get("onset_outer")
        for a_ in (ax, ax_s):
            if onset is not None:
                a_.axvline(onset, color=PLOT_INK2, ls="--", lw=1.0, zorder=1)
            _style_ax(a_)
        if onset is not None:
            handles.append(Line2D([0], [0], color=PLOT_INK2, ls="--", lw=1.0,
                                  label="fault onset"))
        for lim, lab_ in ((S_MIN / S_MAX, "S_MIN"), (1.0, "S_MAX")):
            ax_s.axhline(lim, color=PLOT_INK2, ls=":", lw=0.9)
        ax_s.set_ylabel("storage / S_MAX", fontsize=8.5, color=PLOT_INK2)
        ax_s.set_xlabel("control step", fontsize=9, color=PLOT_INK2)
        ax.set_ylabel(alabel + ("" if reactor < 0 else f", reactor {reactor}"),
                      fontsize=9, color=PLOT_INK2)
        title = n.replace("disturb_", "").replace("_", " ")
        where = "plant-wide" if reactor < 0 else f"reactor {reactor}"
        ax.set_title(f"{title} ({where})\nepisode #{ep}", fontsize=10.5,
                     color=PLOT_INK, fontweight="bold")
        ax.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=7.5,
                  labelcolor=PLOT_INK2)
    fig.suptitle("Disturbance response -- the faulted reactor's answering "
                 "actuator (solid), the faulted quantity (dotted) and the "
                 "storage tank (lower), all arms on one shared episode",
                 fontweight="bold", fontsize=12.5, color=PLOT_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _savefig(fig, "disturbance_response")


def _plot_comparison(results):
    """Render the evaluation figures into PLOT_DIR.

    compare_score (net production), compare_burden (violation burden and
    its make-up), compare_production / compare_quality / compare_reactors
    (per-reactor on-spec ethanol and violation events), compare_adapt,
    distributions, disturbance_response.
    """
    plt = _pyplot()
    if plt is None:
        return
    names = list(results.keys())
    labels = [scenario_label(n) for n in names]
    arms = tuple(a for a in ARM_ORDER if a in results[names[0]])
    x = np.arange(len(names))

    def _style(ax, title, ylab):
        ax.set_title(title, fontsize=11, color=PLOT_INK, fontweight="bold", pad=10)
        ax.set_ylabel(ylab, fontsize=9.5, color=PLOT_INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9,
                           color=PLOT_INK2)
        _style_ax(ax)
        ax.margins(y=0.16)

    def _fmt_short(v):
        av = abs(v)
        if av >= 1e6:
            return f"{v / 1e6:.1f}M"
        if av >= 1e3:
            return f"{v / 1e3:.1f}k"
        return f"{v:.1f}"

    def _label_bars(ax, pos, vals, tops, fmt):
        for xi, v, t in zip(pos, vals, tops):
            ax.annotate(fmt(v), (xi, t), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.8, color=PLOT_INK2, rotation=90)

    w = 0.8 / max(len(arms), 1)
    off = lambda i: (i - (len(arms) - 1) / 2) * w          # noqa: E731
    arm_titles = {"saferl": "Safe RL", "meta": "Meta-RL"}
    suptitle = " vs ".join(arm_titles.get(a, a) for a in arms)
    if "meta" in arms:
        snr = float(np.mean([results[n]["meta"]["z_snr"] for n in names]))
        suptitle += (f"   |   latent separability SNR {snr:.2f} "
                     f"({'above' if snr >= Z_INFORM_MIN_SNR else 'BELOW'} the "
                     f"{Z_INFORM_MIN_SNR:g} gate)")
    suptitle += f"   |   {N_REACTORS} reactors + central storage"

    def _new_fig():
        fig, ax = plt.subplots(figsize=(max(9.0, 0.9 * len(names)), 6.4))
        fig.patch.set_facecolor(PLOT_SURFACE)
        ax.set_facecolor(PLOT_SURFACE)
        return fig, ax

    def _save(fig, stem):
        fig.suptitle(suptitle, fontweight="bold", fontsize=12, color=PLOT_INK)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        _savefig(fig, stem)

    def _bar_panel(ax, key, errkey, ylab, title, fmt=None):
        fmt = fmt or _fmt_short
        for i, arm in enumerate(arms):
            vals = np.asarray([results[n][arm][key] for n in names], dtype=float)
            pos = x + off(i)
            if errkey:
                err = np.asarray([results[n][arm][errkey] for n in names],
                                 dtype=float)
                yerr = np.vstack([np.minimum(err, np.clip(vals, 0, None)), err])
                tops = vals + err
            else:
                yerr, tops = None, vals
            ax.bar(pos, vals, w, label=arm, color=ARM_COLORS[arm], yerr=yerr,
                   capsize=3, ecolor=PLOT_INK2, error_kw={"lw": 1.2})
            _label_bars(ax, pos, vals, tops, fmt)
        _style(ax, title, ylab)
        ax.legend(frameon=False, fontsize=9, labelcolor=PLOT_INK2)

    if all("burden" in results[n][a] for n in names for a in arms):
        _save(_plot_score(results, names, labels, arms), "compare_score")
        _save(_plot_burden(results, names, labels, arms), "compare_burden")
    else:
        print("[plot] compare_score / compare_burden skipped: eval.json "
              "predates the burden metric (re-run evaluate)")
    if "saferl" in arms and "meta" in arms:
        _plot_adapt(results, names, labels)

    fig, ax = _new_fig()
    _bar_panel(ax, "on_spec_ethanol", "on_spec_ethanol_std", "On-spec ethanol",
               "PRODUCTION -- on-spec ethanol (higher is better)")
    _save(fig, "compare_production")

    fig, ax = _new_fig()
    lo_seen, hi_seen = [], []
    for i, arm in enumerate(arms):
        vals = np.asarray([results[n][arm]["mean_titer"] for n in names],
                          dtype=float)
        err = np.asarray([results[n][arm]["mean_titer_std"] for n in names],
                         dtype=float)
        pos = x + off(i)
        ax.errorbar(pos, vals, yerr=err, fmt="o", ms=8, color=ARM_COLORS[arm],
                    label=arm, capsize=4, ecolor=PLOT_INK2, elinewidth=1.2,
                    mec="white", mew=1.0, linestyle="none", zorder=3)
        for xi, v in zip(pos, vals):
            ax.annotate(f"{v:.1f}", (xi, v), xytext=(7, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=7.2, color=PLOT_INK2)
        lo_seen.append(float(np.min(vals - err)))
        hi_seen.append(float(np.max(vals + err)))
    _style(ax, "QUALITY -- mean product titer (higher is better)",
           "Mean product titer (g/L)")
    spec = MIN_PRODUCT_TITER or 0.0
    if spec:
        ax.axhline(spec, color=PLOT_INK2, ls="--", lw=1.2, zorder=0,
                   label=f"purity spec floor ({spec:.0f} g/L)")
    lo, hi = min(lo_seen), max(hi_seen)
    bottom = min(spec - TITER_PANEL_FLOOR_PAD, lo - 1.0) if spec else lo - 1.0
    ax.set_ylim(bottom, max(hi, spec) + 0.10 * max(hi - bottom, 1.0))
    ax.legend(frameon=False, fontsize=9, labelcolor=PLOT_INK2)
    _save(fig, "compare_quality")

    # Per reactor: on-spec ethanol and reactor-local violation events, per arm
    # (averaged over scenarios): which vessel carries the plant's numbers.
    fig, (ax_e, ax_v) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    fig.patch.set_facecolor(PLOT_SURFACE)
    xr = np.arange(N_REACTORS)
    for i, arm in enumerate(arms):
        eth = np.mean([results[n][arm]["reactor_on_spec_ethanol"] for n in names],
                      axis=0)
        vio = np.mean([results[n][arm]["viol_by_reactor"] for n in names], axis=0)
        ax_e.bar(xr + off(i), eth, w, color=ARM_COLORS[arm], label=arm)
        ax_v.bar(xr + off(i), vio, w, color=ARM_COLORS[arm], label=arm)
    for a_, t_ in ((ax_e, "On-spec ethanol per reactor (mean over scenarios)"),
                   (ax_v, "Reactor-local violation events per reactor")):
        _style_ax(a_)
        a_.set_xticks(xr)
        a_.set_xticklabels([f"R{i} ({s:g}x pumps)" for i, s in
                            enumerate(REACTOR_FLOW_SCALE)], color=PLOT_INK2)
        a_.set_title(t_, fontsize=10.5, color=PLOT_INK, fontweight="bold")
        a_.legend(frameon=False, fontsize=9, labelcolor=PLOT_INK2)
    _save(fig, "compare_reactors")

    _plot_distributions(results, names, labels, arms)
    _plot_disturbance_response(results, names, arms)
    print(f"wrote figures to {PLOT_DIR}/")


def _plot_tiers(tiers):
    """compare_tiers.png: meta's pooled gap on saferl in each tier.

    Two panels, one quantity each: the violation-burden cut and the
    net-production gain (``evaluate._tier_summary``). A bar takes the colour
    of the arm it favours; the label beside it gives the significant
    per-scenario wins, meta / saferl / tied.
    """
    if not tiers:
        return
    plt = _pyplot()
    if plt is None:
        return
    from matplotlib.patches import Patch
    names = list(tiers)
    y = np.arange(len(names))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 1.2 + 0.75 * len(names)),
                             sharey=True)
    fig.patch.set_facecolor(PLOT_SURFACE)
    panels = (("burden_reduction", "burden_wins",
               "violation burden cut vs saferl (%)"),
              ("score_gain", "score_wins",
               "net production gain vs saferl\n(% of saferl's on-spec ethanol)"))
    for ax, (key, wkey, title) in zip(axes, panels):
        vals = np.array([100.0 * tiers[n][key] for n in names], dtype=float)
        vals = np.nan_to_num(vals)
        cols = [ARM_COLORS["meta"] if v > 0 else ARM_COLORS["saferl"]
                for v in vals]
        ax.barh(y, vals, height=0.5, color=cols, edgecolor=PLOT_SURFACE, lw=2)
        ax.axvline(0.0, color=PLOT_INK2, lw=1.0)
        span = max(np.max(np.abs(vals)), 1.0)
        ax.set_xlim(-1.9 * span, 1.9 * span)
        for yi, v, n in zip(y, vals, names):
            w = tiers[n][wkey]
            ax.annotate(f"{v:+.1f}%   {w['meta']}/{w['saferl']}/{w['tied']}",
                        (v, yi), xytext=(6 if v >= 0 else -6, 0),
                        textcoords="offset points", va="center",
                        ha="left" if v >= 0 else "right", fontsize=8.5,
                        color=PLOT_INK)
        ax.set_title(title, fontsize=10.5, color=PLOT_INK, fontweight="bold")
        _style_ax(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", alpha=0.5, color=PLOT_GRID, lw=0.8)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{n}  {tiers[n]['label'].split(' (')[0]}\n"
                             f"{tiers[n]['n']} scenarios" for n in names],
                            fontsize=9, color=PLOT_INK2)
    fig.legend(handles=[Patch(color=ARM_COLORS["meta"], label="meta ahead"),
                        Patch(color=ARM_COLORS["saferl"], label="saferl ahead")],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Meta-RL vs Safe RL by evaluation tier   |   labels: pooled "
                 "gap, then significant scenario wins meta / saferl / tied",
                 fontweight="bold", fontsize=11, color=PLOT_INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    _savefig(fig, "compare_tiers")


def _plot_tier_burden_pie(results, tiers):
    """compare_tier_burden_pie.png: what each tier's violation burden is.

    A pie per (arm, tier): rows are the arms, columns the tiers. A slice is a
    constraint's share of the arm's burden summed over the tier's scenarios
    (each scenario's mean per episode, the pooling
    ``evaluate._tier_summary`` uses). Shares are within one pie; its title
    gives the tier's mean burden per scenario and the largest constraint.
    """
    if not tiers or not results:
        return
    plt = _pyplot("tier burden pie")
    if plt is None:
        return
    from matplotlib.patches import Patch
    arms = [a for a in ARM_ORDER if a in next(iter(results.values()))]
    names = list(tiers)
    order = list(CONSTRAINT_NAMES.values())
    parts = {(arm, t): np.array([sum(float(results[s][arm]
                                           ["burden_by_constraint"].get(c, 0.0))
                                     for s in tiers[t]["scenarios"])
                                 for c in order])
             for arm in arms for t in names}
    seen = [c for j, c in enumerate(order) if any(p[j] > 0 for p in
                                                  parts.values())]
    fig, axes = plt.subplots(len(arms), len(names),
                             figsize=(4.6 * len(names), 5.2 * len(arms) + 1.2),
                             squeeze=False)
    fig.patch.set_facecolor(PLOT_SURFACE)
    for r, arm in enumerate(arms):
        for k, t in enumerate(names):
            ax = axes[r][k]
            ax.set_facecolor(PLOT_SURFACE)
            vals = parts[(arm, t)]
            total = float(vals.sum())
            head = (f"{arm}   {t}  {tiers[t]['label'].split(' (')[0]}\n"
                    f"mean B {total / tiers[t]['n']:.2f} per scenario "
                    f"({tiers[t]['n']} scenarios)")
            if total <= 0:
                ax.pie([1.0], colors=[PLOT_GRID],
                       wedgeprops={"edgecolor": PLOT_SURFACE, "lw": 2})
                ax.text(0, 0, "no burden", ha="center", va="center",
                        fontsize=10, color=PLOT_INK2)
                ax.set_title(head, fontsize=9.5, color=ARM_COLORS[arm],
                             fontweight="bold")
                continue
            keep = [j for j, v in enumerate(vals) if v > 0]
            cols = [CONSTRAINT_COLORS[order[j]] for j in keep]
            wedges, _ = ax.pie(vals[keep], colors=cols, startangle=90,
                               counterclock=False,
                               wedgeprops={"edgecolor": PLOT_SURFACE, "lw": 2})
            for w, j, c in zip(wedges, keep, cols):
                pct = 100.0 * vals[j] / total
                if pct < BURDEN_MIX_MIN_LABEL_PCT:
                    continue
                ang = np.deg2rad(0.5 * (w.theta1 + w.theta2))
                ax.text(0.65 * np.cos(ang), 0.65 * np.sin(ang), f"{pct:.0f}%",
                        ha="center", va="center", fontsize=9,
                        color=_ink_on(c), fontweight="bold")
            top = int(np.argmax(vals))
            ax.set_title(f"{head}\nlargest: {order[top].replace('_', ' ')} "
                         f"{100.0 * vals[top] / total:.0f}%", fontsize=9.5,
                         color=ARM_COLORS[arm], fontweight="bold")
            ax.set_aspect("equal")
    if seen:
        fig.legend(handles=[Patch(color=CONSTRAINT_COLORS[c],
                                  label=c.replace("_", " ")) for c in seen],
                   loc="lower center", ncol=min(len(seen), 5), frameon=False,
                   fontsize=9, labelcolor=PLOT_INK2)
    fig.suptitle("What the violation burden is, per evaluation tier "
                 "(share of the tier's summed burden by constraint)",
                 fontweight="bold", fontsize=11.5, color=PLOT_INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    _savefig(fig, "compare_tier_burden_pie")


TRAIN_CURVE_SMOOTH_ITERS = 50


def _moving_average(y, window):
    """Centered moving average that shrinks at the edges and skips NaNs."""
    y = np.asarray(y, dtype=float)
    # np.convolve(mode="same") returns max(len(y), window) samples, so a log
    # shorter than the window (an early-stopped run) must clamp it.
    window = min(int(window), y.size)
    if window <= 1 or y.size == 0:
        return y
    ok = np.isfinite(y)
    k = np.ones(window)
    num = np.convolve(np.where(ok, y, 0.0), k, mode="same")
    den = np.convolve(ok.astype(float), k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def plot_training_curves():
    """Plot each trained arm's training reward and violations."""
    plt = _pyplot("training curves")
    if plt is None:
        return
    logs = {}
    for arm in ARM_ORDER:
        path = os.path.join(LOG_DIR, f"{ARMS[arm].label.lower()}_log.json")
        if os.path.exists(path):
            with open(path) as f:
                logs[arm] = json.load(f)
    if not logs:
        return
    fig, (ax_r, ax_v) = plt.subplots(1, 2, figsize=(17, 6))
    fig.patch.set_facecolor(PLOT_SURFACE)
    w = TRAIN_CURVE_SMOOTH_ITERS
    for arm, lg in logs.items():
        col = ARM_COLORS[arm]
        ax_r.plot(lg["iter"], _moving_average(lg["avg_reward"], w), color=col,
                  lw=1.8, label=arm)
        ax_v.plot(lg["iter"], _moving_average(lg["avg_violations"], w),
                  color=col, lw=1.8, label=arm)
    ax_v.set_yscale("symlog", linthresh=1.0)
    ax_v.set_ylim(bottom=0)
    for ax, title, ylab in ((ax_r, "REWARD (higher is better)", "episode reward"),
                            (ax_v, "VIOLATIONS (lower is better)",
                             "violation events per episode (symlog)")):
        _style_ax(ax)
        ax.grid(alpha=0.5, color=PLOT_GRID, lw=0.8)
        ax.set_title(title, fontsize=11, color=PLOT_INK, fontweight="bold", pad=10)
        ax.set_ylabel(ylab, fontsize=9.5, color=PLOT_INK2)
        ax.set_xlabel("training iteration", fontsize=9.5, color=PLOT_INK2)
        ax.legend(frameon=False, fontsize=9, labelcolor=PLOT_INK2)
    fig.suptitle(f"Training curves -- reward and violations per arm "
                 f"({w}-iteration centered moving average)",
                 fontweight="bold", fontsize=12.5, color=PLOT_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _savefig(fig, "training_curves")
    print(f"wrote {os.path.join(PLOT_DIR, 'training_curves.png')}")
