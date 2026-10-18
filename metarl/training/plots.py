"""Figures: arm comparisons, disturbance responses and training curves."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import json
import os

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.env.config import (MIN_PRODUCT_TITER, N_REACTORS, S_MAX, S_MIN,
                               REACTOR_FLOW_SCALE)
from metarl.training.config import (CONSTRAINT_NAMES, DEPTH_CONSTRAINTS,
                                    Z_INFORM_MIN_SNR, LOG_DIR, PLOT_DIR,
                                    TITER_PANEL_FLOOR_PAD, DISTURB_PROBES)
from metarl.training.arms import ARMS, active_arms
from metarl.training.metrics import episodes_violated, _paired_delta

ARM_COLORS = {"baseline": "#eb6834", "saferl": "#1baf7a", "meta": "#2a78d6"}
PLOT_INK, PLOT_INK2, PLOT_GRID = "#0b0b0b", "#52514e", "#d8d7d2"
PLOT_NEUTRAL = "#b4b3ad"
# Constraints counted per step (product_purity counts per rejected lot).
STEP_CONSTRAINTS = ("overflow", "underflow", "harvest_titer", "instability",
                    "mass_balance", "storage_overflow", "storage_underflow",
                    "turnaround_conflict")
CONSTRAINT_COLORS = {
    "overflow": "#2a78d6",
    "underflow": "#eb6834",
    "harvest_titer": "#1baf7a",
    "product_purity": "#eda100",
    "instability": "#e87ba4",
    "mass_balance": "#008300",
    "storage_overflow": "#6a4fb3",
    "storage_underflow": "#8c6d1f",
    "turnaround_conflict": "#0f8b8d",
}
VIOL_MIX_MIN_LABEL_PCT = 6.0
ACTUATORS = (
    ("stage-time multiplier (x)", lambda a: a * 1.5 + 0.5),
    ("LF feed (frac. of max)", lambda a: a),
    ("C5 feed (frac. of max)", lambda a: a),
    ("discharge (frac. of max)", lambda a: a),
)


def _style_ax(ax, INK2, GRID):
    """Shared axis styling."""
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", alpha=0.5, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def _plot_distributions(results, names, labels, arms, colors, INK, INK2, GRID,
                        plt):
    """Per-episode on-spec ethanol per arm and scenario (box plus points)."""
    cols = min(len(names), 6)
    rows = -(-len(names) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4.1 * cols, 5.0 * rows),
                            squeeze=False, sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axs.flat[len(names):]:
        ax.set_visible(False)
    for ax, n, lab in zip(axs.flat, names, labels):
        _style_ax(ax, INK2, GRID)
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
                        medianprops={"color": INK, "lw": 1.8},
                        whiskerprops={"color": INK2, "lw": 1.1},
                        capprops={"color": INK2, "lw": 1.1},
                        flierprops={"marker": "o", "ms": 3,
                                    "markerfacecolor": INK2,
                                    "markeredgecolor": "none", "alpha": 0.5})
        for patch, arm in zip(bp["boxes"], present):
            patch.set_facecolor(colors[arm])
            patch.set_alpha(0.35)
            patch.set_edgecolor(colors[arm])
            patch.set_linewidth(1.8)
        for i, (v, arm) in enumerate(zip(data, present), start=1):
            jitter = (np.arange(v.size) % 5 - 2) * 0.035
            ax.plot(np.full(v.size, i) + jitter, v, "o", ms=3.5,
                    color=colors[arm], alpha=0.75, markeredgecolor="none")
        ax.set_xticks(range(1, len(present) + 1))
        ax.set_xticklabels(present, rotation=20, ha="right", fontsize=9,
                           color=INK2)
        ax.set_title(lab, fontsize=10.5, color=INK, fontweight="bold")
    for row in axs:
        row[0].set_ylabel("On-spec ethanol per episode  (x10^7)",
                          fontsize=9.5, color=INK2)
    fig.suptitle("Per-episode production distributions", fontweight="bold",
                 fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig.savefig(os.path.join(PLOT_DIR, "distributions.png"), dpi=150)
    plt.close(fig)


def _plot_adapt(results, names, labels, INK, INK2, GRID, plt):
    """ADAPT (meta vs saferl) as paired deltas per scenario, three panels."""
    from matplotlib.patches import Patch

    def _per_ep(r, kind):
        by = r.get("viol_by_constraint_per_ep") or {}
        if kind == "steps":
            parts = [np.asarray(by[c], dtype=float)
                     for c in STEP_CONSTRAINTS if c in by]
            return np.sum(parts, axis=0) if parts else np.zeros(0)
        if kind == "feed":
            return np.asarray(r.get("feed_used_per_ep", []), dtype=float)
        return np.asarray(r.get("on_spec_ethanol_per_ep", []), dtype=float)

    panels = (
        ("steps", "Violation events avoided per episode (inventory, storage, "
                  "harvest titer)", "saferl - meta  (events)",
         lambda v: f"{v:+.1f}"),
        ("eth", "On-spec ethanol gained", "meta - saferl  (% of saferl)",
         lambda v: f"{v:+.1f}%"),
        ("feed", "Feed saved (a win only at equal product)",
         "saferl - meta  (% of saferl)", lambda v: f"{v:+.1f}%"),
    )
    x = np.arange(len(names))
    fig, axs = plt.subplots(len(panels), 1, sharex=True,
                            figsize=(max(9.0, 0.9 * len(names)),
                                     3.5 * len(panels)))
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (kind, title, ylab, fmt) in zip(axs, panels):
        _style_ax(ax, INK2, GRID)
        vals, cis, cols = [], [], []
        for n in names:
            s = _per_ep(results[n]["saferl"], kind)
            m = _per_ep(results[n]["meta"], kind)
            if s.size == 0 or s.size != m.size:
                vals.append(np.nan)
                cis.append(0.0)
                cols.append(PLOT_NEUTRAL)
                continue
            if kind == "eth":
                d = _paired_delta(m, s)
                scale = abs(float(np.mean(s)))
                k = 100.0 / scale if scale > 0 else np.nan
            elif kind == "feed":
                d = _paired_delta(s, m)
                scale = abs(float(np.mean(s)))
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
               ecolor=INK2, error_kw={"lw": 1.1})
        ax.axhline(0.0, color=INK2, lw=1.0, zorder=1)
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
                        fontsize=7.2, color=INK2)
        ax.set_title(title, fontsize=10.5, color=INK, fontweight="bold", pad=8)
        ax.set_ylabel(ylab, fontsize=9.5, color=INK2)
        ax.margins(y=0.22)
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(labels, rotation=25, ha="right", fontsize=9,
                            color=INK2)
    handles = [
        Patch(color=ARM_COLORS["meta"], label="meta better (95% CI excludes 0)"),
        Patch(color=ARM_COLORS["saferl"], label="saferl better (95% CI excludes 0)"),
        Patch(color=PLOT_NEUTRAL, label="no significant or practical difference"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("ADAPT -- meta vs saferl, paired over the same episodes "
                 "(up = meta better)", fontweight="bold", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig.savefig(os.path.join(PLOT_DIR, "compare_adapt.png"), dpi=150)
    plt.close(fig)


def _plot_violation_mix(results, names, labels, INK, INK2, GRID, plt):
    """Per arm, which constraints make up its violations, per scenario."""
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    arms = [a for a in ("saferl", "meta") if a in results[names[0]]]
    order = list(CONSTRAINT_NAMES.values())
    seen = [c for c in order
            if any(float((results[n][a].get("viol_by_constraint") or {})
                         .get(c, 0.0)) > 0 for n in names for a in arms)]

    def _ink_on(fill):
        r, g, b = to_rgb(fill)
        return "#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else INK

    y = np.arange(len(names))
    for arm in arms:
        fig, ax = plt.subplots(figsize=(11.0, 0.42 * len(names) + 2.4))
        fig.patch.set_facecolor("#fcfcfb")
        _style_ax(ax, INK2, GRID)
        ax.grid(axis="y", alpha=0.0)
        ax.grid(axis="x", alpha=0.5, color=GRID, lw=0.8)
        for yi, n in zip(y, names):
            r = results[n][arm]
            by = r.get("viol_by_constraint") or {}
            parts = np.asarray([max(float(by.get(c, 0.0)), 0.0) for c in seen])
            total = parts.sum()
            n_viol, n_eps = episodes_violated(r)
            note = (f"{r['violations']:.1f}/ep · {n_viol}/{n_eps} eps"
                    if total > 0 else f"no violations · 0/{n_eps} eps")
            ax.annotate(note, (100.0, yi), xytext=(6, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=8, color=INK2, annotation_clip=False)
            if total <= 0:
                continue
            left = 0.0
            for c, v in zip(seen, parts):
                pct = 100.0 * v / total
                if pct <= 0:
                    continue
                ax.barh(yi, pct, 0.68, left=left, color=CONSTRAINT_COLORS[c],
                        edgecolor="#fcfcfb", linewidth=1.5)
                if pct >= VIOL_MIX_MIN_LABEL_PCT:
                    ax.text(left + pct / 2, yi, f"{pct:.0f}%", ha="center",
                            va="center", fontsize=7.5,
                            color=_ink_on(CONSTRAINT_COLORS[c]))
                left += pct
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9, color=INK2)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of violation events (%)", fontsize=9.5, color=INK2)
        handles = [Patch(color=CONSTRAINT_COLORS[c], label=c.replace("_", " "))
                   for c in seen]
        if handles:
            fig.legend(handles=handles, loc="upper center",
                       ncol=min(len(handles), 4), frameon=False, fontsize=9,
                       labelcolor=INK2, bbox_to_anchor=(0.5, 0.94))
        title = {"saferl": "Safe RL", "meta": "Meta-RL"}[arm]
        fig.suptitle(f"{title} -- which constraints its violations are, per "
                     f"scenario  (events; a rejected lot counts once)",
                     fontweight="bold", fontsize=12, color=INK)
        fig.tight_layout(rect=(0, 0, 0.88, 0.86))
        os.makedirs(PLOT_DIR, exist_ok=True)
        fig.savefig(os.path.join(PLOT_DIR, f"violation_mix_{arm}.png"), dpi=150)
        plt.close(fig)


def _plot_violation_severity(results, names, labels, INK, INK2, GRID, plt):
    """How far past its limit each arm goes when it violates, per constraint."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    arms = [a for a in ("saferl", "meta") if a in results[names[0]]]

    def _depth(r, c):
        n = float(np.sum(r["viol_by_constraint_per_ep"].get(c, [])))
        if n <= 0:
            return np.nan, np.nan
        s = float(np.sum(r["viol_depth_sum_per_ep"].get(c, [])))
        return 100.0 * s / n, 100.0 * float(
            np.max(r["viol_depth_max_per_ep"].get(c, [0.0])))

    cons = [CONSTRAINT_NAMES[cid] for cid in DEPTH_CONSTRAINTS]
    cons = [c for c in cons
            if any(np.isfinite(_depth(results[n][a], c)[0])
                   for n in names for a in arms)]
    if not cons:
        print("[plot] violation_severity.png skipped: no violation with a depth")
        return
    x = np.arange(len(names))
    w = 0.36
    fig, axs = plt.subplots(len(cons), 1, sharex=True, squeeze=False,
                            figsize=(max(9.0, 0.9 * len(names)),
                                     3.3 * len(cons) + 1.2))
    axs = axs[:, 0]
    fig.patch.set_facecolor("#fcfcfb")
    for ax, c in zip(axs, cons):
        _style_ax(ax, INK2, GRID)
        for i, arm in enumerate(arms):
            pos = x + (i - (len(arms) - 1) / 2) * w
            mean, worst = zip(*(_depth(results[n][arm], c) for n in names))
            mean, worst = np.asarray(mean), np.asarray(worst)
            ax.bar(pos, np.nan_to_num(mean), w * 0.92, color=ARM_COLORS[arm],
                   label=arm)
            ax.plot(pos, worst, "_", ms=11, mew=2.0, color=INK2,
                    linestyle="none")
            for xi, m, wv in zip(pos, mean, worst):
                txt, yv = (f"{m:.1f}", wv) if np.isfinite(m) else ("-", 0.0)
                ax.annotate(txt, (xi, yv), xytext=(0, 4),
                            textcoords="offset points", ha="center",
                            va="bottom", fontsize=6.8, color=INK2, rotation=90)
        ax.set_title(c.replace("_", " "), fontsize=10.5, color=INK,
                     fontweight="bold", pad=6)
        ax.set_ylabel("Depth past limit (%)", fontsize=9.5, color=INK2)
        top = np.nanmax([_depth(results[n][a], c)[1] for n in names for a in arms])
        ax.set_ylim(0, 1.35 * top if top > 0 else 1.0)
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(labels, rotation=25, ha="right", fontsize=9,
                            color=INK2)
    handles = [Patch(color=ARM_COLORS[a], label=f"{a}: mean per violation")
               for a in arms]
    handles.append(Line2D([], [], marker="_", ms=11, mew=2.0, color=INK2,
                          linestyle="none", label="deepest single violation"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, fontsize=9, labelcolor=INK2,
               bbox_to_anchor=(0.5, 1.0 - 0.55 / fig.get_figheight()))
    fig.suptitle("Violation severity -- how far past the limit (lower is "
                 "better; - = none of that kind)", fontweight="bold",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 1.0 - 0.85 / fig.get_figheight()))
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig.savefig(os.path.join(PLOT_DIR, "violation_severity.png"), dpi=150)
    plt.close(fig)


def scenario_probe(name):
    """The DISTURB_PROBES key traced on scenario ``name`` (its step pattern), or None."""
    label, _, pattern = name.rpartition("_")
    return label if pattern == "step" and label in DISTURB_PROBES else None


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


def _plot_disturbance_response(results, names, arms, colors, INK, INK2, GRID,
                               plt):
    """The faulted reactor's answering actuator (solid) vs the faulted quantity."""
    from matplotlib.lines import Line2D

    dist = [n for n in names if scenario_probe(n)
            and all("traces" in results[n][a] for a in arms)]
    if not dist:
        print("[plot] disturbance response skipped: no per-episode traces")
        return
    fig, axs = plt.subplots(2, len(dist), figsize=(4.4 * len(dist), 8.6),
                            squeeze=False, sharex="col",
                            gridspec_kw={"height_ratios": [3, 1.3]})
    fig.patch.set_facecolor("#fcfcfb")
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
            ax.plot(fn(act), color=colors[arm], lw=1.6, zorder=3)
            ax_s.plot(np.asarray(tr["storage"][ep]) / S_MAX, color=colors[arm],
                      lw=1.3)
            handles.append(Line2D([0], [0], color=colors[arm], lw=1.6,
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
            ax2.plot(ref, color=INK, ls=":", lw=1.9)
            handles.append(Line2D([0], [0], color=INK, ls=":", lw=1.9,
                                  label=f"{qlabel} (right)"))
        else:
            for arm, t in traces.items():
                ax2.plot(t, color=colors[arm], ls=":", lw=1.5)
                handles.append(Line2D([0], [0], color=colors[arm], ls=":",
                                      lw=1.5, label=f"{arm} {qlabel} (right)"))
        ax2.set_ylabel(qlabel, fontsize=8.5, color=INK2)
        ax2.tick_params(colors=INK2, labelsize=8)
        for side in ("top", "left", "bottom"):
            ax2.spines[side].set_visible(False)
        ax2.spines["right"].set_color(GRID)
        onset = results[n][arms[0]].get("onset_outer")
        for a_ in (ax, ax_s):
            if onset is not None:
                a_.axvline(onset, color=INK2, ls="--", lw=1.0, zorder=1)
            _style_ax(a_, INK2, GRID)
        if onset is not None:
            handles.append(Line2D([0], [0], color=INK2, ls="--", lw=1.0,
                                  label="fault onset"))
        for lim, lab_ in ((S_MIN / S_MAX, "S_MIN"), (1.0, "S_MAX")):
            ax_s.axhline(lim, color=INK2, ls=":", lw=0.9)
        ax_s.set_ylabel("storage / S_MAX", fontsize=8.5, color=INK2)
        ax_s.set_xlabel("control step", fontsize=9, color=INK2)
        ax.set_ylabel(alabel + ("" if reactor < 0 else f", reactor {reactor}"),
                      fontsize=9, color=INK2)
        title = n.replace("disturb_", "").replace("_", " ")
        where = "plant-wide" if reactor < 0 else f"reactor {reactor}"
        ax.set_title(f"{title} ({where})\nepisode #{ep}", fontsize=10.5,
                     color=INK, fontweight="bold")
        ax.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=7.5,
                  labelcolor=INK2)
    fig.suptitle("Disturbance response -- the faulted reactor's answering "
                 "actuator (solid), the faulted quantity (dotted) and the "
                 "storage tank (lower), all arms on one shared episode",
                 fontweight="bold", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig.savefig(os.path.join(PLOT_DIR, "disturbance_response.png"), dpi=150)
    plt.close(fig)


def _plot_comparison(results):
    """Render the evaluation figures into PLOT_DIR.

    compare_safety / compare_episodes_violated / compare_production /
    compare_quality / compare_reactors (per-reactor on-spec ethanol and
    violation events), compare_adapt, violation_mix_*, violation_severity,
    distributions, disturbance_response.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                             # pragma: no cover
        print(f"[plot] skipped: {exc}")
        return

    names = list(results.keys())
    labels = [n.replace("disturb_", "D ").replace("_", " ") for n in names]
    arms = tuple(a for a in active_arms() if a in results[names[0]])
    colors = ARM_COLORS
    INK, INK2, GRID = PLOT_INK, PLOT_INK2, PLOT_GRID
    x = np.arange(len(names))

    def _style(ax, title, ylab):
        ax.set_title(title, fontsize=11, color=INK, fontweight="bold", pad=10)
        ax.set_ylabel(ylab, fontsize=9.5, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9,
                           color=INK2)
        _style_ax(ax, INK2, GRID)
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
                        fontsize=6.8, color=INK2, rotation=90)

    w = 0.8 / max(len(arms), 1)
    off = lambda i: (i - (len(arms) - 1) / 2) * w          # noqa: E731
    os.makedirs(PLOT_DIR, exist_ok=True)
    arm_titles = {"baseline": "Unconstrained RL", "saferl": "Safe RL",
                  "meta": "Meta-RL"}
    suptitle = " vs ".join(arm_titles.get(a, a) for a in arms)
    if "meta" in arms:
        snr = float(np.mean([results[n]["meta"]["z_snr"] for n in names]))
        suptitle += (f"   |   latent separability SNR {snr:.2f} "
                     f"({'above' if snr >= Z_INFORM_MIN_SNR else 'BELOW'} the "
                     f"{Z_INFORM_MIN_SNR:g} gate)")
    suptitle += f"   |   {N_REACTORS} reactors + central storage"

    def _new_fig():
        fig, ax = plt.subplots(figsize=(max(9.0, 0.9 * len(names)), 6.4))
        fig.patch.set_facecolor("#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        return fig, ax

    def _save(fig, stem):
        fig.suptitle(suptitle, fontweight="bold", fontsize=12, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(os.path.join(PLOT_DIR, f"{stem}.png"), dpi=150)
        plt.close(fig)

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
            ax.bar(pos, vals, w, label=arm, color=colors[arm], yerr=yerr,
                   capsize=3, ecolor=INK2, error_kw={"lw": 1.2})
            _label_bars(ax, pos, vals, tops, fmt)
        _style(ax, title, ylab)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2)

    fig, ax = _new_fig()
    _bar_panel(ax, "violations", None, "Constraint-violation events",
               "SAFETY -- violations (lower is better)", fmt=lambda v: f"{v:.1f}")
    _save(fig, "compare_safety")

    fig, ax = _new_fig()
    counts = {arm: [episodes_violated(results[n][arm]) for n in names]
              for arm in arms}
    n_eps = max((e for arm in arms for _, e in counts[arm]), default=0)
    for i, arm in enumerate(arms):
        vals = np.asarray([k for k, _ in counts[arm]], dtype=float)
        pos = x + off(i)
        ax.bar(pos, vals, w, label=arm, color=colors[arm])
        _label_bars(ax, pos, vals, vals, lambda v: f"{v:.0f}")
    _style(ax, "SAFETY -- episodes with any constraint violation "
               "(lower is better)", f"Episodes with >= 1 violation (of {n_eps})")
    if n_eps:
        ax.axhline(n_eps, color=INK2, ls="--", lw=1.0, zorder=0)
        ax.set_ylim(0, n_eps * 1.12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    _save(fig, "compare_episodes_violated")

    if "saferl" in arms and "meta" in arms:
        _plot_adapt(results, names, labels, INK, INK2, GRID, plt)
    _plot_violation_mix(results, names, labels, INK, INK2, GRID, plt)
    _plot_violation_severity(results, names, labels, INK, INK2, GRID, plt)

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
        ax.errorbar(pos, vals, yerr=err, fmt="o", ms=8, color=colors[arm],
                    label=arm, capsize=4, ecolor=INK2, elinewidth=1.2,
                    mec="white", mew=1.0, linestyle="none", zorder=3)
        for xi, v in zip(pos, vals):
            ax.annotate(f"{v:.1f}", (xi, v), xytext=(7, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=7.2, color=INK2)
        lo_seen.append(float(np.min(vals - err)))
        hi_seen.append(float(np.max(vals + err)))
    _style(ax, "QUALITY -- mean product titer (higher is better)",
           "Mean product titer (g/L)")
    spec = MIN_PRODUCT_TITER or 0.0
    if spec:
        ax.axhline(spec, color=INK2, ls="--", lw=1.2, zorder=0,
                   label=f"purity spec floor ({spec:.0f} g/L)")
    lo, hi = min(lo_seen), max(hi_seen)
    bottom = min(spec - TITER_PANEL_FLOOR_PAD, lo - 1.0) if spec else lo - 1.0
    ax.set_ylim(bottom, max(hi, spec) + 0.10 * max(hi - bottom, 1.0))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    _save(fig, "compare_quality")

    # Per reactor: on-spec ethanol and reactor-local violation events, per arm
    # (averaged over scenarios): which vessel carries the plant's numbers.
    fig, (ax_e, ax_v) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    fig.patch.set_facecolor("#fcfcfb")
    xr = np.arange(N_REACTORS)
    for i, arm in enumerate(arms):
        eth = np.mean([results[n][arm]["reactor_on_spec_ethanol"] for n in names],
                      axis=0)
        vio = np.mean([results[n][arm]["viol_by_reactor"] for n in names], axis=0)
        ax_e.bar(xr + off(i), eth, w, color=colors[arm], label=arm)
        ax_v.bar(xr + off(i), vio, w, color=colors[arm], label=arm)
    for a_, t_ in ((ax_e, "On-spec ethanol per reactor (mean over scenarios)"),
                   (ax_v, "Reactor-local violation events per reactor")):
        _style_ax(a_, INK2, GRID)
        a_.set_xticks(xr)
        a_.set_xticklabels([f"R{i} ({s:g}x pumps)" for i, s in
                            enumerate(REACTOR_FLOW_SCALE)], color=INK2)
        a_.set_title(t_, fontsize=10.5, color=INK, fontweight="bold")
        a_.legend(frameon=False, fontsize=9, labelcolor=INK2)
    _save(fig, "compare_reactors")

    _plot_distributions(results, names, labels, arms, colors, INK, INK2, GRID,
                        plt)
    _plot_disturbance_response(results, names, arms, colors, INK, INK2, GRID,
                               plt)
    print(f"wrote figures to {PLOT_DIR}/")


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
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                             # pragma: no cover
        print(f"[plot] training curves skipped: {exc}")
        return
    logs = {}
    for arm in active_arms():
        path = os.path.join(LOG_DIR, f"{ARMS[arm].label.lower()}_log.json")
        if os.path.exists(path):
            with open(path) as f:
                logs[arm] = json.load(f)
    if not logs:
        return
    INK, INK2, GRID = PLOT_INK, PLOT_INK2, PLOT_GRID
    fig, (ax_r, ax_v) = plt.subplots(1, 2, figsize=(17, 6))
    fig.patch.set_facecolor("#fcfcfb")
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
        _style_ax(ax, INK2, GRID)
        ax.grid(alpha=0.5, color=GRID, lw=0.8)
        ax.set_title(title, fontsize=11, color=INK, fontweight="bold", pad=10)
        ax.set_ylabel(ylab, fontsize=9.5, color=INK2)
        ax.set_xlabel("training iteration", fontsize=9.5, color=INK2)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle(f"Training curves -- reward and violations per arm "
                 f"({w}-iteration centered moving average)",
                 fontweight="bold", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig.savefig(os.path.join(PLOT_DIR, "training_curves.png"), dpi=150)
    plt.close(fig)
    print(f"wrote {os.path.join(PLOT_DIR, 'training_curves.png')}")
