"""Figures: meta vs saferl by evaluation tier, and training curves."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import json
import os

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.training.config import (CONSTRAINT_NAMES, LOG_DIR, PLOT_DIR,
                                    DISTURB_PROBES)
from metarl.training.arms import ARMS, ARM_ORDER
from metarl.training.metrics import _paired_delta

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


def _ink_on(fill):
    """Ink or white, whichever reads on a bar of colour ``fill``."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(fill)
    return "#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else PLOT_INK


def scenario_probe(name):
    """The DISTURB_PROBES key traced on scenario ``name``, or None.

    A kind's stepped scenario is traced: "<label>_extreme" in ``evaluate``
    (a step at EVAL_ONSET), "<label>_step" in the ablation's grid.
    """
    label, _, pattern = name.rpartition("_")
    return (label if pattern in ("step", "extreme") and label in DISTURB_PROBES
            else None)


def _tier_title(tiers, t):
    """A tier's short name, e.g. "T1  parameter OOD"."""
    return f"{t}  {tiers[t]['label'].split(' (')[0]}"


def _tier_order(results, tiers):
    """Scenario names grouped by tier, and each group's (tier, first, last)."""
    order, spans = [], []
    for t, info in tiers.items():
        names = [n for n in info["scenarios"] if n in results]
        if names:
            spans.append((t, len(order), len(order) + len(names) - 1))
            order += names
    rest = [n for n in results if n not in order]
    if rest:
        spans.append(("other", len(order), len(order) + len(rest) - 1))
        order += rest
    return order, spans


def _pooled(results, names, key, arm):
    """``key`` summed over scenarios ``names`` for ``arm``."""
    return float(sum(results[n][arm][key] for n in names))


def _plot_tiers(results, tiers):
    """compare_tiers.png: meta's pooled gap on saferl in each tier.

    One panel per quantity: net production (``evaluate._tier_summary``'s
    ``score_gain``), violation burden (``burden_reduction``, with each arm's
    mean B per scenario, since a large cut of a small burden is little) and
    the flow-change penalty (actuator wear, pooled here from ``results``).
    A bar takes the colour of the arm it favours; "wins" are the tier's
    significant per-scenario verdicts, meta / saferl / tied.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    names = list(tiers)
    y = np.arange(len(names))[::-1]

    def burden_note(t):
        n = tiers[t]["scenarios"]
        s, m = (_pooled(results, n, "burden", a) / len(n)
                for a in ("saferl", "meta"))
        w = tiers[t]["burden_wins"]
        return f"B {s:.2f} -> {m:.2f}   wins {w['meta']}/{w['saferl']}/{w['tied']}"

    def score_note(t):
        w = tiers[t]["score_wins"]
        return f"wins {w['meta']}/{w['saferl']}/{w['tied']}"

    def wear_cut(t):
        n = tiers[t]["scenarios"]
        ref = _pooled(results, n, "smoothing", "saferl")
        return (1.0 - _pooled(results, n, "smoothing", "meta") / ref
                if ref > 0 else float("nan"))

    panels = [
        ("net production gain vs saferl\n(% of saferl's on-spec ethanol)",
         [tiers[t]["score_gain"] for t in names], score_note),
        ("violation burden cut vs saferl (%)",
         [tiers[t]["burden_reduction"] for t in names], burden_note),
    ]
    if all("smoothing" in results[n][a] for t in names
           for n in tiers[t]["scenarios"] for a in ("saferl", "meta")):
        panels.append(("flow-change wear cut vs saferl (%)",
                       [wear_cut(t) for t in names], lambda t: ""))
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(5.4 * len(panels), 1.5 + 0.8 * len(names)),
                             sharey=True)
    fig.patch.set_facecolor(PLOT_SURFACE)
    for ax, (title, raw, note) in zip(axes, panels):
        vals = np.nan_to_num(100.0 * np.asarray(raw, dtype=float))
        cols = [ARM_COLORS["meta"] if v > 0 else ARM_COLORS["saferl"]
                for v in vals]
        ax.barh(y, vals, height=0.5, color=cols, edgecolor=PLOT_SURFACE, lw=2)
        ax.axvline(0.0, color=PLOT_INK2, lw=1.0)
        span = max(np.max(np.abs(vals)), 1.0)
        ax.set_xlim(-2.6 * span, 2.6 * span)
        for yi, v, t in zip(y, vals, names):
            up = v >= 0
            ax.annotate(f"{v:+.1f}%", (v, yi), xytext=(6 if up else -6, 5),
                        textcoords="offset points", va="center",
                        ha="left" if up else "right", fontsize=9.5,
                        color=PLOT_INK, fontweight="bold")
            ax.annotate(note(t), (v, yi), xytext=(6 if up else -6, -7),
                        textcoords="offset points", va="center",
                        ha="left" if up else "right", fontsize=7.8,
                        color=PLOT_INK2)
        ax.set_title(title, fontsize=10.5, color=PLOT_INK, fontweight="bold")
        _style_ax(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", alpha=0.5, color=PLOT_GRID, lw=0.8)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{_tier_title(tiers, t)}\n"
                             f"{tiers[t]['n']} scenarios" for t in names],
                            fontsize=9.5, color=PLOT_INK2)
    fig.legend(handles=[Patch(color=ARM_COLORS["meta"], label="meta ahead"),
                        Patch(color=ARM_COLORS["saferl"], label="saferl ahead")],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Meta-RL vs Safe RL by evaluation tier   |   pooled over the "
                 "tier's scenarios; wins = significant scenarios meta / saferl "
                 "/ tied; B = saferl -> meta burden per scenario",
                 fontweight="bold", fontsize=10.5, color=PLOT_INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    _savefig(fig, "compare_tiers")


def _plot_tier_burden_pie(results, tiers):
    """compare_tier_burden_pie.png: what each tier's violation burden is.

    A pie per (arm, tier): rows are the arms, columns the tiers. A slice is a
    constraint's share of the arm's burden summed over the tier's scenarios
    (each scenario's mean per episode, the pooling
    ``evaluate._tier_summary`` uses). Shares are within one pie; its title
    gives the tier's mean burden per scenario and the largest constraint.
    """
    import matplotlib.pyplot as plt
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
            head = (f"{arm}   {_tier_title(tiers, t)}\n"
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


def _plot_adapt(results, tiers):
    """compare_adapt.png: the per-scenario paired gaps behind the tiers.

    Two panels, violation burden and net production, meta vs saferl over
    the same episodes, scenarios grouped by tier. A bar is coloured by the
    arm it favours when the 95% CI excludes 0 and the gap is not negligible.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    names, spans = _tier_order(results, tiers)
    labels = [scenario_label(n) for n in names]
    panels = (
        ("burden_per_ep", "Violation burden avoided per episode "
         "(limit-hours)", "saferl - meta  (B)", lambda v: f"{v:+.1f}"),
        ("score_per_ep", "Net production gained (on-spec ethanol less "
         "violations and wear)", "meta - saferl\n(% of saferl on-spec)",
         lambda v: f"{v:+.1f}%"),
    )
    x = np.arange(len(names))
    fig, axs = plt.subplots(len(panels), 1, sharex=True,
                            figsize=(max(9.0, 0.62 * len(names) + 2.0), 7.6))
    fig.patch.set_facecolor(PLOT_SURFACE)
    for ax, (key, title, ylab, fmt) in zip(axs, panels):
        _style_ax(ax)
        vals, cis, cols = [], [], []
        for n in names:
            s = np.asarray(results[n]["saferl"].get(key, []), dtype=float)
            m = np.asarray(results[n]["meta"].get(key, []), dtype=float)
            if s.size == 0 or s.size != m.size:
                vals.append(np.nan)
                cis.append(0.0)
                cols.append(PLOT_NEUTRAL)
                continue
            if key == "score_per_ep":
                # Net production can sit near zero: read it as a share of
                # saferl's gross on-spec output.
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
                txt, yv, dy, va = "n/a", 0.0, 3, "bottom"
            else:
                txt = "0" if v == 0 else fmt(v)
                up = v >= 0
                yv = v + ci if up else v - ci
                dy, va = (3, "bottom") if up else (-3, "top")
            ax.annotate(txt, (xi, yv), xytext=(0, dy),
                        textcoords="offset points", ha="center", va=va,
                        fontsize=7.2, color=PLOT_INK2)
        for i, (_, lo, hi) in enumerate(spans):
            if i % 2:
                ax.axvspan(lo - 0.5, hi + 0.5, color=PLOT_GRID, alpha=0.25,
                           lw=0, zorder=0)
        ax.set_title(title, fontsize=10.5, color=PLOT_INK, fontweight="bold",
                     pad=18 if ax is axs[0] else 8)
        ax.set_ylabel(ylab, fontsize=9.5, color=PLOT_INK2)
        ax.margins(y=0.22)
    for t, lo, hi in spans:
        axs[0].text(0.5 * (lo + hi), 1.01, t if t == "other"
                    else _tier_title(tiers, t), ha="center", va="bottom",
                    transform=axs[0].get_xaxis_transform(), fontsize=9,
                    color=PLOT_INK2, fontweight="bold")
    axs[-1].set_xlim(-0.6, len(names) - 0.4)
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(labels, rotation=30, ha="right", fontsize=9,
                            color=PLOT_INK2)
    fig.legend(handles=[
        Patch(color=ARM_COLORS["meta"], label="meta better (95% CI excludes 0)"),
        Patch(color=ARM_COLORS["saferl"], label="saferl better (95% CI excludes 0)"),
        Patch(color=PLOT_NEUTRAL, label="no significant or practical difference"),
    ], loc="upper center", ncol=3, frameon=False, fontsize=9,
        labelcolor=PLOT_INK2, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("Meta-RL vs Safe RL per scenario, paired over the same "
                 "episodes (up = meta better)", fontweight="bold", fontsize=12,
                 color=PLOT_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _savefig(fig, "compare_adapt")


def plot_eval(results, tiers):
    """Render the evaluation figures into PLOT_DIR.

    compare_tiers (meta's gap on saferl per tier), compare_tier_burden_pie
    (what each tier's burden is) and compare_adapt (the per-scenario gaps
    behind the tiers). All three contrast meta with saferl.
    """
    arms = next(iter(results.values())) if results else {}
    if not ("meta" in arms and "saferl" in arms):
        print("[plot] evaluation figures skipped: they need both arms")
        return
    if not all("burden" in r[a] for r in results.values() for a in arms):
        print("[plot] evaluation figures skipped: eval.json predates the "
              "burden metric (re-run evaluate)")
        return
    if _pyplot("evaluation figures") is None:
        return
    if tiers:
        _plot_tiers(results, tiers)
        _plot_tier_burden_pie(results, tiers)
    _plot_adapt(results, tiers or {})
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
