"""The ablation study's figures: learning curves and the final heatmaps."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import math
import os

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from metarl.training.plots import (PLOT_INK, PLOT_INK2, PLOT_SURFACE,
                                   _pyplot, _style_ax, scenario_label)

# One accent series against a gray reference.
_REF = "#898781"
_ACCENT = "#2a78d6"
# Diverging pair, better than full -> no difference -> worse than full.
_BETTER, _NEUTRAL, _WORSE = "#2a78d6", "#f0efec", "#e34948"
CURVE_KEYS = (("heldout_undisturbed_violations", "violation events / episode",
               1.0),
              ("heldout_undisturbed_on_spec_ethanol", "on-spec ethanol (1e7 kg)",
               1e-7))


def _save(fig, plt, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _grid_stats(runs, key):
    """Mean and std over seeds on the union of the runs' check iterations.

    A seed contributes only within its own range (early stopping ends seeds
    at different iterations); the std is 0 where one seed remains.
    """
    xs = np.unique(np.concatenate([r["iter"] for r in runs]))
    m = np.full((len(runs), xs.size), np.nan)
    for i, r in enumerate(runs):
        x, y = r["iter"], r[key]
        ok = np.isfinite(y)
        if ok.sum() < 1:
            continue
        inside = (xs >= x[ok][0]) & (xs <= x[ok][-1])
        m[i, inside] = np.interp(xs[inside], x[ok], y[ok])
    n = np.sum(np.isfinite(m), axis=0)
    keep = n > 0
    xs, m, n = xs[keep], m[:, keep], n[keep]
    mean = np.nansum(m, axis=0) / n
    var = np.nansum((m - mean) ** 2, axis=0) / np.maximum(n - 1, 1)
    return xs, mean, np.where(n > 1, np.sqrt(var), 0.0)


def _band(ax, runs, key, scale, color, label):
    """Plot the seed mean +- 1 std of one curve."""
    xs, mean, std = _grid_stats(runs, key)
    if not xs.size:
        return
    ax.fill_between(xs, (mean - std) * scale, (mean + std) * scale,
                    color=color, alpha=0.18, lw=0)
    ax.plot(xs, mean * scale, color=color, lw=2.0, label=label)


def plot_curves(curves, out_path, label_of):
    """Each ablated variant's held-out learning curves against full's.

    The undisturbed held-out checks, which every variant runs (no_multitask
    has no disturbed panel), as small multiples: one column per variant,
    violations above and on-spec ethanol below.

    Args:
        curves: Config -> per-seed dicts of 'iter' and the CURVE_KEYS arrays.
        out_path: The PNG to write.
        label_of: Display label of a config name.
    """
    plt = _pyplot()
    if plt is None:
        return
    variants = [n for n in curves if n != "full"]
    if "full" not in curves or not variants:
        print("[plot] ablation_curves.png skipped: needs full and at least one "
              "ablated variant")
        return
    cols = min(len(variants), 4)
    groups = math.ceil(len(variants) / cols)
    fig, axs = plt.subplots(2 * groups, cols, figsize=(4.4 * cols,
                                                       6.4 * groups),
                            squeeze=False)
    fig.patch.set_facecolor(PLOT_SURFACE)
    seeds = sorted({len(v) for v in curves.values()})
    for i, name in enumerate(variants):
        g, c = divmod(i, cols)
        for p, (key, ylab, scale) in enumerate(CURVE_KEYS):
            ax = axs[2 * g + p, c]
            _style_ax(ax)
            _band(ax, curves["full"], key, scale, _REF, "Full")
            _band(ax, curves[name], key, scale, _ACCENT, label_of(name))
            if p == 0:
                ax.set_title(label_of(name), fontsize=10.5, color=PLOT_INK,
                             fontweight="bold", pad=8)
                ax.legend(frameon=False, fontsize=8.5, labelcolor=PLOT_INK2)
            else:
                ax.set_xlabel("training iteration", fontsize=9, color=PLOT_INK2)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=9, color=PLOT_INK2)
    for i in range(len(variants), groups * cols):
        g, c = divmod(i, cols)
        axs[2 * g, c].set_visible(False)
        axs[2 * g + 1, c].set_visible(False)
    for p in range(2):
        row_axes = [axs[2 * g + p, c] for g in range(groups)
                    for c in range(cols) if axs[2 * g + p, c].get_visible()]
        lo = min(a.get_ylim()[0] for a in row_axes)
        hi = max(a.get_ylim()[1] for a in row_axes)
        for a in row_axes:
            a.set_ylim(max(lo, 0.0), hi)
    seeds_note = "/".join(str(s) for s in seeds)
    fig.suptitle(f"Ablation learning curves: undisturbed held-out checks "
                 f"(mean +- 1 std over {seeds_note} seeds)",
                 fontweight="bold", fontsize=12, color=PLOT_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, plt, out_path)


def _luminance(rgba):
    """WCAG relative luminance of an RGB(A) tuple in [0, 1]."""
    r, g, b = (((c / 12.92) if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
               for c in rgba[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _text_color(rgba):
    """Ink or surface, whichever contrasts more with the cell."""
    from matplotlib.colors import to_rgba
    cell = _luminance(rgba)
    ink = (cell + 0.05) / (_luminance(to_rgba(PLOT_INK)) + 0.05)
    surface = (_luminance(to_rgba(PLOT_SURFACE)) + 0.05) / (cell + 0.05)
    return PLOT_INK if ink >= surface else PLOT_SURFACE


def _heatmap(ax, values, texts, rows, cols, cmap, norm):
    """Cells with a 2 px surface gap, a text per cell, labels on both axes.

    NaN cells are left as the surface.
    """
    from matplotlib.colors import to_rgba
    mesh = ax.pcolormesh(np.ma.masked_invalid(values), cmap=cmap, norm=norm,
                         edgecolors=PLOT_SURFACE, linewidth=2.0)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = values[i, j]
            rgba = cmap(norm(v)) if np.isfinite(v) else to_rgba(PLOT_SURFACE)
            ax.text(j + 0.5, i + 0.5, texts[i][j], ha="center", va="center",
                    fontsize=8, color=_text_color(rgba), linespacing=1.15)
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(cols, rotation=25, ha="right", fontsize=9,
                       color=PLOT_INK2)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels(rows, fontsize=9, color=PLOT_INK2)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(rows), 0)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    return mesh


def _consistent(row, axis):
    """Whether every paired seed was significant in the same direction."""
    n = row.get(f"seeds_{axis}_paired") or 0
    return bool(n) and (row[f"seeds_{axis}_worse"] == n
                        or row[f"seeds_{axis}_better"] == n)


def plot_final(rows, out_path):
    """Violation burden and net production per config and scenario, vs full.

    Each cell prints the mean over seeds and the paired delta against full
    (``*``: significant in the same direction in every seed); its color is
    that delta, blue where the config does better than full and red where it
    does worse, on both panels.

    Args:
        rows: ``ablation.summary_rows``: one per (config, scenario), with its
            display ``label``.
        out_path: The PNG to write.
    """
    plt = _pyplot()
    if plt is None:
        return
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    configs = list(dict.fromkeys(r["config"] for r in rows))
    labels = {r["config"]: r["label"] for r in rows}
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    get = {(r["config"], r["scenario"]): r for r in rows}
    worse_up = LinearSegmentedColormap.from_list(
        "abl_viol", (_BETTER, _NEUTRAL, _WORSE))
    worse_down = LinearSegmentedColormap.from_list(
        "abl_eth", (_WORSE, _NEUTRAL, _BETTER))
    panels = (
        ("burden", "d_burden_vs_full", "burden", 1.0, "{:.1f}", "{:+.1f}",
         worse_up, 1.0, "VIOLATION BURDEN B PER EPISODE (lower is better)",
         "paired change vs Full (B / episode)"),
        ("score", "d_score_vs_full_pct", "score", 1e-7, "{:.2f}",
         "{:+.1f}%", worse_down, 2.0,
         "NET PRODUCTION Pi, 1e7 kg (higher is better)",
         "paired change vs Full\n(% of Full's on-spec ethanol)"),
    )
    fig, axs = plt.subplots(2, 1, figsize=(max(9.0, 1.15 * len(scenarios) + 3.5),
                                           2 * (1.6 + 0.5 * len(configs))),
                            squeeze=False)
    fig.patch.set_facecolor(PLOT_SURFACE)
    for ax, (key, dkey, axis, scale, vfmt, dfmt, cmap, floor, title,
             cbar_label) in zip(axs[:, 0], panels):
        colors_of = np.full((len(configs), len(scenarios)), np.nan)
        texts = [["-"] * len(scenarios) for _ in configs]
        for i, c in enumerate(configs):
            for j, s in enumerate(scenarios):
                row = get.get((c, s))
                if row is None or row[f"{key}_mean"] is None:
                    continue
                d = 0.0 if c == "full" else row[dkey]
                colors_of[i, j] = np.nan if d is None else d
                delta = ("ref" if c == "full" else "-" if d is None
                         else dfmt.format(d) + ("*" if _consistent(row, axis)
                                                else ""))
                texts[i][j] = (vfmt.format(row[f"{key}_mean"] * scale)
                               + "\n" + delta)
        finite = np.abs(colors_of[np.isfinite(colors_of)])
        lim = max(floor, float(finite.max()) if finite.size else floor)
        norm = Normalize(-lim, lim)
        mesh = _heatmap(ax, colors_of, texts, [labels[c] for c in configs],
                        [scenario_label(s) for s in scenarios], cmap, norm)
        ax.set_title(title, fontsize=11, color=PLOT_INK, fontweight="bold",
                     pad=10, loc="left")
        cb = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.015)
        cb.set_label(cbar_label + "\nblue: better than Full   red: worse",
                     fontsize=8.5, color=PLOT_INK2)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, colors=PLOT_INK2, length=0)
    seeds = sorted({r["seeds"] for r in rows})
    fig.suptitle("Ablation of the meta arm: mean over "
                 + "/".join(str(s) for s in seeds) + " seeds per cell, paired "
                 "delta vs Full beneath (* = significant in every seed)",
                 fontweight="bold", fontsize=12, color=PLOT_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, plt, out_path)
