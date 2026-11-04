"""Summary plot for the multi-head APN validation (python -m safety.validate)."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os

# ─── Third-Party ──────────────────────────────────────────────────────────────
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

matplotlib.use('Agg')

SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, GRAY = '#2a78d6', '#eb6834', '#c9c8c2'
STATUS = {True: ('PASS', '#0ca30c'), False: ('FAIL', '#d03b3b')}


def _badge(ax, x, y, ok, **kw):
    label, color = STATUS[ok]
    ax.text(x, y, label, fontsize=8.5, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', fc=color, ec='none'), **kw)


def _style(ax, title, ok=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', fontsize=11, color=INK, pad=10)
    if ok is not None:
        _badge(ax, 1.0, 1.03, ok, transform=ax.transAxes, ha='right', va='bottom')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRAY)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _footnote(ax, text):
    ax.text(0.5, -0.17, text, transform=ax.transAxes, ha='center', va='top', fontsize=8, color=INK_2)


def _exact(ax, ok, m):
    _style(ax, f'1 · Exact heads vs env  ({m["n"]:,} holds)', ok)
    x = np.arange(3)
    w = 0.38
    ax.bar(x - w / 2 - 0.01, m["mass"], w, color=BLUE, label='reactor mass (of Mmax)')
    ax.bar(x + w / 2 + 0.01, m["storage"], w, color=ORANGE, label='storage (of S_max)')
    ax.axhline(m["limit"], color=INK_2, ls='--', lw=1)
    ax.text(2.6, m["limit"] * 1.4, 'limit: mass p99 < 1e-3', ha='right', fontsize=8, color=INK_2)
    ax.set_yscale('log')
    ax.set_ylim(1e-9, 1e-1)
    ax.set_xticks(x, ['p50', 'p99', 'max'])
    ax.set_ylabel('|predicted − env| after one hold', fontsize=9, color=INK_2)
    ax.grid(axis='x', visible=False)
    ax.legend(fontsize=8, frameon=False, loc='upper left')


def _learned(ax, ok, m):
    _style(ax, f'2 · Learned heads on fresh data  ({m["n"]:,} pairs)', ok)
    rows = [('heat R²', *m["heat_r2"]), ('purity sign\naccuracy', *m["purity_sign_acc"])]
    y = np.arange(len(rows))[::-1]
    for yi, (_, val, thr) in zip(y, rows):
        ax.plot([thr, 1.0], [yi, yi], color=GRID, lw=6, solid_capstyle='butt', zorder=1)
        ax.plot([thr, thr], [yi - 0.22, yi + 0.22], color=INK_2, ls='--', lw=1, zorder=2)
        ax.scatter(val, yi, s=70, color=BLUE, edgecolor='white', linewidth=1.5, zorder=3)
        ax.text(val, yi + 0.3, f'{val:.4f}', ha='center', fontsize=9, color=INK)
        ax.text(thr, yi - 0.38, f'min {thr:.2f}', ha='center', fontsize=8, color=INK_2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0.95, 1.001)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(axis='y', visible=False)
    _footnote(ax, f'heat MAE {m["heat_mae"]:.1f} kg/h · titer MAE {m["titer_mae"]:.2f} g/L · '
                  f'purity sign within 5 g/L of spec {m["purity_sign_acc_near_spec"]:.3f}')


def _gradient(ax, ok, m):
    _style(ax, '4 · Safety-loss gradient', ok)
    vals = [m["before"], m["after"]]
    ax.bar([0, 1], vals, 0.5, color=[GRAY, BLUE])
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.4f}', ha='center', va='top', fontsize=9, color=INK)
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_xticks([0, 1], ['before', 'after one step'])
    ax.set_ylim(min(vals) * 1.2, 0.0)
    ax.set_ylabel('mean violated margin', fontsize=9, color=INK_2)
    ax.grid(axis='x', visible=False)
    _footnote(ax, f'{m["raised"]:.0%} of {m["n"]:,} violated margins raised · pass: mean shortfall shrinks')


def _projection(ax, ok, m):
    _style(ax, f'3 · Projection on {m["n"]:,} plant states with aggressive actions', ok)
    names = list(m["groups"])
    y = np.arange(len(names))[::-1]
    cols = (1.04, 1.2, 1.33)
    for yi, name in zip(y, names):
        g = m["groups"][name]
        cert, past = g["certified"], g["past"]
        left = 0.0
        for width, color in ((cert, BLUE), (past, ORANGE), (max(1.0 - cert - past, 0.0), GRAY)):
            if width > 0:
                ax.barh(yi, max(width - 0.003, 0.0), 0.55, left=left, color=color, zorder=2)
            left += width
        ax.text(cert / 2, yi, f'{cert:.1%}', ha='center', va='center', fontsize=8.5,
                color='white', fontweight='bold', zorder=3)
        ax.text(cols[0], yi, f'{g["violated"]:.2%}', va='center', fontsize=9, color=INK)
        ax.text(cols[1], yi, f'≤ {g["limit"]:.0%}', va='center', fontsize=9, color=INK_2)
        _badge(ax, cols[2], yi, g["violated"] <= g["limit"], va='center')
        if g["raw"] is not None:
            ax.text(0.005, yi - 0.4, f'raw actions violate {g["raw"]:.1%}', ha='left', va='center',
                    fontsize=7.5, color=INK_2)
    top = len(names) - 0.45
    ax.text(cols[0], top, 'violated\nwhen certified', va='bottom', fontsize=8, color=INK_2)
    ax.text(cols[1], top, 'allowed', va='bottom', fontsize=8, color=INK_2)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1.42)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.spines['bottom'].set_bounds(0, 1)
    ax.set_ylim(-0.7, len(names) - 0.2)
    ax.grid(axis='y', visible=False)
    ax.text(0.5 / 1.42, -0.1, 'share of plant states (purity: pre-harvest batches)', transform=ax.transAxes,
            ha='center', va='top', fontsize=9, color=INK_2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, GRAY)]
    ax.legend(handles, ['certified safe', 'already past the limit (no action can fix)', 'not certified'],
              fontsize=8, frameon=False, ncol=3, loc='upper left', bbox_to_anchor=(0, -0.17))


def _latency(ax, ok, m):
    _style(ax, '5 · Projection latency')
    ax.bar(np.arange(3), m["ms"], 0.5, color=BLUE)
    for i, v in enumerate(m["ms"]):
        ax.text(i, v + max(m["ms"]) * 0.02, f'{v:.1f}', ha='center', fontsize=9, color=INK)
    ax.set_xticks(np.arange(3), ['p50', 'p99', 'max'])
    ax.set_ylabel('ms per control step (CPU, 3 reactors)', fontsize=9, color=INK_2)
    ax.set_ylim(0, max(m["ms"]) * 1.2)
    ax.grid(axis='x', visible=False)
    _footnote(ax, 'reported only, no limit')


def plot_validation(results, out_dir='am_plot', name='am_validation.png'):
    """Save the five validation checks as one figure.

    Args:
        results: {check name: (passed, metrics)} as returned by safety.validate's checks.
        out_dir: Output directory.
        name: File name.

    Returns:
        Path of the saved figure.
    """
    fig = plt.figure(figsize=(17, 10.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.42,
                          left=0.08, right=0.98, top=0.87, bottom=0.11)
    top = gs[0].subgridspec(1, 3, wspace=0.32)
    bottom = gs[1].subgridspec(1, 2, width_ratios=[2.6, 1], wspace=0.2)
    _exact(fig.add_subplot(top[0]), *results["exact heads"])
    _learned(fig.add_subplot(top[1]), *results["learned heads"])
    _gradient(fig.add_subplot(top[2]), *results["safety gradient"])
    _projection(fig.add_subplot(bottom[0]), *results["projection"])
    _latency(fig.add_subplot(bottom[1]), *results["latency"])

    n_pass = sum(ok for ok, _ in results.values())
    fig.suptitle(f'Multi-head APN safety layer: validation against the real plant  '
                 f'({n_pass}/{len(results)} checks pass)', x=0.08, y=0.965, ha='left', fontsize=14,
                 fontweight='bold', color=INK)
    fig.text(0.08, 0.912, 'Pass rules: exact heads p99 error < 1e-3 · heat R² > 0.98 and purity sign > 0.97 · '
             'projection never wrong when it certifies (0% overflow, ≤1% underflow/heat, ≤3% purity) · '
             'gradient shrinks the shortfall.\nCoverage (how often it can certify) is shown, not graded.',
             fontsize=9, color=INK_2)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
