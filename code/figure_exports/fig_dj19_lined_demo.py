from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
import origin_style as fs
from engineering_figures import axes_mm, note, WIDTH
import matplotlib.pyplot as plt
KEY1, KEY2, KEY3 = (fs.BLACK, fs.RED, fs.BLUE)
GUIDE = '#8c8c8c'
JOB = 'lined-section-demo'
SRC_SUM = 'results/data/derived/lined_demo_summary.csv'
SRC_ACT = 'results/data/derived/lined_demo_actions.csv'
C_BED = {90.0: KEY1, 45.0: KEY2, 0.0: KEY3}
LS_BED = {90.0: '-', 45.0: (0, (4.5, 1.6)), 0.0: (0, (1.4, 1.4))}
M_RATIO = {1.25: 'o', 2.625: 's', 4.0: '^'}
C_GUIDE = GUIDE
ACTION_RATIO = 4.0

def pin_ticks(ax) -> None:
    """Keep ticks within the displayed axis limits."""
    for axis, lim, setter in ((ax.xaxis, ax.get_xlim, ax.set_xticks), (ax.yaxis, ax.get_ylim, ax.set_yticks)):
        lo, hi = sorted(lim())
        span = hi - lo
        setter([t for t in axis.get_ticklocs() if lo - 1e-09 * span <= t <= hi + 1e-09 * span])

def load(rel: str) -> list[dict]:
    root = Path(__file__).resolve().parents[2]
    with (root / rel).open(encoding='utf-8') as fh:
        return list(csv.DictReader(fh))

def action_series(rows, ratio, dip, route):
    sel = [r for r in rows if float(r['ratio']) == ratio and float(r['material_dip_deg']) == dip and (r['route'] == route)]
    sel.sort(key=lambda r: float(r['station_theta_deg']))
    return (np.array([float(r['station_theta_deg']) for r in sel]), np.array([float(r['thrust_kN_per_m']) for r in sel]), np.array([float(r['moment_kNm_per_m']) for r in sel]))

def panel_actions(ax, rows, key: str, ylabel: str) -> None:
    idx = 1 if key == 'thrust' else 2
    for dip in (0.0, 45.0, 90.0):
        bed = 90.0 - dip
        th, n, m = action_series(rows, ACTION_RATIO, dip, 'condensed')
        y = (n, m)[idx - 1]
        ax.plot(th, y * 0.001 if key == 'thrust' else y, color=C_BED[bed], ls=LS_BED[bed], lw=1.1, label=f'bedding ${bed:.0f}\\degree$')
        tf, nf, mf = action_series(rows, ACTION_RATIO, dip, 'meshed_free')
        if len(tf):
            yf = (nf, mf)[idx - 1]
            ax.plot(tf[::8], (yf * 0.001 if key == 'thrust' else yf)[::8], ls='none', marker='o', ms=2.6, mfc='none', mec=C_BED[bed], mew=0.6)
    ax.set_xlim(0.0, 360.0)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel('Mapping angle $\\theta$ (deg)')
    ax.set_ylabel(ylabel)
    ax.grid(True, lw=0.3, alpha=0.35)

def panel_closure_map(ax, rows) -> None:
    lo, hi = (np.inf, 0.0)
    for dip in (0.0, 45.0, 90.0):
        bed = 90.0 - dip
        for ratio in (1.25, 2.625, 4.0):

            def q(route):
                for r in rows:
                    if r['route'] == route and float(r['ratio']) == ratio and (float(r['material_dip_deg']) == dip):
                        return (float(r['closure_horizontal_mm']), float(r['closure_vertical_mm']))
            xu, yu = q('unlined')
            xl, yl = q('condensed')
            lo = min(lo, xl, yl, xu, yu)
            hi = max(hi, xl, yl, xu, yu)
            ax.annotate('', xy=(xl, yl), xytext=(xu, yu), arrowprops=dict(arrowstyle='->', lw=0.7, color=C_BED[bed], alpha=0.75, linestyle=LS_BED[bed], shrinkA=1.5, shrinkB=1.5))
            ax.plot([xl], [yl], marker=M_RATIO[ratio], ms=4.2, color=C_BED[bed], ls='none')
            ax.plot([xu], [yu], marker=M_RATIO[ratio], ms=4.2, mfc='none', mec=C_BED[bed], mew=0.7, ls='none')
    pad = 0.06 * (hi - lo)
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, ls=(0, (4, 3)), lw=0.7, color=C_GUIDE, zorder=0)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect('equal')
    ax.set_xlabel('horizontal closure (mm)')
    ax.set_ylabel('vertical closure (mm)')
    ax.grid(True, lw=0.3, alpha=0.35)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-dir', type=Path)
    args = parser.parse_args()
    summary, actions = (load(SRC_SUM), load(SRC_ACT))
    fig = fs.figure(WIDTH, 133)
    a = axes_mm(fig, 16, 83, 45, 35)
    b = axes_mm(fig, 84, 83, 45, 35)
    c = axes_mm(fig, 45, 16, 45, 43)
    panel_actions(a, actions, 'thrust', 'Thrust (MN/m)')
    panel_actions(b, actions, 'moment', 'Bending moment (kNm/m)')
    panel_closure_map(c, summary)
    for ax in (a, b, c):
        fs.style_axes(ax, minor=False)
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
        pin_ticks(ax)
    for x, y, title in ((3, 123, '(a)  Thrust'), (73, 123, '(b)  Bending moment'), (29, 63, '(c)  Directional closure')):
        note(fig, x, y, title, fontsize=9)
    note(fig, WIDTH / 2, 70, 'Lines: DtN–FEM; open circles: meshed-exterior reference', ha='center', fontsize=8)
    bedding = [plt.Line2D([], [], color=C_BED[d], ls=LS_BED[d], lw=1.1, label=f'Bedding {d:g}°') for d in (0.0, 45.0, 90.0)]
    fig.legend(handles=bedding, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.999), fontsize=8, frameon=False)
    ratios = [plt.Line2D([], [], color='black', marker=M_RATIO[r], mfc='white', ls='none', ms=4, label=f'$E_h/E_v={r:g}$') for r in M_RATIO]
    fig.legend(handles=ratios, loc='center left', bbox_to_anchor=(0.015, 0.3), title='Panel (c)', title_fontsize=8, fontsize=8, frameon=False)
    states = [plt.Line2D([], [], color='black', marker='o', mfc=fill, ls='none', ms=4, label=label) for fill, label in (('white', 'Unlined'), ('black', 'Lined'))]
    fig.legend(handles=states, loc='center right', bbox_to_anchor=(0.999, 0.32), title='Panel (c)', title_fontsize=8, fontsize=8, frameon=False)
    note(fig, 105, 28, 'Arrows:\nunlined to lined', ha='center', fontsize=8)
    out = ROOT / 'figures/lined-section-demo/lined_demo.pdf'
    preview = args.preview_dir / 'lined_demo.png' if args.preview_dir else None
    fs.save(fig, out, size_mm=(WIDTH, 133), preview_path=preview)
    plt.close(fig)
    print(f'Saved {out}')
if __name__ == '__main__':
    main()
