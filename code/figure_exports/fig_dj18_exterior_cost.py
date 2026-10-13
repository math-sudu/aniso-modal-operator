from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import origin_style as fs
from engineering_figures import axes_mm, note, WIDTH
import matplotlib.pyplot as plt
KEY1, KEY2, KEY3 = (fs.BLACK, fs.RED, fs.BLUE)
GUIDE = '#8c8c8c'
JOB = 'exterior-cost'
SRC_REF = 'results/data/derived/exterior_reference_ladder.csv'
SRC_COST = 'results/data/derived/exterior_cost_ladder.csv'
C_CLEAN = KEY1
C_OP = KEY1
C_FREE = KEY2
C_FIXED = KEY3
C_GUIDE = GUIDE
POINTS = ((1.25, 'o', '-'), (2.625, 's', '--'), (4.0, '^', '-.'))
NEAR_DOF = {'12x64': 1664, '24x128': 6400, '48x256': 25088}

def load(rel: str) -> list[dict]:
    root = Path(__file__).resolve().parents[2]
    with (root / rel).open(encoding='utf-8') as fh:
        return list(csv.DictReader(fh))

def panel_ladder(ax, ref_rows) -> None:
    """(a) the two-treatment difference against near-field resolution."""
    x = [NEAR_DOF[m] for m in ('12x64', '24x128', '48x256')]
    base = 0.0010653
    ax.plot(x, [base * (x[0] / v) for v in x], ls=':', lw=0.7, color=C_GUIDE, zorder=2, label='$h^2$ guide')
    ax.annotate('$h^{2}$', xy=(x[1], base * x[0] / x[1]), xytext=(5, 5), textcoords='offset points', fontsize=6.8, color=C_GUIDE)
    for ratio, marker, ls in POINTS:
        ys = [float(r['diff_free']) for r in ref_rows if float(r['ratio']) == ratio and float(r['s_scale']) == 1.0]
        ax.plot(x, ys, ls=ls, lw=1.0, color=C_CLEAN, marker=marker, ms=3.4, mfc=C_CLEAN, mec=C_CLEAN, zorder=5, label=f'$E_h/E_v={ratio:g}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('near-field mesh', fontsize=7.5, labelpad=1.5)
    ax.set_ylabel('relative difference of\nthe two exterior treatments', fontsize=7.5, labelpad=2)
    ax.set_xlim(1250.0, 34000.0)
    ax.set_ylim(4e-05, 0.0025)
    ax.set_xticks(x)
    ax.set_xticklabels(['$12\\times64$', '$24\\times128$', '$48\\times256$'], fontsize=6.6)
    ax.set_yticks([0.0001, 0.001])
    ax.grid(axis='y', which='major', ls=':', lw=0.4, alpha=0.35)
    ax.tick_params(which='minor', length=0)

def panel_cost(ax, cost_rows) -> None:
    """(b) relative error against displacement degrees of freedom."""
    series = {'operator': (C_OP, 'o', '-', 'condensed\nnear-mesh ladder'), 'far_mesh_free': (C_FREE, 's', '--', 'meshed traction free\nremote-radius ladder'), 'far_mesh_fixed': (C_FIXED, '^', ':', 'meshed clamped\nremote-radius ladder')}
    for key, (colour, marker, ls, label) in series.items():
        rows = [r for r in cost_rows if r['treatment'] == key]
        ax.plot([int(r['total_dof']) for r in rows], [float(r['rel_error_vs_completed_basis']) for r in rows], ls=ls, lw=1.0, color=colour, marker=marker, ms=3.4, mfc=colour, mec=colour, zorder=5, label=label)
    dip = [r for r in cost_rows if r['treatment'] == 'far_mesh_free' and r['rho_far'] == '32'][0]
    ax.annotate('$\\rho_{\\mathrm{far}} = 32$', xy=(int(dip['total_dof']), float(dip['rel_error_vs_completed_basis'])), xytext=(6, -11), textcoords='offset points', fontsize=6.6, color=C_FREE, arrowprops=dict(arrowstyle='-', lw=0.5, color=C_FREE))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('displacement degrees of freedom', fontsize=7.5, labelpad=1.5)
    ax.set_ylabel('relative error against\nthe complex-potential reference', fontsize=7.5, labelpad=2)
    ax.set_xlim(1300.0, 32000.0)
    ax.set_ylim(0.00012, 2.2)
    ax.set_xticks([2000.0, 5000.0, 10000.0, 20000.0])
    ax.set_xticklabels(['2k', '5k', '10k', '20k'], fontsize=6.6)
    ax.set_yticks([0.0001, 0.001, 0.01, 0.1, 1.0])
    ax.grid(axis='both', which='major', ls=':', lw=0.4, alpha=0.35)
    ax.tick_params(which='minor', length=0)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-dir', type=Path)
    args = parser.parse_args()
    fig = fs.figure(WIDTH, 84)
    a = axes_mm(fig, 19, 15, 42, 54)
    b = axes_mm(fig, 86, 15, 45, 54)
    panel_ladder(a, load(SRC_REF))
    panel_cost(b, load(SRC_COST))
    note(fig, 4, 76, '(a)  Mesh convergence', fontsize=9)
    note(fig, 75, 76, '(b)  Accuracy and model size', fontsize=9)
    for ax, loc in ((a, 'lower left'), (b, 'upper right')):
        fs.style_axes(ax, minor=False)
        ax.tick_params(labelsize=7)
        ax.tick_params(axis='y', labelsize=10.1)
        ax.legend(loc=loc, fontsize=7, frameon=False, handlelength=1.5)
    b.set_yticks([0.001, 0.01, 0.1, 1.0])
    from matplotlib.text import Text
    for text in fig.findobj(Text):
        text.set_fontsize(max(7.0, text.get_fontsize()))
    preview = args.preview_dir / 'exterior_cost.png' if args.preview_dir else None
    fs.save(fig, ROOT / 'figures/exterior-cost/exterior_cost.pdf', size_mm=(WIDTH, 84), preview_path=preview)
    plt.close(fig)
if __name__ == '__main__':
    main()
