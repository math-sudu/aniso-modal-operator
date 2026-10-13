from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle, ConnectionPatch
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
import origin_style as fs
from support_screening import closure, screening_rows, verify_saved_partition
WIDTH = 390 * 25.4 / 72.27
INK, BLUE, RED = (fs.BLACK, fs.BLUE, fs.RED)

def axes_mm(fig, x, y, width, height):
    w, h = fig.get_size_inches() * 25.4
    return fig.add_axes([x / w, y / h, width / w, height / h])

def note(fig, x, y, text, **kwargs):
    w, h = fig.get_size_inches() * 25.4
    return fig.text(x / w, y / h, text, **kwargs)

def outline(spec, rho=1.0, theta=None):
    theta = np.linspace(0, 2 * np.pi, 721) if theta is None else np.asarray(theta)
    zeta = rho * np.exp(1j * theta)
    geo = spec['geometry']
    z = geo['radius_m'] * (zeta + sum((c * zeta ** (-j) for j, c in enumerate(geo['coefficients']))))
    return np.array([-z.imag, z.real]).T

def draw_domain(spec):
    """Use original field pixels and the manuscript's actual mapped profile."""
    size = (WIDTH, 148)
    fig = fs.figure(*size)
    assets = ROOT / 'figures/domain-coupling/assets'
    for x, letter, name, title in ((3, 'a', 'moxi_tunnel_face.jpg', 'Excavation face'), (48, 'b', 'moxi_layered_rock.jpg', 'Folded rock layers'), (93, 'c', 'moxi_slate_surface.jpg', 'Interlayer surface')):
        image = plt.imread(assets / name)
        photo_height = 41 * image.shape[0] / image.shape[1]
        photo = axes_mm(fig, x, 107 + (31 - photo_height) / 2, 41, photo_height)
        photo.imshow(image)
        photo.set_axis_off()
        note(fig, x, 141, f'({letter})  {title}', fontsize=8.5)
    note(fig, WIDTH / 2, 101.5, 'Moxi Tunnel, No. 4 inclined shaft', ha='center', fontsize=8.5)
    note(fig, 3, 92, '(d)  Idealized excavation and support', fontsize=9)
    note(fig, 86, 92, '(e)  Exterior coupling', fontsize=9)
    ax = axes_mm(fig, 0, 12, 83, 76)
    wall, outer = (outline(spec), outline(spec, 2))
    for shift in np.arange(-28, 28, 1.9):
        ax.plot([-14, 14], [-14 + shift, 14 + shift], color='#c5c8ca', lw=0.45, ls='-')
    ax.add_patch(Polygon(outer, fc='#f2efe8', ec='none', zorder=2))
    for rho in np.geomspace(1, 2, 13):
        xy = outline(spec, rho)
        ax.plot(*xy.T, color='#a8a6a0', lw=0.35, ls='-', zorder=3)
    for theta in np.linspace(0, 2 * np.pi, 48, endpoint=False):
        xy = np.array([outline(spec, rho, theta) for rho in np.geomspace(1, 2, 32)])
        ax.plot(*xy.T, color='#a8a6a0', lw=0.35, ls='-', zorder=3)
    tangent = np.gradient(wall, axis=0)
    inward = np.array([-tangent[:, 1], tangent[:, 0]]).T
    inward /= np.linalg.norm(inward, axis=1)[:, None]
    inner = wall + 0.4 * inward
    ax.add_patch(Polygon(wall, fc='#efbbb9', ec=RED, lw=0.9, zorder=4))
    ax.add_patch(Polygon(inner, fc='white', ec=RED, lw=0.8, zorder=5))
    ax.plot(*outer.T, color=BLUE, lw=1.35, ls='-', zorder=5)
    nodes = outline(spec, 2, np.linspace(0, 2 * np.pi, 32, endpoint=False))
    ax.plot(*nodes.T, 'o', color=BLUE, mfc='white', ms=2.3, mew=0.65, zorder=6)
    for x in (-5, 0, 5):
        ax.annotate('', (x, 11), (x, 13.2), arrowprops={'arrowstyle': '->', 'lw': 0.8}, zorder=6)
    for y in (-5, 0, 5):
        ax.annotate('', (-10.8, y), (-13, y), arrowprops={'arrowstyle': '->', 'lw': 0.8}, zorder=6)
    ax.text(1.7, 12.1, '$p_v$', fontsize=9)
    ax.text(-13.1, 6.9, '$p_h$', fontsize=9)
    ax.text(-0.3, 7.4, 'FEM near field  $K_h$', ha='center', fontsize=8.5, zorder=7, bbox={'fc': '#f2efe8', 'ec': 'none', 'pad': 0.5})
    ax.annotate('', (0, -2.8), (0, 3.8), arrowprops={'arrowstyle': '<->', 'color': INK, 'lw': 0.85}, zorder=8)
    ax.annotate('', (-4.3, -0.2), (4.3, -0.2), arrowprops={'arrowstyle': '<->', 'color': INK, 'lw': 0.85}, zorder=8)
    ax.text(0.45, -1.65, '$\\Delta_v$', fontsize=9, zorder=9)
    ax.text(-3.6, 0.35, '$\\Delta_h$', fontsize=9, zorder=9)
    ax.annotate('Lining', xy=(4.3, -1.9), xytext=(7.4, -5.8), color=RED, arrowprops={'arrowstyle': '-', 'color': RED, 'lw': 0.7}, fontsize=8.5, zorder=8, bbox={'fc': 'white', 'ec': 'none', 'pad': 0.6})
    ax.text(-8.5, -10.7, '$\\Gamma_\\rho$, $\\rho=2$', color=BLUE, fontsize=9, zorder=8, bbox={'fc': 'white', 'ec': 'none', 'pad': 0.5})
    ax.set(aspect='equal', xlim=(-14, 14), ylim=(-12, 14))
    ax.set_axis_off()
    note(fig, 41, 7, 'Homogeneous elastic exterior', ha='center', fontsize=8.5)
    note(fig, 41, 2, '$\\widetilde{\\sigma}\\rightarrow0$ at infinity', ha='center', fontsize=8.5)
    detail = axes_mm(fig, 87, 55, 48, 31)
    detail.add_patch(Rectangle((0, 0), 0.45, 1, fc='#f2efe8', ec='none'))
    detail.plot([0.45, 0.45], [0, 1], color=BLUE, lw=1.3, ls='-')
    detail.plot([0.1, 0.28], [0, 1], color='#b3afa8', lw=0.5, ls='-')
    detail.plot([0, 0.45], [0.25, 0.25], color='#b3afa8', lw=0.5, ls='-')
    detail.plot([0, 0.45], [0.73, 0.73], color='#b3afa8', lw=0.5, ls='-')
    for endpoint, color, label, pos in (((0.89, 0.49), INK, '$n$', (0.91, 0.49)), ((0.74, 0.89), BLUE, '$g$', (0.76, 0.91)), ((0.11, 0.37), RED, '$t=S_{\\mathrm{f}}g$', (0.05, 0.17))):
        detail.annotate('', endpoint, (0.45, 0.49), arrowprops={'arrowstyle': '->', 'color': color, 'lw': 0.9})
        detail.text(*pos, label, color=color, fontsize=9)
    detail.text(0.15, 1.11, 'Near field', ha='center', fontsize=8)
    detail.text(0.78, 1.11, 'Exterior', ha='center', fontsize=8)
    detail.set(xlim=(0, 1), ylim=(-0.04, 1.15))
    detail.set_axis_off()
    loc = outline(spec, 2, 1.5 * np.pi)
    ax.plot(*loc, 'o', ms=5, mfc='none', mec=RED, zorder=8)
    fig.add_artist(ConnectionPatch(xyA=loc, coordsA=ax.transData, xyB=(0, 0), coordsB=detail.transData, color=RED, lw=0.55, linestyle=(0, (3, 3)), zorder=1))
    note(fig, 111, 50, '$u|_\\Gamma=u^{\\mathrm{ext}}|_\\Gamma=g$', ha='center', fontsize=9)
    block = axes_mm(fig, 91, 20, 34, 22)
    for x, y, text, color in ((0, 1, '$K_{ii}$', '#f2efe8'), (1, 1, '$K_{i\\Gamma}$', '#f2efe8'), (0, 0, '$K_{\\Gamma i}$', '#f2efe8'), (1, 0, '$K_{\\Gamma\\Gamma}-S_{\\mathrm{f}}$', '#dce6f8')):
        block.add_patch(Rectangle((x, y), 1, 1, fc=color, ec='#555555', lw=0.6))
        block.text(x + 0.5, y + 0.5, text, ha='center', va='center', fontsize=8.5)
    block.set(xlim=(0, 2), ylim=(0, 2))
    block.set_axis_off()
    note(fig, 111, 13, '$[K_h-S_{\\mathrm{f}}]\\,u=f$', ha='center', fontsize=10)
    note(fig, 111, 6, 'Reuse exterior stiffness', ha='center', fontsize=8.5)
    for axis in fig.axes:
        axis.set(xticks=[], yticks=[])
    return (fig, [], {'photo_pixel_edits': 'none', 'lining_thickness_m': 0.4, 'mesh': 'schematic mapped quadrilaterals at rho=2', 'field_role': 'geological motivation; independently specified numerical model'})

def draw_screening():
    rows = screening_rows()
    checks = {'case_count': len(rows), 'saved_partition_max_error': verify_saved_partition(rows)}
    fig = fs.figure(WIDTH, 94)
    trade = axes_mm(fig, 14, 23, 48, 58)
    directional = axes_mm(fig, 84, 23, 48, 58)
    releases = np.linspace(0, 1, 501)
    sig0 = max((r['compression_zero_MPa'] for r in rows))
    closures = np.array([max((closure(r, release, d) for r in rows for d in ('vertical', 'horizontal'))) for release in releases])
    lower = max((r['release_min_stress'] for r in rows))
    at_limit = max((closure(r, lower, d) for r in rows for d in ('vertical', 'horizontal')))
    trade.plot(closures, (1 - releases) * sig0, color=INK, lw=1.35)
    trade.axhline(20, color='#555555', ls='--', lw=0.9)
    trade.axvline(24, color='#767676', ls=':', lw=0.85)
    trade.text(24.15, 65, '24 mm allowance', rotation=90, fontsize=8, va='center')
    for release, dx, dy in ((0, 0.65, 1), (0.5, -0.9, 5), (0.82, -3.0, 30), (1, -1.6, 6)):
        x = max((closure(r, release, d) for r in rows for d in ('vertical', 'horizontal')))
        y = (1 - release) * sig0
        trade.plot(x, y, 'o', color=INK, mfc='white', ms=3)
        trade.annotate(f'$\\lambda={release:g}$', xy=(x, y), xytext=(x + dx, y + dy), fontsize=8, arrowprops={'arrowstyle': '-', 'lw': 0.5})
    trade.plot(at_limit, 20, 's', color=INK, ms=3.5, zorder=5)
    trade.annotate(f'{at_limit:.2f} mm', xy=(at_limit, 20), xytext=(19.4, 29), color=INK, fontsize=8, arrowprops={'arrowstyle': '-', 'color': INK, 'lw': 0.6})
    trade.text(20.2, 12, '20 MPa ceiling', color=INK, fontsize=8)
    trade.set(xlim=(19, 24.6), ylim=(-3, 137), xticks=[19, 20, 22, 24], yticks=[0, 20, 60, 100], xlabel='Maximum chord closure (mm)', ylabel='Maximum compressive stress (MPa)')
    fs.style_axes(trade)
    colors = {0.6: INK, 0.9: RED, 1.4: BLUE}
    markers = {1.25: 'o', 2.625: 's', 4.0: '^'}
    directional.plot([0, 25], [0, 25], color='#929292', ls='--', lw=0.8)
    for row in rows:
        directional.plot(row['horizontal_at_stress_limit_mm'], row['vertical_at_stress_limit_mm'], linestyle='none', marker=markers[row['modulus_ratio']], color=colors[row['lateral_pressure_coefficient']], mfc='white', ms=4, mew=0.9)
    directional.text(8, 23, 'Vertical controls', fontsize=8)
    directional.text(15, 3, 'Horizontal\ncontrols', fontsize=8)
    directional.set(xlim=(0, 25), ylim=(0, 26), xticks=[0, 10, 20], yticks=[0, 10, 20], xlabel='Horizontal closure $\\Delta_h$ (mm)', ylabel='Vertical closure $\\Delta_v$ (mm)')
    fs.style_axes(directional)
    note(fig, 3, 88, '(a)  Common release across 27 cases', fontsize=9)
    note(fig, 76, 88, '(b)  Casewise stress-limited release', fontsize=9)
    handles = [Line2D([], [], color=c, lw=1.2, label=f'$p_h/p_v={k:g}$') for k, c in colors.items()]
    handles += [Line2D([], [], color=INK, marker=m, ls='none', mfc='white', label=f'$E_h/E_v={r:g}$') for r, m in markers.items()]
    fig.legend(handles=handles[:3], ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0.08), columnspacing=2.5, handlelength=1.7, fontsize=8)
    fig.legend(handles=handles[3:], ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0.005), columnspacing=2.5, handlelength=1.7, fontsize=8)
    checks.update({'common_release_lower': lower, 'minimum_common_allowance_mm': at_limit})
    return (fig, rows, checks)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-dir', type=Path)
    args = parser.parse_args()
    fig, _, checks = draw_screening()
    output = ROOT / 'figures/support-screening/support_screening.pdf'
    preview = args.preview_dir / 'support_screening.png' if args.preview_dir else None
    fs.save(fig, output, size_mm=(WIDTH, 94), preview_path=preview)
    plt.close(fig)
    print(json.dumps(checks, indent=2))
if __name__ == '__main__':
    main()
