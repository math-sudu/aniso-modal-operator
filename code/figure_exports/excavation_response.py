from __future__ import annotations
import argparse
import csv
import math
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path as PolygonPath
from matplotlib.lines import Line2D
ROOT = Path(__file__).resolve().parents[2]
import origin_style as fs
from engineering_figures import axes_mm, note, WIDTH
from hs_section_geometry import source_polar_angle_deg
DATA = ROOT / 'results/data/derived'
COLORS = {1.25: fs.BLACK, 2.625: fs.RED, 4.0: fs.BLUE}
STYLES = {0.0: ':', 45.0: '--', 90.0: '-'}

def load(name):
    with (DATA / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))

def select(rows, ratio, dip):
    return [r for r in rows if float(r['E_ratio']) == ratio and float(r['bedding_dip_deg']) == dip]

def values(rows, key):
    return np.array([float(r[key]) for r in rows])

def field(ax, rows, key, boundary, limits):
    x, y, z = (values(rows, k) for k in ('x_m', 'y_m', key))
    tri = mtri.Triangulation(x, y)
    centroids = np.column_stack((x[tri.triangles].mean(axis=1), y[tri.triangles].mean(axis=1)))
    tri.set_mask(PolygonPath(boundary).contains_points(centroids))
    artist = ax.tricontourf(tri, z, levels=np.linspace(*limits, 13), cmap='turbo')
    closed = np.vstack((boundary, boundary[0]))
    ax.plot(*closed.T, color='black', lw=0.7)
    ax.set(xlim=(-11, 11), ylim=(-11, 11), aspect='equal', xticks=[-10, 0, 10], yticks=[-10, 0, 10], xlabel='Horizontal distance (m)', ylabel='Vertical distance (m)')
    fs.style_axes(ax, minor=False)
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    return artist

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-dir', type=Path)
    args = parser.parse_args()
    stress = load('near_field_stress_hs100.csv')
    disp = load('near_field_displacement_hs100.csv')
    hoop = load('wall_hoop_hs100.csv')
    central_hoop = select(hoop, 2.625, 45)
    order = np.argsort(values(central_hoop, 'map_theta_deg'))
    boundary = np.column_stack((values(central_hoop, 'x_m'), values(central_hoop, 'y_m')))[order]
    fig = fs.figure(WIDTH, 154)
    a = axes_mm(fig, 15, 91, 43, 43)
    b = axes_mm(fig, 82, 91, 43, 43)
    c = axes_mm(fig, 15, 25, 45, 40)
    d = axes_mm(fig, 82, 25, 45, 40)
    central_stress = select(stress, 2.625, 45)
    central_disp = select(disp, 2.625, 45)
    stress_range = (values(central_stress, 'sigma_3_MPa').min(), values(central_stress, 'sigma_3_MPa').max())
    disp_range = (values(disp, 'u_mag_mm').min(), values(disp, 'u_mag_mm').max())
    ca = field(a, central_stress, 'sigma_3_MPa', boundary, stress_range)
    cb = field(b, central_disp, 'u_mag_mm', boundary, disp_range)
    wall = [r for r in central_disp if int(r['on_wall'])]
    angle = np.arctan2(values(wall, 'y_m'), values(wall, 'x_m'))
    order = np.argsort(angle)
    xy = np.column_stack((values(wall, 'x_m') + 0.13 * values(wall, 'u_h_mm'), values(wall, 'y_m') + 0.13 * values(wall, 'u_v_mm')))[order]
    xy = np.vstack((xy, xy[0]))
    b.plot(*xy.T, color='black', ls='--', lw=0.9)
    for artist, x, label, ticks in ((ca, 15, 'Reference $\\sigma_3$ (MPa)', np.linspace(*stress_range, 4)), (cb, 82, 'DtN–FEM $|u|$ (mm)', np.linspace(*disp_range, 4))):
        bar = fig.colorbar(artist, cax=axes_mm(fig, x, 141, 43, 2.4), orientation='horizontal', ticks=ticks)
        bar.ax.set_xticklabels([f'{t:.1f}' for t in ticks])
        bar.ax.tick_params(labelsize=7, pad=1)
        bar.ax.xaxis.set_ticks_position('top')
        note(fig, x + 21.5, 149.5, label, ha='center', fontsize=8)
    for lo, hi in ((105, 120), (240, 255)):
        c.axvspan(lo, hi, color='#aaaaaa', alpha=0.2, lw=0)
    for ratio, color in COLORS.items():
        for dip, style in STYLES.items():
            rows = select(hoop, ratio, dip)
            delta = source_polar_angle_deg(values(rows, 'map_theta_deg'))
            order = np.argsort(delta)
            c.plot(delta[order], values(rows, 'hoop_sigma_MPa')[order], color=color, ls=style, lw=1)
        positions = []
        for dip in STYLES:
            wall = [r for r in select(disp, ratio, dip) if int(r['on_wall'])]
            index = int(np.argmax(values(wall, 'u_mag_mm')))
            positions.append(math.degrees(math.atan2(float(wall[index]['y_m']), abs(float(wall[index]['x_m'])))))
        d.plot(list(STYLES), positions, '-o', color=color, lw=1, ms=3)
    c.set(xlim=(0, 360), ylim=(-90, -10), xticks=[0, 90, 180, 270, 360], yticks=[-90, -70, -50, -30, -10], xlabel='Physical angle $\\delta$ (deg)', ylabel='Wall hoop stress (MPa)')
    d.axhline(0, color='#999999', ls=':', lw=0.6)
    d.axhline(-90, color='#999999', ls=':', lw=0.6)
    d.text(4, 5, 'Springline', fontsize=7)
    d.text(49, -84, 'Invert', fontsize=7)
    d.set(xlim=(-4, 94), ylim=(-98, 22), xticks=[0, 45, 90], yticks=[-90, -60, -30, 0], xlabel='Bedding dip (deg)', ylabel='Peak angle from horizontal (deg)')
    for ax in (c, d):
        fs.style_axes(ax, minor=False)
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
    for x, y, label in ((3, 132, '(a)'), (70, 132, '(b)'), (3, 69, '(c)'), (70, 69, '(d)')):
        note(fig, x, y, label, fontsize=9)
    ratio_key = [Line2D([], [], color=color, lw=1, label=f'$E_h/E_v={ratio:g}$') for ratio, color in COLORS.items()]
    dip_key = [Line2D([], [], color='black', ls=style, lw=1, label=f'Dip ${dip:g}^\\circ$') for dip, style in STYLES.items()]
    fig.legend(handles=ratio_key, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0.045), fontsize=8, frameon=False)
    fig.legend(handles=dip_key, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0.001), fontsize=8, frameon=False)
    preview = args.preview_dir / 'excavation_response.png' if args.preview_dir else None
    output = ROOT / 'figures/excavation-response/excavation_response.pdf'
    fs.save(fig, output, size_mm=(WIDTH, 154), preview_path=preview)
    plt.close(fig)
    print(f'Saved {output}; displacement color scale ends at {disp_range[1]:.5f} mm')
if __name__ == '__main__':
    main()
