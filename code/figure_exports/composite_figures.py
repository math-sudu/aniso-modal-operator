from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, patheffects
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, ConnectionPatch, Ellipse, Polygon, Rectangle
from matplotlib.path import Path as PolygonPath
import matplotlib.tri as mtri
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
import origin_style as fs
import manuscript_figures as method
from engineering_figures import WIDTH, axes_mm, note, outline
from hs_section_geometry import source_polar_angle_deg
DATA = ROOT / 'results/data/derived'
SPEC = json.loads(method.SPEC.read_text(encoding='utf-8'))
INK, RED, BLUE = (fs.BLACK, fs.RED, fs.BLUE)
TEAL, OCHRE, PURPLE = ('#167d8d', '#c88032', '#82659c')
RATIOS, DIPS = ((1.25, 2.625, 4.0), (0.0, 45.0, 90.0))
RATIO_COLORS = dict(zip(RATIOS, (INK, RED, BLUE)))
DIP_COLORS = dict(zip(DIPS, (BLUE, RED, INK)))
LABELS = {}
HEIGHTS = {entry['figure_id']: entry['dimensions_mm'][1] for entry in SPEC['figures']}

def label(fig, x, y, value, **kwargs):
    return note(fig, x, y, LABELS.get(value, value), fontsize=kwargs.pop('fontsize', 8), **kwargs)

def panel(fig, x, y, letter, title):
    label(fig, x, y, f'({letter})  {title}', fontsize=8.5)

def axis(fig, x, y, w, h, *, diagram=False):
    ax = axes_mm(fig, x, y, w, h)
    if diagram:
        ax.set_axis_off()
        ax.set(xticks=[], yticks=[])
    else:
        fs.style_axes(ax, minor=False)
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
    return ax

def photo(fig, path, x, y, w, h):
    image = plt.imread(ROOT / path)
    height = min(h, w * image.shape[0] / image.shape[1])
    width = height * image.shape[1] / image.shape[0]
    ax = axis(fig, x + (w - width) / 2, y + (h - height) / 2, width, height, diagram=True)
    ax.imshow(image)
    return ax

def read(name):
    return pd.read_csv(DATA / name)

def choose(frame, ratio, dip):
    return frame[(frame.E_ratio == ratio) & (frame.bedding_dip_deg == dip)]

def ring(ax, rho=1, **kwargs):
    xy = outline(SPEC, rho)
    ax.plot(*xy.T, **kwargs)
    return xy

def mesh(ax, rho=2, nr=24, nt=64, color='#b5b1a8'):
    for r in np.geomspace(1, rho, nr + 1):
        ring(ax, r, color=color, lw=0.28, ls='-')
    for theta in np.linspace(0, 2 * np.pi, nt, endpoint=False):
        xy = np.array([outline(SPEC, r, theta) for r in np.geomspace(1, rho, nr + 1)])
        ax.plot(*xy.T, color=color, lw=0.28, ls='-')

def arrow(ax, start, end, color=INK, **kwargs):
    return ax.annotate('', xy=end, xytext=start, arrowprops={'arrowstyle': '-|>', 'lw': 0.8, 'color': color, **kwargs})

def draw_domain():
    from domain_overview import draw_domain as draw_overview
    return draw_overview(SPEC, WIDTH, HEIGHTS['domain-coupling'], LABELS)

def schematic_grid(ax, x, y, w, h, rows, cols, color, *, name, active=None):
    ax.add_patch(Rectangle((x, y), w, h, fc=color, ec='#5b6064', lw=0.6))
    for k in range(1, cols):
        ax.plot([x + k * w / cols] * 2, [y, y + h], color='white', lw=0.45, ls='-')
    for k in range(1, rows):
        ax.plot([x, x + w], [y + k * h / rows] * 2, color='white', lw=0.45, ls='-')
    if active:
        for r, c in active:
            ax.add_patch(Rectangle((x + c * w / cols, y + r * h / rows), w / cols, h / rows, fc=RED, ec='white', lw=0.3))
    ax.text(x + w / 2, y - 3, name, ha='center', fontsize=8)

def field_panel(ax, frame, key, boundary, levels, cmap='turbo'):
    x, y, z = (frame[k].to_numpy() for k in ('x_m', 'y_m', key))
    tri = mtri.Triangulation(x, y)
    centres = np.c_[x[tri.triangles].mean(axis=1), y[tri.triangles].mean(axis=1)]
    tri.set_mask(PolygonPath(boundary).contains_points(centres))
    cf = ax.tricontourf(tri, z, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, matplotlib.colormaps[cmap].N))
    cf.set_edgecolor('face')
    closed = np.vstack((boundary, boundary[0]))
    ax.plot(*closed.T, color=INK, lw=0.6, ls='-')
    ax.set(aspect='equal', xlim=(-11, 11), ylim=(-11, 11), xticks=[-10, 0, 10], yticks=[-10, 0, 10])
    return cf

def shared_bar(fig, artists, x, y, w, label_text, ticks):
    bar = fig.colorbar(artists[0], cax=axes_mm(fig, x, y, w, 2.5), orientation='horizontal', ticks=ticks)
    bar.ax.set_label('<colorbar>')
    bar.ax.tick_params(labelsize=7, pad=1)
    for artist in artists[1:]:
        artist.colorbar = bar
    label(fig, x + w / 2, y + 5, label_text, ha='center')
    return bar

def draw_atlas(kind):
    stress = kind == 'stress'
    dips = (0.0, 22.5, 45.0, 67.5, 90.0)
    job = 'excavation-response' if stress else 'displacement-response'
    data = read('near_field_stress_hs100.csv' if stress else 'near_field_displacement_hs100.csv')
    hoop = read('wall_hoop_hs100.csv')
    wall = choose(hoop, 2.625, 45).sort_values('map_theta_deg')[['x_m', 'y_m']].to_numpy()
    key = 'sigma_3_MPa' if stress else 'u_mag_mm'
    limits = (float(data[key].min()), float(data[key].max()))
    stress_steps = [-80, -70, -60, -50, -40, -32, -28, -25, -22, -20, -18, -16.5, -15.5]
    levels = np.array([limits[0], *[v for v in stress_steps if limits[0] < v < limits[1]], limits[1]]) if stress else np.linspace(*limits, 13)
    fig = fs.figure(WIDTH, HEIGHTS[job])
    artists, peak_rows = ([], [])
    for i, ratio in enumerate(RATIOS):
        y = 105 - 28 * i
        label(fig, 2, y + 11.5, f'$E_h/E_v={ratio:g}$', rotation=90, ha='center', va='center', fontsize=7)
        for j, dip in enumerate(dips):
            x = 10 + 25 * j
            ax = axis(fig, x, y, 23, 23)
            frame = choose(data, ratio, dip)
            artists.append(field_panel(ax, frame, key, wall, levels))
            ax.tick_params(labelsize=7, pad=1, length=2)
            if j:
                ax.set_yticklabels([])
            if i < 2:
                ax.set_xticklabels([])
            else:
                ax.get_xticklabels()[0].set_ha('left')
                ax.get_xticklabels()[-1].set_ha('right')
            a = np.deg2rad(-dip)
            ax.plot([-8 - 1.5 * np.cos(a), -8 + 1.5 * np.cos(a)], [8 - 1.5 * np.sin(a), 8 + 1.5 * np.sin(a)], color='white', lw=2, ls='-')
            ax.plot([-8 - 1.5 * np.cos(a), -8 + 1.5 * np.cos(a)], [8 - 1.5 * np.sin(a), 8 + 1.5 * np.sin(a)], color=INK, lw=0.7, ls='-')
            ax.text(0.04, 0.045, f'({chr(97 + 5 * i + j)})', transform=ax.transAxes, fontsize=7, bbox={'fc': 'white', 'ec': 'none', 'pad': 0.5})
            if stress:
                wr = choose(hoop, ratio, dip)
                for side in (wr[wr.x_m < 0], wr[wr.x_m > 0]):
                    peak = side.loc[side.hoop_sigma_MPa.idxmin()]
                    ax.plot(peak.x_m, peak.y_m, 'D', ms=2.6, mfc='white', mec=INK, mew=0.65)
                    if i == 2:
                        oval = Ellipse((peak.x_m, peak.y_m), 2.7, 3.6, fill=False, ec=INK, lw=0.8, ls=(0, (2, 1.5)))
                        oval.set_path_effects([patheffects.Stroke(linewidth=1.6, foreground='white'), patheffects.Normal()])
                        ax.add_patch(oval)
                    peak_rows.append({'ratio': ratio, 'dip': dip, 'x_m': float(peak.x_m), 'y_m': float(peak.y_m), 'hoop_MPa': float(peak.hoop_sigma_MPa)})
                if i == 2 and j == 2:
                    ax.annotate('Wall foot', xy=(peak.x_m, peak.y_m), xytext=(0, 1), ha='center', fontsize=7, arrowprops={'arrowstyle': '-', 'lw': 0.7, 'color': INK})
            else:
                wr = frame[frame.on_wall == 1].copy()
                wr['angle'] = np.arctan2(wr.y_m, wr.x_m)
                wr = wr.sort_values('angle')
                xy = wr[['x_m', 'y_m']].to_numpy() + 0.13 * wr[['u_h_mm', 'u_v_mm']].to_numpy()
                xy = np.vstack((xy, xy[0]))
                ax.plot(*xy.T, color=INK, lw=0.7, ls='--')
                peak = wr.loc[wr.u_mag_mm.idxmax()]
                angle = float(np.rad2deg(np.arctan2(peak.y_m, abs(peak.x_m))))
                ax.plot(peak.x_m, peak.y_m, 'o', ms=3.5, mfc='white', mec=INK, mew=0.9)
                if i == 2:
                    ax.annotate(f'{angle:.0f}°', xy=(peak.x_m, peak.y_m), xytext=(0, 0.7), fontsize=7, ha='center', va='center', bbox={'fc': 'white', 'ec': 'none', 'pad': 0.3}, arrowprops={'arrowstyle': '-', 'lw': 0.7, 'color': INK})
                peak_rows.append({'ratio': ratio, 'dip': dip, 'magnitude_mm': float(peak.u_mag_mm), 'angle_deg': angle})
            if i == 0:
                label(fig, x + 11.5, 132, f'{dip:g}°', ha='center', fontsize=8)
    bar = shared_bar(fig, artists, 20, 146, 108, 'Minimum in-plane principal stress $\\sigma_{\\min}^{\\rm in}$ (MPa)' if stress else 'DtN–FEM displacement $|u|$ (mm)', [limits[0], -40, -25, -20, limits[1]] if stress else np.linspace(*limits, 5))
    bar.ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.1f}'))
    label(fig, 71.5, 138, 'Bedding dip; diamonds: peak compression on each side' if stress else 'Bedding dip; dashed wall: 130× displacement; circles: wall maxima', ha='center', fontsize=7)
    label(fig, 71.5, 40, 'Coordinates in metres; dashed ovals: wall-foot concentration' if stress else 'Coordinates in metres; labelled peak angles at $E_h/E_v=4$', ha='center', fontsize=7)
    if stress:
        for j, dip in enumerate(dips):
            ax = axis(fig, 10 + 25 * j, 13, 23, 20)
            for lo, hi in ((105, 120), (240, 255)):
                ax.axvspan(lo, hi, color='#dddddd', lw=0, zorder=0)
            for ratio, col in RATIO_COLORS.items():
                frame = choose(hoop, ratio, dip).copy()
                frame['delta'] = source_polar_angle_deg(frame.map_theta_deg.to_numpy())
                frame = frame.sort_values('delta')
                ax.plot(frame.delta, frame.hoop_sigma_MPa, color=col, lw=0.85, ls='-')
                peak = frame.loc[frame.hoop_sigma_MPa.idxmin()]
                ax.plot(peak.delta, peak.hoop_sigma_MPa, 'D', ms=2.3, mfc='white', mec=col, mew=0.6)
                if ratio == 4 and dip == 67.5:
                    ax.annotate(f'{peak.hoop_sigma_MPa:.1f}', xy=(peak.delta, peak.hoop_sigma_MPa), xytext=(205, -80), fontsize=7, ha='center', color=col, arrowprops={'arrowstyle': '-', 'lw': 0.65, 'color': col})
            ax.set(xlim=(0, 360), ylim=(-105, -10), xticks=[0, 180, 360], yticks=[-100, -55, -10])
            ax.tick_params(pad=1, length=2)
            ax.get_xticklabels()[0].set_ha('left')
            ax.get_xticklabels()[-1].set_ha('right')
            if j:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel('Wall hoop stress (MPa)', fontsize=7, labelpad=1)
            label(fig, 10 + 25 * j, 35.5, f'({chr(112 + j)})  {dip:g}°', fontsize=7.5)
        label(fig, 71.5, 6, 'Physical angle $\\delta$ (deg)', ha='center', fontsize=7)
    else:
        peaks = pd.DataFrame(peak_rows)
        axa = axis(fig, 15, 15, 45, 18)
        axb = axis(fig, 87, 15, 45, 18)
        for ratio, col in RATIO_COLORS.items():
            group = peaks[peaks.ratio == ratio]
            axa.plot(group.dip, group.angle_deg, '-o', color=col, lw=1, ms=2.5)
            axb.plot(group.dip, group.magnitude_mm, '-o', color=col, lw=1, ms=2.5)
        axa.set(xlim=(-4, 94), ylim=(-98, 38), xticks=dips, yticks=[-90, -45, 0, 30], xlabel='Bedding dip (deg)', ylabel='Peak angle (deg)')
        axb.set(xlim=(-4, 94), ylim=(8, 12), xticks=dips, yticks=[8, 10, 12], xlabel='Bedding dip (deg)', ylabel='Wall maximum (mm)')
        for ax in (axa, axb):
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
            ax.tick_params(pad=1, length=2)
            ax.xaxis.label.set_size(7)
            ax.yaxis.label.set_size(7)
            ax.xaxis.labelpad = ax.yaxis.labelpad = 1.5
        label(fig, 15, 35.5, '(p)  Peak position', fontsize=7.5)
        label(fig, 87, 35.5, '(q)  Peak magnitude', fontsize=7.5)
    fig.legend(handles=[Line2D([], [], color=col, label=f'$E_h/E_v={ratio:g}$') for ratio, col in RATIO_COLORS.items()], ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.005), fontsize=7, frameon=False)
    return (fig, {'cases': 15, 'spatial_rows': len(data), 'quantity': key, 'actual_range': limits, 'levels': levels.tolist(), 'window_m': [-11, 11, -11, 11], 'row_axis': 'modulus ratio', 'column_axis': 'bedding dip', 'bedding_dips_deg': list(dips), 'annotation_rule': 'Both side peaks; foot ovals in ratio-4 row' if stress else 'Wall maxima in all cases; leader labels in ratio-4 row', 'wall_peaks': peak_rows})

def coloured_lining(ax, actions, key, cmap):
    theta = np.deg2rad(actions.station_theta_deg.to_numpy())
    xy = outline(SPEC, theta=theta)
    segments = np.stack((xy, np.roll(xy, -1, axis=0)), axis=1)
    vals = actions[key].to_numpy() * (0.001 if key == 'thrust_kN_per_m' else 1)
    norm = colors.TwoSlopeNorm(vmin=vals.min(), vcenter=0, vmax=vals.max()) if key == 'moment_kNm_per_m' else colors.Normalize(vals.min(), vals.max())
    artist = LineCollection(segments, cmap=cmap, norm=norm, linewidths=4.5)
    artist.set_array(vals)
    ax.add_collection(artist)
    ring(ax, color='#555555', lw=0.5, ls='-')
    ax.set(aspect='equal', xlim=(-6.5, 6.5), ylim=(-5.6, 5.6), xticks=[], yticks=[])
    return artist

def draw_lining():
    import lining_response_figure as lining
    fig = fs.figure(WIDTH, 210)
    for x, letter, name, title in [(3, 'a', 'moxi_shotcrete_spalling.jpg', 'Shotcrete spalling'), (74, 'b', 'moxi_steel_arch.jpg', 'Steel-arch distortion')]:
        panel(fig, x, 204, letter, title)
        photo(fig, 'figures/lined-section-demo/assets/' + name, x, 166, 59, 34)
    label(fig, WIDTH / 2, 160, 'Moxi Tunnel support conditions', ha='center')
    panel(fig, 3, 153, 'c', 'Idealized contact')
    contact = axis(fig, 3, 117, 36, 30, diagram=True)
    for x, name, col in [(0.15, 'Bonded', RED), (0.73, 'Full slip', BLUE)]:
        contact.add_patch(Rectangle((x - 0.09, 0.1), 0.17, 0.78, fc='#eee9df', ec='#666666', lw=0.6))
        contact.add_patch(Rectangle((x + 0.08, 0.1), 0.07, 0.78, fc='#e3b5aa', ec=col, lw=0.6))
        for y in [0.23, 0.48, 0.73]:
            contact.plot([x + 0.08, x + 0.08], [y - 0.03, y + 0.03], marker='o', ms=1.6, color=col, ls='none')
        arrow(contact, (x - 0.07, 0.47), (x + 0.18, 0.47), col)
        if name == 'Full slip':
            arrow(contact, (x + 0.19, 0.25), (x + 0.19, 0.76), col)
        contact.text(x + 0.04, -0.04, name, ha='center', fontsize=7.5)
    contact.set(xlim=(-0.08, 1.1), ylim=(-0.16, 1))
    actions = read('lined_demo_actions.csv')
    central = actions[(actions.ratio == 4) & (actions.material_dip_deg == 45) & (actions.route == 'condensed')].sort_values('station_theta_deg')
    for x, letter, key, title, cmap in [(49, 'd', 'thrust_kN_per_m', 'Thrust N (MN/m)', 'turbo'), (96, 'e', 'moment_kNm_per_m', 'Moment M (kNm/m)', 'coolwarm')]:
        ax = axis(fig, x, 117, 34, 31, diagram=True)
        artist = coloured_lining(ax, central, key, cmap)
        bar = fig.colorbar(artist, cax=axes_mm(fig, x, 112, 33, 2.2), orientation='horizontal')
        bar.ax.set_label('<colorbar>')
        bar.ax.tick_params(labelsize=7, pad=1)
        bar.set_ticks([artist.norm.vmin, 0, artist.norm.vmax] if key == 'moment_kNm_per_m' else np.linspace(artist.norm.vmin, artist.norm.vmax, 3))
        panel(fig, x - 3, 153, letter, title)
    label(fig, 87, 102, 'Ring maps: $E_h/E_v=4$, dip $45^\\circ$', ha='center', fontsize=7)
    a = axis(fig, 15, 67, 45, 27)
    b = axis(fig, 86, 67, 43, 27)
    lining.panel_actions(a, actions.to_dict('records'), 'thrust', 'Thrust (MN/m)')
    lining.panel_actions(b, actions.to_dict('records'), 'moment', 'Moment (kNm/m)')
    for ax in (a, b):
        fs.style_axes(ax, minor=False)
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        ax.set_xticks([0, 180, 360])
        lining.pin_ticks(ax)
    panel(fig, 3, 97, 'f', 'Circumferential thrust')
    panel(fig, 75, 97, 'g', 'Circumferential moment')
    label(fig, WIDTH / 2, 54, 'Lines: DtN–FEM; open circles: meshed-exterior reference', ha='center', fontsize=7)
    c = axis(fig, 15, 15, 43, 30)
    d = axis(fig, 86, 15, 43, 30)
    lining.panel_closure_map(c, read('lined_demo_summary.csv').to_dict('records'))
    c.set_aspect('auto')
    c.set(xlabel='Horizontal closure (mm)', ylabel='Vertical closure (mm)')
    fs.style_axes(c, minor=False)
    c.tick_params(labelsize=7)
    c.set_xticks([5, 10, 15])
    c.set_yticks([5, 15, 25])
    for ax in (c, d):
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
    sensitivity = read('lined_contact_sensitivity.csv')
    for row in sensitivity.itertuples():
        d.plot(row.thrust_shift_pct, row.moment_shift_pct, linestyle='none', color=DIP_COLORS[row.bedding_dip_deg], marker={1.25: 'o', 2.625: 's', 4.0: '^'}[row.ratio], ms=3.5, mfc='white')
    d.set(xlim=(-48, -30), ylim=(31, 39), xticks=[-45, -40, -35], yticks=[32, 35, 38], xlabel='Peak thrust change (%)', ylabel='Peak moment change (%)')
    d.xaxis.label.set_size(7)
    d.yaxis.label.set_size(7)
    panel(fig, 3, 49, 'h', 'Unlined to lined')
    panel(fig, 75, 49, 'i', 'Full slip vs bonding')
    fig.legend(handles=[Line2D([], [], color=DIP_COLORS[v], label=f'Dip {v:g}°') for v in DIPS], loc='lower center', bbox_to_anchor=(0.29, 0.491), ncol=3, fontsize=7, frameon=False, handlelength=1.2, columnspacing=0.7)
    fig.legend(handles=[Line2D([], [], color=INK, marker=m, mfc='white', ls='none', label=f'$E_h/E_v={r:g}$') for r, m in zip(RATIOS, ['o', 's', '^'])], ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.002), fontsize=7, frameon=False)
    return (fig, {'photos': [13575, 13580], 'ring_material': {'ratio': 4, 'bedding_dip_deg': 45}, 'ring_source_rows': len(central), 'contact_summary_cases': len(sensitivity), 'thrust_change_percent': [sensitivity.thrust_shift_pct.min(), sensitivity.thrust_shift_pct.max()], 'moment_change_percent': [sensitivity.moment_shift_pct.min(), sensitivity.moment_shift_pct.max()]})

def draw_friction():
    evidence = ROOT / 'results/data/derived'
    actions = pd.read_csv(evidence / 'frictional_contact_mesh_refined_actions.csv')
    contact = pd.read_csv(evidence / 'frictional_contact_mesh_refined_interface.csv')
    summary = pd.read_csv(evidence / 'frictional_contact_mesh_refined.csv')
    actions = actions[(actions.ratio == 4) & (actions.material_dip_deg == 45)]
    contact = contact[(contact.ratio == 4) & (contact.material_dip_deg == 45)]
    fig = fs.figure(WIDTH, 120)
    axn = axis(fig, 16, 71, 45, 29)
    axm = axis(fig, 86, 71, 45, 29)
    axs = axis(fig, 16, 14, 45, 29)
    axp = axis(fig, 86, 14, 45, 29)
    styles = [('bonded', 0.0, '#888888', '--', 'Bonded'), ('Coulomb', 0.0, '#555555', ':', '$\\mu=0$'), ('Coulomb', 0.53, INK, '-', '$\\mu=0.53$'), ('Coulomb', 0.73, RED, '-', '$\\mu=0.73$'), ('Coulomb', 1.15, BLUE, '-', '$\\mu=1.15$')]
    for law, mu, color, ls, text in styles:
        a = actions[(actions.law == law) & (actions.mu == mu)].sort_values('theta_deg')
        c = contact[(contact.law == law) & (contact.mu == mu)].sort_values('theta_deg')
        for ax, frame, key in ((axn, a, 'thrust_MN_per_m'), (axm, a, 'moment_kNm_per_m'), (axs, c, 'slip_mm'), (axp, c, 'pressure_MPa')):
            ax.plot(frame.theta_deg, frame[key], color=color, ls=ls, lw=1.0, label=text)
    for ax, ylabel in ((axn, 'Thrust (MN/m)'), (axm, 'Moment (kNm/m)'), (axs, 'Relative slip (mm)'), (axp, 'Contact pressure (MPa)')):
        ax.set(xlim=(0, 360), xticks=[0, 90, 180, 270, 360], xlabel='Mapping angle $\\theta$ (deg)', ylabel=ylabel)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
    axm.axhline(0, color='#bbbbbb', lw=0.5, zorder=0)
    axs.axhline(0, color='#bbbbbb', lw=0.5, zorder=0)
    axp.axhspan(0.5, 1.5, color='#dddddd', alpha=0.7, zorder=0)
    panel(fig, 3, 105, 'a', 'Thrust redistribution')
    panel(fig, 73, 105, 'b', 'Bending response')
    panel(fig, 3, 48, 'c', 'Tangential slip')
    panel(fig, 73, 48, 'd', 'Normal compression')
    label(fig, WIDTH / 2, 116, '$E_h/E_v=4$, bedding dip $45^\\circ$, $p_h/p_v=0.9$', ha='center')
    fig.legend(*axn.get_legend_handles_labels(), ncol=5, loc='center', bbox_to_anchor=(0.5, 0.48), frameon=False, fontsize=7, handlelength=1.9, columnspacing=1.2)
    label(fig, WIDTH / 2, 1.5, 'Pressure shading: normal-stress range of the cited direct-shear tests', ha='center', fontsize=7)
    finite = summary[(summary.law == 'Coulomb') & (summary.mu > 0)]
    return (fig, {'material_ratio': 4, 'bedding_dip_deg': 45, 'source_action_rows': len(actions), 'source_contact_rows': len(contact), 'pressure_range_MPa': [finite.minimum_pressure_MPa.min(), finite.maximum_pressure_MPa.max()], 'test_normal_stress_MPa': [0.5, 1.5]})

def draw_density():
    from density_overview import draw_density as draw_density_overview
    return draw_density_overview(WIDTH, HEIGHTS['density-invariance'], LABELS)

def draw_sampling():
    from sampling_overview import draw_sampling as draw_overview
    return draw_overview(SPEC, WIDTH, HEIGHTS['analyticity-sampling'], LABELS)

def draw_kernels():
    from kernel_overview import draw_kernels as draw_overview
    return draw_overview(SPEC, WIDTH, HEIGHTS['kernel-validity'], LABELS)

def draw_exterior():
    from exterior_overview import draw_exterior as draw_overview
    return draw_overview(SPEC, WIDTH, HEIGHTS['exterior-cost'], LABELS)
DRAWERS = {'domain-coupling': draw_domain, 'excavation-response': lambda: draw_atlas('stress'), 'displacement-response': lambda: draw_atlas('displacement'), 'lined-section-demo': draw_lining, 'frictional-contact': draw_friction, 'density-invariance': draw_density, 'analyticity-sampling': draw_sampling, 'kernel-validity': draw_kernels, 'exterior-cost': draw_exterior}
STEMS = {'domain-coupling': 'domain_coupling', 'lined-section-demo': 'lined_demo'}

def finalize_svg(pdf, report):
    """Keep generated SVG whitespace stable in the byte-preserved figure tree."""
    svg = pdf.with_suffix('.svg')
    content = '\n'.join((line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())) + '\n'
    svg.write_text(content, encoding='utf-8', newline='\n')
    report['outputs']['svg']['sha256'] = hashlib.sha256(svg.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=list(DRAWERS))
    parser.add_argument('--preview-dir', type=Path)
    parser.add_argument('--labels', type=Path)
    args = parser.parse_args()
    if args.labels:
        LABELS.update(json.loads(args.labels.read_text(encoding='utf-8')))
    for job, draw in DRAWERS.items():
        if args.only and args.only != job:
            continue
        fig, checks = draw()
        fig.set_size_inches(WIDTH / 25.4, HEIGHTS[job] / 25.4)
        out = ROOT / 'figures' / job / (STEMS.get(job, job.replace('-', '_')) + '.pdf')
        preview = args.preview_dir / (job + '.png') if args.preview_dir else None
        entry = next((entry for entry in SPEC['figures'] if entry['figure_id'] == job))
        report = fs.save(fig, out, size_mm=fig.get_size_inches() * 25.4, preview_path=preview, spec={'series_colors': entry.get('series_colors', [])})
        finalize_svg(out, report)
        plt.close(fig)
        print(job, report['ok'], flush=True)
if __name__ == '__main__':
    main()
