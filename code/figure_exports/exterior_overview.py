from __future__ import annotations
from pathlib import Path
import sys
import matplotlib as mpl
from matplotlib import colors, patheffects
from matplotlib.patches import Polygon, Rectangle
from matplotlib.tri import Triangulation
from matplotlib.ticker import FuncFormatter, NullLocator
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
import origin_style as fs
from engineering_figures import axes_mm, outline
INK, BLUE = (fs.BLACK, fs.BLUE)
GREY, LIGHT, OCHRE, TEAL = ('#70787D', '#D8DCDF', '#B36624', '#16858A')
DATA = ROOT / 'results/data/derived'

def periodic_triangles(nr, nt):
    """Split each mapped Q4 cell, retaining the actual hole and outer edge."""
    cells = []
    for i in range(nr):
        for j in range(nt):
            a, b = (i * nt + j, i * nt + (j + 1) % nt)
            c, d = (a + nt, b + nt)
            cells.extend(((a, b, d), (a, d, c)))
    return np.asarray(cells)

def draw_exterior(spec, width, height, labels):
    fields = pd.read_csv(DATA / 'near_field_interface_error_hs100.csv')
    fields = fields[(fields.E_ratio == 2.625) & (fields.bedding_dip_deg == 45)]
    profiles = pd.read_csv(DATA / 'exterior_wall_profiles.csv')
    costs = pd.read_csv(DATA / 'exterior_cost_ladder.csv')
    assert len(fields) == 6400 and len(profiles) == 576
    assert fields.abs_diff_um.between(0, 25).all()
    fig = fs.figure(width, height)
    canvas = fig.add_axes([0, 0, 1, 1], zorder=5)
    canvas.set(xlim=(0, width), ylim=(0, height), aspect='equal')
    canvas.set_axis_off()
    serial = 0

    def tag(artist, role):
        nonlocal serial
        serial += 1
        artist.set_gid(f'{role}-{serial:03d}')
        return artist

    def text(x, y, value, *, size=8, color=INK, **kwargs):
        return tag(canvas.text(x, y, labels.get(value, value), fontsize=size, color=color, va=kwargs.pop('va', 'center'), zorder=30, **kwargs), 'label')

    def line(points, *, role, color=INK, **kwargs):
        artist, = canvas.plot(*np.asarray(points).T, color=color, ls=kwargs.pop('ls', '-'), **kwargs)
        return tag(artist, role)

    def leader(points, *, color=GREY, role='correspondence'):
        obj = line(points, role=role, color=color, lw=0.6, zorder=22)
        obj.set_path_effects([patheffects.Stroke(linewidth=1.8, foreground='white'), patheffects.Normal()])
        return obj

    def mark(point, name, *, color=BLUE, offset=(0, 2.5), symbol='o'):
        line([point], role=f'state-{name}', color=color, ls='none', marker=symbol, ms=4, mfc='white', mec=color, mew=0.85, zorder=24)
        if name:
            label = text(*np.asarray(point) + offset, name, color=color, size=7.5, ha='center')
            label.set_path_effects([patheffects.Stroke(linewidth=2, foreground='white'), patheffects.Normal()])

    def axes(x, y, w, h):
        ax = axes_mm(fig, x, y, w, h)
        fs.style_axes(ax, minor=False)
        ax.tick_params(labelsize=7, pad=2, length=3, width=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)
        return ax

    def chart_point(ax, point):
        return fig.transFigure.inverted().transform(ax.transData.transform(point)) * [width, height]

    def draw_mesh(centre, scale, rho, nr, nt, *, color, role):
        for r in np.geomspace(1, rho, nr + 1):
            line(centre + scale * outline(spec, r), role=role + '-ring', color=color, lw=0.24, zorder=7)
        for theta in np.linspace(0, 2 * np.pi, nt, endpoint=False):
            xy = [outline(spec, r, theta) for r in np.geomspace(1, rho, nr + 1)]
            line(centre + scale * np.asarray(xy), role=role + '-radial', color=color, lw=0.24, zorder=7)
        line(centre + scale * outline(spec), role=role + '-wall', lw=0.7, zorder=8)
    text(2, 152, '(a)  Exterior elimination', size=9)
    far_c, near_c = (np.array([16.0, 128.0]), np.array([48.0, 128.0]))
    far_s, near_s = (0.08, 1.28)
    far_boundary = far_c + far_s * outline(spec, 32)
    tag(canvas.add_patch(Polygon(far_boundary, fc='#F3EAE1', ec='none', zorder=6)), 'omitted-exterior-area')
    draw_mesh(far_c, far_s, 32, 60, 64, color='#B9A795', role='far-mesh')
    line(far_boundary, role='remote-boundary-32', color=OCHRE, lw=0.9, zorder=9)
    tag(canvas.add_patch(Polygon(far_c + far_s * outline(spec, 2), fc='#DCE7FA', ec=BLUE, lw=0.55, zorder=12)), 'shared-near-block')
    draw_mesh(near_c, near_s, 2, 12, 64, color='#93A4BD', role='near-mesh')
    line(near_c + near_s * outline(spec, 2), role='DtN-interface', color=BLUE, lw=1.1, zorder=9)
    for centre, scale in [(far_c, far_s), (near_c, near_s)]:
        tag(canvas.add_patch(Rectangle(centre - 10.6 * scale, 21.2 * scale, 21.2 * scale, fc='none', ec=BLUE, lw=0.5, ls=(0, (3, 2)), zorder=13)), 'exact-enlargement-window')
    for sign in (-1, 1):
        leader([far_c + [far_s * 10.6, sign * far_s * 10.6], near_c + [-near_s * 10.6, sign * near_s * 10.6]], color=BLUE, role='exact-near-block-enlargement')
    text(16, 146, 'Meshed, $\\rho_{\\rm far}=32$', size=7.5, ha='center')
    text(48, 146, 'DtN, $\\rho=2$', size=7.5, color=BLUE, ha='center')
    text(48, 128, 'A', size=9, color=BLUE, ha='center')
    text(16, 119.5, 'B', size=9, color=OCHRE, ha='center', bbox=dict(fc='white', ec='none', pad=0.5))
    text(31.8, 142.4, '×16', size=7, color=BLUE, ha='center')
    text(16, 111.4, '7,808 DOFs', size=7.5, ha='center', color=OCHRE)
    text(48, 111.4, '1,664 DOFs', size=7.5, ha='center', color=BLUE)
    text(31, 106.5, 'Identical $12\\times64$ near block', size=7.5, ha='center')
    text(16, 100.8, 'Shaded mesh: omitted exterior', size=7, color=OCHRE)
    text(69, 152, '(b)  Accuracy at common nodes', size=9)
    cost_ax = axes(78, 114, 54, 29)
    styles = [('operator', BLUE, 'o', '-', 'DtN'), ('far_mesh_free', OCHRE, 's', '-', 'Free'), ('far_mesh_fixed', GREY, '^', '--', 'Clamped')]
    for treatment, color, marker, style, name in styles:
        group = costs[costs.treatment == treatment]
        tag(cost_ax.plot(group.total_dof / 1000, 100 * group.rel_error_vs_completed_basis, color=color, ls=style, marker=marker, ms=2.8, lw=0.9, mfc='white', mec=color, mew=0.65, label=name)[0], treatment)
    cost_ax.set_yscale('log')
    cost_ax.set(xlim=(0, 27), ylim=(0.012, 180), xticks=[2, 8, 16, 24], yticks=[0.1, 1, 10, 100])
    cost_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    cost_ax.yaxis.set_minor_locator(NullLocator())
    cost_ax.set_xlabel('Displacement DOFs (×1,000)', fontsize=7.5, labelpad=2)
    cost_ax.set_ylabel('Relative error (%)', fontsize=7.5, labelpad=2)
    cost_ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=7, frameon=False, handlelength=1.3, handletextpad=0.4, columnspacing=0.9, borderaxespad=0)
    first = costs[costs.treatment == 'operator'].iloc[0]
    dip = costs[(costs.treatment == 'far_mesh_free') & (costs.rho_far == 32)].iloc[0]
    mark(chart_point(cost_ax, (first.total_dof / 1000, 100 * first.rel_error_vs_completed_basis)), 'A', color=BLUE, offset=(0, 2.8))
    mark(chart_point(cost_ax, (dip.total_dof / 1000, 100 * dip.rel_error_vs_completed_basis)), 'B', color=OCHRE, offset=(1.9, -2.1), symbol='s')
    cost_ax.annotate('$\\rho_{\\rm far}=32$', xy=(dip.total_dof / 1000, 100 * dip.rel_error_vs_completed_basis), xytext=(13.4, 0.12), fontsize=7, color=OCHRE, arrowprops=dict(arrowstyle='-', color=OCHRE, lw=0.6))
    cost_ax.text(14.8, 0.73, 'Near-mesh plateau', fontsize=7, color=GREY)
    cost_ax.text(10.2, 0.021, 'Near mesh refined', fontsize=7, color=BLUE)
    text(104.5, 101, 'A → B: 4.7× DOFs at ≈0.3% error', size=7.5, ha='center')
    text(2, 95.5, '(c)  Spatial error, $24\\times128$', size=9)
    text(135, 95.5, '$E_h/E_v=2.625,\\ \\phi=45^\\circ$', size=8, ha='right')
    levels = np.linspace(0, 25, 17)
    norm = colors.BoundaryNorm(levels, mpl.colormaps['turbo'].N)
    artists, peaks = ([], [])
    wall = outline(spec)
    for cx, rho, color in [(30.0, 2.0, BLUE), (88.0, 1.5, TEAL)]:
        ax = axes_mm(fig, cx - 20.9, 50.5, 41.8, 41.8)
        ax.set(xlim=(-11, 11), ylim=(-11, 11), aspect='equal')
        ax.set_axis_off()
        group = fields[fields.rho_out == rho].reset_index(drop=True)
        assert len(group) == 25 * 128
        for i, grade in enumerate(np.geomspace(1, rho, 25)):
            xy = outline(spec, grade, np.linspace(0, 2 * np.pi, 128, endpoint=False))
            assert np.allclose(group.iloc[i * 128:(i + 1) * 128][['x_m', 'y_m']], xy, rtol=0, atol=0.000501)
        tri = Triangulation(group.x_m, group.y_m, periodic_triangles(24, 128))
        cf = ax.tricontourf(tri, group.abs_diff_um, levels=levels, norm=norm, cmap='turbo')
        cf.set_edgecolor('face')
        tag(cf, f'error-field-rho-{rho:g}')
        ax.plot(*wall.T, color=INK, lw=0.7, gid=f'wall-rho-{rho:g}')
        ax.plot(*outline(spec, rho).T, color=color, lw=0.9, gid=f'interface-rho-{rho:g}')
        peak = group.loc[group.abs_diff_um.idxmax()]
        assert peak.node_grade == 1
        wall_profile = profiles[(profiles.rho_out == rho) & (profiles.near_mesh == '24x128')]
        assert np.allclose(wall_profile.abs_diff_um, group.iloc[:128].abs_diff_um, rtol=0, atol=5.1e-06)
        point = chart_point(ax, (peak.x_m, peak.y_m))
        mark(point, 'W', color=color, offset=(-2.7, -2.7), symbol='D')
        leader([point + [1, 0.8], [cx - 5, 69.5], [cx - 3, 69.5]], color=color, role=f'wall-foot-value-rho-{rho:g}')
        text(cx + 3.5, 70, f'{peak.abs_diff_um:.2f} μm', size=7, ha='center', color=color)
        text(cx, 75.8, f'$\\rho={rho:g}$', size=8, color=color, ha='center')
        for theta, name, delta in [(0, 'C', [0, -2.0]), (np.pi, 'I', [0, 2.2])]:
            p = chart_point(ax, outline(spec, 1, theta))
            text(*p + delta, name, size=7, ha='center')
        artists.append(cf)
        peaks.append(dict(rho=rho, x_m=float(peak.x_m), y_m=float(peak.y_m), value_um=float(peak.abs_diff_um)))
    bar = fig.colorbar(artists[0], cax=axes_mm(fig, 120, 58, 2.5, 29), orientation='vertical', ticks=[0, 5, 10, 15, 20, 25])
    bar.ax.set_label('<colorbar>')
    bar.ax.tick_params(labelsize=7, pad=2, length=2.2)
    artists[1].colorbar = bar
    text(122.4, 91.3, '$|\\Delta u|$ (μm)', size=7.5, ha='center')
    line([(116, 52.5), (125.5, 52.5)], role='five-metre-scale', lw=1)
    for x in (116, 125.5):
        line([(x, 52), (x, 53)], role='scale-end', lw=0.7)
    text(120.75, 49.9, '5 m', size=7, ha='center')
    text(58, 48.5, 'Same physical scale; W is the same wall node', size=7.5, ha='center')
    text(2, 43, '(d)  Wall convergence', size=9)
    profile_ax = axes(14, 10, 118, 26)
    profile_ax.set_yscale('log')
    profile_ax.set(xlim=(0, 360), ylim=(0.08, 130), xticks=[0, 90, 180, 270, 360], yticks=[0.1, 1, 10, 100])
    profile_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    profile_ax.yaxis.set_minor_locator(NullLocator())
    profile_ax.set_ylabel('$|\\Delta u|$ (μm)', fontsize=7.5, labelpad=2)
    profile_ax.set_xlabel('Wall parameter $\\theta$ (deg)', fontsize=7.5, labelpad=2)
    series = [(2.0, '12x64', GREY, '-', '$12\\times64$'), (2.0, '24x128', BLUE, '-', '$24\\times128$'), (2.0, '48x256', INK, '-', '$48\\times256$'), (1.5, '24x128', TEAL, '--', '$24\\times128,\\ \\rho=1.5$')]
    for rho, mesh, color, style, title in series:
        group = profiles[(profiles.rho_out == rho) & (profiles.near_mesh == mesh)]
        theta = np.r_[group.theta_deg, 360]
        values = np.r_[group.abs_diff_um, group.abs_diff_um.iloc[0]]
        tag(profile_ax.plot(theta, values, color=color, lw=0.95, ls=style, label=title)[0], f'wall-profile-{mesh}-rho-{rho:g}')
        point = group[group.theta_deg == 118.125].iloc[0]
        profile_ax.plot(point.theta_deg, point.abs_diff_um, marker='D', ms=3.2, color=color, mfc='white', mew=0.7, ls='none', gid=f'same-wall-node-{mesh}-rho-{rho:g}')
    profile_ax.axvline(118.125, color=LIGHT, ls=':', lw=0.7, zorder=0)
    profile_ax.text(124, 75, 'W', fontsize=7.5, color=INK)
    profile_ax.text(4, 0.12, 'Crown C', fontsize=7, color=GREY)
    profile_ax.text(180, 0.12, 'Invert I', fontsize=7, color=GREY, ha='center')
    profile_ax.text(356, 0.12, 'Crown C', fontsize=7, color=GREY, ha='right')
    for x, (_, _, color, style, title) in zip([45, 64, 84, 105], series):
        line([(x - 4, 42), (x, 42)], role='profile-key', color=color, ls=style, lw=0.95)
        text(x + 1, 42, title, size=7)
    near_xy = np.array([outline(spec, r, np.linspace(0, 2 * np.pi, 64, endpoint=False)) for r in np.geomspace(1, 2, 13)])
    far_xy = np.array([outline(spec, r, np.linspace(0, 2 * np.pi, 64, endpoint=False)) for r in np.geomspace(1, 32, 61)])
    deviation = float(np.max(np.abs(near_xy - far_xy[:13])))
    assert deviation < 1e-12
    return (fig, {'material_ratio': 2.625, 'bedding_dip_deg': 45, 'field_mesh': '24x128', 'cost_near_base': '12x64', 'cost_evaluation_nodes': 832, 'field_node_count': 6400, 'field_connectivity': 'Two triangles per actual periodic Q4 element', 'field_scale_mm_per_m': 1.9, 'field_levels_um': levels.tolist(), 'field_limits_um': [0, 25], 'clipped_field_nodes': 0, 'field_alignment': 'Mean translation on each complete near mesh', 'field_peaks': peaks, 'wall_profile_nodes': len(profiles), 'wall_marker_theta_deg': 118.125, 'near_block_max_difference_m': deviation, 'near_block_enlargement': near_s / far_s, 'DOF_ratio_B_to_A': float(dip.total_dof / first.total_dof)})
