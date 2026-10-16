from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FuncFormatter, NullLocator
ROOT = Path(__file__).resolve().parents[2]
import origin_style as fs
from engineering_figures import axes_mm, outline
INK, RED, BLUE = (fs.BLACK, fs.RED, fs.BLUE)
GREY, LIGHT, OCHRE = ('#69747B', '#CDD2D5', '#AE791C')
MATERIALS = (('F0', '(1.5, 90°)', INK, '-'), ('F1', '(3, 45°)', RED, '--'), ('F2', '(2, 60°)', OCHRE, '-.'), ('SABNGS', 'Queenston', GREY, ':'), ('WAI(d=45.0)', 'Waichecheng', BLUE, '-'))

def draw_kernels(spec, width, height, labels):
    data = pd.read_csv(ROOT / 'results/data/derived/closed_kernel_paths.csv')
    wall = data[data.path == 'shape']
    exterior = data[data.path == 'interface']
    assert len(wall) == len(exterior) == 155
    assert set(data.material) == {row[0] for row in MATERIALS}
    assert (data[['rel0', 'rel1', 'rel2']] > 0).all().all()
    assert np.allclose(wall.rho_out, 1) and np.allclose(exterior.t, 1)
    fig = fs.figure(width, height)
    canvas = fig.add_axes([0, 0, 1, 1], zorder=3)
    canvas.set(xlim=(0, width), ylim=(0, height), aspect='equal')
    canvas.set_axis_off()
    serial = 0

    def tag(artist, role):
        nonlocal serial
        serial += 1
        artist.set_gid(f'{role}-{serial:03d}')
        return artist

    def text(x, y, value, *, size=8, color=INK, **kwargs):
        return tag(canvas.text(x, y, labels.get(value, value), fontsize=size, color=color, va=kwargs.pop('va', 'center'), zorder=25, **kwargs), 'label')

    def line(points, *, role, color=INK, **kwargs):
        artist, = canvas.plot(*np.asarray(points).T, color=color, ls=kwargs.pop('ls', '-'), **kwargs)
        return tag(artist, role)

    def route(points, *, role):
        path = MplPath(points, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        arrow = FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=6, color=GREY, lw=0.7, zorder=10)
        canvas.add_patch(arrow)
        return tag(arrow, role)

    def mark(point, name, *, offset=(0, 3), color=BLUE):
        line([point], role=f'state-{name}', ls='none', marker='o', ms=4, mfc='white', mec=color, mew=1.0, zorder=20)
        text(*np.asarray(point) + offset, name, size=7.5, color=color, ha='center')

    def physical(t, rho=1, theta=None):
        geometry = {**spec['geometry'], 'coefficients': [t * c for c in spec['geometry']['coefficients']]}
        return outline({'geometry': geometry}, rho, theta)

    def plot_axes(x, y, w, h, xlabel, ylim, yticks):
        ax = axes_mm(fig, x, y, w, h)
        fs.style_axes(ax, minor=False)
        ax.set_yscale('log')
        ax.set(ylim=ylim, yticks=yticks, xlabel=xlabel)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.0e}'))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(labelsize=7, pad=2, length=3, width=0.7)
        ax.tick_params(axis='y', labelleft=False, labelright=True)
        ax.xaxis.label.set_size(8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)
        return ax

    def chart_point(ax, x, y):
        return fig.transFigure.inverted().transform(ax.transData.transform((x, y))) * [width, height]

    def scale_bar(x, y, scale):
        line([(x, y), (x + 5 * scale, y)], role='five-metre-scale', lw=0.9)
        for a in (x, x + 5 * scale):
            line([(a, y - 0.6), (a, y + 0.6)], role='scale-end', lw=0.7)
        text(x + 2.5 * scale, y - 2.8, '5 m', ha='center', size=7)
    text(2, 120, '(a)  Shape departure', size=9)
    text(62, 120, 'Relative wall error, $\\rho=1$', size=9)
    centre, scale = (np.array([25.0, 97.0]), 3.8)
    styles = [(0, GREY, (0, (2, 2))), (0.25, OCHRE, ':'), (0.5, RED, '--'), (1, BLUE, '-')]
    for t, color, style in styles:
        line(centre + scale * physical(t), role=f'shape-t-{t:g}', color=color, lw=1.0 if t == 1 else 0.7, ls=style, zorder=5)
    text(25, 102, '$c_j(t)=t\\,c_j$', ha='center', size=8.5)
    text(25, 96.5, 'Circle → horseshoe', ha='center', size=7.5)
    for x, y, t, col, style in [(7, 76, 0, GREY, (0, (2, 2))), (29, 76, 0.25, OCHRE, ':'), (7, 71, 0.5, RED, '--'), (29, 71, 1, BLUE, '-')]:
        line([(x, y), (x + 5, y)], role='shape-key', color=col, lw=0.9, ls=style)
        text(x + 6.5, y, f'$t={t:g}$', size=7.5)
    scale_bar(14, 65, scale)
    upper = plot_axes(62, 81, 62, 32, 'Mapping strength $t$', (7e-07, 0.025), [1e-06, 1e-05, 0.0001, 0.001, 0.01])
    upper.set(xlim=(0.25, 1.04), xticks=[0.25, 0.5, 0.75, 1], xticklabels=['0.25', '0.5', '0.75', '1'])
    for order, color, style in [(0, INK, '-'), (1, RED, '--'), (2, BLUE, '-.')]:
        grouped = wall.groupby('t')[f'rel{order}']
        tag(upper.fill_between(grouped.min().index, grouped.min(), grouped.max(), color=color, alpha=0.12, lw=0), f'order-{order}-material-range')
        artist, = upper.plot(grouped.max().index, grouped.max(), color=color, lw=1.1, ls=style, label=f'{order}th' if order == 0 else f"{order}{('st' if order == 1 else 'nd')}")
        tag(artist, f'order-{order}-maximum')
    upper.legend(loc='lower right', fontsize=7.5, frameon=False, handlelength=1.7, borderaxespad=0.6, handletextpad=0.5, labelspacing=0.4)
    text(93, 115.8, 'Lines: maximum; bands: five-material range', size=7, ha='center')
    theta_mark = 3 * np.pi / 2
    a = centre + scale * physical(0.25, theta=theta_mark)
    a_error = float(wall[wall.t == 0.25].rel2.max())
    a_curve = chart_point(upper, 0.25, a_error)
    route([a + [1.2, 0], [52, 94], [54, a_curve[1]], a_curve + [-1, 0]], role='shape-A-to-operator')
    mark(a, 'A', offset=(0, 3.2))
    mark(a_curve, 'A', offset=(2.6, 3))
    b_error = float(wall[wall.t == 1].rel2.max())
    b_curve = chart_point(upper, 1, b_error)
    mark(b_curve, 'B', offset=(-3, -3.0))
    text(2, 57.5, '(b)  Interface expansion', size=9)
    text(62, 62, 'Second-order relative error, $t=1$', size=9)
    centre2, scale2 = (np.array([25.0, 32.5]), 2.1)
    outer = centre2 + scale2 * physical(1, 2)
    fill = Polygon(outer, fc='#EDF2F6', ec='none', zorder=0)
    canvas.add_patch(fill)
    tag(fill, 'retained-near-field')
    for rho in np.linspace(1, 2, 9):
        line(centre2 + scale2 * physical(1, rho), role='mapped-annulus-ring', color=LIGHT, lw=0.25, zorder=1)
    for theta in np.linspace(0, 2 * np.pi, 32, endpoint=False):
        xy = np.array([physical(1, rho, theta) for rho in np.linspace(1, 2, 41)])
        line(centre2 + scale2 * xy, role='mapped-annulus-ray', color=LIGHT, lw=0.25, zorder=1)
    fill = Polygon(centre2 + scale2 * physical(1), fc='white', ec='none', zorder=2)
    canvas.add_patch(fill)
    tag(fill, 'excavation')
    interface_points = []
    for rho, col, style, name in [(1, INK, '-', 'B'), (1.5, OCHRE, '--', 'C'), (2, BLUE, '-', 'D')]:
        line(centre2 + scale2 * physical(1, rho), role=f'physical-interface-{rho:g}', color=col, lw=1.0, ls=style, zorder=5)
        point = centre2 + scale2 * physical(1, rho, theta_mark)
        interface_points.append(point)
        mark(point, name, offset=(0, -3.4))
    text(25, 35, 'Wall', size=7.5, ha='center')
    for rho, col, x, y in [(1, INK, 14, 22), (1.5, OCHRE, 12, 48), (2, BLUE, 37, 51.5)]:
        text(x, y, f'$\\rho={rho:g}$', size=7.5, color=col, ha='center', bbox=dict(fc='white', ec='none', pad=0.2))
    scale_bar(8, 7.5, scale2)
    text(32, 6.5, 'Physical scale', size=7, color=GREY, ha='center')
    lower = plot_axes(62, 16, 62, 39, 'Interface radius $\\rho$', (3e-09, 0.0015), [1e-08, 1e-07, 1e-06, 1e-05, 0.0001, 0.001])
    lower.set(xlim=(1, 2.04), xticks=[1, 1.25, 1.5, 1.75, 2], xticklabels=['1', '1.25', '1.5', '1.75', '2'])
    for material, title, color, style in MATERIALS:
        subset = exterior[exterior.material == material].sort_values('rho_out')
        assert len(subset) == 31 and np.all(np.diff(subset.rel2) < 0)
        artist, = lower.plot(subset.rho_out, subset.rel2, color=color, lw=1.15 if material.startswith('WAI') else 0.85, ls=style, label=title)
        tag(artist, f'second-order-{material}')
    lower.legend(loc='upper right', fontsize=7, frameon=False, handlelength=1.65, handletextpad=0.45, borderaxespad=0.6, labelspacing=0.35)
    marked_errors = {}
    for rho, name in [(1, 'B'), (1.5, 'C'), (2, 'D')]:
        error = float(exterior[exterior.rho_out == rho].rel2.max())
        point = chart_point(lower, rho, error)
        mark(point, name, offset=(2.5, 2.5) if rho < 2 else (-2.8, -2.8))
        marked_errors[name] = error
    low_b = chart_point(lower, 1, marked_errors['B'])
    route([interface_points[0] + [0, 1.3], [39, 48], [54, 57], low_b + [-1, 0]], role='wall-B-to-operator')
    identity_errors = []
    for rho in (1, 1.5, 2):
        normalized = {'geometry': {'radius_m': spec['geometry']['radius_m'] * rho, 'coefficients': [c * rho ** (-(j + 1)) for j, c in enumerate(spec['geometry']['coefficients'])]}}
        identity_errors.append(float(np.max(abs(outline(normalized) - physical(1, rho)))))
    assert max(identity_errors) < 1e-12
    assert np.isclose(marked_errors['B'], b_error, rtol=1e-12)
    refinement = abs(data.rel2_check - data.rel2) / data.rel2
    return (fig, {'source_data': 'results/data/derived/closed_kernel_paths.csv', 'path_samples': {'shape_per_material': 31, 'interface_per_material': 31}, 'unique_operator_states': 305, 'retained_band': 8, 'resultant_cutoffs': [64, 96], 'worst_reference_relative_drift': float(refinement.max()), 'geometry_scales_mm_per_m': [scale, scale2], 'interface_reparameterization_error_m': max(identity_errors), 'marked_maximum_second_order_errors': {'A': a_error, **marked_errors}, 'marks': 'A: t=0.25 at the wall; B: t=rho=1; C/D: rho=1.5/2 at t=1', 'bands': 'Five-material extrema, not uncertainty intervals', 'reference_composition': 'Baxi F23 curve and state connectors; physical-coordinate vector primitives from F21'})
