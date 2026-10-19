import csv
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
from matplotlib.path import Path
import origin_style as fs
import manuscript_figures as method
INK, BLUE, RED = (fs.BLACK, fs.BLUE, fs.RED)
GREY, LIGHT, GOLD, PURPLE = ('#747d85', '#d7dce0', '#a66d22', '#80529b')

def branch_preimages(series, target):
    lo, hi = (min(series), max(series))
    polynomial = np.array([series.get(j, 0) for j in range(lo, hi + 1)], complex)
    polynomial[-lo] -= target
    roots = np.roots(polynomial[::-1])
    residual = float(np.max(abs(method.eval_series(series, roots) - target)))
    assert residual < 1e-10
    assert np.count_nonzero(abs(roots) < 1) == 1 - lo
    assert np.count_nonzero(abs(roots) > 1) == hi - 1
    return (roots, residual)

def draw_sampling(spec, width, height, labels):
    rows, checks = method.branch_geometry(spec)
    theta = np.linspace(0, 2 * np.pi, 721)
    angles = 2 * np.pi * (np.arange(24) + checks['point_offset']) / 24
    phase = np.exp(1j * theta)
    marked = 3
    fig = fs.figure(width, height)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(0, height), aspect='equal')
    ax.set_axis_off()
    serial = 0

    def tag(artist, role):
        nonlocal serial
        serial += 1
        artist.set_gid(f'{role}-{serial:03d}')
        return artist

    def text(x, y, value, *, size=8, color=INK, **kwargs):
        return tag(ax.text(x, y, labels.get(value, value), fontsize=size, color=color, ha=kwargs.pop('ha', 'left'), va=kwargs.pop('va', 'center'), zorder=30, **kwargs), 'label')

    def line(points, *, role, color=GREY, **kwargs):
        artist, = ax.plot(*np.asarray(points).T, color=color, ls=kwargs.pop('ls', '-'), **kwargs)
        return tag(artist, role)

    def circle(centre, radius, *, role, **kwargs):
        artist = Circle(centre, radius, **kwargs)
        ax.add_patch(artist)
        return tag(artist, role)

    def arrow(start, end, *, role, color=GREY):
        artist = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=7, color=color, lw=0.8, shrinkA=0, shrinkB=0, zorder=15)
        ax.add_patch(artist)
        return tag(artist, role)

    def xy(values, centre, scale):
        values = np.atleast_1d(values)
        return np.c_[values.real, values.imag] * scale + centre

    def points(values, centre, scale, *, role, **kwargs):
        return line(xy(values, centre, scale), role=role, ls='none', **kwargs)

    def crosshair(centre, radius, name):
        x, y = centre
        line([(x - radius, y), (x + radius, y)], role=f'{name}-real-axis', color=LIGHT, lw=0.45, zorder=2)
        line([(x, y - radius), (x, y + radius)], role=f'{name}-imaginary-axis', color=LIGHT, lw=0.45, zorder=2)
    text(24, 125, 'Parameter plane $\\sigma$', ha='center', size=9)
    text(71, 125, 'Characteristic image $z_k/R$', ha='center', size=9)
    text(116, 125, 'Algebraic preimages', ha='center', size=9)
    physical = phase + sum((c * phase ** (-j) for j, c in enumerate(spec['geometry']['coefficients'])))
    for fact, colour, light, cy in zip(checks['branches'], (BLUE, RED), ('#e8effb', '#fbeceb'), (94.0, 41.0)):
        k, eta, fold = (fact['k'], fact['eta'], fact['fold_radius'])
        mu = complex(*fact['root'])
        series = method.branch_series(mu, spec)
        boundary = method.eval_series(series, phase)
        boundary_error = float(np.max(abs(boundary - (physical.real + mu * physical.imag))))
        assert boundary_error < 1e-12
        sampling_parameters = eta * np.exp(1j * angles)
        samples = method.eval_series(series, sampling_parameters)
        target = samples[marked]
        roots, residual = branch_preimages(series, target)
        inner, outer = (roots[abs(roots) < 1], roots[abs(roots) > 1])
        selected = int(np.argmin(abs(inner - sampling_parameters[marked])))
        selected_error = float(abs(inner[selected] - sampling_parameters[marked]))
        assert selected_error < 1e-10
        other_inner = np.delete(inner, selected)
        critical = np.array([complex(row['real'], row['imag']) for row in rows if row['branch'] == k and row['kind'] == 'critical'])
        collar = np.array([method.eval_series(series, radius * phase) for radius in np.linspace(eta, 1, 7)])
        interior = Path(np.c_[boundary.real, boundary.imag])
        assert np.all(interior.contains_points(np.c_[collar[:-1].real.ravel(), collar[:-1].imag.ravel()]))
        pc, bc, rc = ((24.0, cy), (71.0, cy), (116.0, cy))
        ps, bs, rs = (18.0, 18.0, 6.0)
        text(5, cy + 22, f'Branch {k}', color=colour, size=9)
        circle(pc, ps, role=f'branch-{k}-unit-disc', fc=light, ec=INK, lw=1.0, zorder=0)
        circle(pc, ps * fold, role=f'branch-{k}-critical-shell', fc='#f0f0ee', ec=GREY, lw=0.65, ls=(0, (2, 2)), zorder=1)
        crosshair(pc, ps, f'branch-{k}-parameter')
        line(xy(eta * phase, pc, ps), role=f'branch-{k}-sampling-ring', color=colour, lw=1.0, zorder=3)
        points(critical, pc, ps, role=f'branch-{k}-derivative-zeros', color=GREY, marker='x', ms=3.2, mew=0.75, zorder=6)
        points(sampling_parameters, pc, ps, role=f'branch-{k}-sampling-parameters', color=colour, marker='o', ms=2.7, mew=0, zorder=7)
        points(other_inner, pc, ps, role=f'branch-{k}-enlarged-inner-preimages', color=INK, marker='o', ms=3.4, mew=0.45, mec='white', zorder=8)
        ray = np.linspace(eta, 1, 81) * np.exp(1j * angles[marked])
        line(xy(ray, pc, ps), role=f'branch-{k}-marked-parameter-ray', color=PURPLE, lw=1.2, zorder=9)
        points(sampling_parameters[marked], pc, ps, role=f'branch-{k}-marked-parameter', color=PURPLE, marker='s', ms=5.0, mfc='white', mew=1.15, zorder=10)
        point = xy(sampling_parameters[marked], pc, ps)[0]
        text(*point + [3.0, 2.7], f'$A_{k}$', color=PURPLE, ha='center')
        text(24, cy - 23, f'$r_{k}={fold:.3f}\\quad\\eta_{k}={eta:.3f}$', ha='center', size=8)
        text(24, cy + 1.8, '0', size=7, color=GREY, ha='center')
        text(24 - ps - 2, cy, '−1', size=7, color=GREY, ha='right')
        text(24, cy + ps + 2.4, '1', size=7, color=GREY, ha='center')
        wall = Polygon(xy(boundary, bc, bs), fc='#fafaf9', ec='none', zorder=0)
        ax.add_patch(wall)
        tag(wall, f'branch-{k}-interior')
        band = Polygon(np.vstack((xy(collar[-1], bc, bs), xy(collar[0][::-1], bc, bs))), fc=light, ec='none', zorder=1)
        ax.add_patch(band)
        tag(band, f'branch-{k}-mapped-collar')
        crosshair(bc, bs * 1.22, f'branch-{k}-image')
        for number, curve in enumerate(collar[1:-1]):
            line(xy(curve, bc, bs), role=f'branch-{k}-radial-level-{number}', color=colour, lw=0.38, alpha=0.4, zorder=3)
        for number, angle in enumerate(angles[::2]):
            curve = method.eval_series(series, np.linspace(eta, 1, 81) * np.exp(1j * angle))
            line(xy(curve, bc, bs), role=f'branch-{k}-angular-line-{number}', color=colour, lw=0.45, alpha=0.65, zorder=4)
        line(xy(boundary, bc, bs), role=f'branch-{k}-characteristic-boundary', color=INK, lw=1.0, zorder=6)
        line(xy(collar[0], bc, bs), role=f'branch-{k}-mapped-sampling-ring', color=colour, lw=1.0, zorder=5)
        points(samples, bc, bs, role=f'branch-{k}-interior-test-points', color=colour, marker='o', ms=2.7, mew=0.3, mec='white', zorder=8)
        line(xy(method.eval_series(series, ray), bc, bs), role=f'branch-{k}-marked-mapped-ray', color=PURPLE, lw=1.2, zorder=9)
        points(target, bc, bs, role=f'branch-{k}-marked-image', color=PURPLE, marker='s', ms=5.0, mfc='white', mew=1.15, zorder=10)
        point = xy(target, bc, bs)[0]
        text(*point + [2.0, 8.5], '$w_0/R$', color=PURPLE, ha='center')
        line([point + [1.2, 5.7], point + [0.2, 1.3]], role=f'branch-{k}-target-label', color=PURPLE, lw=0.6, zorder=12)
        text(71, cy - 1.5, f'$\\mathrm{{Int}}(\\Gamma_{k})$', size=8.5, ha='center')
        text(71, cy - 23, 'Mapped sampling collar', size=8, color=colour, ha='center')
        arrow((43.5, cy), (47, cy), color=colour, role=f'branch-{k}-mapping')
        arrow((94, cy), (98.5, cy), color=PURPLE, role=f'branch-{k}-preimage-solve')
        circle(rc, rs, role=f'branch-{k}-preimage-unit-disc', fc=light, ec=INK, lw=0.8, zorder=1)
        circle(rc, 2 * rs, role=f'branch-{k}-radius-two-guide', fc='none', ec=LIGHT, lw=0.5, ls=(0, (2, 3)), zorder=0)
        crosshair(rc, rs * 3.0, f'branch-{k}-full-preimage')
        points(other_inner, rc, rs, role=f'branch-{k}-inner-preimages', color=INK, marker='o', ms=2.8, mew=0.3, mec='white', zorder=7)
        points(outer, rc, rs, role=f'branch-{k}-outer-preimages', color=GOLD, marker='o', ms=3.8, mfc='white', mew=0.85, zorder=8)
        points(inner[selected], rc, rs, role=f'branch-{k}-selected-preimage', color=PURPLE, marker='s', ms=4.5, mfc='white', mew=1.0, zorder=9)
        text(116, cy + 23, f'$z_{k}(\\sigma_r)=w_0$', ha='center', size=8)
        text(116 + rs * 2, cy - 2.4, '2', color=GREY, size=7, ha='center')
        text(116, cy - 23, '10 inside · 8 outside', size=8, ha='center')
        vertices = [(110.8, cy - 3.0), (99, cy - 8), (101, cy - 25.5), (91, cy - 25.5), (48, cy - 25.5), (45, cy - 25.5), (39.8, cy - 9.0)]
        path = Path(vertices, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE3, Path.CURVE3])
        connector = FancyArrowPatch(path=path, arrowstyle='-', color=GREY, lw=0.6, ls=(0, (3, 3)), zorder=0)
        ax.add_patch(connector)
        tag(connector, f'branch-{k}-unit-disc-enlargement')
        text(70, cy - 29, 'Unit disc enlarged ×3', color=GREY, size=7.5, ha='center')
        fact.update({'polynomial_degree': len(roots), 'preimages': [[float(z.real), float(z.imag)] for z in roots], 'target_normalized': [float(target.real), float(target.imag)], 'preimage_outside_count': len(outer), 'winding_number': len(inner) + min(series), 'preimage_evaluation_residual': residual, 'selected_preimage_error': selected_error, 'boundary_affine_error': boundary_error, 'displayed_collar_inside': True})
    for x, marker, colour, face, value in [(5, 'o', BLUE, BLUE, 'Sampling point'), (39, 'x', GREY, GREY, 'Derivative zero'), (76, 'o', INK, INK, 'Inner preimage'), (110, 'o', GOLD, 'white', 'Outer preimage')]:
        line([(x, 4)], role='shared-symbol-key', color=colour, ls='none', marker=marker, ms=3.5, mfc=face, mew=0.75)
        text(x + 2.2, 4, value, size=7.5)
    checks.update({'marked_sample_index': marked, 'parameter_scale_mm': 18.0, 'branch_image_scale_mm': 18.0, 'full_preimage_scale_mm': 6.0, 'unit_disc_enlargement': 3.0, 'reference_adaptation': 'Baxi F21 object connections and F23 corresponding state markers', 'composition': 'Two connected branch paths with mapped collars and complete preimage planes'})
    source_data = method.ROOT / 'figures/analyticity-sampling/source_data.csv'
    source_data.parent.mkdir(parents=True, exist_ok=True)
    with source_data.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return (fig, checks)
