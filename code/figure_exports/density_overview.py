import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
import numpy as np
import origin_style as fs
from engineering_figures import axes_mm
import manuscript_figures as method
INK, RED, BLUE = (fs.BLACK, fs.RED, fs.BLUE)
GREY = '#737b83'
LOCATIONS = [('A', 180.0, 'D', '#a76c1e'), ('B', 60.0, 'o', '#7b519b')]

def draw_density(width, height, labels):
    rows, checks = method.density_data()
    data = {c: {key: np.array([row[key] for row in rows if row['c1'] == c]) for key in rows[0]} for c in (0.0, 0.3)}
    theta = data[0.0]['theta_deg']
    phase = np.deg2rad(theta)
    _, mat_m, _, _ = method.material()
    coefficients = np.linalg.solve(mat_m, np.array([0.5, 0.0]))
    displacement = 2 * np.real((mat_m @ coefficients)[:, None] * np.exp(-3j * phase))
    displacement_error = float(np.max(abs(displacement - [np.cos(3 * phase), np.zeros(len(phase))])))
    assert displacement_error < 1e-12
    density_error = max((float(np.max(abs(data[0.0][f'density_{c}_over_Ev_u0'] - data[0.3][f'density_{c}_over_Ev_u0']))) for c in 'xy'))
    assert density_error < 1e-12
    for part in data.values():
        for component in 'xy':
            assert np.allclose(part[f'density_{component}_over_Ev_u0'], part['jacobian_over_R'] * part[f'R_traction_{component}_over_Ev_u0'], rtol=1e-12, atol=1e-12)
    fig = fs.figure(width, height)
    canvas = fig.add_axes([0, 0, 1, 1], zorder=0)
    canvas.set(xlim=(0, width), ylim=(0, height), aspect='equal')
    canvas.set_axis_off()

    def text(x, y, value, *, size=8, **kw):
        return canvas.text(x, y, labels.get(value, value), fontsize=size, ha=kw.pop('ha', 'left'), va=kw.pop('va', 'center'), color=kw.pop('color', INK), zorder=15, **kw)

    def line(points, *, role, color=GREY, **kw):
        xy = np.asarray(points)
        artist, = canvas.plot(*xy.T, color=color, ls=kw.pop('ls', '-'), **kw)
        artist.set_gid(role)
        return artist

    def arrow(start, end, *, color=GREY, **kw):
        return canvas.annotate('', xy=end, xytext=start, arrowprops={'arrowstyle': '-|>', 'color': color, 'lw': 0.75, 'shrinkA': 0, 'shrinkB': 0, **kw})

    def graph(x, y, w, h):
        ax = axes_mm(fig, x, y, w, h)
        fs.style_axes(ax, minor=False)
        ax.tick_params(labelsize=7.5, pad=2, length=2.6, width=0.7)
        ax.set(xlim=(0, 360), xticks=[0, 90, 180, 270, 360])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        return ax
    text(3, 112, 'One displacement harmonic', size=9)
    input_ax = graph(10, 98, 40, 10)
    input_ax.plot(theta, displacement[0], color=BLUE, lw=1.05)
    input_ax.axhline(0, color='#d3d8dc', lw=0.5, zorder=0)
    input_ax.set(ylim=(-1.3, 1.3), yticks=[-1, 0, 1], xticks=[0, 180, 360])
    input_ax.set_ylabel('$u_x/u_0$', fontsize=8, labelpad=1)
    text(30, 91.5, '$u_x=u_0\\cos3\\theta,\\quad u_y=0$', size=7.5, ha='center')
    arrow((52, 103), (66, 103), color=BLUE)
    text(59, 107, '$m=3$', size=7.5, color=BLUE, ha='center')
    levels = np.linspace(-4, 4, 17)
    spectrum = colors.LinearSegmentedColormap.from_list('boundary_traction', ['#1724d5', '#00b8f5', '#27da83', '#d9ee38', '#ff9c22', '#d8131c'])
    norm = colors.BoundaryNorm(levels, spectrum.N)
    field_artists = []
    scale = 12.0
    for c1, centre, title in [(0.0, (27.0, 73.0), 'Circle'), (0.3, (27.0, 37.0), 'Ellipse')]:
        part = data[c1]
        xy = np.c_[part['x_over_R'], part['y_over_R']] * scale + centre
        canvas.add_patch(Polygon(xy, fc='#f3f3f0', ec='none', zorder=1))
        line(xy, color=INK, lw=4.2, role=f'boundary-outline-{c1}', zorder=2)
        field = LineCollection(np.stack((xy[:-1], xy[1:]), axis=1), cmap=spectrum, norm=norm, linewidths=3.3, zorder=3)
        field.set_array((part['R_traction_x_over_Ev_u0'][:-1] + part['R_traction_x_over_Ev_u0'][1:]) / 2)
        field.set_gid(f'boundary-x-traction-{c1}')
        canvas.add_collection(field)
        field_artists.append(field)
        nodes = xy[:-1:30]
        line(nodes, color=INK, ls='none', marker='.', ms=1.8, role=f'equal-parameter-samples-{c1}', zorder=5)
        for index in range(0, 720, 60):
            dx = 5.0 * displacement[0, index]
            if abs(dx) > 1e-08:
                start = xy[index]
                arrow(start, start + [dx, 0], color=BLUE)
        text(centre[0], centre[1], title, size=9, ha='center')
        for name, angle, marker, col in LOCATIONS:
            index = int(np.argmin(abs(theta - angle)))
            p = xy[index]
            line([p], color=col, ls='none', marker=marker, ms=4.7, mfc='white', mew=1.0, role=f'boundary-location-{name}-{c1}', zorder=10)
            offset = (-2.6, 3.2) if name == 'A' else (3.4, 3.3)
            text(*p + offset, name, color=col, size=8, ha='center')
        text(centre[0], centre[1] - 4.5, f'$c_1={c1:g}$', size=7.5, ha='center', color=GREY)
    line([(9, 20), (21, 20)], role='common-length-scale', color=INK, lw=1)
    line([(9, 19), (9, 21)], role='scale-left', color=INK, lw=0.7)
    line([(21, 19), (21, 21)], role='scale-right', color=INK, lw=0.7)
    text(15, 17.5, '$R$', size=8, ha='center')
    arrow((43, 18), (49, 18), color=INK)
    arrow((43, 18), (43, 24), color=INK)
    text(50.5, 18, '$x$', size=8)
    text(43, 26, '$y$', size=8, ha='center')
    text(27, 13, 'Boundary color: $Rt_x/(E_vu_0)$', size=7.5, ha='center')
    bar = fig.colorbar(field_artists[0], cax=axes_mm(fig, 6, 7.5, 44, 2.4), orientation='horizontal', ticks=[-4, -2, 0, 2, 4])
    bar.ax.set_label('<colorbar>')
    bar.ax.tick_params(labelsize=7.5, pad=2, length=2)
    for field in field_artists[1:]:
        field.colorbar = bar
    x0, w = (72.0, 60.0)
    density_ax = graph(x0, 82, w, 23)
    jac_ax = graph(x0, 56, w, 14)
    traction_ax = graph(x0, 18, w, 25)
    text(x0, 109.5, 'Coincident force densities', size=9)
    text(x0, 74, 'Arc-length Jacobian', size=9)
    text(x0, 47, 'Geometry-dependent traction', size=9)
    for ax in (density_ax, jac_ax):
        ax.tick_params(labelbottom=False)
    for ax, ylim, ylabel in [(density_ax, (-3.4, 3.4), '$q/(E_vu_0)$'), (traction_ax, (-4.3, 4.3), '$Rt/(E_vu_0)$')]:
        ax.set(ylim=ylim, yticks=[-3, 0, 3] if ax is density_ax else [-4, -2, 0, 2, 4])
        ax.set_ylabel(ylabel, fontsize=8.5, labelpad=3)
        ax.axhline(0, color='#cdd3d7', lw=0.55, zorder=0)
    jac_ax.set(ylim=(0.6, 1.4), yticks=[0.7, 1, 1.3])
    jac_ax.set_ylabel('$J/R$', fontsize=8.5, labelpad=3)
    traction_ax.set_xlabel('Boundary parameter $\\theta$ (deg)', fontsize=8.5, labelpad=3)
    for c1, ls in [(0.0, '-'), (0.3, (0, (3.5, 2.2)))]:
        part = data[c1]
        jac_ax.plot(theta, part['jacobian_over_R'], color=GREY if c1 == 0 else BLUE, ls=ls, lw=1.1)
        for component, col in [('x', INK), ('y', RED)]:
            for ax, key in [(density_ax, f'density_{component}_over_Ev_u0'), (traction_ax, f'R_traction_{component}_over_Ev_u0')]:
                artist, = ax.plot(theta, part[key], color=col, ls=ls, lw=2.5 if c1 == 0 else 1.0, alpha=0.3 if c1 == 0 else 1.0)
                artist.set_gid(f'{key}-{c1}')
    jac_ax.fill_between(theta, 1, data[0.3]['jacobian_over_R'], color=BLUE, alpha=0.07, lw=0)
    for name, angle, marker, col in LOCATIONS:
        index = int(np.argmin(abs(theta - angle)))
        px = x0 + w * angle / 360
        line([(px, 18), (px, 105)], role=f'aligned-parameter-{name}', color=col, lw=0.55, ls=(0, (1.5, 3)), alpha=0.45, zorder=0)
        for ax, key in [(density_ax, 'density_x_over_Ev_u0'), (jac_ax, 'jacobian_over_R'), (traction_ax, 'R_traction_x_over_Ev_u0')]:
            ax.axvline(angle, color=col, lw=0.55, ls=(0, (1.5, 3)), alpha=0.45, zorder=0)
            ax.plot(angle, data[0.3][key][index], marker=marker, ms=4.3, mfc='white', mec=col, mew=1, ls='none', zorder=8)
        density_ax.text(angle + 8, 2.8, name, color=col, fontsize=8)
        j = data[0.3]['jacobian_over_R'][index]
        jac_ax.annotate(f'{j:.2f}', (angle, j), (angle + 23, j + 0.08), fontsize=7.5, color=col, va='center')
        factor = 1 / j
        traction_ax.annotate(f'{factor:.2f}×', (angle, data[0.3]['R_traction_x_over_Ev_u0'][index]), (angle + 37, 3.3 if name == 'B' else 2.6), fontsize=8, color=col, arrowprops={'arrowstyle': '-', 'lw': 0.6, 'color': col})
    arrow((46, 73), (69, 74), color=GREY)
    arrow((49, 37), (69, 74), color=BLUE, connectionstyle='arc3,rad=-.23', linestyle=(0, (3.5, 2.2)))
    text(101, 51.5, 'Density ÷ Jacobian', size=7.5, ha='center', color=GREY)
    line([(132, 93), (135, 93), (135, 32)], role='density-to-traction-route', lw=0.7)
    line([(132, 63), (135, 63)], role='jacobian-to-traction-route', lw=0.7)
    canvas.add_patch(Circle((135, 63), 1.1, fc='white', ec=GREY, lw=0.6, zorder=10))
    text(135, 63, '÷', size=8, ha='center')
    arrow((135, 32), (132.5, 32), color=GREY)
    fig.legend(handles=[Line2D([], [], color=INK, lw=1.2, label='$x$'), Line2D([], [], color=RED, lw=1.2, label='$y$'), Line2D([], [], color=GREY, lw=2.5, alpha=0.5, label='Circle'), Line2D([], [], color=GREY, lw=1, ls=(0, (3.5, 2.2)), label='Ellipse')], loc='center', bbox_to_anchor=(102 / width, 3.5 / height), ncol=4, fontsize=7.5, frameon=False, handlelength=1.6, columnspacing=0.9)
    checks.update(samples_per_shape=len(theta), displacement_reconstruction_error=displacement_error, density_difference_between_shapes=density_error, jacobian_range_over_R=[float(data[0.3]['jacobian_over_R'].min()), float(data[0.3]['jacobian_over_R'].max())], boundary_traction_color={'quantity': 'R*t_x/(E_v*u0)', 'levels': levels.tolist(), 'actual_range': [float(min((part['R_traction_x_over_Ev_u0'].min() for part in data.values()))), float(max((part['R_traction_x_over_Ev_u0'].max() for part in data.values())))], 'clipping': False, 'location': 'actual physical boundary'}, highlighted_locations=[{'label': name, 'theta_deg': angle, 'ellipse_J_over_R': float(data[0.3]['jacobian_over_R'][int(np.argmin(abs(theta - angle)))]), 'ellipse_to_circle_traction': float(1 / data[0.3]['jacobian_over_R'][int(np.argmin(abs(theta - angle)))])} for name, angle, _, _ in LOCATIONS], geometry_scale_mm_per_R=scale, displacement_arrow_mm_per_u0=5.0)
    return (fig, checks)
