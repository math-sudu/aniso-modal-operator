from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from matplotlib.path import Path as CurvePath
import matplotlib.patheffects as pe
import numpy as np
import origin_style as fs
import manuscript_figures as method
ROOT = Path(__file__).resolve().parents[2]
INK, BLUE, GREY = ('#20262b', '#285b79', '#737b80')
YELLOW = '#ffe36f'
VIEWS = [{'id': 11997, 'asset': 'moxi_severe_steel_ribs.jpg', 'position': [1.5, 5], 'crop': [0, 0, 842, 473], 'site': 'Moxi · steel ribs', 'ellipses': [[197, 225, 265, 228, 0], [672, 301, 239, 270, 24]], 'callout': ['Bent ribs', [608, 271], [0.55, 0.12]]}, {'id': 13334, 'asset': 'moxi_i18_replacement.jpg', 'position': [104.5, 5], 'crop': [0, 0, 475, 290], 'site': 'Moxi · I18 joints', 'ellipses': [[144, 147, 101, 171, 9], [351, 133, 83, 164, 3]], 'callout': ['Flange bending', [148, 133], [0.03, 0.9]]}, {'id': 14822, 'asset': 'moxi_distorted_rib_row.jpg', 'position': [1.5, 50], 'crop': [5, 140, 795, 630], 'site': 'Moxi · rib row', 'ellipses': [[325, 322, 238, 255, 0], [528, 330, 144, 261, 0]], 'callout': ['Bent segments', [385, 325], [0.03, 0.87]]}, {'id': 14819, 'asset': 'chaluo_bent_rib.jpg', 'position': [104.5, 50], 'crop': [35, 90, 445, 350], 'site': 'Chaluo · steel rib', 'ellipses': [[256, 211, 119, 213, -8]], 'callout': ['Rib bending', [278, 238], [0.03, 0.86]]}, {'id': 14800, 'asset': 'guodashan_crown_ribs.jpg', 'position': [1.5, 94], 'crop': [0, 45, 550, 382], 'site': 'Guodashan · crown', 'ellipses': [[307, 167, 293, 114, 33]], 'callout': ['Crown distortion', [369, 196], [0.03, 0.89]]}, {'id': 11600, 'asset': 'moxi_floor_heave.jpg', 'position': [104.5, 94], 'crop': [5, 55, 530, 380], 'site': 'Moxi · floor heave', 'ellipses': [[177, 250, 339, 112, -38]], 'callout': ['Heave and crack', [245, 200], [0.38, 0.12]]}]

def draw_domain(spec, width, height, labels):
    fig = fs.figure(width, height)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(height, 0), aspect='equal')
    ax.set_axis_off()

    def text(x, y, value, size=7.2, color=INK, **kw):
        return ax.text(x, y, labels.get(value, value), fontsize=size, color=color, ha=kw.pop('ha', 'center'), va=kw.pop('va', 'center'), zorder=kw.pop('zorder', 10), **kw)

    def line(points, color=GREY, lw=0.55, **kw):
        points = np.asarray(points)
        return ax.plot(points[:, 0], points[:, 1], color=color, lw=lw, ls=kw.pop('ls', '-'), **kw)[0]

    def arrow(points, color=BLUE, lw=0.8):
        points = np.asarray(points)
        if len(points) > 2:
            line(points[:-1], color, lw, zorder=7)
        return ax.annotate('', xy=points[-1], xytext=points[-2], zorder=8, arrowprops={'arrowstyle': '-|>', 'lw': lw, 'color': color, 'shrinkA': 0, 'shrinkB': 0})

    def backed(x, y, value, **kw):
        return text(x, y, value, bbox={'fc': 'white', 'ec': 'none', 'pad': 0.2}, **kw)

    def dot(points, *, marker='o', color=BLUE, size=3.5, face='white', zorder=8):
        points = np.atleast_2d(points)
        ax.scatter(points[:, 0], points[:, 1], s=size, marker=marker, facecolors=face, edgecolors=color, linewidths=0.4, zorder=zorder)
    geometry = spec['geometry']
    coeff = np.asarray(geometry['coefficients'])
    radius = geometry['radius_m']
    theta = np.linspace(0, 2 * np.pi, 721)
    sigma = np.exp(1j * theta)
    centre = np.array([51.0, 66.5])
    physical_scale = 1.4
    parameter_centre = np.array([50.5, 43.5])
    parameter_scale = 3.1
    rho_interface = 2.0
    theta_mark = 1.5 * np.pi

    def mapped(angle, rho=1):
        return method.omega(rho * np.exp(1j * np.asarray(angle)), coeff, radius)

    def physical(z):
        return np.stack((-np.asarray(z).imag, -np.asarray(z).real), axis=-1) * physical_scale + centre

    def parameter(z):
        return np.stack((np.asarray(z).real, -np.asarray(z).imag), axis=-1) * parameter_scale + parameter_centre
    wall_z, exterior_z = (mapped(theta), mapped(theta, rho_interface))
    wall, interface = (physical(wall_z), physical(exterior_z))
    text(51.5, 4.5, 'Density base', size=8, color=BLUE)
    for c1, cxy in ((0.0, (43.5, 16)), (0.3, (59.5, 16))):
        z = sigma + c1 / sigma
        xy = np.c_[z.real, -z.imag] * 3.8 + cxy
        line(xy, INK, 0.65)
        for angle in np.arange(8) * np.pi / 4:
            w = np.exp(1j * angle) + c1 * np.exp(-1j * angle)
            p = np.array(cxy) + 3.8 * np.array([w.real, -w.imag])
            length = 1.8 * np.cos(3 * angle)
            if abs(length) > 0.2:
                arrow([p, p + [length, 0]], BLUE, 0.45)
    text(51.5, 25, '$S_{\\rm dens}^{(m)}=m\\Lambda$', size=8.1, color=BLUE)
    text(51.5, 30, 'Classical impedance $\\Lambda$', size=7, color=GREY)
    text(86, 4.5, 'Shape corrections', size=8, color=BLUE)
    ray_centre = np.array([86, 16])
    for t, col, lw, ls in ((0, '#9ba1a5', 0.55, '--'), (0.5, '#7f929f', 0.6, '-'), (1, BLUE, 0.9, '-')):
        z = method.omega(sigma, t * coeff, radius)
        line(np.c_[-z.imag, -z.real] * 1.08 + ray_centre, col, lw, ls=ls)
    text(75.8, 17, '$t=0$', size=7, color=GREY)
    text(96.3, 20, '$t=1$', size=7, color=BLUE)
    text(86, 27, '$S_0+tS_1+t^2S_2$', size=8.1, color=BLUE)
    line([(64.4, 25), (71.8, 25)], GREY, 0.45)
    line([(79.2, 22), (69, 32.5), (56.3, 39.7)], GREY, 0.45, ls=(0, (2, 2)))
    backed(66.7, 33.5, '$c_j$', size=7.4, color=GREY)
    for rho in (1, rho_interface):
        line(parameter(rho * sigma), BLUE if rho == 2 else INK, 0.65)
    for angle in np.arange(16) * np.pi / 8:
        line(parameter(np.array([1, 2]) * np.exp(1j * angle)), '#d1d4d7', 0.28)
    text(41.8, 43.5, '$\\zeta$', size=8)
    text(50.5, 42, '$1$', size=7)
    text(50.5, 35.7, '$\\rho$', size=7.2, color=BLUE)
    p_parameter = parameter(rho_interface * np.exp(1j * theta_mark))
    p_physical = physical(mapped(theta_mark, rho_interface))
    arrow([p_parameter, (60.5, 53), p_physical], GREY, 0.55)
    backed(61.5, 49.8, '$z=\\omega(\\zeta)$', size=7.6)
    dot([p_parameter, p_physical], marker='s', size=7, face=BLUE)
    text(55.7, 49, '$\\theta_0$', size=7, color=BLUE)
    ax.add_patch(Polygon(interface, fc='#fafafa', ec='none', zorder=0))
    radii = np.geomspace(1, rho_interface, 9)
    for rho in radii[1:-1]:
        line(physical(mapped(theta, rho)), '#c4c8ca', 0.22, zorder=1)
    for angle in np.arange(48) * 2 * np.pi / 48:
        line(physical(mapped(np.full_like(radii, angle), radii)), '#c4c8ca', 0.22, zorder=1)
    tangent = 1j * radius * (sigma - sum((j * c * sigma ** (-j) for j, c in enumerate(coeff))))
    intrados = wall_z + 0.4 * 1j * tangent / np.abs(tangent)
    ax.add_patch(Polygon(wall, fc='#aeb7bd', ec=INK, lw=0.6, zorder=3))
    ax.add_patch(Polygon(physical(intrados), fc='white', ec=INK, lw=0.4, zorder=4))
    line(interface, BLUE, 0.9, zorder=5)
    dot(physical(mapped(np.arange(24) * 2 * np.pi / 24, rho_interface)), size=1.8)
    backed(51, 56.2, 'FEM', size=7.5)
    text(51, 65.8, 'Lining', size=7.1)
    line([(51, 67.3), physical(mapped(np.pi))], INK, 0.4, zorder=5)
    backed(63.5, 55.5, '$\\Gamma$', size=8.3, color=BLUE)
    text(50.8, 85.4, 'Anisotropic exterior', size=7.1)
    arrow([(33.8, 66.5), (36.2, 66.5)], INK, 0.55)
    text(34.6, 63.3, '$p_h$', size=7)
    text(86, 35.3, 'Same $\\Gamma$, two branches', size=7.5)
    text(86, 39.4, '$z_k=x+\\mu_k y$', size=7.8)
    mus, *_ = method.material()
    interface_spec = {'geometry': {'coefficients': [float(c / rho_interface ** (j + 1)) for j, c in enumerate(coeff)]}}
    branch_facts = []
    for k, (mu, cy) in enumerate(zip(mus, (51, 72)), 1):
        series = method.branch_series(mu, interface_spec)
        lo, hi = (min(series), max(series))
        derivative = np.array([j * series.get(j, 0) for j in range(lo, hi + 1)])
        roots = np.roots(derivative[::-1])
        fold = float(max(abs(roots[abs(roots) < 1]), default=0.0))
        eta = method.cauchy_radius(series)
        image = method.eval_series(series, sigma)
        test = method.eval_series(series, eta * np.exp(2j * np.pi * (np.arange(12) + 0.31) / 12))
        trace = exterior_z / (rho_interface * radius)
        error = float(np.max(abs(image - (trace.real + mu * trace.imag))))
        assert error < 1e-12
        assert all(CurvePath(np.c_[image.real, image.imag]).contains_points(np.c_[test.real, test.imag]))

        def branch(z):
            return np.stack((np.asarray(z).real, -np.asarray(z).imag), axis=-1) * 7.5 + [86, cy]
        line(branch(image), INK, 0.7, zorder=4)
        dot(branch(test), color=GREY, size=3)
        marked = method.eval_series(series, np.exp(1j * theta_mark))
        dot(branch(marked), marker='s', size=7, face=BLUE)
        backed(98, cy - 6.5, f'$\\varphi_{k}$', size=8, color=BLUE)
        text(76, cy + 6.6, f'$\\Gamma_{k}$', size=7.4)
        branch_facts.append({'branch': k, 'mu': [float(mu.real), float(mu.imag)], 'fold_radius': fold, 'sampling_radius': eta, 'shown_interior_points': 12, 'affine_identity_error': error})
    line([(66.2, 62), (71.2, 62), (74.9, 51)], GREY, 0.5)
    line([(71.2, 62), (76.8, 72)], GREY, 0.5)
    text(86, 83, '$C_k[g_k]=a_k$', size=7.7, color=GREY)
    line([(96, 51), (101.8, 51), (101.8, 72)], BLUE, 0.65)
    arrow([(96, 72), (101.8, 72), (101.8, 91.2), (96, 94)], BLUE, 0.65)
    u_point = physical(mapped(1.25 * np.pi, rho_interface))
    arrow([u_point, (70.7, 82.2), (72.5, 92)], BLUE, 0.8)
    backed(70, 81.6, '$u_\\Gamma$', size=8, color=BLUE)
    text(85.5, 89, 'Spectral assembly', size=7.7, color=BLUE)
    phase = np.linspace(0, 2 * np.pi, 241)
    for m, cy in enumerate((94, 98, 102), 1):
        xx = 76 + 20 * phase / (2 * np.pi)
        line(np.c_[xx, np.full_like(xx, cy)], '#d4d8da', 0.3)
        line(np.c_[xx, cy - 1.15 * np.sin(m * phase)], '#9ba9b2', 0.45, ls=(0, (2, 1.8)))
        line(np.c_[xx, cy - 1.15 * np.cos(m * phase)], BLUE, 0.6)
        text(99.5, cy, f'${m}$', size=7, color=GREY)
    arrow([(86, 105.8), (69, 105.8)], BLUE, 0.8)
    backed(76, 109, '$S_{\\rm mod}$', size=8, color=BLUE)
    text(52.5, 92, 'P1 boundary work', size=7.5, color=BLUE)
    nodes = np.arange(41.5, 67, 5)
    for j in range(1, len(nodes) - 1):
        line([(nodes[j - 1], 101.5), (nodes[j], 96.3), (nodes[j + 1], 101.5)], INK if j == 2 else '#a8afb4', 0.65 if j == 2 else 0.4)
    line([(nodes[0], 101.5), (nodes[-1], 101.5)], GREY, 0.4)
    xx = np.linspace(nodes[0], nodes[-1], 201)
    line(np.c_[xx, 99 - 1.35 * np.cos(2 * np.pi * (xx - nodes[0]) / np.ptp(nodes))], BLUE, 0.7)
    text(51.5, 95, '$N_j$', size=7)
    text(54, 108, '$f_j=\\int_\\Gamma N_j\\,t\\,{\\rm d}s$', size=8)
    force_point = physical(mapped(2.4, rho_interface))
    arrow([(41.5, 99), (35.2, 94), (35.2, 77.5), force_point], BLUE, 0.8)
    backed(36.2, 82, '$f_\\Gamma$', size=8, color=BLUE)
    dot(physical(mapped(np.array([2.2, 2.4, 2.6]), rho_interface)), size=5, face=BLUE)
    text(68, 115, '$(K_h-S_h)u=f$', size=9, color=BLUE)
    text(68, 120.5, 'Reuse: support · contact · release', size=7.6, color=BLUE)
    text(68, 125, 'Fixed exterior material and $\\Gamma$', size=7, color=GREY)
    line([(32.5, 60), (36.5, 60), physical(mapped(2.15))], GREY, 0.5, ls=(0, (1.3, 2.4)), zorder=2)
    photo_width = 31.0
    for view in VIEWS:
        pixels = plt.imread(ROOT / 'figures/domain-coupling/assets' / view['asset'])
        x0, y0, x1, y1 = view['crop']
        x, y = view['position']
        photo_height = photo_width * (y1 - y0) / (x1 - x0)
        pa = fig.add_axes([x / width, (height - y - photo_height) / height, photo_width / width, photo_height / height])
        pa.imshow(pixels)
        pa.set(xlim=(x0, x1), ylim=(y1, y0))
        pa.set_axis_off()
        for cx, cy, ew, eh, angle in view['ellipses']:
            contour = Ellipse((cx, cy), ew, eh, angle=angle, fc='none', ec=YELLOW, lw=0.75, zorder=5)
            contour.set_path_effects([pe.Stroke(linewidth=1.4, foreground='#332c1c'), pe.Normal()])
            pa.add_patch(contour)
        name, target, label_fraction = view['callout']
        anchor = (x0 + label_fraction[0] * (x1 - x0), y0 + label_fraction[1] * (y1 - y0))
        pa.annotate(labels.get(name, name), xy=target, xytext=anchor, fontsize=7, color=YELLOW, ha='left', va='center', zorder=8, bbox={'fc': '#202020', 'alpha': 0.8, 'ec': 'none', 'pad': 0.7}, arrowprops={'arrowstyle': '-', 'lw': 0.6, 'color': YELLOW})
        text(x + photo_width / 2, y + photo_height + 2.8, view['site'], size=7)
    return (fig, {'photos': [v['id'] for v in VIEWS], 'photo_groups': len(VIEWS), 'photo_width_mm': photo_width, 'source_bound_views': VIEWS, 'pixel_edits': 'Source files unchanged; display crops and editable vector contours/callouts', 'layout': 'Field observations flank a mapped physical/characteristic-domain construction and a boundary-work coupling loop', 'physical_model_width_mm': float(np.ptp(interface[:, 0])), 'physical_model_scale_mm_per_m': physical_scale, 'geometry': 'Exact Laurent wall, parameter annulus and rho=2 artificial interface', 'illustrative_mapped_mesh': [8, 48], 'lining_normal_thickness_m': 0.4, 'characteristic_branches': branch_facts, 'branch_material': {'E_h_over_E_v': 3.0, 'phi_deg': 45.0}, 'branch_plot_scale_mm_per_normalized_unit': 7.5, 'corresponding_boundary_point_theta': theta_mark, 'shape_rays': [0, 0.5, 1], 'illustrative_spectral_modes': [1, 2, 3], 'model_scope': 'Independently specified elastic model; field support damage motivates deformation and lining assessment', 'classical_basis': 'Complex potentials, the two material branches and surface impedance', 'paper_constructions': 'Circle/ellipse density base, general-map analyticity, explicit shape corrections and consistent FEM assembly', 'coupling': 'The blue tunnel route uses spectral assembly and consistent P1 boundary work to return FEM nodal force; analytic constraints and shape corrections are distinct constructions', 'reuse': 'Exterior stiffness -S_h is reusable for fixed exterior material and interface'})
