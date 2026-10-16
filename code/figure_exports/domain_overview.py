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
VIEWS = [{'id': 11997, 'asset': 'moxi_severe_steel_ribs.jpg', 'position': [1.5, 5], 'crop': [0, 0, 842, 473], 'site': 'Moxi · steel ribs', 'ellipses': [[197, 225, 265, 228, 0], [672, 301, 239, 270, 24]], 'callout': ['Bent ribs', [608, 271], [0.55, 0.12]]}, {'id': 13334, 'asset': 'moxi_i18_replacement.jpg', 'position': [104.5, 5], 'crop': [0, 0, 475, 290], 'site': 'Moxi · I18 joints', 'ellipses': [[144, 147, 101, 171, 9], [351, 133, 83, 164, 3]], 'callout': ['Flange bending', [148, 133], [0.03, 0.9]]}, {'id': 14822, 'asset': 'moxi_distorted_rib_row.jpg', 'position': [1.5, 37], 'crop': [5, 140, 795, 630], 'site': 'Moxi · rib row', 'ellipses': [[325, 322, 238, 255, 0], [528, 330, 144, 261, 0]], 'callout': ['Bent segments', [385, 325], [0.03, 0.87]]}, {'id': 14819, 'asset': 'chaluo_bent_rib.jpg', 'position': [104.5, 37], 'crop': [35, 90, 445, 350], 'site': 'Chaluo · steel rib', 'ellipses': [[256, 211, 119, 213, -8]], 'callout': ['Rib bending', [278, 238], [0.03, 0.86]]}, {'id': 14800, 'asset': 'guodashan_crown_ribs.jpg', 'position': [1.5, 66], 'crop': [0, 45, 550, 382], 'site': 'Guodashan · crown', 'ellipses': [[307, 167, 293, 114, 33]], 'callout': ['Crown distortion', [369, 196], [0.03, 0.89]]}, {'id': 11600, 'asset': 'moxi_floor_heave.jpg', 'position': [104.5, 66], 'crop': [5, 55, 530, 380], 'site': 'Moxi · floor heave', 'ellipses': [[177, 250, 339, 112, -38]], 'callout': ['Heave and crack', [245, 200], [0.38, 0.12]]}]

def draw_domain(spec, width, height, labels):
    """One mapped section connects field motivation to exterior boundary work."""
    fig = fs.figure(width, height)
    plt.rcParams['image.composite_image'] = False
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(height, 0), aspect='equal')
    ax.set_axis_off()
    serial = 0

    def tag(artist, role):
        nonlocal serial
        serial += 1
        artist.set_gid(f'{role}-{serial:03d}')
        return artist

    def text(x, y, value, size=7.2, color=INK, **kw):
        return tag(ax.text(x, y, labels.get(value, value), fontsize=size, color=color, ha=kw.pop('ha', 'center'), va=kw.pop('va', 'center'), zorder=30, **kw), 'label')

    def line(points, color=GREY, lw=0.55, **kw):
        points = np.asarray(points)
        role = kw.pop('role', 'edge')
        artist, = ax.plot(points[:, 0], points[:, 1], color=color, lw=lw, ls=kw.pop('ls', '-'), **kw)
        return tag(artist, role)

    def poly(points, *, fc='none', ec=INK, lw=0.6, zorder=3, role='geometry'):
        patch = Polygon(points, fc=fc, ec=ec, lw=lw, zorder=zorder)
        ax.add_patch(patch)
        return tag(patch, role)

    def arrow(points, color=BLUE, lw=0.8):
        points = np.asarray(points)
        if len(points) > 2:
            line(points[:-1], color, lw, zorder=18, role='force-route')
        artist = ax.annotate('', xy=points[-1], xytext=points[-2], zorder=20, arrowprops={'arrowstyle': '-|>', 'lw': lw, 'color': color, 'shrinkA': 0, 'shrinkB': 0})
        return tag(artist, 'directed-route')

    def backed(x, y, value, **kw):
        return text(x, y, value, bbox={'fc': 'white', 'ec': 'none', 'pad': 0.35}, **kw)

    def dot(points, *, marker='o', color=BLUE, size=4, face='white', zorder=21):
        points = np.atleast_2d(points)
        return tag(ax.scatter(points[:, 0], points[:, 1], s=size, marker=marker, facecolors=face, edgecolors=color, linewidths=0.45, zorder=zorder), 'boundary-point')
    geometry = spec['geometry']
    coeff = np.asarray(geometry['coefficients'])
    radius = geometry['radius_m']
    theta = np.linspace(0, 2 * np.pi, 721)
    sigma = np.exp(1j * theta)
    centre = np.array([68.0, 55.0])
    physical_scale = 2.2
    rho_interface = 2.0
    theta_mark = 1.5 * np.pi

    def mapped(angle, rho=1):
        return method.omega(rho * np.exp(1j * np.asarray(angle)), coeff, radius)

    def physical(z):
        return np.stack((-np.asarray(z).imag, -np.asarray(z).real), axis=-1) * physical_scale + centre
    wall_z, exterior_z = (mapped(theta), mapped(theta, rho_interface))
    wall, interface = (physical(wall_z), physical(exterior_z))
    rock = plt.imread(ROOT / 'figures/domain-coupling/assets/moxi_layered_rock.jpg')
    rock_box = [37.5, 4.0, 25.5, 25.5 * rock.shape[0] / rock.shape[1]]
    x, y, w, h = rock_box
    tag(ax.imshow(rock, extent=(x, x + w, y + h, y), interpolation='none', zorder=10), 'photo-bedded-slate')
    poly([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], ec='#b7bec3', lw=0.4, zorder=11, role='photo-frame')
    text(50.25, 26.0, 'Bedded slate · Moxi', size=7)
    arrow([(50, 29), (50, 32), (42, 40)], color='#9a7957', lw=0.7)
    parameter_centre = np.array([83.5, 15.0])
    parameter_scale = 3.5

    def parameter(z):
        return np.stack((np.asarray(z).real, -np.asarray(z).imag), axis=-1) * parameter_scale + parameter_centre
    text(83.5, 3.5, 'Laurent geometry', size=8, color=BLUE)
    for r in (1, rho_interface):
        line(parameter(r * sigma), BLUE if r == rho_interface else INK, 0.7)
    for angle in np.arange(16) * np.pi / 8:
        line(parameter(np.array([1, 2]) * np.exp(1j * angle)), '#c8ced2', 0.3)
    text(83.5, 14.8, '$\\zeta$', size=8.4)
    text(91.7, 12, '$\\rho$', size=7.5, color=BLUE)
    p_parameter = parameter(rho_interface * np.exp(1j * theta_mark))
    p_physical = physical(mapped(theta_mark, rho_interface))
    arrow([p_parameter, (97, 25), (97, 44), p_physical], GREY, 0.65)
    backed(82, 26, '$z=\\omega(\\zeta)$', size=8)
    dot([p_parameter, p_physical], marker='s', size=11, face=BLUE)
    exterior_window = poly([(37, 35), (99, 35), (99, 80), (37, 80)], fc='#f8f5ef', ec='none', zorder=0, role='exterior-display-window')
    for intercept in np.arange(-25, 105, 4):
        band = line([(37, intercept), (99, intercept + 62)], '#d3bfa5', 0.8, zorder=1, role='schematic-bedding')
        band.set_clip_path(exterior_window)
    poly(interface, fc='#fafcfd', ec='none', zorder=2, role='finite-element-near-field')
    radii = np.geomspace(1, rho_interface, 13)
    for r in radii[1:-1]:
        line(physical(mapped(theta, r)), '#aebcc6', 0.24, zorder=3, role='mapped-radial-layer')
    for angle in np.arange(64) * 2 * np.pi / 64:
        line(physical(mapped(np.full_like(radii, angle), radii)), '#aebcc6', 0.24, zorder=3, role='mapped-circumferential-edge')
    tangent = 1j * radius * (sigma - sum((j * c * sigma ** (-j) for j, c in enumerate(coeff))))
    intrados = wall_z + 0.4j * tangent / np.abs(tangent)
    poly(wall, fc='#b8c1c8', ec=INK, lw=0.7, zorder=5, role='lining')
    poly(physical(intrados), fc='white', ec=INK, lw=0.45, zorder=6, role='excavation')
    line(interface, BLUE, 1.05, zorder=7, role='artificial-interface')
    dot(physical(mapped(np.arange(32) * 2 * np.pi / 32, rho_interface)), size=3)
    dot(p_physical, marker='s', size=11, face=BLUE)
    backed(68, 39.7, 'FEM near field', size=7.7, color=BLUE)
    text(68, 53.7, 'Excavation', size=7.4)
    text(68, 59.0, 'Lining', size=7.1, color=GREY)
    line([(75.5, 59), physical(mapped(1.2 * np.pi))], INK, 0.5, zorder=8)
    backed(89.5, 40.8, '$\\Gamma$', size=9, color=BLUE)
    backed(68, 83.6, 'Unbounded TI exterior', size=7.5, color='#806344')
    text(95, 78, '$\\infty$', size=11, color='#806344')
    arrow([(68, 28.7), (68, 32.4)], INK, 0.75)
    backed(72.5, 30, '$p_v$', size=7.5)
    arrow([(36, 56), (44.4, 56)], INK, 0.75)
    backed(39.5, 53, '$p_h$', size=7.5)
    origin = np.array([67.0, 68.8])
    for end, name in [(origin + [0, -5.5], '$x$'), (origin + [-5.5, 0], '$y$')]:
        arrow([origin, end], INK, 0.55)
        text(*end + ([1.7, 0] if name == '$x$' else [0, 2.1]), name, size=7)
    for angle in np.linspace(0.32 * np.pi, 0.63 * np.pi, 4):
        p = physical(mapped(angle, rho_interface))
        normal = (p - centre) / np.linalg.norm(p - centre)
        arrow([p + 3.0 * normal, p + 0.4 * normal], BLUE, 0.65)
    line([(32.5, 48), (35, 48), (46, 48)], GREY, 0.5, ls=(0, (1.4, 2.2)), zorder=4, role='field-motivation')
    line([(104.5, 79), (101.5, 79), (77, 65)], GREY, 0.5, ls=(0, (1.4, 2.2)), zorder=4, role='field-motivation')
    mus, *_ = method.material()
    interface_spec = {'geometry': {'coefficients': [float(c / rho_interface ** (j + 1)) for j, c in enumerate(coeff)]}}
    branch_facts = []
    text(33.5, 95.8, 'Two material planes', size=8, color=BLUE)
    for k, (mu, cx) in enumerate(zip(mus, (18.5, 49)), 1):
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
            return np.stack((np.asarray(z).real, -np.asarray(z).imag), axis=-1) * 8.4 + [cx, 108.5]
        boundary = branch(image)
        poly(boundary, fc='#f0f5fa', ec=INK, lw=0.7, role=f'characteristic-plane-{k}')
        dot(branch(test), color=GREY, size=4)
        marked = method.eval_series(series, np.exp(1j * theta_mark))
        dot(branch(marked), marker='s', size=9, face=BLUE)
        backed(cx + 10, 102, f'$\\Gamma_{k}$', size=7.7)
        text(cx, 120.2, f'$\\varphi_{k}(z_{k})$', size=8, color=BLUE)
        line([(cx, 116.8), (cx, 117.8)], BLUE, 0.7)
        line([(cx, 122.8), (cx, 125)], BLUE, 0.7)
        branch_facts.append({'branch': k, 'mu': [float(mu.real), float(mu.imag)], 'fold_radius': fold, 'sampling_radius': eta, 'shown_interior_points': 12, 'affine_identity_error': error})
    text(33.5, 130, '$z_k=x+\\mu_k y$', size=8)
    line([(18.5, 125), (59, 125), (65, 111)], BLUE, 0.8, role='parallel-branch-readout')
    arrow([(65, 111), (69, 111)], BLUE, 0.8)
    input_point = physical(mapped(2.22, rho_interface))
    arrow([input_point, (35.5, 79.5), (35.5, 91.5), (33.5, 91.5)], BLUE, 0.85)
    backed(38.4, 85.2, '$u_\\Gamma$', size=8, color=BLUE)
    line([(18.5, 91.5), (49, 91.5)], BLUE, 0.7)
    line([(18.5, 91.5), (18.5, 98.8)], BLUE, 0.7)
    arrow([(18.5, 98.8), (18.5, 101)], BLUE, 0.7)
    line([(49, 91.5), (61.2, 91.5), (61.2, 105), (58.5, 105)], BLUE, 0.7)
    arrow([(58.5, 105), (56, 105)], BLUE, 0.7)
    text(82, 96, 'Modal response', size=8, color=BLUE)
    phase = np.linspace(0, 2 * np.pi, 241)
    for m, cy in enumerate((104.5, 109.2, 113.9), 1):
        xx = 71 + 22 * phase / (2 * np.pi)
        line(np.c_[xx, np.full_like(xx, cy)], '#c8d1d8', 0.3)
        line(np.c_[xx, cy - 1.25 * np.sin(m * phase)], '#9aaab5', 0.5, ls=(0, (2, 1.5)))
        line(np.c_[xx, cy - 1.25 * np.cos(m * phase)], BLUE, 0.65, role='harmonic-trace')
    text(82, 120.2, '$S_{\\rm mod}$', size=8.5, color=BLUE)
    arrow([(94.2, 109.2), (101, 109.2)], BLUE, 0.9)
    text(118, 96, 'Nodal boundary work', size=7.8, color=BLUE)
    nodes = np.arange(103, 134, 6)
    for j in range(1, len(nodes) - 1):
        line([(nodes[j - 1], 113), (nodes[j], 104), (nodes[j + 1], 113)], INK if j == 2 else '#bac2c7', 0.8 if j == 2 else 0.5, role='P1-trace-basis')
    xx = np.linspace(nodes[0], nodes[-1], 201)
    yy = 109 - 2.0 * np.cos(2 * np.pi * (xx - nodes[0]) / np.ptp(nodes))
    line(np.c_[xx, yy], BLUE, 0.8, role='traction-density')
    text(115, 101.5, '$N_j$', size=7.5)
    text(118, 120.2, '$f_j=\\int_\\Gamma N_j\\,t\\,{\\rm d}s$', size=8)
    force_point = physical(mapped(1.3 * np.pi, rho_interface))
    arrow([(132, 104), (136, 99), (136, 92), (101.4, 92), (101.4, 64), force_point], BLUE, 0.85)
    backed(98, 62, '$f_\\Gamma$', size=8, color=BLUE)
    text(103, 130, '$(K_h-S_h)u=f$', size=9, color=BLUE)
    photo_width = 31.0
    for view in VIEWS:
        pixels = plt.imread(ROOT / 'figures/domain-coupling/assets' / view['asset'])
        x0, y0, x1, y1 = view['crop']
        x, y = view['position']
        photo_height = photo_width * (y1 - y0) / (x1 - x0)
        pa = fig.add_axes([x / width, (height - y - photo_height) / height, photo_width / width, photo_height / height])
        tag(pa.imshow(pixels), 'photo-' + view['asset'].removesuffix('.jpg'))
        pa.set(xlim=(x0, x1), ylim=(y1, y0))
        pa.set_axis_off()
        for cx, cy, ew, eh, angle in view['ellipses']:
            contour = Ellipse((cx, cy), ew, eh, angle=angle, fc='none', ec=YELLOW, lw=0.75, zorder=5)
            contour.set_path_effects([pe.Stroke(linewidth=1.4, foreground='#332c1c'), pe.Normal()])
            pa.add_patch(contour)
            tag(contour, 'observed-damage-outline')
        name, target, label_fraction = view['callout']
        anchor = (x0 + label_fraction[0] * (x1 - x0), y0 + label_fraction[1] * (y1 - y0))
        tag(pa.annotate(labels.get(name, name), xy=target, xytext=anchor, fontsize=7, color=YELLOW, ha='left', va='center', zorder=8, bbox={'fc': '#202020', 'alpha': 0.8, 'ec': 'none', 'pad': 0.7}, arrowprops={'arrowstyle': '-', 'lw': 0.6, 'color': YELLOW}), 'observed-damage-callout')
        backed(x + photo_width / 2, y + photo_height + 2.8, view['site'], size=7)
    return (fig, {'photos': [13683, *[v['id'] for v in VIEWS]], 'photo_groups': 7, 'photo_width_mm': photo_width, 'source_bound_views': VIEWS, 'rock_context': {'figure_id': 13683, 'asset': 'moxi_layered_rock.jpg', 'display_box_mm': rock_box, 'crop': None}, 'pixel_edits': 'Original files unchanged; display crops and editable overlays', 'layout': 'Enlarged shared interface with field context, exact parameter mapping, parallel material planes and nodal-force return', 'physical_model_width_mm': float(np.ptp(interface[:, 0])), 'physical_model_scale_mm_per_m': physical_scale, 'geometry': 'Exact Laurent wall and rho=2 interface in a plane-strain section', 'illustrative_mapped_mesh': [12, 64], 'lining_normal_thickness_m': 0.4, 'characteristic_branches': branch_facts, 'branch_material': {'E_h_over_E_v': 3.0, 'phi_deg': 45.0}, 'branch_plot_scale_mm_per_normalized_unit': 8.4, 'corresponding_boundary_point_theta': theta_mark, 'illustrative_spectral_modes': [1, 2, 3], 'model_scope': 'Field photographs motivate independently specified homogeneous TI ground and support calculations', 'coupling': 'Parallel complex-potential branches supply modal response, paired with P1 traces to return nodal forces to the same FEM interface', 'reference_organization': 'baxi-cohesive F21 dominant physical object and semantic leaders; F12 source-pixel annotations and linked local objects'})
