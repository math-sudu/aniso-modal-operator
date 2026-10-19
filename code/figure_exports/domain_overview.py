from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as CurvePath
import matplotlib.patheffects as pe
import numpy as np
import origin_style as fs
import manuscript_figures as method
ROOT = Path(__file__).resolve().parents[2]
INK, BLUE, GREY = ('#353535', '#397e94', '#7d858a')
GREEN, PURPLE, ORANGE = ('#85b94b', '#b8a2cc', '#f2a132')
YELLOW, CREAM, CYAN, RED = ('#fff033', '#fbf4df', '#97cde1', '#c2766b')
VIEWS = [{'id': 13683, 'asset': 'moxi_layered_rock.jpg', 'x': 1.5, 'crop': None, 'site': 'Bedded slate · Moxi', 'ellipses': [], 'callout': None}, {'id': 14819, 'asset': 'chaluo_bent_rib.jpg', 'x': 36.0, 'crop': [35, 90, 445, 350], 'site': 'Bent rib · Chaluo', 'ellipses': [[256, 211, 119, 213, -8]], 'callout': ['Rib bending', [278, 238], [0.03, 0.86]]}, {'id': 14800, 'asset': 'guodashan_crown_ribs.jpg', 'x': 70.5, 'crop': [0, 45, 550, 382], 'site': 'Crown · Guodashan', 'ellipses': [[307, 167, 293, 114, 33]], 'callout': ['Distortion', [369, 196], [0.03, 0.89]]}, {'id': 11600, 'asset': 'moxi_floor_heave.jpg', 'x': 105.0, 'crop': [5, 55, 530, 380], 'site': 'Floor heave · Moxi', 'ellipses': [[177, 250, 339, 112, -38]], 'callout': ['Heave and crack', [245, 200], [0.27, 0.12]]}]

def rounded_path(points, radius=0.9):
    vertices, codes = ([points[0]], [CurvePath.MOVETO])
    for before, corner, after in zip(points, points[1:], points[2:]):
        incoming, outgoing = (np.asarray(corner) - before, np.asarray(after) - corner)
        a, b = (np.linalg.norm(incoming), np.linalg.norm(outgoing))
        r = min(radius, a / 2, b / 2)
        vertices.extend((np.asarray(corner) - r * incoming / a, corner, np.asarray(corner) + r * outgoing / b))
        codes.extend((CurvePath.LINETO, CurvePath.CURVE3, CurvePath.CURVE3))
    vertices.append(points[-1])
    codes.append(CurvePath.LINETO)
    return CurvePath(vertices, codes)

def draw_domain(spec, width, height, labels):
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

    def text(x, y, value, size=7.5, color=INK, **kw):
        return tag(ax.text(x, y, labels.get(value, value), fontsize=size, color=color, ha=kw.pop('ha', 'center'), va='center', zorder=30, linespacing=1.15, **kw), 'label')

    def backed(x, y, value, **kw):
        return text(x, y, value, bbox={'fc': 'white', 'ec': 'none', 'pad': 0.3}, **kw)

    def line(points, color=GREY, lw=0.55, role='edge', **kw):
        points = np.asarray(points)
        artist, = ax.plot(points[:, 0], points[:, 1], color=color, lw=lw, ls=kw.pop('ls', '-'), **kw)
        return tag(artist, role)

    def arrow(points, color=INK, lw=0.75, role='route', dashed=False, tip=True, head=5, zorder=3):
        artist = FancyArrowPatch(path=rounded_path(points), arrowstyle='-|>' if tip else '-', mutation_scale=head, linewidth=lw, color=color, linestyle=(0, (3, 2)) if dashed else '-', zorder=zorder)
        ax.add_patch(artist)
        return tag(artist, role)

    def poly(points, fc='none', ec=INK, lw=0.6, zorder=3, role='geometry'):
        artist = Polygon(points, fc=fc, ec=ec, lw=lw, zorder=zorder)
        ax.add_patch(artist)
        return tag(artist, role)

    def dot(points, marker='o', color=BLUE, size=4, face='white', zorder=21):
        points = np.atleast_2d(points)
        return tag(ax.scatter(points[:, 0], points[:, 1], s=size, marker=marker, facecolors=face, edgecolors=color, linewidths=0.45, zorder=zorder), 'boundary-point')

    def grid(x, y, w, h, columns, rows, color, name, zorder=10):
        p = Rectangle((x, y), w, h, fc=color, ec='#777174', lw=0.4, zorder=zorder)
        ax.add_patch(p)
        tag(p, name)
        for k in range(1, columns):
            line([(x + k * w / columns, y), (x + k * w / columns, y + h)], '#777174', 0.25, role=name + '-column', zorder=zorder + 0.2)
        for k in range(1, rows):
            line([(x, y + k * h / rows), (x + w, y + k * h / rows)], '#777174', 0.25, role=name + '-row', zorder=zorder + 0.2)

    def node(x, y, color, role, radius=2.6):
        p = Circle((x, y), radius, fc=color, ec='none', zorder=12)
        ax.add_patch(p)
        tag(p, role)

    def brace(x, y0, y1):
        mid = (y0 + y1) / 2
        vertices = [(x, y0), (x + 1.5, y0), (x + 1.5, y0 + 1.5), (x + 1.5, mid - 1.5), (x + 1.5, mid), (x + 3, mid), (x + 1.5, mid), (x + 1.5, mid + 1.5), (x + 1.5, y1 - 1.5), (x + 1.5, y1), (x, y1)]
        codes = [CurvePath.MOVETO, CurvePath.CURVE3, CurvePath.CURVE3, CurvePath.LINETO, CurvePath.CURVE3, CurvePath.CURVE3, CurvePath.CURVE3, CurvePath.CURVE3, CurvePath.LINETO, CurvePath.CURVE3, CurvePath.CURVE3]
        p = PathPatch(CurvePath(vertices, codes), fill=False, ec=INK, lw=0.7, zorder=3)
        ax.add_patch(p)
        tag(p, 'displacement-output-brace')
    photo_width = 30.5
    for view in VIEWS:
        pixels = plt.imread(ROOT / 'figures/domain-coupling/assets' / view['asset'])
        x0, y0, x1, y1 = view['crop'] or [0, 0, pixels.shape[1], pixels.shape[0]]
        x, y = (view['x'], 2.0)
        photo_height = photo_width * (y1 - y0) / (x1 - x0)
        pa = fig.add_axes([x / width, (height - y - photo_height) / height, photo_width / width, photo_height / height])
        tag(pa.imshow(pixels), 'photo-' + view['asset'].removesuffix('.jpg'))
        pa.set(xlim=(x0, x1), ylim=(y1, y0))
        pa.set_axis_off()
        for cx, cy, ew, eh, angle in view['ellipses']:
            p = Ellipse((cx, cy), ew, eh, angle=angle, fc='none', ec=YELLOW, lw=0.75, zorder=5)
            p.set_path_effects([pe.Stroke(linewidth=1.4, foreground='#332c1c'), pe.Normal()])
            pa.add_patch(p)
            tag(p, 'observed-damage-outline')
        if view['callout']:
            name, target, fraction = view['callout']
            anchor = (x0 + fraction[0] * (x1 - x0), y0 + fraction[1] * (y1 - y0))
            tag(pa.annotate(name, xy=target, xytext=anchor, fontsize=7, color=YELLOW, ha='left', va='center', zorder=8, bbox={'fc': '#202020', 'alpha': 0.8, 'ec': 'none', 'pad': 0.7}, arrowprops={'arrowstyle': '-', 'lw': 0.6, 'color': YELLOW}), 'observed-damage-callout')
        text(x + photo_width / 2, 27.5, view['site'], size=7)
    geometry = spec['geometry']
    coeff = np.asarray(geometry['coefficients'])
    radius = geometry['radius_m']
    theta = np.linspace(0, 2 * np.pi, 721)
    sigma = np.exp(1j * theta)
    centre = np.array([24.0, 53.0])
    physical_scale = 1.7
    rho_interface = 2.0
    theta_mark = 1.5 * np.pi

    def mapped(angle, rho=1):
        return method.omega(rho * np.exp(1j * np.asarray(angle)), coeff, radius)

    def physical(z):
        return np.stack((-np.asarray(z).imag, -np.asarray(z).real), axis=-1) * physical_scale + centre

    def stack(x, y, w, h, colors, name, cols=6, rows=6, offset=(1.1, -1.1)):
        for i, color in enumerate(colors):
            grid(x + i * offset[0], y + i * offset[1], w, h, cols, rows, color, f'{name}-{i}', zorder=10 + i * 0.4)
    wall_z, exterior_z = (mapped(theta), mapped(theta, rho_interface))
    wall, interface = (physical(wall_z), physical(exterior_z))
    window = poly([(2, 34), (45, 34), (45, 72), (2, 72)], fc='#faf6ed', ec='none', zorder=0)
    for intercept in np.arange(-15, 90, 3.5):
        band = line([(2, intercept), (45, intercept + 43)], '#d1bb9a', 0.7, zorder=1)
        band.set_clip_path(window)
    poly(interface, fc='#fbfcfd', ec='none', zorder=2, role='near-field')
    radii = np.geomspace(1, rho_interface, 13)
    for r in radii[1:-1]:
        line(physical(mapped(theta, r)), '#b9c7ce', 0.22, zorder=3, role='mapped-coordinate-net')
    for angle in np.arange(64) * 2 * np.pi / 64:
        line(physical(mapped(np.full_like(radii, angle), radii)), '#b9c7ce', 0.22, zorder=3)
    tangent = 1j * radius * (sigma - sum((j * c * sigma ** (-j) for j, c in enumerate(coeff))))
    intrados = wall_z + 0.4j * tangent / np.abs(tangent)
    poly(wall, fc='#c0c7cc', ec=INK, lw=0.65, zorder=5, role='lining')
    poly(physical(intrados), fc='white', ec=INK, lw=0.45, zorder=6, role='excavation')
    line(interface, INK, 0.9, zorder=7, role='shared-artificial-interface')
    dot(physical(mapped(np.arange(32) * 2 * np.pi / 32, rho_interface)), size=2.6)
    p_physical = physical(mapped(theta_mark, rho_interface))
    dot(p_physical, marker='s', size=10, face=BLUE)
    backed(24, 39.5, 'PINN ground', size=7.5)
    text(24, 51, 'Excavation', size=7)
    text(24, 57, 'FEM lining', size=7, color=GREY)
    line([(31, 57), physical(mapped(1.2 * np.pi))], INK, 0.5, zorder=8)
    backed(40, 40, '$\\Gamma$', size=8.5)
    text(24, 75, 'Unbounded TI exterior', size=7, color='#806344')
    text(42, 69, '$\\infty$', size=10, color='#806344')
    arrow([(24, 30.5), (24, 35.5)], role='vertical-pressure')
    text(28, 32, '$p_v$', size=7.5)
    arrow([(1, 53), (7, 53)], role='horizontal-pressure')
    backed(3, 49.5, '$p_h$', size=7.5)
    origin = np.array([6.5, 69.0])
    for end, name, offset in [(origin + [0, -3.8], '$x$', [1.5, 0]), (origin + [-3.8, 0], '$y$', [0, 1.8])]:
        arrow([origin, end], lw=0.5)
        text(*end + offset, name, size=7)
    line([(16, 29.5), (16, 31), (11, 36)], '#a68b6c', 0.55, ls=(0, (1.4, 2.2)), role='rock-context')
    mus, *_ = method.material()
    interface_spec = {'geometry': {'coefficients': [float(c / rho_interface ** (j + 1)) for j, c in enumerate(coeff)]}}
    branch_facts = []
    text(74, 32, 'Characteristic planes', size=7.5)
    for k, (mu, cx) in enumerate(zip(mus, (62.0, 85.0)), 1):
        series = method.branch_series(mu, interface_spec)
        eta = method.cauchy_radius(series, minimum=0.97)
        image = method.eval_series(series, sigma)
        test = method.eval_series(series, eta * np.exp(2j * np.pi * (np.arange(12) + 0.31) / 12))
        trace = exterior_z / (rho_interface * radius)
        error = float(np.max(abs(image - (trace.real + mu * trace.imag))))
        assert error < 1e-12
        assert all(CurvePath(np.c_[image.real, image.imag]).contains_points(np.c_[test.real, test.imag]))

        def branch(z):
            return np.stack((np.asarray(z).real, -np.asarray(z).imag), axis=-1) * 4.9 + [cx, 43.5]
        poly(branch(image), fc='#eef5f6', ec=INK, lw=0.6, role=f'characteristic-plane-{k}')
        dot(branch(test), color=GREY, size=2.6)
        marked = method.eval_series(series, np.exp(1j * theta_mark))
        dot(branch(marked), marker='s', size=8, face=BLUE)
        backed(cx + 6.5, 38.5, f'$\\Gamma_{k}$', size=7.5)
        text(cx, 53, f'$\\varphi_{k}(z_k)$', size=7.5)
        arrow([(cx, 54.5), (cx, 56), (98, 56), (98, 43), (103, 43)], lw=0.6, role=f'material-branch-{k}-response')
        branch_facts.append({'branch': k, 'mu': [float(mu.real), float(mu.imag)], 'sampling_radius': eta, 'shown_interior_points': 12, 'affine_identity_error': error})
    arrow([p_physical, (49, 53), (49, 35.5), (62, 35.5), (62, 38)], role='interface-construction')
    arrow([(49, 35.5), (85, 35.5), (85, 38)], lw=0.6, role='second-material-interface')
    backed(49, 46.5, '$\\Gamma$', size=7.5)
    phase = np.linspace(0, 2 * np.pi, 241)
    for m, cy in enumerate((39, 43, 47), 1):
        xx = 104 + 24 * phase / (2 * np.pi)
        line(np.c_[xx, np.full_like(xx, cy)], '#d1d8dc', 0.3)
        line(np.c_[xx, cy - 1.05 * np.sin(m * phase)], '#99acb3', 0.45, ls=(0, (2, 1.5)))
        line(np.c_[xx, cy - 1.05 * np.cos(m * phase)], INK, 0.65, role='modal-trace')
    text(116, 32, 'Modal response', size=7.3)
    arrow([(129, 43), (133, 43), (133, 60), (130, 60)], role='modal-to-work', lw=0.6)
    nodes = np.linspace(106, 129, 5)
    for j in range(1, 4):
        line([(nodes[j - 1], 65), (nodes[j], 58), (nodes[j + 1], 65)], INK if j == 2 else '#b8c2c8', 0.7 if j == 2 else 0.45, role='P1-trace')
    xx = np.linspace(106, 129, 201)
    line(np.c_[xx, 61.5 - 1.4 * np.cos(2 * np.pi * (xx - 106) / 23)], BLUE, 0.65, role='traction')
    text(117, 70, 'P1 boundary work', size=7)
    arrow([(104, 61.5), (74, 61.5), (74, 65.5), (71.5, 65.5)], role='construct-sh-from-boundary-work', lw=0.65)
    outer_point = physical(mapped(1.37 * np.pi, rho_interface))
    arrow([outer_point, (48, 61), (48, 65.5), (51, 65.5)], role='section-to-exterior-trace', lw=0.65)
    stack(52, 61, 1.8, 9, [CYAN, PURPLE], 'exterior-trace', cols=1, rows=6, offset=(0.55, -0.55))
    text(53, 57.5, '$g_\\Gamma$', size=7.5)
    arrow([(55.5, 65.5), (59.5, 65.5)], role='exterior-trace-input', lw=0.6)
    grid(61, 61, 9, 9, 5, 5, CYAN, 'shared-exterior-matrix')
    text(65.5, 65.5, '$S_h$', size=9)
    text(65.5, 73.5, 'Fixed exterior', size=7)
    arrow([(71.5, 67), (85.5, 67)], role='exterior-projected-work', lw=0.65)
    text(79, 64, '$Q_\\Gamma^{\\mathsf{T}}$', size=7)
    grid(87, 62, 2.3, 10, 1, 6, CYAN, 'exterior-residual')
    text(93, 67, '$R_\\Gamma$', size=7.5, ha='left')
    lining_point = physical(mapped(0.95 * np.pi))
    arrow([lining_point, (34, 70), (49, 70), (49, 82.5), (51, 82.5)], role='section-to-lining-trace', lw=0.6)
    stack(52, 78, 1.8, 9, [CYAN, PURPLE], 'lining-relative-trace', cols=1, rows=6, offset=(0.55, -0.55))
    text(53, 74.5, '$\\Delta g_\\ell$', size=7)
    arrow([(55.5, 82.5), (59.5, 82.5)], role='lining-trace-input', lw=0.6)
    grid(61, 78, 9, 9, 5, 5, '#c8d397', 'fixed-lining-work')
    text(65.5, 82.5, '$K_\\ell$', size=9)
    text(78, 75.5, '$a\\geq0.82$', size=7)
    arrow([(71.5, 82.5), (85.5, 82.5)], role='lining-projected-work', lw=0.65)
    text(79, 79.5, '$Q_\\ell^{\\mathsf{T}}$', size=7)
    grid(87, 78, 2.3, 10, 1, 6, '#c8d397', 'lining-residual')
    text(93, 83, '$R_\\ell$', size=7.5, ha='left')
    arrow([(90.5, 69), (101, 69), (101, 87), (134, 87), (134, 160), (81, 160), (81, 145), (74, 145)], role='exterior-residual-collection', lw=0.7)
    arrow([(90.5, 85), (101, 85)], role='lining-residual-collection', lw=0.7, tip=False)
    text(117, 79, 'Boundary work', size=7.3)
    input_nodes = [(8, y) for y in (98, 108)]
    feature_ys = (92, 99, 106, 117)
    feature_layers = [[(x, y) for y in feature_ys] for x in (22, 35, 48)]
    output_nodes = [(62, y) for y in (98, 108)]
    layers = [input_nodes, *feature_layers, output_nodes]
    for layer, (left, right) in enumerate(zip(layers, layers[1:])):
        pairs = ((i, j) for i in range(len(left)) for j in range(len(right)) if layer != 1 or i == j)
        for i, j in pairs:
            start, end = (np.asarray(left[i]), np.asarray(right[j]))
            unit = (end - start) / np.linalg.norm(end - start)
            arrow([start + 2.35 * unit, end - 2.35 * unit], '#999999', lw=0.3, head=3, zorder=2, role=f'feature-stage-{layer}-{i}-{j}')
    for i, (x, y) in enumerate(input_nodes):
        node(x, y, YELLOW, f'coordinate-{i}', radius=2.4)
        text(x, y, ['$s$', '$\\theta$'][i], size=8.5)
    for k, layer in enumerate(feature_layers):
        for i, (x, y) in enumerate(layer):
            node(x, y, GREEN, f'fixed-stage-{k}-{i}', radius=2.35)
        text(layer[0][0], 111.5, '$\\vdots$', size=11, color=GREEN)
    for i, (x, y) in enumerate(output_nodes):
        node(x, y, YELLOW, f'displacement-{i}', radius=2.4)
        text(x, y, ['$u_x$', '$u_y$'][i], size=8)
    for x, label in [(22, '$d_j$'), (35, '$G_j$'), (48, '$H_j$')]:
        text(x, 87, label, size=8.5)
    backed(56, 94, '$W$', size=8.5)
    text(8, 116, '$s=\\log\\rho$', size=7)
    text(22, 123, 'Distances', size=7)
    text(35, 123, 'Gaussians', size=7)
    text(48, 123, 'Normalize', size=7)
    line([(18.5, 126), (18.5, 127.5), (51.5, 127.5), (51.5, 126)], INK, 0.6)
    text(35, 131, 'Fixed feature transform', size=7)
    sample_point = physical(mapped(0.35 * np.pi, 1.55))
    arrow([sample_point, (5, 65), (5, 82), (8, 82), (8, 94)], role='physical-section-to-neural-coordinates', lw=0.7)
    backed(19, 81, '$z=\\omega(e^{s+i\\theta})$', size=8)
    brace(66, 98, 108)
    arrow([(69, 103), (74.5, 103)], role='field-sampling')
    stack(76, 98, 10, 10, [CYAN, PURPLE], 'displacement-fields', offset=(1.5, -1.5))
    text(81.5, 114, '$u_W(s,\\theta)$', size=8)
    field_point = physical(mapped(1.18 * np.pi, 1.45))
    arrow([(72, 103), (72, 89), (40, 89), (40, 74), (40, 71), field_point], color=BLUE, lw=0.8, role='neural-field-on-shared-section', zorder=8)
    backed(72, 93, '$u_W$', size=7.5, color=BLUE)
    arrow([(89, 103), (94, 103)], role='analytic-derivatives', lw=0.65)
    stack(96, 98, 10, 10, [YELLOW, PURPLE, '#a9a9ad', ORANGE], 'parametric-derivatives')
    text(102, 114, '$\\partial_s u,\\ \\partial_\\theta u$', size=7.5)
    text(96, 91.5, 'Analytic', size=7)
    arrow([(111, 103), (117, 103)], role='coordinate-transform', lw=0.65)
    grid(119, 98, 10, 10, 2, 2, CREAM, 'geometry-jacobian')
    text(124, 103, '$J_\\omega^{-1}$', size=9)
    text(123.5, 114, 'Map metric', size=7)
    arrow([(124, 109.5), (124, 119)], role='physical-strain', lw=0.7)
    stack(117, 123, 10, 10, [CREAM, CYAN, PURPLE, ORANGE], 'Mandel-strain')
    text(122, 137, '$\\varepsilon(u_W)$', size=8)
    text(113, 119, '$\\varepsilon_{zz}=0$', size=7)
    arrow([(116, 129), (103, 129), (103, 136)], role='implicit-return', lw=0.7)
    arrow([(103, 129), (114.5, 129), (114.5, 143), (115, 143)], role='updated-plastic-state', lw=0.6)
    backed(100, 126, 'Implicit DP return', size=7)
    stack(97, 139, 9, 10, [RED, YELLOW, '#9b9b97', ORANGE], 'Mandel-stress')
    stack(116, 142, 8, 9, [CREAM, CYAN, '#c8d397', PURPLE], 'plastic-history')
    text(102, 153, '$\\sigma-\\sigma^0$', size=8)
    text(121, 154, '$\\varepsilon^p$', size=9)
    backed(112, 134, '$\\varepsilon^p_{\\rm old}$', size=7.5)
    arrow([(115, 147), (112, 147), (112, 131), (104, 131)], color='#c9826d', role='accepted-history-commit', lw=0.6, dashed=True)
    grid(89, 139, 2.3, 11, 1, 7, ORANGE, 'internal-force-residual')
    arrow([(96, 144), (92.5, 144)], role='internal-work', lw=0.6)
    text(89, 154, '$R_{\\rm int}$', size=7.5)
    arrow([(87.5, 145), (74, 145)], role='internal-residual-collection', lw=0.7)
    node(69, 145, YELLOW, 'weak-equilibrium', radius=3.6)
    text(69, 145, '$R(W)$', size=8.5)
    text(65, 153, 'Weak equilibrium', size=7)
    text(44, 157, '2 translation constraints', size=7)
    arrow([(58, 145), (64.5, 145)], role='release-load', lw=0.65)
    text(54, 145, '$a f_0$', size=8)
    arrow([(69, 140.5), (69, 128), (58, 128), (58, 114)], role='newton-weight-update', lw=0.8)
    text(71, 133, 'Newton /', size=7, ha='left')
    text(71, 137, 'line search', size=7, ha='left')
    grid(17, 140, 7.5, 7.5, 4, 4, '#d6b186', 'consistent-tangent')
    text(20.75, 143.75, '$C_{\\rm alg}$', size=7.5)
    text(22, 152, 'Consistent tangent', size=7)
    arrow([(26, 144), (42, 144), (42, 136), (66.5, 136)], role='tangent-to-newton', lw=0.6)
    text(113, 157, 'Accepted history', size=7, color='#a66851')
    return (fig, {'photos': [v['id'] for v in VIEWS], 'photo_groups': len(VIEWS), 'photo_width_mm': photo_width, 'source_bound_views': VIEWS, 'physical_model_width_mm': float(np.ptp(interface[:, 0])), 'physical_model_scale_mm_per_m': physical_scale, 'geometry': 'Exact Laurent wall and rho=2 interface in plane strain', 'illustrative_mapped_coordinate_net': [12, 64], 'lining_normal_thickness_m': 0.4, 'characteristic_branches': branch_facts, 'branch_material': {'E_h_over_E_v': 3.0, 'phi_deg': 45.0}, 'branch_plot_scale_mm_per_normalized_unit': 4.9, 'corresponding_boundary_point_theta': theta_mark, 'illustrative_spectral_modes': [1, 2, 3], 'neural_architecture': 'Coordinates -> scaled distances d_j -> Gaussian responses G_j -> normalized features H_j -> two displacement outputs; only output weights W are trainable', 'neural_stage_meaning': 'Three green columns expose operations within the implemented single fixed Gaussian feature layer, not extra trainable hidden layers', 'neural_constraints': 'Weak equilibrium with two translations removed; fixed DtN and FEM lining activated unstressed at release 0.82; history committed after stage equilibrium', 'model_scope': 'Field photographs motivate independently prescribed homogeneous TI ground and support calculations', 'array_role': 'Schematic sampled field planes and residual vectors; grid cells do not encode measured values', 'derivative_channels': ['d_s u_x', 'd_s u_y', 'd_theta u_x', 'd_theta u_y'], 'constitutive_channels': ['xx', 'yy', 'zz', 'sqrt(2) xy']})
