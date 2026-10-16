from __future__ import annotations
import argparse
import csv
import importlib
from pathlib import Path
from types import SimpleNamespace
import fitz
import gmsh
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from matplotlib.tri import Triangulation, LinearTriInterpolator
from threadpoolctl import threadpool_limits
HERE = Path(__file__).resolve().parents[2] / 'results/data/derived'
ROOT = HERE.parents[2]
base = importlib.import_module('orthotropic_plate')

def write_rows(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def material(e1, e2, g, nu, angle=90.0):
    """Rotate engineering compliance; angle is material 1 from global x."""
    local = np.array([[1 / e1, -nu / e1, 0.0], [-nu / e1, 1 / e2, 0.0], [0.0, 0.0, 1 / g]])
    c, s = (np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)))
    q = np.array([[c, -s], [s, c]])
    cols = []
    for stress in (np.diag([1.0, 0.0]), np.diag([0.0, 1.0]), np.array([[0.0, 1.0], [1.0, 0.0]])):
        sl = q.T @ stress @ q
        el = local @ np.array([sl[0, 0], sl[1, 1], sl[0, 1]])
        eg = q @ np.array([[el[0], el[2] / 2], [el[2] / 2, el[1]]]) @ q.T
        cols.append([eg[0, 0], eg[1, 1], 2 * eg[0, 1]])
    compliance = np.array(cols).T
    assert np.allclose(compliance, compliance.T, atol=1e-15)
    assert np.linalg.eigvalsh(compliance).min() > 0
    beta = dict(zip(('11', '22', '12', '66', '16', '26'), (compliance[0, 0], compliance[1, 1], compliance[0, 1], compliance[2, 2], compliance[0, 2], compliance[1, 2])))
    isotropic = np.isclose(e1, e2) and np.isclose(g, e1 / (2 * (1 + nu)))
    roots = None if isotropic else base.mu_roots(beta)
    return SimpleNamespace(compliance=compliance, stiffness=np.linalg.inv(compliance), beta=beta, mus=roots, pq=[base.pq_coeffs(beta, m) for m in roots] if roots else None)

def mesh(radius, width, length, kind, refinement, outer_scale=3.5):
    count = 256 * refinement
    theta = np.arange(count) * 2 * np.pi / count
    wall = radius * np.c_[np.cos(theta), np.sin(theta)]
    section = base.SectionMap(outer_scale * radius, [0.0, 0.12, 0.025])
    if kind == 'infinite':
        z = section.boundary(theta)
        outer = np.c_[z.real, z.imag]
    else:
        corners = np.array([[-width / 2, -length / 2], [width / 2, -length / 2], [width / 2, length / 2], [-width / 2, length / 2]])
        outer = np.concatenate([a + (b - a) * np.arange(n)[:, None] / n for a, b, n in zip(corners, np.roll(corners, -1, axis=0), (48 * refinement, 160 * refinement) * 2)])
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.add('independent_hole')
        loops, groups = ([], [])
        for points, size in ((outer, radius * 0.18 / refinement), (wall, radius * 0.025 / refinement)):
            tags = [gmsh.model.geo.addPoint(float(x), float(y), 0.0, size) for x, y in points]
            lines = [gmsh.model.geo.addLine(a, b) for a, b in zip(tags, tags[1:] + tags[:1])]
            loops.append(gmsh.model.geo.addCurveLoop(lines))
            groups.append(tags)
        gmsh.model.geo.addPlaneSurface(loops)
        gmsh.model.geo.synchronize()
        for _, tag in gmsh.model.getEntities(1):
            gmsh.model.mesh.setTransfiniteCurve(tag, 2)
        gmsh.option.setNumber('Mesh.Algorithm', 6)
        gmsh.model.mesh.generate(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        xy = np.asarray(xyz).reshape(-1, 3)[:, :2]
        index = {int(tag): i for i, tag in enumerate(tags)}
        types, _, elements = gmsh.model.mesh.getElements(2)
        triangles, = [nodes for typ, nodes in zip(types, elements) if typ == 2]
        cells = np.array([index[int(tag)] for tag in triangles]).reshape(-1, 3)
        boundaries = [np.array([index[int(gmsh.model.mesh.getNodes(0, t)[0][0])] for t in group]) for group in groups]
        return (xy, cells, *boundaries, section)
    finally:
        gmsh.finalize()

def solve(data, mat, kind, sigma, harmonics):
    xy, cells, outer, wall, section = data
    b, area, dofs = base.cst(xy, cells)
    ke = np.einsum('eai,ab,ebj,e->eij', b, mat.stiffness, b, area)
    nd = 2 * len(xy)
    k = sp.coo_matrix((ke.ravel(), (np.repeat(dofs, 6, axis=1).ravel(), np.tile(dofs, (1, 6)).ravel())), shape=(nd, nd)).tocsc()
    f = np.zeros(nd)
    affine = np.zeros_like(xy)
    stress = np.diag([0.0, sigma])
    constraints = np.zeros((nd, 3))
    constraints[0::2, 0], constraints[1::2, 1] = (1.0, 1.0)
    constraints[0::2, 2], constraints[1::2, 2] = (-xy[:, 1], xy[:, 0])
    if kind == 'infinite':
        kd = base.exterior(mat, len(outer), harmonics, section)
        od = (2 * outer[:, None] + [0, 1]).ravel()
        k += sp.coo_matrix((kd.ravel(), (np.repeat(od, len(od)), np.tile(od, len(od)))), shape=(nd, nd)).tocsc()
        base.traction(f, xy, wall, stress)
        e = mat.compliance @ np.array([0.0, sigma, 0.0])
        affine = xy @ np.array([[e[0], e[2] / 2], [e[2] / 2, e[1]]])
        constraints = constraints[:, :2]
    else:
        base.traction(f, xy, outer, stress)
    classes = np.arange(nd)
    fixed = np.array([], dtype=int)
    if kind == 'clamped':
        half_length = abs(xy[outer, 1]).max()
        ends = outer[np.isclose(abs(xy[outer, 1]), half_length)]
        fixed = 2 * ends
        for y in (-half_length, half_length):
            ds = 2 * outer[np.isclose(xy[outer, 1], y)] + 1
            classes[ds] = ds[0]
        constraints = constraints[:, 1:2]
    _, inv = np.unique(classes, return_inverse=True)
    transform = sp.coo_matrix((np.ones(nd), (np.arange(nd), inv)), shape=(nd, inv.max() + 1)).tocsc()
    free = np.setdiff1d(np.arange(transform.shape[1]), inv[fixed])
    transform = transform[:, free]
    constraints /= np.linalg.norm(constraints, axis=0)
    c = sp.csc_matrix(transform.T @ constraints)
    system = sp.bmat([[transform.T @ k @ transform, c], [c.T, None]], format='csc')
    rhs = np.r_[transform.T @ f, np.zeros(c.shape[1])]
    result = spsolve(system, rhs)
    residual = np.linalg.norm(system @ result - rhs) / np.linalg.norm(rhs)
    if not np.isfinite(residual) or residual > 1e-08:
        raise RuntimeError(f'Equilibrium failure: {residual}')
    u = (transform @ result[:len(free)]).reshape(-1, 2) + affine
    stresses = np.einsum('ab,ebi,ei->ea', mat.stiffness, b, u.ravel()[dofs])
    return (u, stresses, dict(vertices=len(xy), triangles=len(cells), residual=residual))

def sample_displacement(data, u, samples):
    tri = Triangulation(*data[0].T, data[1])
    values = np.zeros((len(samples), 2))
    for j in (0, 1):
        interp = LinearTriInterpolator(tri, u[:, j])
        for dx in (-0.525 / np.sqrt(3), 0.525 / np.sqrt(3)):
            for dy in (-0.525 / np.sqrt(3), 0.525 / np.sqrt(3)):
                got = interp(samples[:, 0] + dx, samples[:, 1] + dy)
                if np.any(np.ma.getmaskarray(got)):
                    raise ValueError('DIC support outside material')
                values[:, j] += np.asarray(got) / 4
    return values

def digitize_fatima():
    pdf = ROOT / 'references/papers/fatima2017/paper.pdf'
    with fitz.open(pdf) as doc:
        xref = doc[8].get_images(full=True)[0][0]
        pix = fitz.Pixmap(doc, xref)
        a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    assert a.shape == (254, 512, 3)
    palette = np.median(a[21:211, 450:466], axis=1)
    tree = cKDTree(palette)
    rows = []
    for iy in range(23, 208, 3):
        for ix in range(52, 239, 3):
            x, y = ((210 - iy) / 95.25, (241 - ix) / 95.25)
            radius = np.hypot(x, y)
            if not 1.1 <= radius <= 1.98:
                continue
            rgb = np.median(a[iy - 1:iy + 2, ix - 1:ix + 2].reshape(-1, 3), axis=0)
            distance, j = tree.query(rgb)
            valid = distance < 12
            lower = np.clip(round((1 - j / 189) * 10) / 10, 0.0, 0.9) * 0.001
            rows.append(dict(pixel_x=ix, pixel_y=iy, transverse_mm=x * 12.7, axial_mm=y * 12.7, valid=int(valid), rgb_distance=distance, u_low_mm=lower * 12.7, u_high_mm=(lower + 0.0001) * 12.7, u_mid_mm=(lower + 5e-05) * 12.7))
    write_rows(HERE / 'fatima2017_observations.csv', rows)
    return [r for r in rows if r['valid']]

def assess_fatima(refinement, harmonics, suffix, from_data=False):
    if from_data:
        with (HERE / 'fatima2017_observations.csv').open(newline='', encoding='utf-8') as stream:
            obs = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream) if int(float(row['valid']))]
    else:
        obs = digitize_fatima()
    samples = np.array([[r['transverse_mm'], r['axial_mm']] for r in obs])
    measured = np.array([r['u_mid_mm'] for r in obs])
    lo, hi = (np.array([r[key] for r in obs]) for key in ('u_low_mm', 'u_high_mm'))
    orth = material(104100.0, 28100.0, 3000.0, 0.16)
    iso = material(104100.0, 104100.0, 104100.0 / 2.32, 0.16)
    sigma = 4450 / (76.2 * 5.28)
    rows, fields, saved = ([], [], {})
    for kind, mat, label in [('traction', orth, 'finite_orthotropic'), ('clamped', orth, 'clamped_orthotropic'), ('infinite', orth, 'dtn_orthotropic'), ('traction', iso, 'finite_isotropic')]:
        data = mesh(12.7, 76.2, 279.4, kind, refinement)
        u, stress, info = solve(data, mat, kind, sigma, harmonics)
        predicted = sample_displacement(data, u, samples)[:, 1]
        zero_samples = np.c_[np.full(5, 2.5 * 12.7), np.linspace(-0.4, 0.4, 5)]
        origin = sample_displacement(data, u, zero_samples)[:, 1].mean()
        predicted -= origin
        saved[label] = predicted
        distance = np.maximum(lo - predicted, 0) + np.maximum(predicted - hi, 0)
        row = dict(case=label, samples=len(obs), measured_rms_um=1000 * np.sqrt(np.mean(measured ** 2)), predicted_rms_um=1000 * np.sqrt(np.mean(predicted ** 2)), bias_um=1000 * np.mean(predicted - measured), rms_error_um=1000 * np.sqrt(np.mean((predicted - measured) ** 2)), rms_error_relative=np.linalg.norm(predicted - measured) / np.linalg.norm(measured), fraction_in_printed_bin=np.mean(distance == 0), rms_distance_from_bin_um=1000 * np.sqrt(np.mean(distance ** 2)), **info)
        rows.append(row)
        print(row, flush=True)
    for i, ob in enumerate(obs):
        fields.append(dict(**ob, **{key + '_mm': value[i] for key, value in saved.items()}))
    write_rows(HERE / f'fatima2017_forward{suffix}.csv', rows)
    write_rows(HERE / f'fatima2017_fields{suffix}.csv', fields)
    for a, b in [('finite_orthotropic', 'dtn_orthotropic'), ('finite_orthotropic', 'finite_isotropic'), ('finite_orthotropic', 'clamped_orthotropic')]:
        print('field difference', a, b, 'RMS um', 1000 * np.sqrt(np.mean((saved[a] - saved[b]) ** 2)), flush=True)

def assess_saved():
    coarse = np.genfromtxt(HERE / 'fatima2017_fields.csv', delimiter=',', names=True)
    fine = np.genfromtxt(HERE / 'fatima2017_fields_refined.csv', delimiter=',', names=True)
    assert np.array_equal(coarse['pixel_x'], fine['pixel_x'])
    assert np.array_equal(coarse['pixel_y'], fine['pixel_y'])
    rows = []

    def record(comparison, region, delta):
        rows.append(dict(comparison=comparison, region=region, samples=len(delta), rms_um=1000 * np.sqrt(np.mean(delta ** 2)), mean_um=1000 * np.mean(delta)))
    models = ['finite_orthotropic', 'clamped_orthotropic', 'dtn_orthotropic', 'finite_isotropic']
    distance = np.hypot(fine['transverse_mm'], fine['axial_mm']) - 12.7
    angle = np.rad2deg(np.arctan2(fine['transverse_mm'], fine['axial_mm']))
    regions = {'all': distance > 0, 'edge_distance_below_half_t': distance < 2.64, 'edge_distance_half_t_to_t': (distance >= 2.64) & (distance < 5.28), 'edge_distance_above_t': distance >= 5.28, 'axial_0_to_30_deg': angle < 30, 'diagonal_30_to_60_deg': (angle >= 30) & (angle < 60), 'transverse_60_to_90_deg': angle >= 60}
    for model in models:
        key = model + '_mm'
        record(model + '_mesh_change', 'all', fine[key] - coarse[key])
        record(model + '_minus_palette_value', 'all', fine[key] - fine['u_low_mm'])
        for name, mask in regions.items():
            record(model + '_minus_band_midpoint', name, fine[key][mask] - fine['u_mid_mm'][mask])
    for model in models[1:]:
        record('finite_orthotropic_minus_' + model, 'all', fine['finite_orthotropic_mm'] - fine[model + '_mm'])
    mat = material(104100.0, 28100.0, 3000.0, 0.16)
    sigma = 4450 / (76.2 * 5.28)
    ref = base.solve_neumann_twoterm(mat, base.SectionMap(12.7, [0.0]), 0.0, sigma, 0.0, 4)

    def exact(offset=(0.0, 0.0), average=True):
        terms = []
        steps = (-0.525 / np.sqrt(3), 0.525 / np.sqrt(3)) if average else (0.0,)
        for dx in steps:
            for dy in steps:
                x = fine['transverse_mm'] + dx + offset[0]
                y = fine['axial_mm'] + dy + offset[1]
                terms.append(ref.fields_xy(x, y)['u_y'] + sigma * mat.compliance[1, 1] * y)
        return np.mean(terms, axis=0)
    exact_mean = exact()
    record('coarse_dtn_minus_exact', 'all', coarse['dtn_orthotropic_mm'] - exact_mean)
    record('refined_dtn_minus_exact', 'all', fine['dtn_orthotropic_mm'] - exact_mean)
    record('exact_subset_mean_minus_point', 'all', exact_mean - exact(average=False))
    for direction in (-1.0, 1.0):
        for axis in (0, 1):
            shift = np.zeros(2)
            shift[axis] = direction * 12.7 / 95.25
            record(f'exact_registration_axis{axis}_{direction:+g}px', 'all', exact(shift) - exact_mean)
    for angle in (45.0, -45.0):
        rotated = material(22540.0, 10940.0, 3540.0, 0.3, 90 - angle)
        other = material(22540.0, 10940.0, 3540.0, 0.3, 90 + angle)
        flip = np.diag([1.0, 1.0, -1.0])
        assert np.allclose(rotated.compliance, flip @ other.compliance @ flip, atol=1e-15)
    write_rows(HERE / 'fatima2017_sensitivity.csv', rows)
    for row in rows:
        if row['region'] == 'all':
            print(row, flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refinement', type=int, default=1)
    parser.add_argument('--harmonics', type=int, default=48)
    parser.add_argument('--case', choices=('fatima',), default='fatima')
    parser.add_argument('--assess-only', action='store_true')
    parser.add_argument('--from-data', action='store_true', help='Use supplied extracted observations instead of the source PDF')
    args = parser.parse_args()
    suffix = '' if args.refinement == 1 else '_refined'
    with threadpool_limits(limits=1):
        if args.assess_only:
            assess_saved()
            return
        if args.case in ('all', 'fatima'):
            assess_fatima(args.refinement, args.harmonics, suffix, args.from_data)
if __name__ == '__main__':
    main()
