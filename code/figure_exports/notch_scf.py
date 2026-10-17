from __future__ import annotations
import argparse
import csv
import importlib
from pathlib import Path
import gmsh
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits
hole = importlib.import_module('orthotropic_hole')
from exterior_operator import assemble_modal, projected_work
HERE = Path(__file__).resolve().parents[2] / 'results/data/derived'

def wall_points(width, height, radius, shape, refinement, corner_segments=None):
    if shape == 'circle':
        theta = np.arange(256 * refinement) * 2 * np.pi / (256 * refinement)
        return radius * np.c_[np.cos(theta), np.sin(theta)]
    centers = np.array([[width / 2 - radius, height / 2 - radius], [-width / 2 + radius, height / 2 - radius], [-width / 2 + radius, -height / 2 + radius], [width / 2 - radius, -height / 2 + radius]])
    arcs = []
    count = 32 * refinement if corner_segments is None else corner_segments
    for j, center in enumerate(centers):
        theta = (j + np.arange(count + 1) / count) * np.pi / 2
        arcs.append(center + radius * np.c_[np.cos(theta), np.sin(theta)])
    points = []
    for arc, following in zip(arcs, arcs[1:] + arcs[:1]):
        points.extend(arc)
        a, b = (arc[-1], following[0])
        count = int(np.ceil(np.linalg.norm(b - a) / (0.1 / refinement)))
        points.extend(a + (b - a) * np.arange(1, count)[:, None] / count)
    return np.asarray(points)

def mesh(observation, kind, refinement, corner_segments=None):
    width, height, radius = (float(observation[k]) for k in ('width_mm', 'height_mm', 'corner_radius_mm'))
    wall = wall_points(width, height, radius, observation['shape'], refinement, corner_segments)
    section = hole.base.SectionMap(1.8 * max(width, height), [0.0, 0.12, 0.025])
    if kind == 'infinite':
        theta = np.arange(256 * refinement) * 2 * np.pi / (256 * refinement)
        z = section.boundary(theta)
        outer = np.c_[z.real, z.imag]
    else:
        corners = np.array([[-12.5, -125.0], [12.5, -125.0], [12.5, 125.0], [-12.5, 125.0]])
        outer = np.concatenate([a + (b - a) * np.arange(n)[:, None] / n for a, b, n in zip(corners, np.roll(corners, -1, axis=0), (40 * refinement, 160 * refinement) * 2)])
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.add('notch_scf')
        loops, groups = ([], [])
        for points, size in ((outer, 0.8 / refinement), (wall, 0.04 / refinement)):
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

def quadratic_mesh(data):
    """Add shared straight-edge midpoints without changing the P1 geometry."""
    xy, cells, outer, wall, section = data
    edges, inverse = np.unique(np.sort(cells[:, [[0, 1], [1, 2], [2, 0]]], axis=2).reshape(-1, 2), axis=0, return_inverse=True)
    mids = len(xy) + np.arange(len(edges))
    return (np.vstack((xy, xy[edges].mean(axis=1))), np.c_[cells, mids[inverse].reshape(-1, 3)], outer, wall, section)

def strain_matrix(grad, barycentric, order):
    if order == 1:
        derivatives = grad
    else:
        local = np.asarray(barycentric)
        vertex = (4 * local - 1)[None, :, None] * grad
        edge = 4 * (local[[0, 1, 2]][None, :, None] * grad[:, [1, 2, 0]] + local[[1, 2, 0]][None, :, None] * grad[:, [0, 1, 2]])
        derivatives = np.concatenate((vertex, edge), axis=1)
    b = np.zeros((len(grad), 3, 2 * derivatives.shape[1]))
    b[:, 0, 0::2], b[:, 1, 1::2] = (derivatives[:, :, 0], derivatives[:, :, 1])
    b[:, 2, 0::2], b[:, 2, 1::2] = (derivatives[:, :, 1], derivatives[:, :, 0])
    return b

def boundary_edges(cells, nodes):
    """Cell and local-edge indices in the order of a supplied vertex loop."""
    sides = cells[:, [[0, 1], [1, 2], [2, 0]]]
    touching = np.all(np.isin(sides, nodes), axis=2)
    elements, local = np.where(touching)
    lookup = {tuple(sorted(pair)): (e, j) for pair, e, j in zip(sides[touching], elements, local)}
    return np.array([lookup[tuple(sorted((a, b)))] for a, b in zip(nodes, np.roll(nodes, -1))]).T

def solve(data, mat, kind, sigma, harmonics, order, return_fields=False):
    """P1/P2 elasticity, with a P1 trace restriction on the DtN interface."""
    if order == 2:
        data = quadratic_mesh(data)
    xy, cells, outer, wall, section = data
    b1, area, _ = hole.base.cst(xy, cells[:, :3])
    grad = np.stack((b1[:, 0, 0::2], b1[:, 1, 1::2]), axis=2)
    dofs = (2 * cells[:, :, None] + [0, 1]).reshape(len(cells), -1)
    nd, ed = (2 * len(xy), dofs.shape[1])
    ke = np.zeros((len(cells), ed, ed))
    for point in ((2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3)):
        b = strain_matrix(grad, point, order)
        ke += np.einsum('eai,ab,ebj,e->eij', b, mat.stiffness, b, area / 3)
    k = sp.coo_matrix((ke.ravel(), (np.repeat(dofs, ed, axis=1).ravel(), np.tile(dofs, (1, ed)).ravel())), shape=(nd, nd)).tocsc()
    del ke
    f = np.zeros(nd)
    affine = np.zeros_like(xy)
    stress = np.diag([0.0, sigma])
    load_nodes = wall if kind == 'infinite' else outer
    elements, local = boundary_edges(cells, load_nodes)
    for a, z, e, j in zip(load_nodes, np.roll(load_nodes, -1), elements, local):
        dx, dy = xy[z] - xy[a]
        load = stress @ np.array([dy, -dx])
        if kind == 'fixed' and (not np.isclose(xy[[a, z], 1], 125.0).all()):
            continue
        if order == 1:
            for node in (a, z):
                f[2 * node:2 * node + 2] += load / 2
        else:
            for node, weight in ((a, 1 / 6), (z, 1 / 6), (cells[e, 3 + j], 2 / 3)):
                f[2 * node:2 * node + 2] += weight * load
    transform = sp.eye(nd, format='csc')
    if kind == 'infinite':
        theta = np.arange(len(outer)) * 2 * np.pi / len(outer)
        kd = -projected_work(assemble_modal(mat, section, harmonics), theta)
        od = (2 * outer[:, None] + [0, 1]).ravel()
        k += sp.coo_matrix((kd.ravel(), (np.repeat(od, len(od)), np.tile(od, len(od)))), shape=(nd, nd)).tocsc()
        strain = mat.compliance @ np.array([0.0, sigma, 0.0])
        affine = xy @ np.array([[strain[0], strain[2] / 2], [strain[2] / 2, strain[1]]])
        if order == 2:
            oe, oj = boundary_edges(cells, outer)
            md = (2 * cells[oe, 3 + oj][:, None] + [0, 1]).ravel()
            keep = np.setdiff1d(np.arange(nd), md)
            columns = np.full(nd, -1)
            columns[keep] = np.arange(len(keep))
            following = (2 * np.roll(outer, -1)[:, None] + [0, 1]).ravel()
            transform = sp.coo_matrix((np.r_[np.ones(len(keep)), np.full(2 * len(md), 0.5)], (np.r_[keep, md, md], np.r_[columns[keep], columns[od], columns[following]])), shape=(nd, len(keep))).tocsc()
    if kind == 'fixed':
        bottom = np.flatnonzero(np.isclose(xy[:, 1], -125.0))
        free = np.setdiff1d(np.arange(nd), (2 * bottom[:, None] + [0, 1]).ravel())
        transform = transform[:, free]
        system, rhs = (transform.T @ k @ transform, transform.T @ f)
    else:
        constraints = np.zeros((nd, 2 if kind == 'infinite' else 3))
        if kind == 'infinite':
            constraints[2 * outer[0], 0] = constraints[2 * outer[0] + 1, 1] = 1.0
        else:
            constraints[0::2, 0], constraints[1::2, 1] = (1.0, 1.0)
            constraints[0::2, 2], constraints[1::2, 2] = (-xy[:, 1], xy[:, 0])
        constraints /= np.linalg.norm(constraints, axis=0)
        c = sp.csc_matrix(transform.T @ constraints)
        system = sp.bmat([[transform.T @ k @ transform, c], [c.T, None]], format='csc')
        rhs = np.r_[transform.T @ f, np.zeros(c.shape[1])]
    result = spsolve(system.tocsc(), rhs)
    residual = np.linalg.norm(system @ result - rhs) / np.linalg.norm(rhs)
    if not np.isfinite(residual) or residual > 1e-08:
        raise RuntimeError(f'Equilibrium failure: {residual}')
    u = (transform @ result[:transform.shape[1]]).reshape(-1, 2) + affine
    values = u.ravel()[dofs]
    we, wj = boundary_edges(cells, wall)
    center_stress = np.einsum('ab,ebi,ei->ea', mat.stiffness, strain_matrix(grad[we], (1 / 3,) * 3, order), values[we])
    edge_stress = np.zeros((len(we), 3))
    for j, pair in enumerate(((0, 1), (1, 2), (2, 0))):
        selected = wj == j
        bary = np.zeros(3)
        bary[list(pair)] = 0.5
        edge_stress[selected] = np.einsum('ab,ebi,ei->ea', mat.stiffness, strain_matrix(grad[we[selected]], bary, order), values[we[selected]])
    sums, weights = (np.zeros((len(xy), 3)), np.zeros(len(xy)))
    adjacent = np.any(np.isin(cells[:, :3], wall), axis=1)
    for j in range(3):
        stresses = np.einsum('ab,ebi,ei->ea', mat.stiffness, strain_matrix(grad[adjacent], np.eye(3)[j], order), values[adjacent])
        np.add.at(sums, cells[adjacent, j], area[adjacent, None] * stresses)
        np.add.at(weights, cells[adjacent, j], area[adjacent])
    peak = np.argmax(edge_stress[:, 1])
    point = xy[[wall[peak], np.roll(wall, -1)[peak]]].mean(axis=0)
    info = dict(axial_scf=edge_stress[peak, 1] / sigma, hoop_scf=np.max(edge_stress[:, 0] + edge_stress[:, 1]) / sigma, centroid_scf=center_stress[:, 1].max() / sigma, recovered_node_scf=np.max(sums[wall, 1] / weights[wall]) / sigma, peak_x_mm=point[0], peak_y_mm=point[1], vertices=len(xy), triangles=len(cells), residual=residual)
    if return_fields:
        midpoint_u = u[cells[we, 3 + wj]] if order == 2 else (u[wall] + u[np.roll(wall, -1)]) / 2
        return (info, dict(xy=xy, displacement=u, edge_xy=(xy[wall] + xy[np.roll(wall, -1)]) / 2, edge_stress=edge_stress, edge_displacement=midpoint_u, external_work=float(f @ u.ravel())))
    return info

def assess(refinement, harmonics, selected, order=2, boundaries=('fixed', 'infinite'), output=None):
    with (HERE / 'khechai2018_table3.csv').open(encoding='utf-8', newline='') as stream:
        observations = list(csv.DictReader(stream))
    mat = hole.material(22540.0, 10940.0, 3540.0, 0.3)
    rows = []
    for i, ob in enumerate(observations):
        if selected and i not in selected:
            continue
        for kind in boundaries:
            data = mesh(ob, kind, refinement)
            info = solve(data, mat, kind, 10.0, harmonics, order)
            row = dict(**ob, boundary=kind, refinement=refinement, element_order=order, harmonics=harmonics if kind == 'infinite' else 0, relative_difference=info['axial_scf'] / float(ob['observed_scf']) - 1, source_fem_difference=info['axial_scf'] / float(ob['source_fem_scf']) - 1, **info)
            rows.append(row)
            print(row, flush=True)
    suffix = '_refined' if refinement == 2 else ''
    if selected:
        suffix += '_selected'
    hole.write_rows(output or HERE / f'khechai2018_notch_forward{suffix}.csv', rows)

def check_elements():
    """Check exact quadratic strains, not just the solved linear system."""
    xy = np.array([[0.0, 0.0], [2.0, 0.2], [0.1, 1.3]])
    cells = np.array([[0, 1, 2]])
    mesh2 = quadratic_mesh((xy, cells, np.arange(3), np.arange(3), None))
    b, _, _ = hole.base.cst(xy, cells)
    grad = np.stack((b[:, 0, 0::2], b[:, 1, 1::2]), axis=2)
    x, y = mesh2[0].T
    u = np.c_[2 * x * x + 3 * x * y - y * y, x * x - 2 * x * y + 4 * y * y]
    for point in ((1 / 3,) * 3, (0.1, 0.3, 0.6), (0.0, 0.5, 0.5)):
        xp, yp = np.asarray(point) @ xy
        actual = strain_matrix(grad, point, 2) @ u[mesh2[1]].reshape(1, -1, 1)
        expected = [4 * xp + 3 * yp, -2 * xp + 8 * yp, 5 * xp - 4 * yp]
        np.testing.assert_allclose(actual.ravel(), expected, atol=4e-14)
    print('Quadratic displacement/strain reproduction passed.')

def protocol():
    """Separate boundary, polynomial-order and recovery effects on two holes."""
    with (HERE / 'khechai2018_table3.csv').open(encoding='utf-8', newline='') as stream:
        observations = list(csv.DictReader(stream))
    mat = hole.material(22540.0, 10940.0, 3540.0, 0.3)
    rows = []
    for refinement, order, corner_segments in ((1, 2, 32), (2, 1, 64), (2, 2, 64), (2, 2, 20)):
        for i in (5, 11):
            ob = observations[i]
            data = mesh(ob, 'fixed', refinement, corner_segments)
            for kind in ('fixed',) if corner_segments == 20 else ('fixed', 'traction'):
                info = solve(data, mat, kind, 10.0, 0, order)
                row = dict(**ob, boundary=kind, refinement=refinement, element_order=order, corner_segments=corner_segments, harmonics=0, relative_difference=info['axial_scf'] / float(ob['observed_scf']) - 1, source_fem_difference=info['axial_scf'] / float(ob['source_fem_scf']) - 1, **info)
                rows.append(row)
                print(row, flush=True)
    hole.write_rows(HERE / 'khechai2018_notch_protocol.csv', rows)

def assess_saved():

    def read(name):
        with (HERE / name).open(encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream))
    coarse = read('khechai2018_notch_forward.csv')
    fine = read('khechai2018_notch_forward_refined.csv')
    keys = ('shape', 'width_mm', 'height_mm', 'boundary', 'observed_scf')
    assert len(coarse) == len(fine) == 24
    assert [tuple((r[k] for k in keys)) for r in coarse] == [tuple((r[k] for k in keys)) for r in fine]
    for boundary in ('fixed', 'infinite'):
        selected = [(a, b) for a, b in zip(coarse, fine) if b['boundary'] == boundary and b['shape'] != 'circle']
        differences = [float(b['relative_difference']) for _, b in selected]
        mesh_change = max((abs(float(b['axial_scf']) / float(a['axial_scf']) - 1) for a, b in selected))
        print(boundary, 'relative differences', min(differences), max(differences), 'maximum refinement change', mesh_change)
        for width in ('4', '6', '8'):
            square = next((b for _, b in selected if b['shape'] == 'square' and b['width_mm'] == width))
            rectangle = next((b for _, b in selected if b['shape'] == 'rectangle' and b['width_mm'] == width))
            print('width', width, 'aspect-ratio reductions: measured, computed', 1 - float(rectangle['observed_scf']) / float(square['observed_scf']), 1 - float(rectangle['axial_scf']) / float(square['axial_scf']))
    mat = hole.material(22540.0, 10940.0, 3540.0, 0.3)
    ref = hole.base.solve_neumann_twoterm(mat, hole.base.SectionMap(1.0, [0.0]), 0.0, 10.0, 0.0, 4)
    stress = ref.boundary_fields(np.arange(8192) * 2 * np.pi / 8192)
    exact = (stress['s_yy'] + 10.0).max() / 10.0
    circle = [float(r['axial_scf']) for r in fine if r['shape'] == 'circle' and r['boundary'] == 'infinite']
    print('exact infinite circular SCF', exact, 'refined differences', np.asarray(circle) / exact - 1)
    for boundary in ('fixed', 'infinite'):
        rows = [r for r in fine if r['boundary'] == boundary and r['shape'] != 'circle']
        print(boundary, 'source FEM differences', [float(r['source_fem_difference']) for r in rows])
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refinement', type=int, choices=(1, 2), default=1)
    parser.add_argument('--harmonics', type=int, default=48)
    parser.add_argument('--rows', type=int, nargs='*')
    parser.add_argument('--order', type=int, choices=(1, 2), default=2)
    parser.add_argument('--boundaries', nargs='+', choices=('fixed', 'traction', 'infinite'), default=('fixed', 'infinite'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--assess-only', action='store_true')
    parser.add_argument('--check-elements', action='store_true')
    parser.add_argument('--protocol', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.check_elements:
            check_elements()
        elif args.protocol:
            protocol()
        elif args.assess_only:
            assess_saved()
        else:
            assess(args.refinement, args.harmonics, args.rows, args.order, args.boundaries, args.output)
