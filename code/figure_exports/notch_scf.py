from __future__ import annotations
import argparse
import csv
import importlib
from pathlib import Path
import gmsh
import numpy as np
from threadpoolctl import threadpool_limits
hole = importlib.import_module('orthotropic_hole')
HERE = Path(__file__).resolve().parents[2] / 'results/data/derived'

def wall_points(width, height, radius, shape, refinement):
    if shape == 'circle':
        theta = np.arange(256 * refinement) * 2 * np.pi / (256 * refinement)
        return radius * np.c_[np.cos(theta), np.sin(theta)]
    centers = np.array([[width / 2 - radius, height / 2 - radius], [-width / 2 + radius, height / 2 - radius], [-width / 2 + radius, -height / 2 + radius], [width / 2 - radius, -height / 2 + radius]])
    arcs = []
    for j, center in enumerate(centers):
        theta = (j + np.arange(32 * refinement + 1) / (32 * refinement)) * np.pi / 2
        arcs.append(center + radius * np.c_[np.cos(theta), np.sin(theta)])
    points = []
    for arc, following in zip(arcs, arcs[1:] + arcs[:1]):
        points.extend(arc)
        a, b = (arc[-1], following[0])
        count = int(np.ceil(np.linalg.norm(b - a) / (0.1 / refinement)))
        points.extend(a + (b - a) * np.arange(1, count)[:, None] / count)
    return np.asarray(points)

def mesh(observation, kind, refinement):
    width, height, radius = (float(observation[k]) for k in ('width_mm', 'height_mm', 'corner_radius_mm'))
    wall = wall_points(width, height, radius, observation['shape'], refinement)
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

def assess(refinement, harmonics, selected):
    with (HERE / 'khechai2018_table3.csv').open(encoding='utf-8', newline='') as stream:
        observations = list(csv.DictReader(stream))
    mat = hole.material(22540.0, 10940.0, 3540.0, 0.3)
    rows = []
    for i, ob in enumerate(observations):
        if selected and i not in selected:
            continue
        for kind in ('traction', 'infinite'):
            data = mesh(ob, kind, refinement)
            u, stresses, info = hole.solve(data, mat, kind, 10.0, harmonics)
            xy, cells, _, wall, _ = data
            touching = np.sum(np.isin(cells, wall), axis=1) == 2
            edge_cells, edge_stresses = (cells[touching], stresses[touching])
            axial_index = np.argmax(edge_stresses[:, 1])
            centers = xy[edge_cells].mean(axis=1)
            axial = edge_stresses[axial_index, 1] / 10.0
            hoop = np.max(edge_stresses[:, 0] + edge_stresses[:, 1]) / 10.0
            row = dict(**ob, boundary=kind, refinement=refinement, harmonics=harmonics if kind == 'infinite' else 0, axial_scf=axial, hoop_scf=hoop, peak_x_mm=centers[axial_index, 0], peak_y_mm=centers[axial_index, 1], relative_difference=axial / float(ob['observed_scf']) - 1, **info)
            rows.append(row)
            print(row, flush=True)
    suffix = '_refined' if refinement == 2 else ''
    if selected:
        suffix += '_selected'
    hole.write_rows(HERE / f'khechai2018_notch_forward{suffix}.csv', rows)

def assess_saved():

    def read(name):
        with (HERE / name).open(encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream))
    coarse = read('khechai2018_notch_forward.csv')
    fine = read('khechai2018_notch_forward_refined.csv')
    keys = ('shape', 'width_mm', 'height_mm', 'boundary', 'observed_scf')
    assert len(coarse) == len(fine) == 24
    assert [tuple((r[k] for k in keys)) for r in coarse] == [tuple((r[k] for k in keys)) for r in fine]
    for boundary in ('traction', 'infinite'):
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
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refinement', type=int, choices=(1, 2), default=1)
    parser.add_argument('--harmonics', type=int, default=48)
    parser.add_argument('--rows', type=int, nargs='*')
    parser.add_argument('--assess-only', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.assess_only:
            assess_saved()
        else:
            assess(args.refinement, args.harmonics, args.rows)
