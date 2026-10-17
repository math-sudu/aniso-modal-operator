from __future__ import annotations
import argparse
import csv
import importlib
from pathlib import Path
import numpy as np
import skfem
from scipy.spatial import cKDTree
from skfem import Basis, BilinearForm, ElementTriP2, ElementVector, FacetBasis, LinearForm, MeshTri, asm, condense, solve
from skfem.helpers import sym_grad
from threadpoolctl import threadpool_limits
production = importlib.import_module('notch_scf')
HERE = Path(__file__).resolve().parents[2] / 'results/data/derived'
SIGMA = 10.0
E_X, E_Y, G_XY, NU_YX = (10940.0, 22540.0, 3540.0, 0.3)
NU_XY = NU_YX * E_X / E_Y
DENOMINATOR = 1.0 - NU_XY * NU_YX
Q_XX, Q_YY, Q_XY = (E_X / DENOMINATOR, E_Y / DENOMINATOR, NU_YX * E_X / DENOMINATOR)

@BilinearForm
def elasticity(u, v, w):
    eu, ev = (sym_grad(u), sym_grad(v))
    return Q_XX * eu[0, 0] * ev[0, 0] + Q_YY * eu[1, 1] * ev[1, 1] + Q_XY * (eu[0, 0] * ev[1, 1] + eu[1, 1] * ev[0, 0]) + 4.0 * G_XY * eu[0, 1] * ev[0, 1]

@LinearForm
def top_traction(v, w):
    return SIGMA * v[1]

def independent_solve(data):
    xy, cells, _, wall, _ = data
    mesh = MeshTri(xy.T, cells.T)
    element = ElementVector(ElementTriP2())
    basis = Basis(mesh, element, intorder=4)
    stiffness = asm(elasticity, basis)
    top = mesh.facets_satisfying(lambda x: np.isclose(x[1], 125.0))
    load_basis = FacetBasis(mesh, element, facets=top, intorder=4)
    load = asm(top_traction, load_basis)
    fixed = basis.get_dofs(lambda x: np.isclose(x[1], -125.0)).all()
    displacement = solve(*condense(stiffness, load, D=fixed))
    free = basis.complement_dofs(fixed)
    residual = np.linalg.norm((stiffness @ displacement - load)[free]) / np.linalg.norm(load[free])
    if not np.isfinite(residual) or residual > 1e-08:
        raise RuntimeError(f'Independent equilibrium failure: {residual}')
    facets = {tuple(pair): i for i, pair in enumerate(mesh.facets.T)}
    hole_facets = np.array([facets[tuple(sorted((a, b)))] for a, b in zip(wall, np.roll(wall, -1))])
    edge_basis = FacetBasis(mesh, element, facets=hole_facets, quadrature=(np.array([[0.5]]), np.array([1.0])))
    field = edge_basis.interpolate(displacement)
    gradient = field.grad[:, :, :, 0]
    edge_stress = np.column_stack((Q_XX * gradient[0, 0] + Q_XY * gradient[1, 1], Q_XY * gradient[0, 0] + Q_YY * gradient[1, 1], G_XY * (gradient[0, 1] + gradient[1, 0])))
    edge_xy = np.asarray(edge_basis.global_coordinates())[:, :, 0].T
    edge_displacement = np.asarray(field)[:, :, 0].T
    component_dofs = basis.split_indices()
    node_xy = basis.doflocs[:, component_dofs[0]].T
    if not np.array_equal(node_xy, basis.doflocs[:, component_dofs[1]].T):
        raise ValueError('Vector component coordinates do not correspond')
    node_displacement = np.column_stack([displacement[dofs] for dofs in component_dofs])
    peak = np.argmax(edge_stress[:, 1])
    info = dict(axial_scf=edge_stress[peak, 1] / SIGMA, peak_x_mm=edge_xy[peak, 0], peak_y_mm=edge_xy[peak, 1], residual=residual, external_work=float(load @ displacement), top_force_y=float(load[component_dofs[1]].sum()), bottom_reaction_y=float((stiffness @ displacement - load)[np.intersect1d(fixed, component_dofs[1])].sum()))
    return (info, dict(xy=node_xy, displacement=node_displacement, edge_xy=edge_xy, edge_stress=edge_stress, edge_displacement=edge_displacement))

def write_rows(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def compare(refinement):
    with (HERE / 'khechai2018_table3.csv').open(encoding='utf-8', newline='') as stream:
        observations = list(csv.DictReader(stream))
    mat = production.hole.material(22540.0, 10940.0, 3540.0, 0.3)
    summaries, profiles = ([], [])
    for ob in (observations[5], observations[11]):
        label = f"{ob['shape']}_{ob['width_mm']}x{ob['height_mm']}"
        print(f'{label}: generating refinement {refinement} mesh', flush=True)
        data = production.mesh(ob, 'fixed', refinement)
        print(f'{label}: production P2 solution', flush=True)
        primary, primary_field = production.solve(data, mat, 'fixed', SIGMA, 0, 2, return_fields=True)
        print(f'{label}: independent scikit-fem P2 solution', flush=True)
        independent, independent_field = independent_solve(data)
        coordinates = primary_field['edge_xy']
        if not np.allclose(coordinates, independent_field['edge_xy'], rtol=0.0, atol=1e-12):
            raise ValueError('Hole midpoint coordinates or ordering differ')
        distance, order = cKDTree(independent_field['xy']).query(primary_field['xy'])
        if distance.max() > 1e-12 or len(np.unique(order)) != len(order):
            raise ValueError('Quadratic displacement nodes do not correspond')
        displacement_delta = independent_field['displacement'][order] - primary_field['displacement']
        stress_delta = independent_field['edge_stress'] - primary_field['edge_stress']
        summary = dict(case=label, width_mm=ob['width_mm'], height_mm=ob['height_mm'], refinement=refinement, corner_segments=32 * refinement, triangles=len(data[1]), displacement_nodes=len(primary_field['xy']), edge_samples=len(coordinates), skfem_version=skfem.__version__, source_fem_scf=ob['source_fem_scf'], observed_scf=ob['observed_scf'], production_axial_scf=primary['axial_scf'], independent_axial_scf=independent['axial_scf'], peak_relative_difference=independent['axial_scf'] / primary['axial_scf'] - 1.0, production_peak_x_mm=primary['peak_x_mm'], production_peak_y_mm=primary['peak_y_mm'], independent_peak_x_mm=independent['peak_x_mm'], independent_peak_y_mm=independent['peak_y_mm'], axial_profile_max_abs_scf_difference=np.max(abs(stress_delta[:, 1])) / SIGMA, axial_profile_relative_l2_difference=np.linalg.norm(stress_delta[:, 1]) / np.linalg.norm(primary_field['edge_stress'][:, 1]), stress_profile_max_abs_mpa_difference=np.max(abs(stress_delta)), displacement_max_abs_mm_difference=np.max(abs(displacement_delta)), displacement_relative_l2_difference=np.linalg.norm(displacement_delta) / np.linalg.norm(primary_field['displacement']), production_external_work_nmm_per_mm=primary_field['external_work'], independent_external_work_nmm_per_mm=independent['external_work'], work_relative_difference=independent['external_work'] / primary_field['external_work'] - 1.0, top_force_y_n_per_mm=independent['top_force_y'], bottom_reaction_y_n_per_mm=independent['bottom_reaction_y'], production_residual=primary['residual'], independent_residual=independent['residual'])
        summaries.append(summary)
        print(summary, flush=True)
        for i, point in enumerate(coordinates):
            row = dict(case=label, edge_index=i, x_mm=point[0], y_mm=point[1])
            for name, values in (('production', primary_field), ('independent', independent_field)):
                for j, component in enumerate(('xx', 'yy', 'xy')):
                    row[f'{name}_sigma_{component}_mpa'] = values['edge_stress'][i, j]
                for j, component in enumerate(('x', 'y')):
                    row[f'{name}_u_{component}_mm'] = values['edge_displacement'][i, j]
            profiles.append(row)
        write_rows(HERE / 'khechai2018_notch_independent.csv', summaries)
        write_rows(HERE / 'khechai2018_notch_independent_profiles.csv', profiles)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refinement', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        compare(args.refinement)
if __name__ == '__main__':
    main()
