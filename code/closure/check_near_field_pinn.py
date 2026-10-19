from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.stats import qmc
from threadpoolctl import threadpool_limits
from near_field_pinn import ROOT, NeuralSection, lined, plastic_law, quadrature, strain_matrix, write_csv, SQRT2

def replay(features, s, theta, weights, law):
    """Constitutive history at new points, without interpolating plastic states."""
    b, h = strain_matrix(features, s, theta)
    old = np.zeros((len(s), 4))
    snapshots, work = ([], np.zeros(len(s)))
    minimum, axial, violation = (0.0, 0.0, 0.0)
    s0 = np.array([-15.0, -9.0, -10.95, 0.0])
    inverse = np.linalg.inv(law.c)
    for coefficient in weights:
        strain = np.asarray(b @ coefficient).reshape(-1, 4)
        stress, dep, _ = law.update(s0 + (strain - old) @ law.c.T)
        old += dep
        increment = np.sum(stress * dep, axis=1)
        work += increment
        minimum = min(minimum, float(increment.min()))
        axial = max(axial, float(np.abs(((stress - s0) @ inverse.T + old)[:, 2]).max()))
        violation = max(violation, float(np.maximum(law.yield_value(stress), 0.0).max()))
        snapshots.append(dict(stress=stress.copy(), plastic=old.copy(), work=work.copy()))
    return (snapshots, b, h, dict(minimum_incremental_dissipation_MPa=minimum, axial_total_strain_error=axial, yield_violation_MPa=violation))

def p1_integrate(theta, values, weights, nt):
    cell = np.floor(theta / (2 * np.pi) * nt).astype(int) % nt
    xi = theta / (2 * np.pi) * nt - np.floor(theta / (2 * np.pi) * nt)
    result = np.zeros((nt, 2))
    np.add.at(result, cell, (1 - xi)[:, None] * values * weights[:, None])
    np.add.at(result, (cell + 1) % nt, xi[:, None] * values * weights[:, None])
    return result.ravel()

def boundary_check(model, weights, releases, installation):
    nt = len(model.boundary_theta)
    g, w = np.polynomial.legendre.leggauss(4)
    theta = ((np.arange(nt)[:, None] + (g + 1) / 2) * 2 * np.pi / nt).ravel()
    dtheta = np.tile(w * np.pi / nt, nt)
    s = np.r_[np.zeros(len(theta)), np.full(len(theta), np.log(model.features.rho))]
    angles = np.tile(theta, 2)
    states, _, h, checks = replay(model.features, s, angles, weights, model.law)
    rows = []
    for k, (coefficient, release, state) in enumerate(zip(weights, releases, states)):
        split = len(theta)
        for name, radius, slc, trace in (('wall', 1.0, slice(0, split), model.wall), ('exterior', model.features.rho, slice(split, None), model.outer)):
            normal, jac = lined.SMAP.wall_normal(theta, radius)
            normals = np.c_[normal.real, normal.imag] * (1 if name == 'exterior' else -1)
            stress = state['stress'][slc] - model.s0
            traction = np.c_[stress[:, 0] * normals[:, 0] + stress[:, 3] / SQRT2 * normals[:, 1], stress[:, 3] / SQRT2 * normals[:, 0] + stress[:, 1] * normals[:, 1]]
            force = p1_integrate(theta, traction, dtheta * jac, nt)
            nodal = np.asarray(trace @ coefficient)
            if name == 'exterior':
                target = model.work @ nodal
            else:
                target = release * model.wall_load
                if release > installation:
                    install_index = np.flatnonzero(np.isclose(releases, installation))[0]
                    target -= model.ring_wall @ (model.wall @ (coefficient - weights[install_index]))
            cell = np.floor(theta / (2 * np.pi) * nt).astype(int) % nt
            xi = theta / (2 * np.pi) * nt - np.floor(theta / (2 * np.pi) * nt)
            nodal = nodal.reshape(-1, 2)
            p1 = (1 - xi[:, None]) * nodal[cell] + xi[:, None] * nodal[(cell + 1) % nt]
            smooth = np.c_[h[slc] @ coefficient[:model.features.size], h[slc] @ coefficient[model.features.size:]]
            gap = smooth - p1
            eqp = np.sqrt(2 / 3 * np.sum((state['plastic'][slc] @ plastic_law.DEV) ** 2, axis=1))
            rows.append(dict(release=float(release), boundary=name, traction_force_relative=float(np.linalg.norm(force - target) / np.linalg.norm(target)), traction_force_max_MN_per_m=float(np.abs(force - target).max()), displacement_interpolation_max_mm=float(np.linalg.norm(gap, axis=1).max() * 1000), yield_max_MPa=float(model.law.yield_value(state['stress'][slc]).max()), plastic_strain_max=float(eqp.max())))
    return (rows, checks)

def check(args):
    folder = Path(args.run)
    config = json.loads((folder / 'config.json').read_text())
    data = np.load(folder / 'checkpoint.npz')
    weights, releases, installed = (data['weights'], data['release'], data['installed'])
    model = NeuralSection(config['radial'], config['angular'], config['boundary'], config['quadrature'], config['rho'], config['width'], config['padding'])
    features = model.features
    s, theta, weight = quadrature(config['radial'], config['angular'], config['quadrature'] + 1, config['rho'])
    states, b, h, summary = replay(features, s, theta, weights, model.law)
    integration = []
    for release, coefficient, state in zip(releases, weights, states):
        residual = b.T @ ((state['stress'] - model.s0) * weight[:, None]).ravel()
        residual += model.exterior @ coefficient - release * model.load
        if release > config['install']:
            residual += model.lining @ (coefficient - installed)
        eqp = np.sqrt(2 / 3 * np.sum((state['plastic'] @ plastic_law.DEV) ** 2, axis=1))
        integration.append(dict(release=float(release), independent_weak_residual_relative=float(np.linalg.norm(residual) / np.linalg.norm(model.load)), plastic_work_kJ_per_m=float(state['work'] @ weight * 1000), plastic_area_m2=float(weight[eqp > 1e-10].sum()), plastic_rho_max=float(np.exp(s[eqp > 1e-10]).max()) if np.any(eqp > 1e-10) else 1.0))
    write_csv(folder / 'independent_integration.csv', integration)
    boundary, bchecks = boundary_check(model, weights, releases, config['install'])
    write_csv(folder / 'boundary_checks.csv', boundary)
    summary['independent_weak_residual_max'] = max((r['independent_weak_residual_relative'] for r in integration))
    summary['independent_quadrature_points'] = len(s)
    summary['outer_yield_max_MPa'] = max((r['yield_max_MPa'] for r in boundary if r['boundary'] == 'exterior'))
    summary['outer_plastic_strain_max'] = max((r['plastic_strain_max'] for r in boundary if r['boundary'] == 'exterior'))
    summary['wall_traction_relative_final'] = boundary[-2]['traction_force_relative']
    summary['exterior_traction_relative_final'] = boundary[-1]['traction_force_relative']
    summary['wall_trace_interpolation_max_mm'] = max((r['displacement_interpolation_max_mm'] for r in boundary if r['boundary'] == 'wall'))
    summary['outer_trace_interpolation_max_mm'] = max((r['displacement_interpolation_max_mm'] for r in boundary if r['boundary'] == 'exterior'))
    install_index = np.flatnonzero(np.isclose(releases, config['install']))[0]
    summary['installation_displacement_jump_m'] = float(np.abs(model.wall @ (weights[install_index] - installed)).max())
    summary['installation_lining_force_MN_per_m'] = float(np.abs(model.ring_wall @ (model.wall @ (weights[install_index] - installed))).max())
    native, _, _, _ = replay(features, data['s'], data['theta'], weights, model.law)
    summary['history_replay_plastic_error'] = float(np.abs(native[-1]['plastic'] - data['plastic']).max())
    summary['history_replay_stress_error_MPa'] = float(np.abs(native[-1]['stress'] - data['stress']).max())
    previous = native[-2]['plastic']
    r, k, _, _ = model.evaluate(weights[-1], previous, 1.0, installed)
    rng = np.random.default_rng(72038)
    direction = rng.normal(size=model.ndof)
    direction *= 1e-08 / np.max(np.abs(model.b @ direction))
    plus = model.evaluate(weights[-1] + direction, previous, 1.0, installed, False)[0]
    minus = model.evaluate(weights[-1] - direction, previous, 1.0, installed, False)[0]
    summary['residual_tangent_relative_error'] = float(np.linalg.norm((plus - minus) / 2 - k @ direction) / np.linalg.norm(k @ direction))
    points = qmc.Sobol(2, scramble=True, seed=72038).random_base2(10)
    ps = (0.002 + 0.996 * points[:, 0]) * np.log(config['rho'])
    pt = points[:, 1] * 2 * np.pi
    eps = 1e-05
    qs = np.concatenate([ps + eps, ps - eps, ps, ps])
    qt = np.concatenate([pt, pt, pt + eps, pt - eps])
    probes, _, _, _ = replay(features, qs, qt, weights, model.law)
    a, bminus, c, d = np.split(probes[-1]['stress'], 4)
    ds, dt = ((a - bminus) / (2 * eps), (c - d) / (2 * eps))
    zeta = np.exp(ps + 1j * pt)
    v = zeta * lined.SMAP.domega(zeta)
    dx = (ds * v.real[:, None] - dt * v.imag[:, None]) / np.abs(v[:, None]) ** 2
    dy = (ds * v.imag[:, None] + dt * v.real[:, None]) / np.abs(v[:, None]) ** 2
    divergence = np.c_[dx[:, 0] + dy[:, 3] / SQRT2, dx[:, 3] / SQRT2 + dy[:, 1]]
    summary['strong_equilibrium_rms_MPa_per_m'] = float(np.sqrt(np.mean(np.sum(divergence ** 2, axis=1))))
    summary['strong_equilibrium_p95_MPa_per_m'] = float(np.quantile(np.linalg.norm(divergence, axis=1), 0.95))
    hs = features.evaluate(ps + eps, pt)[0] - features.evaluate(ps - eps, pt)[0]
    ht = features.evaluate(ps, pt + eps)[0] - features.evaluate(ps, pt - eps)[0]
    value, analytic_s, analytic_t = features.evaluate(ps, pt)
    summary['feature_derivative_relative_error'] = max(float(sp.linalg.norm(hs / (2 * eps) - analytic_s) / sp.linalg.norm(analytic_s)), float(sp.linalg.norm(ht / (2 * eps) - analytic_t) / sp.linalg.norm(analytic_t)))
    summary['constant_feature_error'] = float(np.abs(np.asarray(value.sum(1)) - 1).max())
    import importlib.util
    spec = importlib.util.spec_from_file_location('lining_capacity', ROOT / 'code/figure_exports/lining_capacity.py')
    capacity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capacity)
    wall = model.wall @ (weights[-1] - installed)
    lining, profiles = capacity.recover(model.ring, np.r_[model.ring_transfer @ wall, wall], {})
    summary.update(lining)
    write_csv(folder / 'lining_profiles.csv', profiles)
    (folder / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        check(args)
if __name__ == '__main__':
    main()
