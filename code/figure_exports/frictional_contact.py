from __future__ import annotations
import argparse
import csv
import gc
import sys
import time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/figure_exports'))
import lined_section_demo as lined

def contact_solve(k, f, b, c, mu, previous=None, clearance=None, bonded=False):
    """Solve KU + B.T L = F with closed/stick/slip/open active sets.

    B U is (lining - ground) in interleaved outward-normal/tangent frames.
    L_n is compressive; -L_t is the traction on the lining. Coulomb sliding
    obeys L_t = mu L_n sign(delta relative tangential displacement).
    C fixes only the two common rigid translations in the section problem.
    """
    nd, nc = (len(f), len(b) // 2)
    ng = c.shape[1]
    old = np.zeros(nc) if previous is None else np.asarray(previous)
    gap0 = np.zeros(nc) if clearance is None else np.asarray(clearance)
    scale = float(np.abs(np.diag(k)).mean())
    state = np.ones(nc, dtype=int)
    a = np.zeros((nd + 2 * nc + ng, nd + 2 * nc + ng))
    a[:nd, :nd], a[:nd, nd:nd + 2 * nc] = (k, b.T)
    a[:nd, nd + 2 * nc:], a[nd + 2 * nc:, :nd] = (c, c.T)
    rhs = np.concatenate([f, np.zeros(2 * nc + ng)])

    def residual(x):
        u = x[:nd]
        force = x[nd:nd + 2 * nc].reshape(nc, 2)
        jump = (b @ u).reshape(nc, 2)
        trial = force[:, 0] + scale * (jump[:, 0] - gap0)
        limit = mu * np.maximum(force[:, 0], 0.0)
        local = np.empty_like(force)
        local[:, 0] = force[:, 0] - np.maximum(trial, 0.0)
        local[:, 1] = force[:, 1] - np.where(trial > 0.0, np.clip(force[:, 1] + scale * (jump[:, 1] - old), -limit, limit), 0.0)
        return np.concatenate([k @ u + b.T @ force.ravel() + c @ x[nd + 2 * nc:] - f, local.ravel(), c.T @ u])
    previous_solution = None
    backtracks = 0
    for iteration in range(80):
        a[nd:nd + 2 * nc] = 0.0
        rhs[nd:nd + 2 * nc] = 0.0
        for j, mode in enumerate(state):
            rn, rt = (nd + 2 * j, nd + 2 * j + 1)
            if mode == 0:
                a[rn, rn] = a[rt, rt] = 1.0
            else:
                a[rn, :nd], rhs[rn] = (b[2 * j], gap0[j])
                if mode == 1:
                    a[rt, :nd], rhs[rt] = (b[2 * j + 1], old[j])
                else:
                    a[rt, rt], a[rt, rn] = (1.0, -np.sign(mode) * mu)
        solution = la.solve(a, rhs, assume_a='gen', check_finite=False)
        if previous_solution is not None:
            old_norm = la.norm(residual(previous_solution))
            step = 1.0
            direction = solution - previous_solution
            for _ in range(20):
                candidate = previous_solution + step * direction
                if la.norm(residual(candidate)) <= (1 - 0.0001 * step) * old_norm:
                    solution = candidate
                    break
                step *= 0.5
                backtracks += 1
            else:
                raise RuntimeError('Coulomb semismooth Newton line search failed')
        u, force = (solution[:nd], solution[nd:nd + 2 * nc].reshape(nc, 2))
        jump = (b @ u).reshape(nc, 2)
        if bonded:
            break
        normal_trial = force[:, 0] + scale * (jump[:, 0] - gap0)
        shear_trial = force[:, 1] + scale * (jump[:, 1] - old)
        closed = normal_trial > 0.0
        slip = np.abs(shear_trial) > mu * np.maximum(force[:, 0], 0.0) + 1e-10
        updated = np.where(closed, np.where(slip, 2 * np.sign(shear_trial), 1), 0).astype(int)
        if la.norm(residual(solution)) < 1e-10 * max(1.0, la.norm(f)):
            break
        state = updated
        previous_solution = solution
    else:
        raise RuntimeError('Coulomb contact active set did not converge')
    equilibrium = k @ u + b.T @ force.ravel() + c @ solution[nd + 2 * nc:] - f
    gap = gap0 - jump[:, 0]
    delta = jump[:, 1] - old
    if not bonded:
        defect = max(float(np.max(np.maximum(-force[:, 0], 0.0))), float(scale * np.max(np.maximum(-gap, 0.0))), float(np.max(np.abs(np.minimum(force[:, 0], scale * gap)))), float(np.max(np.maximum(np.abs(force[:, 1]) - mu * force[:, 0], 0.0))))
        if defect > 1e-07 * max(1.0, np.linalg.norm(f)):
            raise RuntimeError(f'Contact admissibility defect {defect:g}')
        if np.min(force[:, 1] * delta) < -1e-10:
            raise RuntimeError('Negative frictional dissipation')
    norm_load = max(1.0, np.linalg.norm(f))
    diagnostics = dict(contact_backtracks=backtracks)
    if not bonded:
        diagnostics.update(contact_residual_relative=float(la.norm(residual(solution)) / norm_load), contact_admissibility_relative=defect / norm_load, normal_force_violation_MN_per_m=float(np.maximum(-force[:, 0], 0.0).max()), penetration_m=float(np.maximum(-gap, 0.0).max()), complementarity_MN_per_m=float(np.abs(np.minimum(force[:, 0], scale * gap)).max()), friction_cone_violation_MN_per_m=float(np.maximum(np.abs(force[:, 1]) - mu * force[:, 0], 0.0).max()), minimum_friction_work_MN=float((force[:, 1] * delta).min()))
    return dict(u=u, force=force, jump=jump, gap=gap, state=state, diagnostics=diagnostics, iterations=iteration + 1, equilibrium=np.linalg.norm(equilibrium) / norm_load)

class ContactSection:

    def __init__(self, nr=24, nt=128, ratio=4.0, dip=45.0, pressure=0.9, exterior_system=None):
        self.ratio, self.dip, self.pressure = (ratio, dip, pressure)
        self.mat = lined.material_F(ratio, dip)
        self.mesh = lined.MappedMesh(lined.SMAP, lined.RHO, nr, nt)
        self.fe = lined.AnisoAnnulusFE(self.mesh, self.mat)
        assemble = lined.condensed_mapped_system if exterior_system is None else exterior_system
        kg, _ = assemble(self.fe, self.mat, lined.SMAP, lined.RHO)
        kg = kg.tocsr() / 1000000.0
        self.fg = lined.mapped_released_load(self.mesh, *lined.insitu(pressure)) / 1000000.0
        nw = 2 * nt
        if np.any(self.fg[nw:]):
            raise ValueError('Ground condensation expects excavation loading at the wall')
        ground_lu = sla.splu(kg[nw:, nw:].tocsc())
        self.ground_transfer = -ground_lu.solve(kg[nw:, :nw].toarray())
        sg = kg[:nw, :nw].toarray() + kg[:nw, nw:] @ self.ground_transfer
        self.ring = lined.LiningRing(lined.SMAP, self.mesh.theta_nodes, lined.LINING_T, lined.LINING_NL)
        kr = lined.lining_stiffness(self.ring, lined.E_LIN, lined.NU_LIN).tocsr() / 1000000.0
        self.ring_split = self.ring.n_dof - nw
        ni = self.ring_split
        self.ring_transfer = -sla.splu(kr[:ni, :ni].tocsc()).solve(kr[:ni, ni:].toarray())
        sr = kr[ni:, ni:].toarray() + kr[ni:, :ni] @ self.ring_transfer
        self.k = la.block_diag(sg, sr)
        self.f = np.concatenate([self.fg[:nw], np.zeros(nw)])
        normal, _ = lined.SMAP.wall_normal(self.mesh.theta_nodes)
        self.b = np.zeros((nw, 2 * nw))
        for j, n in enumerate(normal):
            frame = np.array([[n.real, n.imag], [-n.imag, n.real]])
            self.b[2 * j:2 * j + 2, 2 * j:2 * j + 2] = -frame
            self.b[2 * j:2 * j + 2, nw + 2 * j:nw + 2 * j + 2] = frame
        self.c = np.zeros((2 * nw, 2))
        self.c[:nw:2, 0], self.c[1:nw:2, 1] = (1 / np.sqrt(nt), 1 / np.sqrt(nt))
        self.c *= np.abs(np.diag(self.k)).mean()
        wall = self.mesh.xy[:nt]
        edges = np.linalg.norm(np.roll(wall, -1, axis=0) - wall, axis=1)
        self.length = 0.5 * (edges + np.roll(edges, 1))

    def solve(self, mu, fractions=(1.0,), bonded=False):
        old = None
        for fraction in fractions:
            result = contact_solve(self.k, fraction * self.f, self.b, self.c, mu, previous=old, bonded=bonded)
            old = result['jump'][:, 1].copy()
        nw = 2 * self.mesh.n_t
        ug, ul = (result['u'][:nw], result['u'][nw:])
        result['ground'] = np.concatenate([ug, self.ground_transfer @ ug])
        result['lining'] = np.concatenate([self.ring_transfer @ ul, ul])
        result['thrust'], result['moment'] = lined.ring_actions(self.ring, result['lining'], lined.E_LIN, lined.NU_LIN)
        result['pressure'] = result['force'][:, 0] / self.length
        result['shear'] = result['force'][:, 1] / self.length
        return result

    def records(self, result, law, mu, steps):
        dv, dh = lined.closures(self.mesh, result['ground'])
        n, m = (result['thrust'] / 1000000.0, result['moment'] / 1000.0)
        identity = dict(ratio=self.ratio, material_dip_deg=self.dip, bedding_dip_deg=90 - self.dip, pressure_coefficient=self.pressure, n_r=self.mesh.n_r, n_t=self.mesh.n_t, law=law, mu=mu, steps=steps)
        state = result['state']
        row = dict(**identity, peak_thrust_MN_per_m=float(np.max(np.abs(n))), peak_moment_kNm_per_m=float(np.max(np.abs(m))), closure_vertical_mm=dv, closure_horizontal_mm=dh, minimum_pressure_MPa=float(result['pressure'].min()), maximum_pressure_MPa=float(result['pressure'].max()), slip_length_fraction=float(np.sum(self.length[np.abs(state) == 2]) / self.length.sum()), open_length_fraction=float(np.sum(self.length[state == 0]) / self.length.sum()), maximum_slip_mm=float(np.max(np.abs(result['jump'][:, 1])) * 1000.0), maximum_gap_mm=float(result['gap'].max() * 1000.0), equilibrium_relative=result['equilibrium'], active_set_iterations=result['iterations'])
        row.update(result['diagnostics'])
        angles = lined.station_theta(self.mesh.n_t)
        actions = [dict(**identity, theta_deg=t, thrust_MN_per_m=nn, moment_kNm_per_m=mm) for t, nn, mm in zip(angles, n, m)]
        contact = [dict(**identity, theta_deg=np.degrees(t), pressure_MPa=p, shear_MPa=q, slip_mm=d[1] * 1000.0, gap_mm=g * 1000.0, state=int(s)) for t, p, q, d, g, s in zip(self.mesh.theta_nodes, result['pressure'], result['shear'], result['jump'], result['gap'], state)]
        return (row, actions, contact)

def checks():
    k, b, c = (np.diag([10.0, 5.0]), np.eye(2), np.empty((2, 0)))
    cases = [([2.0, 0.5], [0.0, 0.0], [2.0, 0.5]), ([2.0, 3.0], [0.0, 0.4], [2.0, 1.0]), ([2.0, -3.0], [0.0, -0.4], [2.0, -1.0]), ([-2.0, 3.0], [-0.2, 0.6], [0.0, 0.0])]
    for f, expected_u, expected_force in cases:
        r = contact_solve(k, np.array(f), b, c, 0.5)
        np.testing.assert_allclose(r['u'], expected_u, atol=1e-12)
        np.testing.assert_allclose(r['force'].ravel(), expected_force, atol=1e-12)
    r = contact_solve(k, np.array([2.0, 2.0]), b, c, 0.5, previous=np.array([0.4]))
    np.testing.assert_allclose(r['u'], [0.0, 0.4], atol=1e-12)
    r = contact_solve(k, np.array([2.0, 0.0]), b, c, 0.5, previous=np.array([0.4]))
    np.testing.assert_allclose(r['u'], [0.0, 0.2], atol=1e-12)
    np.testing.assert_allclose(r['force'].ravel(), [2.0, -1.0], atol=1e-12)
    model = ContactSection(6, 32)
    r = model.solve(0.0, bonded=True)
    _, ring, _, ur = lined.merged_operator_solve(model.mat, 6, 32, lined.E_LIN)
    n, m = lined.ring_actions(ring, ur, lined.E_LIN, lined.NU_LIN)
    np.testing.assert_allclose(r['thrust'], n, rtol=1e-08, atol=0.0001)
    np.testing.assert_allclose(r['moment'], m, rtol=1e-08, atol=0.0001)
    high = model.solve(10.0)
    np.testing.assert_allclose(high['thrust'], n, rtol=1e-08, atol=0.0001)
    np.testing.assert_allclose(high['moment'], m, rtol=1e-08, atol=0.0001)
    free = model.solve(0.0)
    _, ring, _, ur = lined.merged_operator_solve(model.mat, 6, 32, lined.E_LIN, contact='slip')
    n, m = lined.ring_actions(ring, ur, lined.E_LIN, lined.NU_LIN)
    np.testing.assert_allclose(free['thrust'], n, rtol=1e-08, atol=0.0001)
    np.testing.assert_allclose(free['moment'], m, rtol=1e-08, atol=0.0001)
    r = model.solve(0.53)
    staged = model.solve(0.53, fractions=(0.25, 0.5, 0.75, 1.0))
    np.testing.assert_allclose(r['thrust'], staged['thrust'], rtol=1e-09, atol=0.0001)
    np.testing.assert_allclose(r['moment'], staged['moment'], rtol=1e-09, atol=0.0001)
    print('Spring-slider opening/reversal, bonded and zero/high-friction limits, and proportional-step checks passed.', flush=True)

def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        fields = list(dict.fromkeys((key for row in rows for key in row)))
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', nargs=2, type=int, default=(24, 128))
    parser.add_argument('--ratios', nargs='+', type=float, default=lined.RATIOS)
    parser.add_argument('--dips', nargs='+', type=float, default=lined.MAT_DIPS)
    parser.add_argument('--mu', nargs='+', type=float, default=(0.0, 0.53, 0.73, 1.15))
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/data/derived/frictional_contact.csv')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        checks()
        return
    rows, actions, contacts = ([], [], [])
    for ratio in args.ratios:
        for dip in args.dips:
            start = time.perf_counter()
            model = ContactSection(*args.mesh, ratio, dip)
            print(f'Assembled ratio={ratio:g}, dip={dip:g}, mesh={args.mesh} in {time.perf_counter() - start:.1f}s', flush=True)
            for law, mu in [('bonded', 0.0)] + [('Coulomb', x) for x in args.mu]:
                result = model.solve(mu, np.linspace(0, 1, args.steps + 1)[1:], bonded=law == 'bonded')
                row, a, c = model.records(result, law, mu, args.steps)
                rows.append(row)
                actions.extend(a)
                contacts.extend(c)
                print(f"  {law} mu={mu:g}: N={row['peak_thrust_MN_per_m']:.4f}, M={row['peak_moment_kNm_per_m']:.3f}, p={row['minimum_pressure_MPa']:.3f}..{row['maximum_pressure_MPa']:.3f}, slip={row['slip_length_fraction']:.3f}, open={row['open_length_fraction']:.3f}", flush=True)
            del model, result
            gc.collect()
    write_csv(args.output, rows)
    write_csv(args.output.with_name(args.output.stem + '_actions.csv'), actions)
    write_csv(args.output.with_name(args.output.stem + '_interface.csv'), contacts)
    print(f'Saved {args.output}', flush=True)
if __name__ == '__main__':
    main()
