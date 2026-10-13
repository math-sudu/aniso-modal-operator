from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/figure_exports'))
import lined_section_demo as lined
from exec20_src.material import compliance_voigt_principal, _voigt_to_tensor, _rotate_z, _tensor_to_voigt
IDENTITY = np.array([1.0, 1.0, 1.0, 0.0])
DEV = np.eye(4) - np.outer(IDENTITY, IDENTITY) / 3
SQRT2 = np.sqrt(2.0)

def stiffness_3d(mat):
    a = _tensor_to_voigt(_rotate_z(_voigt_to_tensor(compliance_voigt_principal(mat.Eh, mat.Ev, mat.nu_vh, mat.nu_hh, mat.Gvh)), mat.dip))
    c = np.linalg.inv(a)[np.ix_([0, 1, 2, 5], [0, 1, 2, 5])] / 1000000.0
    m = np.diag([1.0, 1.0, 1.0, SQRT2])
    return m @ c @ m

def q_derivatives(stress):
    s = stress @ DEV
    q = np.sqrt(1.5 * np.sum(s * s, axis=-1))
    if np.any(q <= 1e-12):
        raise ValueError('Drucker-Prager cone apex requires a separate return')
    n = 1.5 * s / q[..., None]
    h = (1.5 * DEV - n[..., :, None] * n[..., None, :]) / q[..., None, None]
    return (q, n, h)

class DruckerPrager:

    def __init__(self, cmat, cohesion=10.0, friction=30.0, dilation=0.0):
        self.c = cmat
        phi, psi = np.deg2rad([friction, dilation])
        self.alpha = 2 * np.sin(phi) / (3 - np.sin(phi))
        self.beta = 2 * np.sin(psi) / (3 - np.sin(psi))
        self.intercept = 6 * cohesion * np.cos(phi) / (3 - np.sin(phi))

    def yield_value(self, stress):
        s = stress @ DEV
        return np.sqrt(1.5 * np.sum(s * s, axis=-1)) + self.alpha * (stress @ IDENTITY) - self.intercept

    def update(self, trial):
        shape = trial.shape
        stress = trial.reshape(-1, 4).copy()
        dep = np.zeros_like(stress)
        tangent = np.broadcast_to(self.c, (len(stress), 4, 4)).copy()
        active = np.flatnonzero(self.yield_value(stress) > 1e-09)
        self.last_diagnostics = dict(return_residual_MPa=0.0, active_yield_residual_MPa=0.0, yield_violation_MPa=float(np.maximum(self.yield_value(stress), 0.0).max()), minimum_plastic_multiplier=0.0, active_return_points=len(active), return_iterations=0)
        if not len(active):
            return (stress.reshape(shape), dep.reshape(shape), tangent.reshape((*shape[:-1], 4, 4)))
        t = stress[active].copy()
        s = t.copy()
        lam = np.zeros(len(s))
        for iteration in range(30):
            q, n, h = q_derivatives(s)
            flow = n + self.beta * IDENTITY
            normal = n + self.alpha * IDENTITY
            cg = flow @ self.c.T
            residual = np.column_stack((s - t + lam[:, None] * cg, q + self.alpha * (s @ IDENTITY) - self.intercept))
            jac = np.zeros((len(s), 5, 5))
            jac[:, :4, :4] = np.eye(4) + lam[:, None, None] * np.einsum('ab,nbc->nac', self.c, h)
            jac[:, :4, 4], jac[:, 4, :4] = (cg, normal)
            if np.max(np.abs(residual)) < 1e-09:
                break
            delta = np.linalg.solve(jac, -residual[..., None])[..., 0]
            s += delta[:, :4]
            lam += delta[:, 4]
        else:
            raise RuntimeError('Drucker-Prager local return did not converge')
        if np.min(lam) < -1e-12:
            raise RuntimeError('Negative plastic multiplier')
        rhs = np.zeros((len(s), 5, 4))
        rhs[:, :4, :] = self.c
        tangent[active] = np.linalg.solve(jac, rhs)[:, :4, :]
        stress[active], dep[active] = (s, lam[:, None] * flow)
        self.last_diagnostics = dict(return_residual_MPa=float(np.abs(residual).max()), active_yield_residual_MPa=float(np.abs(self.yield_value(s)).max()), yield_violation_MPa=float(np.maximum(self.yield_value(stress), 0.0).max()), minimum_plastic_multiplier=float(lam.min()), active_return_points=len(active), return_iterations=iteration)
        return (stress.reshape(shape), dep.reshape(shape), tangent.reshape((*shape[:-1], 4, 4)))

class CoupledSection:

    def __init__(self, nr, nt, rho, cohesion, exterior_system=None):
        self.cohesion = cohesion
        self.mat = lined.material_F(4.0, 0.0)
        self.mesh = lined.MappedMesh(lined.SMAP, rho, nr, nt)
        self.fe = lined.AnisoAnnulusFE(self.mesh, self.mat)
        assemble = lined.condensed_mapped_system if exterior_system is None else exterior_system
        elastic, _ = assemble(self.fe, self.mat, lined.SMAP, rho, n_modes=lined.P.N_MODES_GENERAL)
        self.exterior = (elastic - self.fe.K).tocsr() / 1000000.0
        self.elastic = elastic / 1000000.0
        self.law = DruckerPrager(stiffness_3d(self.mat), cohesion)
        self.s0 = np.array([-15.0, -9.0, -10.95, 0.0])
        if self.law.yield_value(self.s0) >= 0:
            raise ValueError('Initial stress is outside the chosen yield surface')
        self.load = lined.mapped_released_load(self.mesh, *lined.insitu(0.6)) / 1000000.0
        self.edof = np.empty((len(self.mesh.elems), 8), dtype=int)
        self.edof[:, 0::2] = 2 * self.mesh.elems
        self.edof[:, 1::2] = 2 * self.mesh.elems + 1
        self.b, self.weight, self.xy, self.rho_gp = ([], [], [], [])
        for xi in lined._G2:
            for eta in lined._G2:
                b2, w = lined._bmat_all(self.mesh, xi, eta)
                b = np.zeros((len(b2), 4, 8))
                b[:, [0, 1, 3], :] = b2
                b[:, 3, :] /= SQRT2
                self.b.append(b)
                self.weight.append(w)
                n = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
                self.xy.append(np.einsum('a,eai->ei', n, self.mesh.xy[self.mesh.elems]))
                self.rho_gp.append(self.mesh.r_node[self.mesh.elems] @ n)
        self.b = np.stack(self.b, axis=1)
        self.weight = np.stack(self.weight, axis=1)
        self.xy = np.stack(self.xy, axis=1)
        self.rho_gp = np.stack(self.rho_gp, axis=1)
        self.rows = np.repeat(self.edof, 8, axis=1).ravel()
        self.cols = np.tile(self.edof, (1, 8)).ravel()
        self.wall = self.mesh.ring_dofs('inner')
        self.ring = lined.LiningRing(lined.SMAP, self.mesh.theta_nodes, lined.LINING_T, lined.LINING_NL)
        kr = lined.lining_stiffness(self.ring, lined.E_LIN, lined.NU_LIN) / 1000000.0
        split = 2 * self.ring.n_l * nt
        self.ring_transfer = -sla.splu(kr[:split, :split].tocsc()).solve(kr[:split, split:].toarray())
        self.ring_wall = kr[split:, split:].toarray() + kr[split:, :split] @ self.ring_transfer
        self.lining = sp.coo_matrix((self.ring_wall.ravel(), (np.repeat(self.wall, len(self.wall)), np.tile(self.wall, len(self.wall)))), shape=self.elastic.shape).tocsr()
        c = np.zeros((self.mesh.n_dof, 2))
        c[0::2, 0], c[1::2, 1] = (1.0, 1.0)
        c /= np.sqrt(self.mesh.n_nodes)
        self.constraint = sp.csc_matrix(c * np.abs(self.elastic.diagonal()).mean())
        _, assembled, _, _ = self.evaluate(np.zeros(self.mesh.n_dof), np.zeros((*self.weight.shape, 4)), 0.0, None)
        gap = assembled - self.elastic
        if np.max(np.abs(gap.data), initial=0.0) > 1e-10 * np.max(np.abs(self.elastic.data)):
            raise AssertionError('3D integration disagrees with the original elastic FE block')

    def strain(self, u):
        return np.einsum('egai,ei->ega', self.b, u[self.edof])

    def linear_solve(self, matrix, load):
        c = self.constraint
        ka = sp.bmat([[matrix, c], [c.T, None]], format='csc')
        return sla.splu(ka).solve(np.concatenate([load, np.zeros(2)]))[:self.mesh.n_dof]

    def evaluate(self, u, plastic_old, release, installed_at):
        trial = self.s0 + (self.strain(u) - plastic_old) @ self.law.c.T
        stress, dep, tangent = self.law.update(trial)
        ef = np.einsum('egai,ega,eg->ei', self.b, stress - self.s0, self.weight)
        residual = np.bincount(self.edof.ravel(), weights=ef.ravel(), minlength=self.mesh.n_dof)
        residual += self.exterior @ u - release * self.load
        ek = np.einsum('egai,egab,egbj,eg->eij', self.b, tangent, self.b, self.weight, optimize=True)
        k = sp.coo_matrix((ek.ravel(), (self.rows, self.cols)), shape=self.elastic.shape).tocsr() + self.exterior
        if installed_at is not None:
            residual += self.lining @ (u - installed_at)
            k += self.lining
        return (residual, k, stress, plastic_old + dep)

    def solve_step(self, u, plastic_old, release, installed_at):
        u = u.copy()
        norm_load = np.linalg.norm(self.load)
        backtracks = 0
        for iteration in range(35):
            r, k, stress, plastic = self.evaluate(u, plastic_old, release, installed_at)
            error = np.linalg.norm(r) / norm_load
            if error < 1e-08:
                recovered_strain = (stress - self.s0) @ np.linalg.inv(self.law.c).T + plastic
                self.last_diagnostics = dict(equilibrium_relative=float(error), equilibrium_backtracks=backtracks, axial_total_strain_error=float(np.abs(recovered_strain[..., 2]).max()), **self.law.last_diagnostics)
                return (u, plastic, stress, iteration)
            c = self.constraint
            ka = sp.bmat([[k, c], [c.T, None]], format='csc')
            du = sla.splu(ka).solve(-np.concatenate([r, np.asarray(c.T @ u)]))[:len(u)]
            step = 1.0
            for backtrack in range(12):
                candidate = u + step * du
                rr, _, _, _ = self.evaluate(candidate, plastic_old, release, installed_at)
                if np.linalg.norm(rr) < np.linalg.norm(r):
                    u = candidate
                    break
                step *= 0.5
                backtracks += 1
            else:
                raise RuntimeError(f'Equilibrium line search failed at release {release:g}')
        raise RuntimeError(f'Equilibrium did not converge at release {release:g}')

    def row(self, u, plastic, stress, release, stage, installed_at, work, iterations):
        dv, dh = lined.closures(self.mesh, u)
        ep_eq = np.sqrt(2 / 3 * np.sum((plastic @ DEV) ** 2, axis=-1))
        yielded = ep_eq > 1e-10
        zone = self.rho_gp[yielded]
        outer = slice(-self.mesh.n_t, None)
        boundary_stress = []
        for eta in lined._G2:
            b2, _ = lined._bmat_all(self.mesh, 1.0, eta)
            strain = np.zeros((self.mesh.n_t, 4))
            strain[:, [0, 1, 3]] = np.einsum('eai,ei->ea', b2[outer], u[self.edof[outer]])
            strain[:, 3] /= SQRT2
            boundary_stress.append(self.s0 + strain @ self.law.c.T)
        row = dict(n_r=self.mesh.n_r, n_t=self.mesh.n_t, rho=self.mesh.rho_out, cohesion_MPa=self.cohesion, friction_deg=30.0, dilation_deg=0.0, release=release, stage=stage, closure_vertical_mm=dv, closure_horizontal_mm=dh, peak_thrust_MN_per_m=0.0, peak_moment_kNm_per_m=0.0, plastic_area_m2=float(np.sum(self.weight[yielded])), plastic_rho_max=float(zone.max()) if len(zone) else 1.0, max_equivalent_plastic_strain=float(ep_eq.max()), outer_layer_yield_MPa=float(self.law.yield_value(stress[outer]).max()), outer_layer_plastic_strain=float(ep_eq[outer].max()), boundary_elastic_yield_MPa=float(self.law.yield_value(np.concatenate(boundary_stress)).max()), plastic_work_kJ_per_m=work * 1000, newton_iterations=iterations)
        if stage != 'elastic_superposition':
            row.update(self.last_diagnostics)
        if installed_at is not None:
            wall_increment = (u - installed_at)[self.wall]
            ur = np.concatenate([self.ring_transfer @ wall_increment, wall_increment])
            n, m = lined.ring_actions(self.ring, ur, lined.E_LIN, lined.NU_LIN)
            row['peak_thrust_MN_per_m'] = float(np.max(np.abs(n)) / 1000000.0)
            row['peak_moment_kNm_per_m'] = float(np.max(np.abs(m)) / 1000.0)
        return row

def checks():
    young, nu = (20000.0, 0.25)
    bulk, shear = (young / (3 * (1 - 2 * nu)), young / (2 * (1 + nu)))
    c = 2 * shear * DEV + bulk * np.outer(IDENTITY, IDENTITY)
    law = DruckerPrager(c)
    trial = np.array([[-100.0, -5.0, -10.0, 7.0]])
    q, n, _ = q_derivatives(trial)
    dl = law.yield_value(trial) / (3 * shear + 9 * bulk * law.alpha * law.beta)
    expected = trial - dl[:, None] * (n + law.beta * IDENTITY) @ c.T
    stress, dep, _ = law.update(trial)
    np.testing.assert_allclose(stress, expected, atol=1e-09)
    np.testing.assert_allclose(stress + dep @ c.T, trial, atol=1e-09)
    for dip in (0.0, 31.0, 90.0):
        mat = lined.material_F(4.0, dip)
        c = stiffness_3d(mat)
        reduce = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1 / SQRT2]])
        np.testing.assert_allclose(reduce.T @ c @ reduce, mat.D() / 1000000.0, rtol=1e-12, atol=1e-10)
        law = DruckerPrager(c)
        stress, dep, tangent = law.update(trial)
        np.testing.assert_allclose(law.yield_value(stress), 0, atol=1e-09)
        numerical = np.empty((4, 4))
        for j in range(4):
            perturb = 1e-07 * c[:, j]
            plus = law.update(trial + perturb)[0]
            minus = law.update(trial - perturb)[0]
            numerical[:, j] = ((plus - minus) / 2e-07)[0]
        np.testing.assert_allclose(tangent[0], numerical, rtol=2e-06, atol=0.02)
        unloaded = stress - 0.1 * (stress @ DEV)
        np.testing.assert_allclose(law.update(unloaded)[1], 0, atol=1e-12)
    print('Isotropic return, TI plane strain, anisotropic tangent and unloading checks passed.', flush=True)

def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

def coupling_checks():
    model = CoupledSection(6, 32, 2.0, 1000.0)
    u0 = np.zeros(model.mesh.n_dof)
    ep0 = np.zeros((*model.weight.shape, 4))
    u, ep, _, _ = model.solve_step(u0, ep0, 0.82, None)
    final, ep, _, _ = model.solve_step(u, ep, 1.0, u)
    uu = model.linear_solve(model.elastic, model.load)
    ul = model.linear_solve(model.elastic + model.lining, model.load)
    np.testing.assert_allclose(final, 0.82 * uu + 0.18 * ul, atol=1e-12)
    np.testing.assert_allclose(ep, 0.0, atol=1e-12)
    model = CoupledSection(6, 32, 2.0, 5.0)
    u, ep, stress, _ = model.solve_step(u0, ep0, 0.82, None)
    assert np.max(np.abs(ep[..., 2])) > 1e-07
    total_strain = (stress - model.s0) @ np.linalg.inv(model.law.c).T + ep
    np.testing.assert_allclose(total_strain, model.strain(u), atol=1e-12)
    print('Coupled elastic staging and full constitutive plane-strain balance passed.', flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', nargs=2, type=int, default=(24, 128))
    parser.add_argument('--rho', type=float, default=2.0)
    parser.add_argument('--cohesion', type=float, default=5.0)
    parser.add_argument('--step', type=float, default=0.02)
    parser.add_argument('--install', type=float, default=0.82)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/data/derived/localized_plasticity.csv')
    parser.add_argument('--field-output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    checks()
    if args.check_only:
        coupling_checks()
        return
    start = time.perf_counter()
    model = CoupledSection(*args.mesh, args.rho, args.cohesion)
    print(f'Assembled {args.mesh}, rho={args.rho:g}, {time.perf_counter() - start:.1f}s', flush=True)
    u = np.zeros(model.mesh.n_dof)
    plastic = np.zeros((*model.weight.shape, 4))
    stress = np.broadcast_to(model.s0, plastic.shape).copy()
    rows, work = ([], 0.0)
    installed_at = None
    for lo, hi, stage in ((0.0, args.install, 'unsupported'), (args.install, 1.0, 'supported')):
        if stage == 'supported':
            installed_at = u.copy()
        targets = np.linspace(lo, hi, int(np.ceil((hi - lo) / args.step)) + 1)[1:]
        for release in targets:
            old = plastic
            u, plastic, stress, iterations = model.solve_step(u, old, release, installed_at)
            increment = np.sum(stress * (plastic - old), axis=-1)
            if np.min(increment) < -1e-08:
                raise RuntimeError('Negative plastic dissipation')
            work += float(np.sum(increment * model.weight))
            row = model.row(u, plastic, stress, release, stage, installed_at, work, iterations)
            row.update(installation_release=args.install, maximum_load_step=args.step)
            rows.append(row)
            print(f"{stage} a={release:.4f} closure={row['closure_vertical_mm']:.4f}/{row['closure_horizontal_mm']:.4f} plastic_rho={row['plastic_rho_max']:.4f} outer_F={row['outer_layer_yield_MPa']:.3f} iterations={iterations}", flush=True)
    uu = model.linear_solve(model.elastic, model.load)
    ul = model.linear_solve(model.elastic + model.lining, model.load)
    ue = args.install * uu + (1 - args.install) * ul
    elastic_row = model.row(ue, np.zeros_like(plastic), model.s0 + model.strain(ue) @ model.law.c.T, 1.0, 'elastic_superposition', args.install * uu, 0.0, 1)
    elastic_row.update(installation_release=args.install, maximum_load_step=args.step)
    rows.append(elastic_row)
    write_csv(args.output, rows)
    if args.field_output is not None:
        write_fields(args.field_output, model, plastic, stress)
    print(f'Saved {args.output}; {time.perf_counter() - start:.1f} s', flush=True)

def write_fields(path, model, plastic, stress):
    fields = []
    for xy, rho, ep, sig in zip(model.xy.reshape(-1, 2), model.rho_gp.ravel(), plastic.reshape(-1, 4), stress.reshape(-1, 4)):
        fields.append(dict(x_physical_m=-xy[1], y_physical_m=xy[0], rho=rho, equivalent_plastic_strain=float(np.sqrt(2 / 3 * np.sum((ep @ DEV) ** 2))), sigma_raw_xx_MPa=sig[0], sigma_raw_yy_MPa=sig[1], sigma_zz_MPa=sig[2], sigma_raw_xy_MPa=sig[3] / SQRT2))
    write_csv(path, fields)
if __name__ == '__main__':
    main()
