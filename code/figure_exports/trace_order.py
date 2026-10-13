from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/closure'))
from closure_env import bridge_pinned
bridge_pinned()
sys.path.insert(0, str(ROOT / 'code/closure/numeric'))
sys.path.insert(0, str(ROOT / 'code/figure_exports'))
from exec20_src.material import material_F
from exec26_src import protocol as P
from exec26_src.geometry import horseshoe_map
from exec26_src.fe_mapped import theta_ring_mass
from exec26_src.fe_mapped import MappedMesh, AnisoAnnulusFE, mapped_released_load, solve_condensed
from modal_condensation import nodal_from_modal
from modal_condensation import condense_with
from p21_exact_evaluation import assemble_exact
from near_field_clouds import RingReference
SMAP = horseshoe_map(1.0, P.R_HS, P.C_HS)
LOAD = np.array([-15000000.0, -13500000.0, 0.0])

def lagrange(p, x):
    """Values and derivatives at x for equispaced nodes on [0, 1]."""
    x = np.atleast_1d(x)
    nodes = np.linspace(0.0, 1.0, p + 1)
    values, derivatives = ([], [])
    for i in range(p + 1):
        polynomial = np.poly1d([1.0])
        for j in range(p + 1):
            if j != i:
                polynomial *= np.poly1d([1.0, -nodes[j]]) / (nodes[i] - nodes[j])
        values.append(polynomial(x))
        derivatives.append(np.polyder(polynomial)(x))
    return (np.array(values).T, np.array(derivatives).T)

def harmonics(theta, n_h):
    th = np.asarray(theta)
    modes = th[:, None] * np.arange(1, n_h + 1)
    e = np.zeros((len(th), 2, 4 * n_h))
    e[:, 0, 0::4], e[:, 0, 1::4] = (np.cos(modes), np.sin(modes))
    e[:, 1, 2::4], e[:, 1, 3::4] = (np.cos(modes), np.sin(modes))
    return e

class CurvedFE:

    def __init__(self, mat, rho, nr, nt, p, nq=8):
        self.mat, self.rho, self.nr, self.nt, self.p = (mat, rho, nr, nt, p)
        self.ns, self.nth = (nr * p + 1, nt * p)
        self.ndof = 2 * self.ns * self.nth
        self.ds, self.dt = (np.log(rho) / nr, 2 * np.pi / nt)
        self.s0 = np.repeat(np.arange(nr) * self.ds, nt)
        self.t0 = np.tile(np.arange(nt) * self.dt, nr)
        radial = np.repeat(np.arange(nr), nt) * p
        angular = np.tile(np.arange(nt), nr) * p
        self.conn = np.array([(radial + i) * self.nth + (angular + j) % self.nth for i in range(p + 1) for j in range(p + 1)]).T
        self.dofs = (2 * self.conn[:, :, None] + np.arange(2)).reshape(len(self.conn), -1)
        self.outer = np.arange(self.ndof - 2 * self.nth, self.ndof)
        q, w = np.polynomial.legendre.leggauss(nq)
        q, w = ((q + 1) / 2, w / 2)
        val, der = lagrange(p, q)
        ke = np.zeros((len(self.conn), self.dofs.shape[1], self.dofs.shape[1]))
        for i, a in enumerate(q):
            for j, b in enumerate(q):
                ns = np.outer(der[i] / self.ds, val[j]).ravel()
                ntv = np.outer(val[i], der[j] / self.dt).ravel()
                zeta = np.exp(self.s0 + a * self.ds + 1j * (self.t0 + b * self.dt))
                dz = zeta * SMAP.domega(zeta)
                jac = abs(dz) ** 2
                gx = (dz.real[:, None] * ns - dz.imag[:, None] * ntv) / jac[:, None]
                gy = (dz.imag[:, None] * ns + dz.real[:, None] * ntv) / jac[:, None]
                bm = np.zeros((len(self.conn), 3, self.dofs.shape[1]))
                bm[:, 0, 0::2], bm[:, 1, 1::2] = (gx, gy)
                bm[:, 2, 0::2], bm[:, 2, 1::2] = (gy, gx)
                ke += np.einsum('eia,ij,ejb->eab', bm, mat.D(), bm) * (jac * self.ds * self.dt * w[i] * w[j])[:, None, None]
        rows = np.repeat(self.dofs, self.dofs.shape[1], axis=1).ravel()
        cols = np.tile(self.dofs, (1, self.dofs.shape[1])).ravel()
        self.k = sp.coo_matrix((ke.ravel(), (rows, cols)), shape=(self.ndof, self.ndof)).tocsr()
        self.f = np.zeros(self.ndof)
        for j in range(nt):
            theta = (j + q) * self.dt
            normal, jac = SMAP.wall_normal(theta)
            density = np.column_stack([LOAD[0] * normal.real + LOAD[2] * normal.imag, LOAD[2] * normal.real + LOAD[1] * normal.imag]) * jac[:, None]
            forces = val.T @ (density * (w * self.dt)[:, None])
            nodes = (j * p + np.arange(p + 1)) % self.nth
            np.add.at(self.f, (2 * nodes[:, None] + np.arange(2)).ravel(), forces.ravel())

    def projection(self, n_h):
        """Real Fourier coefficients of every periodic Pp shape function."""
        nq = max(16, int(2 * n_h / self.nt) + 8)
        q, w = np.polynomial.legendre.leggauss(nq)
        q, w = ((q + 1) / 2, w / 2)
        val, _ = lagrange(self.p, q)
        transform = np.zeros((4 * n_h, 2 * self.nth))
        for j in range(self.nt):
            e = harmonics((j + q) * self.dt, n_h)
            block = np.einsum('qcm,qa,q->mac', e, val, w * self.dt / np.pi)
            nodes = (j * self.p + np.arange(self.p + 1)) % self.nth
            for a, node in enumerate(nodes):
                transform[:, 2 * node:2 * node + 2] += block[:, a, :]
        return transform

    def solve(self, modal, pairing):
        n_h = len(modal) // 4
        transform = self.projection(n_h)
        if pairing == 'galerkin':
            exterior = np.pi * transform.T @ modal @ transform
        elif pairing == 'original':
            if self.p != 1:
                raise ValueError('The original pairing is P1 only')
            band = self.nth // 2 - 1
            density = nodal_from_modal(modal[:4 * band, :4 * band], np.arange(self.nth) * self.dt, band)
            exterior = np.kron(theta_ring_mass(self.nth), np.eye(2)) @ density
        else:
            raise ValueError(pairing)
        row = np.repeat(self.outer, len(self.outer))
        col = np.tile(self.outer, len(self.outer))
        ks = self.k - sp.coo_matrix((exterior.ravel(), (row, col)), shape=self.k.shape).tocsr()
        constraint = np.tile(np.eye(2), (self.ndof // 2, 1)) / np.sqrt(self.ndof / 2)
        c = sp.csc_matrix(constraint)
        system = sp.bmat([[ks, c], [c.T, None]], format='csc')
        u = sla.splu(system).solve(np.r_[self.f, 0.0, 0.0])[:self.ndof]
        return (u, self.trace_coefficients(u[self.outer], transform, pairing))

    def trace_coefficients(self, outer_u, transform, pairing):
        if pairing == 'original':
            band = self.nth // 2 - 1
            coefficients = np.zeros(len(transform))
            e = harmonics(np.arange(self.nth) * self.dt, band).reshape(2 * self.nth, -1)
            coefficients[:4 * band] = 2 / self.nth * e.T @ outer_u
        else:
            coefficients = transform @ outer_u
        return coefficients

    def evaluate(self, u, s, theta):
        s, theta = np.broadcast_arrays(s, theta)
        s, theta = (s.ravel(), np.mod(theta.ravel(), 2 * np.pi))
        ir = np.minimum((s / self.ds).astype(int), self.nr - 1)
        it = np.minimum((theta / self.dt).astype(int), self.nt - 1)
        vr, _ = lagrange(self.p, (s - ir * self.ds) / self.ds)
        vt, _ = lagrange(self.p, (theta - it * self.dt) / self.dt)
        shape = (vr[:, :, None] * vt[:, None, :]).reshape(len(s), -1)
        return np.einsum('qa,qac->qc', shape, u.reshape(-1, 2)[self.conn[ir * self.nt + it]])

def reference_values(ref, s, theta):
    z = SMAP.omega(np.exp(np.asarray(s) + 1j * np.asarray(theta)))
    return np.column_stack(ref.displacements(z.real, z.imag))

def study_case(ratio, phi, rho, ref, modal, n_h, meshes):
    mat = material_F(ratio, phi, 'trace_order')
    theta = 2 * np.pi * (np.arange(512) + 0.37) / 512
    sample_s, sample_t = np.meshgrid(np.linspace(0.0, np.log(rho), 9), theta[::4], indexing='ij')
    reference_u = reference_values(ref, sample_s.ravel(), sample_t.ravel())
    reference_u -= reference_u.mean(axis=0)
    z = SMAP.boundary(theta, rho)
    normal, jac = SMAP.wall_normal(theta, rho)
    sx, sy, txy = ref.stresses_perturbation(z.real, z.imag)
    traction_ref = np.column_stack([sx * normal.real + txy * normal.imag, txy * normal.real + sy * normal.imag])
    e = harmonics(theta, n_h)

    def recover_traction(coefficients, pairing, interface_nodes):
        density_coefficients = modal @ coefficients
        if pairing == 'original':
            density_coefficients[4 * (interface_nodes // 2 - 1):] = 0.0
        return np.einsum('qcm,m->qc', e, density_coefficients) / jac[:, None]
    wall_theta = 2 * np.pi * (np.arange(1024) + 0.37) / 1024
    normal_wall, jac_wall = SMAP.wall_normal(wall_theta)
    wall_load = np.column_stack([LOAD[0] * normal_wall.real, LOAD[1] * normal_wall.imag]) * jac_wall[:, None]
    reference_work = 2 * np.pi * np.mean(np.sum(reference_values(ref, 0.0, wall_theta) * wall_load, axis=1))
    wall_angle = np.abs(np.degrees(np.angle(SMAP.boundary(theta))))
    foot = (wall_angle >= 105.0) & (wall_angle <= 120.0)
    rows, profiles = ([], [])
    for nr, nt in meshes:
        for p in (1, 2, 3):
            fe = CurvedFE(mat, rho, nr, nt, p)
            reference_outer = reference_values(ref, np.log(rho), np.arange(fe.nth) * 2 * np.pi / fe.nth).ravel()
            transform = fe.projection(n_h)
            for pairing in ('original', 'galerkin') if p == 1 else ('galerkin',):
                u, coefficients = fe.solve(modal, pairing)
                values = fe.evaluate(u, sample_s, sample_t)
                values -= values.mean(axis=0)
                traction = recover_traction(coefficients, pairing, fe.nth)
                diff = traction - traction_ref
                relative_traction = np.sqrt(np.sum(jac[:, None] * diff ** 2) / np.sum(jac[:, None] * traction_ref ** 2))
                interpolation_coefficients = fe.trace_coefficients(reference_outer, transform, pairing)
                interpolated_traction = recover_traction(interpolation_coefficients, pairing, fe.nth)
                interpolation_difference = interpolated_traction - traction_ref
                work = float(u @ fe.f)
                row = dict(ratio=ratio, phi_deg=phi, rho=rho, n_r=nr, n_t=nt, order=p, interface_nodes=fe.nth, pairing=pairing, n_h=n_h, displacement_error=np.linalg.norm(values - reference_u) / np.linalg.norm(reference_u), traction_error=relative_traction, work_J_per_m=work, interpolated_trace_traction_error=np.sqrt(np.sum(jac[:, None] * interpolation_difference ** 2) / np.sum(jac[:, None] * traction_ref ** 2)), foot_sector_traction_error=np.sqrt(np.sum(jac[foot, None] * diff[foot] ** 2) / np.sum(jac[foot, None] * traction_ref[foot] ** 2)), reference_work_J_per_m=reference_work, work_relative_difference=(work - reference_work) / reference_work, traction_peak_error_MPa=np.max(np.linalg.norm(diff, axis=1)) / 1000000.0, traction_error_peak_theta_deg=np.degrees(theta[np.argmax(np.linalg.norm(diff, axis=1))]), q4_to_curved_displacement_difference='', q4_work_relative_difference='')
                if pairing == 'original':
                    q4 = AnisoAnnulusFE(MappedMesh(SMAP, rho, nr, nt), mat)
                    band = nt // 2 - 1
                    density = nodal_from_modal(modal[:4 * band, :4 * band], q4.mesh.theta_nodes, band)
                    q4_load = mapped_released_load(q4.mesh, *LOAD)
                    q4_u = solve_condensed(q4, condense_with(q4, density), q4_load)
                    q4_u = q4_u.reshape(-1, 2)
                    q4_u -= q4_u.mean(axis=0)
                    row['q4_to_curved_displacement_difference'] = np.linalg.norm(q4_u - u.reshape(-1, 2)) / np.linalg.norm(u)
                    row['q4_work_relative_difference'] = (q4_u.ravel() @ q4_load - reference_work) / reference_work
                cc = fe.evaluate(u, np.zeros(4), np.arange(4) * np.pi / 2)
                row.update(vertical_closure_mm=(cc[2, 0] - cc[0, 0]) * 1000, horizontal_closure_mm=(cc[3, 1] - cc[1, 1]) * 1000)
                rows.append(row)
                for j, th in enumerate(theta):
                    profiles.append(dict(ratio=ratio, phi_deg=phi, rho=rho, n_r=nr, n_t=nt, order=p, pairing=pairing, n_h=n_h, theta_deg=np.degrees(th), tx_MPa=traction[j, 0] / 1000000.0, ty_MPa=traction[j, 1] / 1000000.0, reference_tx_MPa=traction_ref[j, 0] / 1000000.0, reference_ty_MPa=traction_ref[j, 1] / 1000000.0))
                print({k: row[k] for k in ('ratio', 'rho', 'n_t', 'order', 'pairing', 'displacement_error', 'traction_error', 'work_relative_difference')}, flush=True)
    return (rows, profiles)

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT / 'results/data/derived/trace_order.csv')
    ap.add_argument('--case', choices=('central', 'tight', 'strong', 'all'), default='all')
    ap.add_argument('--harmonics', type=int, default=127)
    ap.add_argument('--fine-only', action='store_true')
    args = ap.parse_args()
    cases = {'central': (2.625, 45.0, 2.0), 'tight': (2.625, 45.0, 1.25), 'strong': (4.0, 0.0, 1.25)}
    rows, profiles, refs = ([], [], {})
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        for name, (ratio, phi, rho) in cases.items():
            if args.case not in ('all', name):
                continue
            key = (ratio, phi)
            mat = material_F(ratio, phi, 'trace_order')
            if key not in refs:
                print('Reference', key, flush=True)
                refs[key] = RingReference(mat, SMAP, *LOAD)
            print('Exterior', name, args.harmonics, flush=True)
            operator = assemble_exact(mat, SMAP.scaled(rho), args.harmonics, args.harmonics + 32, eta_override=0.97)
            new, prof = study_case(ratio, phi, rho, refs[key], operator['S'], args.harmonics, [(12, 64)] if args.fine_only else [(6, 32), (12, 64)])
            for row in new:
                row.update(cauchy_test_radius=0.97, cauchy_resultant_harmonics=args.harmonics + 32, operator_reciprocity_error=np.linalg.norm(operator['S'] - operator['S'].T) / np.linalg.norm(operator['S']))
            rows.extend(new)
            profiles.extend(prof)
            write_csv(args.output, rows)
            write_csv(args.output.with_name(args.output.stem + '_profiles.csv'), profiles)
    print('Wrote', args.output, 'in', time.perf_counter() - start, 'seconds', flush=True)
if __name__ == '__main__':
    main()
