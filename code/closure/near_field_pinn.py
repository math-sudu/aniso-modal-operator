from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/figure_exports'))
import lined_section_demo as lined
from exterior_operator import assemble_modal, projected_work
_spec = importlib.util.spec_from_file_location('plastic_law', ROOT / 'code/figure_exports/localized_plasticity.py')
plastic_law = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plastic_law)
SQRT2 = np.sqrt(2.0)

def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

def quadrature(nr, nt, order, rho):
    """Tensor Gaussian integration in log-radius and periodic angle."""
    g, w = np.polynomial.legendre.leggauss(order)
    s = ((np.arange(nr)[:, None] + (g + 1) / 2) / nr * np.log(rho)).ravel()
    t = ((np.arange(nt)[:, None] + (g + 1) / 2) / nt * 2 * np.pi).ravel()
    ws = np.tile(w * np.log(rho) / (2 * nr), nr)
    wt = np.tile(w * np.pi / nt, nt)
    ss, tt = np.meshgrid(s, t, indexing='ij')
    zeta = np.exp(ss + 1j * tt)
    jac = np.abs(zeta * lined.SMAP.domega(zeta)) ** 2
    return (ss.ravel(), tt.ravel(), (ws[:, None] * wt[None, :] * jac).ravel())

class GaussianFeatures:
    """Normalized periodic Gaussian RBF hidden layer with analytic gradients.

    Centers span log-radius [0, log(rho)] with two padding layers on each
    radial side and periodic angle [0, 2*pi). Terms smaller
    than exp(-40) are omitted in sparse evaluation; the normalization makes
    constant displacement exact. Width is measured in center spacings.
    """

    def __init__(self, nr, nt, rho, width=1.0, padding=2):
        self.nr, self.nt, self.rho, self.width = (nr, nt, rho, width)
        self.padding = padding
        self.hs, self.ht = (np.log(rho) / nr, 2 * np.pi / nt)
        self.size = (nr + 1 + 2 * padding) * nt

    def evaluate(self, s, t):
        s, t = (np.asarray(s).ravel(), np.asarray(t).ravel())
        span = int(np.ceil(np.sqrt(80) * self.width)) + 1
        offset = np.arange(-span, span + 1)
        ir = np.floor(s / self.hs).astype(int)[:, None] + offset
        jt = np.floor(t / self.ht).astype(int)[:, None] + offset
        ds = (s[:, None] - ir * self.hs) / (self.hs * self.width)
        dt = (t[:, None] - jt * self.ht) / (self.ht * self.width)
        valid_r = (ir >= -self.padding) & (ir <= self.nr + self.padding)
        ar = np.exp(-0.5 * ds ** 2) * valid_r
        at = np.exp(-0.5 * dt ** 2)
        dar = -ds / (self.hs * self.width) * ar
        dat = -dt / (self.ht * self.width) * at
        sumr, sumt = (ar.sum(1)[:, None], at.sum(1)[:, None])
        dar = (dar - ar * (dar.sum(1)[:, None] / sumr)) / sumr
        dat = (dat - at * (dat.sum(1)[:, None] / sumt)) / sumt
        ar, at = (ar / sumr, at / sumt)
        value = ar[:, :, None] * at[:, None, :]
        keep = value > np.exp(-40)
        row, ri, ti = np.nonzero(keep)
        col = (ir[row, ri] + self.padding) * self.nt + jt[row, ti] % self.nt
        shape = (len(s), self.size)

        def matrix(data):
            return sp.coo_matrix((data[keep], (row, col)), shape=shape).tocsr()
        h = matrix(value)
        hs = matrix(dar[:, :, None] * at[:, None, :])
        ht = matrix(ar[:, :, None] * dat[:, None, :])
        return (h, hs, ht)

    def physical(self, s, t):
        h, hs, ht = self.evaluate(s, t)
        zeta = np.exp(np.asarray(s) + 1j * np.asarray(t))
        tangent = zeta * lined.SMAP.domega(zeta)
        metric = np.abs(tangent) ** 2
        hx = sp.diags(tangent.real / metric) @ hs - sp.diags(tangent.imag / metric) @ ht
        hy = sp.diags(tangent.imag / metric) @ hs + sp.diags(tangent.real / metric) @ ht
        return (h, hx.tocsr(), hy.tocsr())

def strain_matrix(features, s, t):
    h, hx, hy = features.physical(s, t)
    zero = sp.csr_matrix(h.shape)
    blocks = [sp.hstack([hx, zero]), sp.hstack([zero, hy]), sp.hstack([zero, zero]), sp.hstack([hy / SQRT2, hx / SQRT2])]
    b = sp.vstack(blocks).tocsr()
    order = np.arange(4 * len(s)).reshape(4, -1).T.ravel()
    return (b[order], h)

def trace_matrix(features, radius, theta):
    h = features.evaluate(np.full(len(theta), np.log(radius)), theta)[0]
    return sp.kron(h, sp.eye(2), format='csr')[:, np.r_[np.arange(features.size) * 2, np.arange(features.size) * 2 + 1]]

class NeuralSection:

    def __init__(self, nr=12, nt=96, boundary=256, order=4, rho=2.0, width=1.0, padding=2):
        self.features = GaussianFeatures(nr, nt, rho, width, padding)
        self.mat = lined.material_F(4.0, 0.0)
        self.law = plastic_law.DruckerPrager(plastic_law.stiffness_3d(self.mat), 5.0)
        self.s0 = np.array([-15.0, -9.0, -10.95, 0.0])
        self.s, self.theta, self.weight = quadrature(nr, nt, order, rho)
        self.b, self.h = strain_matrix(self.features, self.s, self.theta)
        self.ndof = 2 * self.features.size
        self.boundary_theta = 2 * np.pi * np.arange(boundary) / boundary
        self.wall = trace_matrix(self.features, 1.0, self.boundary_theta)
        self.outer = trace_matrix(self.features, rho, self.boundary_theta)
        self.modal = assemble_modal(self.mat, lined.SMAP.scaled(rho), boundary // 2 - 1) / 1000000.0
        self.work = projected_work(self.modal, self.boundary_theta)
        self.exterior = -(self.outer.T @ sp.csr_matrix(self.work) @ self.outer).tocsr()
        self.ring = lined.LiningRing(lined.SMAP, self.boundary_theta, lined.LINING_T, lined.LINING_NL)
        kr = lined.lining_stiffness(self.ring, lined.E_LIN, lined.NU_LIN) / 1000000.0
        split = 2 * self.ring.n_l * boundary
        self.ring_transfer = -sla.splu(kr[:split, :split].tocsc()).solve(kr[:split, split:].toarray())
        self.ring_wall = kr[split:, split:].toarray() + kr[split:, :split] @ self.ring_transfer
        self.lining = (self.wall.T @ sp.csr_matrix(self.ring_wall) @ self.wall).tocsr()
        mesh = lined.MappedMesh(lined.SMAP, rho, 1, boundary)
        self.wall_load = lined.mapped_released_load(mesh, *lined.insitu(0.6))[:2 * boundary] / 1000000.0
        self.load = np.asarray(self.wall.T @ self.wall_load)
        mean = np.asarray(self.h.mean(axis=0)).ravel()
        self.constraint = sp.block_diag((mean[None, :], mean[None, :]), format='csr').T
        self.elastic = self.tangent(np.broadcast_to(self.law.c, (len(self.s), 4, 4)))
        self.constraint *= np.abs(self.elastic.diagonal()).mean()
        self.elastic_factors = {}

    def tangent(self, tangent, correction=False):
        if correction:
            tangent = tangent - self.law.c
            active = np.any(tangent != 0.0, axis=(1, 2))
            tangent = tangent[active]
            weight = self.weight[active]
            select = (4 * np.flatnonzero(active)[:, None] + np.arange(4)).ravel()
            b = self.b[select]
        else:
            weight, b = (self.weight, self.b)
        nq = len(tangent)
        index = np.arange(4 * nq).reshape(-1, 4)
        rows = np.repeat(index, 4, axis=1).ravel()
        cols = np.tile(index, (1, 4)).ravel()
        c = sp.coo_matrix(((tangent * weight[:, None, None]).ravel(), (rows, cols)), shape=(4 * nq, 4 * nq)).tocsr()
        matrix = (b.T @ c @ b).tocsr()
        return matrix if correction else matrix + self.exterior

    def evaluate(self, weights, old, release, installed, need_tangent=True):
        strain = np.asarray(self.b @ weights).reshape(-1, 4)
        stress, dep, tangent = self.law.update(self.s0 + (strain - old) @ self.law.c.T)
        residual = np.asarray(self.b.T @ ((stress - self.s0) * self.weight[:, None]).ravel())
        residual += self.exterior @ weights - release * self.load
        if installed is not None:
            residual += self.lining @ (weights - installed)
        matrix = None
        if need_tangent:
            if self.law.last_diagnostics['active_return_points']:
                matrix = self.elastic + self.tangent(tangent, correction=True)
            else:
                matrix = self.elastic
            if installed is not None:
                matrix = matrix + self.lining
        return (residual, matrix, stress, old + dep)

    def train_step(self, weights, old, release, installed, tolerance=1e-08):
        weights = weights.copy()
        scale = np.linalg.norm(self.load)
        trace = []
        for iteration in range(40):
            r, k, stress, plastic = self.evaluate(weights, old, release, installed)
            error = np.linalg.norm(r) / scale
            trace.append(dict(iteration=iteration, weak_residual_relative=float(error)))
            if error < tolerance:
                return (weights, plastic, stress, trace)
            c = self.constraint
            elastic = self.law.last_diagnostics['active_return_points'] == 0
            key = installed is not None
            if key not in self.elastic_factors:
                base = self.elastic + (self.lining if key else sp.csr_matrix(self.elastic.shape))
                self.elastic_factors[key] = sla.splu(sp.bmat([[base, c], [c.T, None]], format='csc'))
            factor = self.elastic_factors[key]
            rhs = -np.r_[r, c.T @ weights]
            if elastic:
                correction = factor.solve(rhs)
            else:
                matrix = sp.bmat([[k, c], [c.T, None]], format='csr')
                preconditioner = sla.LinearOperator(matrix.shape, matvec=factor.solve)
                correction, info = sla.gmres(matrix, rhs, M=preconditioner, rtol=1e-10, atol=0.0, restart=50, maxiter=100)
                if info:
                    raise RuntimeError(f'Neural tangent solve failed: GMRES {info}')
            delta = correction[:self.ndof]
            step = 1.0
            for backtrack in range(16):
                candidate = weights + step * delta
                rr = self.evaluate(candidate, old, release, installed, False)[0]
                if np.linalg.norm(rr) < np.linalg.norm(r):
                    weights = candidate
                    trace[-1]['step'] = step
                    break
                step *= 0.5
            else:
                raise RuntimeError(f'Neural residual line search failed at {release}: {error}')
        raise RuntimeError(f'Neural training did not converge at {release}: {error}')

    def response(self, weights, plastic, stress, installed):
        wall = np.asarray(self.wall @ weights)
        z = lined.SMAP.boundary(self.boundary_theta)
        mesh = SimpleNamespace(xy=np.c_[z.real, z.imag], ring_nodes=lambda side: np.arange(len(self.boundary_theta)))
        dv, dh = lined.closures(mesh, wall)
        eqp = np.sqrt(2 / 3 * np.sum((plastic @ plastic_law.DEV) ** 2, axis=1))
        yielded = eqp > 1e-10
        edge = self.s > np.log(self.features.rho) - self.features.hs
        row = dict(closure_vertical_mm=float(dv), closure_horizontal_mm=float(dh), peak_thrust_MN_per_m=0.0, peak_moment_kNm_per_m=0.0, plastic_area_m2=float(self.weight[yielded].sum()), plastic_rho_max=float(np.exp(self.s[yielded]).max()) if yielded.any() else 1.0, max_equivalent_plastic_strain=float(eqp.max()), max_axial_plastic_strain=float(np.abs(plastic[:, 2]).max()), outer_layer_yield_MPa=float(self.law.yield_value(stress[edge]).max()), outer_layer_plastic_strain=float(eqp[edge].max()), axial_total_strain_error=float(np.abs(((stress - self.s0) @ np.linalg.inv(self.law.c).T + plastic)[:, 2]).max()))
        if installed is not None:
            dw = np.asarray(self.wall @ (weights - installed))
            ur = np.r_[self.ring_transfer @ dw, dw]
            n, m = lined.ring_actions(self.ring, ur, lined.E_LIN, lined.NU_LIN)
            row['peak_thrust_MN_per_m'] = float(np.abs(n).max() / 1000000.0)
            row['peak_moment_kNm_per_m'] = float(np.abs(m).max() / 1000.0)
        return row

def run(args):
    start = time.perf_counter()
    model = NeuralSection(args.radial, args.angular, args.boundary, args.quadrature, args.rho, args.width, args.padding)
    print(f'Assembled {model.features.size} neurons, {len(model.s)} quadrature points in {time.perf_counter() - start:.1f}s', flush=True)
    weights = np.zeros(model.ndof)
    plastic = np.zeros((len(model.s), 4))
    installed = None
    rows, training, snapshots = ([], [], [])
    work = 0.0
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = {key: value for key, value in vars(args).items() if key not in ('output', 'resume')}
    if args.resume:
        saved_config = json.loads((out / 'config.json').read_text(encoding='utf-8'))
        if saved_config != config:
            raise ValueError('Resume parameters differ from the saved training configuration')
        with (out / 'history.csv').open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        with (out / 'training.csv').open(encoding='utf-8') as stream:
            training = list(csv.DictReader(stream))
        saved = np.load(out / 'checkpoint.npz')
        snapshots = list(saved['weights'])
        weights, plastic = (snapshots[-1].copy(), saved['plastic'].copy())
        value = saved['installed']
        installed = value.copy() if value.shape == weights.shape else None
        work = float(rows[-1]['plastic_work_kJ_per_m']) / 1000
        print(f"Resumed at a={float(rows[-1]['release']):g}", flush=True)
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    for lo, hi, stage in ((0.0, args.install, 'unsupported'), (args.install, 1.0, 'supported')):
        if stage == 'supported' and installed is None:
            installed = weights.copy()
        for release in np.linspace(lo, hi, int(np.ceil((hi - lo) / args.step)) + 1)[1:]:
            if rows and release <= float(rows[-1]['release']) + 1e-12:
                continue
            previous = plastic
            weights, plastic, stress, trace = model.train_step(weights, previous, release, installed, args.tolerance)
            dissipation = np.sum(stress * (plastic - previous), axis=1)
            if dissipation.min() < -1e-08:
                raise RuntimeError('Negative constitutive dissipation')
            work += float(dissipation @ model.weight)
            row = dict(release=float(release), stage=stage, **model.response(weights, plastic, stress, installed), plastic_work_kJ_per_m=work * 1000, minimum_incremental_dissipation_MPa=float(dissipation.min()), weak_residual_relative=trace[-1]['weak_residual_relative'], training_iterations=len(trace) - 1, **model.law.last_diagnostics)
            rows.append(row)
            training.extend((dict(release=float(release), stage=stage, step=record.get('step', 0.0), **{k: v for k, v in record.items() if k != 'step'}) for record in trace))
            snapshots.append(weights.copy())
            write_csv(out / 'history.csv', rows)
            write_csv(out / 'training.csv', training)
            np.savez_compressed(out / 'checkpoint.npz', weights=np.array(snapshots), release=np.array([float(r['release']) for r in rows]), installed=np.empty(0) if installed is None else installed, plastic=plastic, stress=stress, s=model.s, theta=model.theta, quadrature_weight=model.weight)
            print(f"{stage} a={release:.4f} u={row['closure_vertical_mm']:.6f}/{row['closure_horizontal_mm']:.6f} N={row['peak_thrust_MN_per_m']:.6f} M={row['peak_moment_kNm_per_m']:.6f} Wp={work * 1000:.6f} rho_p={row['plastic_rho_max']:.5f} iterations={len(trace) - 1} elapsed={time.perf_counter() - start:.1f}s", flush=True)
    (out / 'runtime.json').write_text(json.dumps(dict(wall_seconds_this_invocation=time.perf_counter() - start, resumed=args.resume, linear_correction='elastic_preconditioned_gmres', neurons=model.features.size, trainable_weights=model.ndof, quadrature_points=len(model.s), software=dict(numpy=np.__version__)), indent=2) + '\n', encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--radial', type=int, default=12)
    parser.add_argument('--angular', type=int, default=96)
    parser.add_argument('--boundary', type=int, default=256)
    parser.add_argument('--quadrature', type=int, default=4)
    parser.add_argument('--rho', type=float, default=2.0)
    parser.add_argument('--width', type=float, default=1.0)
    parser.add_argument('--padding', type=int, default=2)
    parser.add_argument('--step', type=float, default=0.02)
    parser.add_argument('--install', type=float, default=0.82)
    parser.add_argument('--tolerance', type=float, default=1e-08)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', default=str(ROOT / 'results/data/derived/near_field_pinn/medium'))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args)
if __name__ == '__main__':
    main()
