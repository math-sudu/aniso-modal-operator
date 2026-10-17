from __future__ import annotations
import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.linalg import solve_triangular
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/closure'))
from closure_env import bridge_pinned
bridge_pinned()
sys.path.insert(0, str(ROOT / 'code/closure/numeric'))
from exec20_src.material import material_F
from exec26_src import protocol as P
from exec26_src.geometry import horseshoe_map, SectionMap
from exec26_src.sf_general import GeneralExterior, ring_poles, _real_design, _spectral_dtheta, assemble_modal_dtn, rho_crit_of_map
from exec26_src.fe_mapped import MappedMesh, AnisoAnnulusFE, mapped_released_load, solve_condensed, strip_translation
from modal_condensation import nodal_from_modal, condense_with, condense_work
from p21_exact_evaluation import assemble_exact
from exterior_operator import assemble_modal, p1_projection, projected_work
SMAP = horseshoe_map(1.0, P.R_HS, P.C_HS)
LAYOUTS = {'default': {}, 'deep_ring': {'eta': 0.6}, 'near_ring': {'eta': 0.97}, 'tight_clusters': {'cluster_ratio': 0.1}, 'spread_clusters': {'cluster_ratio': 0.8}, 'ring_only': {'lightning': False}, 'basis_refined': {'eta': 0.97, 'n_modes': 192, 'n_grid': 1024}}

class PlacedExterior(GeneralExterior):
    """Expose the existing pole builder's layout parameters for this experiment."""

    def __init__(self, mat, smap, n_modes, **layout):
        self.mat, self.smap, self.n_modes = (mat, smap, n_modes)
        self.poles, self.light, self.w_dft, self.s, self.eta, self.J = ring_poles(smap, mat, n_modes, **layout)
        self.n_light = len(self.light[0])
        self.n_cols = n_modes + self.n_light

def harmonic_data(theta, n_h):
    b = np.zeros((2 * len(theta), 4 * n_h))
    for n in range(1, n_h + 1):
        c, s = (np.cos(n * theta), np.sin(n * theta))
        b[:len(theta), 4 * n - 4:4 * n - 2] = np.column_stack([c, s])
        b[len(theta):, 4 * n - 2:4 * n] = np.column_stack([c, s])
    return b

def design(ext, theta):
    tr = ext.wall_traces(theta)
    dec = np.vstack([_real_design(tr['Ux']), _real_design(tr['Uy'])])
    rig = np.zeros((2 * len(theta), 2))
    rig[:len(theta), 0], rig[len(theta):, 1] = (1.0, 1.0)
    rig *= np.median(np.linalg.norm(dec, axis=0)) / np.sqrt(len(theta))
    return (np.column_stack([dec, rig]), tr)

def assemble(mat, tier, n_h, layout, n_modes=96, n_grid=512):
    ext = PlacedExterior(mat, tier, n_modes, **layout)
    theta = 2 * np.pi * np.arange(n_grid) / n_grid
    a, tr = design(ext, theta)
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    keep = s > s[0] * max(a.shape) * np.finfo(float).eps
    b = harmonic_data(theta, n_h)
    coef = vh[keep].T @ (u[:, keep].T @ b / s[keep, None])
    nd = 4 * ext.n_cols
    tx = _spectral_dtheta(_real_design(tr['X']) @ coef[:nd])
    ty = _spectral_dtheta(_real_design(tr['Y']) @ coef[:nd])
    modal = 2 / n_grid * b.T @ np.vstack([tx, ty])
    unit_a = a / np.linalg.norm(a, axis=0)
    unit_s = np.linalg.svd(unit_a, compute_uv=False)
    th_check = 2 * np.pi * (np.arange(2 * n_grid) + 0.37) / (2 * n_grid)
    tr_check = ext.wall_traces(th_check)
    dec_check = np.vstack([_real_design(tr_check['Ux']), _real_design(tr_check['Uy'])])
    fit_check = dec_check @ coef[:nd]
    fit_check[:2 * n_grid] += a[0, nd] * coef[nd]
    fit_check[2 * n_grid:] += a[n_grid, nd + 1] * coef[nd + 1]
    b_check = harmonic_data(th_check, n_h)
    stats = dict(eta=ext.eta, quadrature_poles=ext.J, cluster_poles_per_branch=ext.n_light, fit_rows=a.shape[0], fit_columns=a.shape[1], fit_rank=int(keep.sum()), svd_cutoff_relative=max(a.shape) * np.finfo(float).eps, smallest_retained_singular_ratio=s[keep][-1] / s[0], fit_condition=s[0] / s[-1], fit_effective_condition=s[0] / s[keep][-1], unit_column_condition=unit_s[0] / unit_s[-1], fit_residual=np.max(np.linalg.norm(a @ coef - b, axis=0) / np.linalg.norm(b, axis=0)), heldout_residual=np.max(np.linalg.norm(fit_check - b_check, axis=0) / np.linalg.norm(b_check, axis=0)), heldout_max_residual=np.max(np.abs(fit_check - b_check)), heldout_grid=2 * n_grid, heldout_shift=0.37)
    return (modal, stats)

def band_resolution():
    """Resolve each used band independently at the support interface."""
    rows = []
    for ratio, phi in ((2.625, 45.0), (4.0, 0.0)):
        mat = material_F(ratio, phi, 'band_resolution')
        tier = SMAP.scaled(2.0)
        for n_h in (31, 63, 127):
            reference = assemble_exact(mat, tier, n_h, n_h + 32, eta_override=0.97)['S']
            refined = assemble_exact(mat, tier, n_h, n_h + 64, eta_override=0.97)['S']
            reference_gap = np.linalg.norm(reference - refined) / np.linalg.norm(refined)
            resolved = (96, 512) if n_h == 31 else (384, 2048)
            for refined_basis, (n_modes, n_grid) in enumerate(((96, 512), resolved)):
                layout = {'eta': 0.97} if refined_basis else {}
                modal, stats = assemble(mat, tier, n_h, layout, n_modes, n_grid)
                row = dict(ratio=ratio, phi_deg=phi, bedding_dip_deg=90 - phi, rho=2.0, n_h=n_h, n_modes=n_modes, n_grid=n_grid, layout='default' if not layout else 'band_refined', reference_nc=n_h + 32, reference_refined_nc=n_h + 64, reference_radius=0.97, reference_operator_gap=reference_gap, **stats, operator_error=np.linalg.norm(modal - refined) / np.linalg.norm(refined))
                rows.append(row)
                print('Band', ratio, phi, n_h, row['layout'], row['operator_error'], flush=True)
    return rows

def legacy_work_condition(modal, n_t):
    """Historical one-sided mass/DFT condition on the retained Fourier range."""
    h = 2 * np.pi / n_t
    mass = np.repeat(h / 3 * (2 + np.cos(np.arange(1, len(modal) // 4 + 1) * h)), 4)
    return np.linalg.cond(mass[:, None] * modal)

def projected_range(modal, n_t):
    """Actual P1 work in the orthonormal nodal harmonic basis."""
    weights = np.repeat(np.sinc(np.arange(1, len(modal) // 4 + 1) / n_t) ** 2, 4)
    return 2 * np.pi / n_t * weights[:, None] * modal * weights[None, :]

def work_metrics(work):
    """Diagnostics of an unmodified work matrix with its null space removed."""
    energy = np.linalg.eigvalsh(-(work + work.T) / 2)
    return dict(projected_condition=np.linalg.cond(work), work_reciprocity=np.linalg.norm(work - work.T) / np.linalg.norm(work), minimum_energy_Pa=energy[0], maximum_energy_Pa=energy[-1])

def p1_diagnostics(modal, n_t):
    theta = 2 * np.pi * np.arange(n_t) / n_t
    transform = p1_projection(theta, len(modal) // 4)
    weights = np.repeat(np.sinc(np.arange(1, len(modal) // 4 + 1) / n_t) ** 2, 4)
    basis = np.sqrt(n_t / 2) * transform.T / weights
    full = projected_work(modal, theta)
    direct = basis.T @ full @ basis
    reduced = projected_range(modal, n_t)
    return dict(**work_metrics(reduced), retained_rank=len(reduced), direct_condition=np.linalg.cond(direct), range_identity_error=np.linalg.norm(direct - reduced) / np.linalg.norm(reduced), range_reconstruction_error=np.linalg.norm(basis @ reduced @ basis.T - full) / np.linalg.norm(full))

def projected_diagnostics():
    """Production P1 bands, Table 3 controls and actual higher-order trace ranges."""
    rows = []
    for ratio, phi in ((2.625, 45.0), (4.0, 0.0)):
        mat = material_F(ratio, phi, 'projected_work')
        for rho in (1.25, 2.0):
            for n_h in (31, 63, 127):
                modal = assemble_modal(mat, SMAP.scaled(rho), n_h)
                row = dict(ratio=ratio, phi_deg=phi, rho=rho, n_h=n_h, n_t=2 * (n_h + 1), trace_order=1, layout='cauchy', **p1_diagnostics(modal, 2 * (n_h + 1)))
                rows.append(row)
                print('Projected', ratio, rho, n_h, row['projected_condition'], flush=True)
    mat = material_F(2.625, 45.0, 'projected_work')
    reference = assemble_modal(mat, SMAP.scaled(1.25), 63)
    fe = AnisoAnnulusFE(MappedMesh(SMAP, 1.25, 24, 128), mat)
    uref = coupled(fe, reference, projected=True)
    for name in ('deep_ring', 'default', 'near_ring', 'basis_refined'):
        settings = LAYOUTS[name].copy()
        nm, ng = (settings.pop('n_modes', 96), settings.pop('n_grid', 512))
        modal, stats = assemble(mat, SMAP.scaled(1.25), 63, settings, nm, ng)
        solution = coupled(fe, modal, projected=True)
        rows.append(dict(ratio=2.625, phi_deg=45.0, rho=1.25, n_h=63, n_t=128, trace_order=1, layout=name, n_modes=nm, n_grid=ng, eta=stats['eta'], legacy_condition=legacy_work_condition(modal, 128), fit_condition=stats['fit_condition'], operator_error=np.linalg.norm(modal - reference) / np.linalg.norm(reference), field_error=np.linalg.norm(solution - uref) / np.linalg.norm(uref), **p1_diagnostics(modal, 128)))
        print('Table 3', name, rows[-1]['projected_condition'], flush=True)
    modal, _ = assemble(mat, SMAP.scaled(2.0), 127, {})
    rows.append(dict(ratio=2.625, phi_deg=45.0, rho=2.0, n_h=127, n_t=256, trace_order=1, layout='default', n_modes=96, n_grid=512, **p1_diagnostics(modal, 256)))
    spec = importlib.util.spec_from_file_location('trace', Path(__file__).with_name('trace_order.py'))
    trace = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trace)
    modal = assemble_modal(mat, SMAP.scaled(1.25), 127)
    for p in (2, 3):
        fe = SimpleNamespace(p=p, nt=64, nth=64 * p, dt=2 * np.pi / 64)
        transform = trace.CurvedFE.projection(fe, 127)
        _, singular, vt = np.linalg.svd(transform, full_matrices=False)
        keep = singular > max(transform.shape) * np.finfo(float).eps * singular[0]
        basis = vt[keep].T
        full = np.pi * transform.T @ modal @ transform
        reduced = basis.T @ full @ basis
        rows.append(dict(ratio=2.625, phi_deg=45.0, rho=1.25, n_h=127, n_t=fe.nth, trace_order=p, layout='cauchy', retained_rank=int(keep.sum()), **work_metrics(reduced), range_reconstruction_error=np.linalg.norm(basis @ reduced @ basis.T - full) / np.linalg.norm(full)))
        print('P', p, rows[-1]['projected_condition'], flush=True)
    return rows

class PoleSelectionError(RuntimeError):

    def __init__(self, rows):
        super().__init__('No admissible resolved pole configuration in the requested refinement range')
        self.rows = rows

def energy_relative_error(work, reference, factor):
    """Spectral error in the positive reference exterior-energy norm."""
    left = solve_triangular(factor, work - reference, lower=True)
    scaled = solve_triangular(factor, left.T, lower=True).T
    return np.linalg.norm(scaled, 2)

def select_poles(mat, tier, n_h, *, trace_tolerance=None, max_refinements=3):
    """Select admissible poles using a resolved reference and held-out traces.

    The default accuracy scale is the first-harmonic attenuation of the
    matched P1 trace (nt=2*(N+1)). A caller using another trace discretization
    can supply its accuracy scale. Refinement limits bound this search; an
    unresolved search raises PoleSelectionError with its candidate records.
    """
    nt = 2 * (n_h + 1)
    tolerance = 1 - np.sinc(1 / nt) ** 2 if trace_tolerance is None else trace_tolerance
    if not 0 < tolerance < 1 or max_refinements < 1:
        raise ValueError('Require a relative trace accuracy in (0,1) and at least one refinement')
    reference_modal = assemble_modal(mat, tier, n_h)
    reference = projected_range(reference_modal, nt)
    factor = np.linalg.cholesky(-(reference + reference.T) / 2)
    check = projected_range(assemble_modal(mat, tier, n_h, n_h + 32), nt)
    reference_gap = energy_relative_error(check, reference, factor)
    if reference_gap >= tolerance:
        raise ValueError('Refine the Cauchy resultant order before selecting poles')
    eta_geometry = max(0.85, (1 + rho_crit_of_map(tier)) / 2)
    if eta_geometry >= 1:
        raise ValueError('Pole geometry requires all physical-map derivative zeros inside the unit circle')
    rows = []
    initial_modes = max(96, 3 * (n_h + 1))
    for level in range(max_refinements):
        nm = initial_modes * 2 ** level
        ng = 1 << int(np.ceil(np.log2(max(512, 4 * nm))))
        eta_base = max(eta_geometry, (np.finfo(float).eps / tolerance) ** (1 / (nm - 1)))
        for radius_step in range(3):
            eta = 1 - (1 - eta_base) / 2 ** radius_step
            row = dict(candidate=len(rows) + 1, n_modes=nm, n_grid=ng, eta=eta, eta_geometry=eta_geometry, trace_tolerance=tolerance, reference_energy_gap=reference_gap)
            try:
                modal, stats = assemble(mat, tier, n_h, {'eta': eta}, nm, ng)
            except ValueError as exc:
                if 'basis poles leave the mapped hole' not in str(exc):
                    raise
                row['status'] = 'rejected_outside_poles'
                rows.append(row)
                continue
            work = projected_range(modal, nt)
            error = energy_relative_error(work, reference, factor)
            row.update(**stats, **work_metrics(work), energy_relative_error=error, operator_error=np.linalg.norm(modal - reference_modal) / np.linalg.norm(reference_modal))
            failures = []
            if row['minimum_energy_Pa'] <= 0:
                failures.append('energy')
            if error > tolerance:
                failures.append('band')
            if row['heldout_residual'] > tolerance:
                failures.append('trace')
            row['status'] = 'rejected_' + '_'.join(failures) if failures else 'selected'
            rows.append(row)
            print('Candidate', n_h, nm, ng, eta, row['status'], 'energy error', error, 'held-out', row['heldout_residual'], flush=True)
            if not failures:
                return (modal, rows)
    raise PoleSelectionError(rows)

def automatic_study():
    rows = []
    cases = [('horseshoe', ratio, phi, rho, n_h, SMAP.scaled(rho)) for ratio, phi in ((2.625, 45.0), (4.0, 0.0)) for rho in (1.25, 2.0) for n_h in (31, 63, 127)]
    cases.append(('ellipse', 2.625, 45.0, 1.0, 31, SectionMap(1.0, [0.0, 0.81])))
    for shape, ratio, phi, rho, n_h, tier in cases:
        identity = dict(shape=shape, ratio=ratio, phi_deg=phi, rho=rho, n_h=n_h)
        print('Automatic', identity, flush=True)
        started = time.perf_counter()
        try:
            _, candidates = select_poles(material_F(ratio, phi, 'automatic_poles'), tier, n_h)
        except PoleSelectionError as error:
            error.rows = rows + [dict(**identity, **row) for row in error.rows]
            raise
        for row in candidates:
            rows.append(dict(**identity, **row))
        rows[-1]['selection_seconds'] = time.perf_counter() - started
    return rows

def coupled(fe, modal, projected=False):
    if projected:
        k = condense_work(fe, projected_work(modal, fe.mesh.theta_nodes))
    else:
        sd = nodal_from_modal(modal, fe.mesh.theta_nodes, fe.mesh.n_t // 2 - 1)
        k = condense_with(fe, sd)
    f = mapped_released_load(fe.mesh, -15000000.0, -13500000.0, 0.0)
    return strip_translation(solve_condensed(fe, k, f))

def closures(mesh, u):
    inner = mesh.ring_nodes('inner')
    j = mesh.n_t // 4
    uv = u.reshape(-1, 2)[inner]
    return np.array([uv[2 * j, 0] - uv[0, 0], uv[3 * j, 1] - uv[j, 1]]) * 1000

def run_case(ratio, phi, rho, n_r, n_t, layouts):
    mat = material_F(ratio, phi, 'pole_sensitivity')
    tier = SMAP.scaled(rho)
    n_h = n_t // 2 - 1
    nc = max(64, n_h + 32)
    ref = assemble_exact(mat, tier, n_h, nc)
    ref_fine = assemble_exact(mat, tier, n_h, nc + 32)
    sref = ref_fine['S']
    ref_gap = np.linalg.norm(ref['S'] - sref) / np.linalg.norm(sref)
    fe = AnisoAnnulusFE(MappedMesh(SMAP, rho, n_r, n_t), mat) if n_r else None
    uref = coupled(fe, sref) if fe else None
    cref = closures(fe.mesh, uref) if fe else None
    ref_u_gap = np.linalg.norm(coupled(fe, ref['S']) - uref) / np.linalg.norm(uref) if fe else ''
    rows = []
    for name in layouts:
        settings = LAYOUTS[name].copy()
        n_modes, n_grid = (settings.pop('n_modes', 96), settings.pop('n_grid', 512))
        row = dict(ratio=ratio, phi_deg=phi, bedding_dip_deg=90 - phi, rho=rho, n_r=n_r, n_t=n_t, n_h=n_h, n_modes=n_modes, n_grid=n_grid, layout=name, reference_nc=nc + 32, reference_operator_gap=ref_gap, reference_field_gap=ref_u_gap)
        try:
            modal, stats = assemble(mat, tier, n_h, settings, n_modes, n_grid)
        except ValueError as exc:
            if 'basis poles leave the mapped hole' not in str(exc):
                raise
            row.update(status='rejected_outside_poles')
            rows.append(row)
            print(rho, n_t, name, row['status'], flush=True)
            continue
        row.update(status='assembled', **stats, operator_error=np.linalg.norm(modal - sref) / np.linalg.norm(sref), work_condition=legacy_work_condition(modal, n_t), reference_work_condition=legacy_work_condition(sref, n_t))
        if fe:
            u = coupled(fe, modal)
            cc = closures(fe.mesh, u)
            row.update(field_error=np.linalg.norm(u - uref) / np.linalg.norm(uref), vertical_closure_mm=cc[0], horizontal_closure_mm=cc[1], reference_vertical_closure_mm=cref[0], reference_horizontal_closure_mm=cref[1], closure_max_difference_mm=np.max(np.abs(cc - cref)))
        rows.append(row)
        print(rho, n_t, name, {k: row[k] for k in ('fit_condition', 'operator_error', 'work_condition')}, 'field', row.get('field_error'), flush=True)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT / 'results/data/derived/pole_placement.csv')
    ap.add_argument('--case', choices=('all', 'wall', 'coarse', 'fine', 'anisotropic', 'refined', 'bands', 'projected', 'automatic'), default='all')
    args = ap.parse_args()
    cases = {'wall': [(2.625, 45.0, 1.0, 0, 34, tuple(LAYOUTS)[:-1])], 'coarse': [(2.625, 45.0, rho, 12, 64, tuple(LAYOUTS)[:-1]) for rho in (1.25, 2.0)], 'fine': [(2.625, 45.0, rho, 24, 128, ('default', 'deep_ring', 'near_ring')) for rho in (1.25, 2.0)], 'anisotropic': [(4.0, 0.0, rho, 24, 128, ('default', 'near_ring')) for rho in (1.25, 2.0)], 'refined': [(2.625, 45.0, 1.0, 0, 34, ('basis_refined',))] + [(ratio, phi, rho, 24, 128, ('basis_refined',)) for ratio, phi in ((2.625, 45.0), (4.0, 0.0)) for rho in (1.25, 2.0)]}
    rows = []
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        if args.case == 'bands':
            rows = band_resolution()
        elif args.case == 'projected':
            rows = projected_diagnostics()
        elif args.case == 'automatic':
            try:
                rows = automatic_study()
            except PoleSelectionError as error:
                fields = list(dict.fromkeys((k for row in error.rows for k in row)))
                with args.output.open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(error.rows)
                raise
        for group, specs in cases.items():
            if args.case not in ('all', group):
                continue
            for spec in specs:
                rows.extend(run_case(*spec))
    fields = list(dict.fromkeys((k for row in rows for k in row)))
    with args.output.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print('Wrote', args.output, 'in', time.perf_counter() - start, 'seconds')
if __name__ == '__main__':
    main()
