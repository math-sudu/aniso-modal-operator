from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/closure'))
from closure_env import bridge_pinned
bridge_pinned()
sys.path.insert(0, str(ROOT / 'code/closure/numeric'))
from exec20_src.material import material_F
from exec26_src import protocol as P
from exec26_src.geometry import horseshoe_map
from exec26_src.sf_general import GeneralExterior, ring_poles, _real_design, _spectral_dtheta, assemble_modal_dtn
from exec26_src.fe_mapped import MappedMesh, AnisoAnnulusFE, mapped_released_load, solve_condensed, strip_translation
from modal_condensation import nodal_from_modal, condense_with
from p21_exact_evaluation import assemble_exact
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

def work_condition(modal, n_t):
    """Condition of S_h on its retained Fourier range, excluding null modes."""
    h = 2 * np.pi / n_t
    mass = np.repeat(h / 3 * (2 + np.cos(np.arange(1, len(modal) // 4 + 1) * h)), 4)
    return np.linalg.cond(mass[:, None] * modal)

def coupled(fe, modal):
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
        row.update(status='assembled', **stats, operator_error=np.linalg.norm(modal - sref) / np.linalg.norm(sref), work_condition=work_condition(modal, n_t), reference_work_condition=work_condition(sref, n_t))
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
    ap.add_argument('--case', choices=('all', 'wall', 'coarse', 'fine', 'anisotropic', 'refined', 'bands'), default='all')
    args = ap.parse_args()
    cases = {'wall': [(2.625, 45.0, 1.0, 0, 34, tuple(LAYOUTS)[:-1])], 'coarse': [(2.625, 45.0, rho, 12, 64, tuple(LAYOUTS)[:-1]) for rho in (1.25, 2.0)], 'fine': [(2.625, 45.0, rho, 24, 128, ('default', 'deep_ring', 'near_ring')) for rho in (1.25, 2.0)], 'anisotropic': [(4.0, 0.0, rho, 24, 128, ('default', 'near_ring')) for rho in (1.25, 2.0)], 'refined': [(2.625, 45.0, 1.0, 0, 34, ('basis_refined',))] + [(ratio, phi, rho, 24, 128, ('basis_refined',)) for ratio, phi in ((2.625, 45.0), (4.0, 0.0)) for rho in (1.25, 2.0)]}
    rows = []
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        if args.case == 'bands':
            rows = band_resolution()
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
