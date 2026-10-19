from __future__ import annotations
import csv
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code' / 'closure'))
sys.path.insert(0, str(ROOT / 'code' / 'closure' / 'numeric'))
sys.path.insert(0, str(ROOT / 'code' / 'figure_exports'))
from closure_env import bridge_pinned
bridge_pinned()
from anisotropic_elasticity.material import material_F
from coarsened_far_comparator import GROWTH_LADDER, RHO_FAR, coarsened_mesh, coarsened_solve, near_block_gate, shipped_row
from exterior_treatment_sweep import NEAR_LADDER, PIN_CASE, REF_N_COLLOCATION, REF_N_ORDER, certified_reference, operator_solve, rel_to_ls
OUT = ROOT / 'results/data/derived/coarsened_far_comparator_clamped.csv'
SIBLING = ROOT / 'results/data/derived/coarsened_far_comparator.csv'
FREE_REGRESSION_RTOL = 1e-09
CLAMPED_ANCHOR_RTOL = 0.15

def sibling_rows() -> list[dict]:
    with SIBLING.open(encoding='utf-8', newline='') as fh:
        return list(csv.DictReader(fh))

def main() -> None:
    mat = material_F(*PIN_CASE, f'r{PIN_CASE[0]}d{PIN_CASE[1]}')
    ref, ref_dev, gate_solve = certified_reference(mat, f'ratio={PIN_CASE[0]} dip={PIN_CASE[1]}')
    n_r0, n_t0 = NEAR_LADDER[0]
    n_near0 = (n_r0 + 1) * n_t0
    idx = np.arange(n_near0)
    mesh_near, _, _ = operator_solve(mat, n_r0, n_t0)
    probe, _, _ = coarsened_mesh(n_r0, n_t0, RHO_FAR, GROWTH_LADDER[0])
    near_dev = near_block_gate(probe, mesh_near, n_near0)
    k = int(round(np.log2(RHO_FAR)))
    anchor = shipped_row('far_mesh_fixed', f'rho_far=2^{k}')
    err_anchor = float(anchor['rel_error_vs_completed_basis'])
    print(f'[anchor] far_mesh_fixed rho_far=2^{k} read from exterior_cost_ladder.csv: {err_anchor:.6e}')
    ship = next((r for r in sibling_rows() if r['treatment'] == 'coarsened_free' and r['shipped_row'] == 'True'))
    ship_growth = next((g for g in GROWTH_LADDER if f'growth={g:.2f}' == ship['configuration']))
    ship_err = float(ship['rel_error_vs_completed_basis'])
    mesh_f, u_f, _, _ = coarsened_solve(mat, n_r0, n_t0, RHO_FAR, ship_growth, 'free')
    err_f = rel_to_ls(mesh_f, u_f, ref, idx)
    rel = abs(err_f - ship_err) / ship_err
    print(f'[gate] free half at growth={ship_growth:.2f}: {err_f:.9e} against the shipped {ship_err:.9e} (rel dev {rel:.2e})')
    if rel >= FREE_REGRESSION_RTOL:
        raise SystemExit("free-half regression failed: this file's meshes are not the shipped comparator's. Nothing written.")
    rows: list[dict] = []
    for growth in GROWTH_LADDER:
        mesh_c, u_c, _, meta = coarsened_solve(mat, n_r0, n_t0, RHO_FAR, growth, 'fixed')
        err = rel_to_ls(mesh_c, u_c, ref, idx)
        rows.append({'treatment': 'coarsened_fixed', 'configuration': f'growth={growth:.2f}', 'total_dof': mesh_c.n_dof, 'interface_dof': 2 * mesh_c.n_t, 'n_rings': mesh_c.n_r, 'rel_error_vs_completed_basis': err, 'ref_n_order': REF_N_ORDER, 'ref_n_collocation': REF_N_COLLOCATION, 'rho_far': RHO_FAR, 'near_mesh': f'{n_r0}x{n_t0}', 'growth_ratio': round(meta[0], 6), 'n_exterior_rings': meta[1], 'grading': 'coarsened', 'shipped_counterpart': growth == ship_growth, 'anchor_far_mesh_fixed': err_anchor, 'near_block_dev': near_dev, 'ref_window_dev': f'{ref_dev:.3e}'})
        print(f'[X] coarsened_fixed growth={growth:.2f}: dof={mesh_c.n_dof:6d} rings={mesh_c.n_r:3d} err={err:.4e}', flush=True)
    densest = rows[0]['rel_error_vs_completed_basis']
    dev = abs(densest - err_anchor) / err_anchor
    print(f'[gate] clamped half at growth={GROWTH_LADDER[0]:.2f}: {densest:.6e} against the self-similar clamped anchor {err_anchor:.6e} (rel dev {dev:.2%})')
    if dev >= CLAMPED_ANCHOR_RTOL:
        raise SystemExit('clamped-half sanity failed: the densest coarsening does not track the clamped anchor. Nothing written.')
    with OUT.open('w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    pick = next((r for r in rows if r['shipped_counterpart']))
    print(f"\n[pair] shipped comparator at growth {pick['growth_ratio']:.4f}: free {ship_err:.4e}, clamped {pick['rel_error_vs_completed_basis']:.4e}; self-similar bracket at the same radius free {float(shipped_row('far_mesh_free', f'rho_far=2^{k}')['rel_error_vs_completed_basis']):.4e}, clamped {err_anchor:.4e}")
    print(f'  wrote {OUT.relative_to(ROOT)} ({len(rows)} rows)')
if __name__ == '__main__':
    main()
