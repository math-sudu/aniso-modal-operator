from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import near_field_clouds as clouds
import exterior_treatment_sweep as sweep
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'results/data/derived/exterior_wall_profiles.csv'

def main():
    saved = pd.read_csv(clouds.OUT_DIR / 'near_field_interface_error_hs100.csv')
    saved = saved[(saved.E_ratio == 2.625) & (saved.bedding_dip_deg == 45)]
    cost = pd.read_csv(clouds.OUT_DIR / 'exterior_cost_ladder.csv')
    mat = clouds.material_F(2.625, 45, 'exterior-wall')
    reference = clouds.RingReference(mat, sweep.SMAP, *clouds.insitu_raw())
    cost_ref = sweep.LSReference(mat, sweep.SMAP, *clouds.insitu_raw(), n_order=sweep.REF_N_ORDER, n_collocation=sweep.REF_N_COLLOCATION)
    rows = []
    for nr, nt in sweep.NEAR_LADDER:
        mesh, u, _ = sweep.operator_solve(mat, nr, nt)
        ux, uy = reference.displacements(*mesh.xy.T)
        expected = np.column_stack((ux, uy)).ravel()
        expected = clouds.strip_translation(expected)
        delta = (u - expected).reshape(-1, 2)
        error = np.linalg.norm(delta, axis=1) * 1000000.0
        x, y = clouds.to_phys(*mesh.xy.T)
        if (nr, nt) == (24, 128):
            prior = saved[saved.rho_out == 2]
            assert np.allclose(prior.x_m, x, rtol=0, atol=0.000501)
            assert np.allclose(prior.y_m, y, rtol=0, atol=0.000501)
            deviation = np.max(np.abs(prior.abs_diff_um.to_numpy() - error))
            assert deviation <= 5.1e-06, deviation
            print(f'Saved field agreement: {deviation:.3e} micrometres', flush=True)
        cost_error = sweep.rel_to_ls(mesh, u, cost_ref, sweep.shared_index(nr, nt))
        previous = float(cost[(cost.treatment == 'operator') & (cost.configuration == f'{nr}x{nt}')].rel_error_vs_completed_basis.iloc[0])
        assert np.isclose(cost_error, previous, rtol=1e-08, atol=1e-12)
        for j in range(nt):
            rows.append(dict(rho_out=2.0, near_mesh=f'{nr}x{nt}', node_index=j, theta_deg=j * 360 / nt, x_m=x[j], y_m=y[j], abs_diff_um=error[j]))
        print(f'{nr}x{nt}: wall maximum {error[:nt].max():.6f} micrometres; shared-node cost error {cost_error:.9e}', flush=True)
    inner = saved[(saved.rho_out == 1.5) & (saved.node_grade == 1)]
    assert len(inner) == 128
    for j, row in enumerate(inner.itertuples()):
        rows.append(dict(rho_out=1.5, near_mesh='24x128', node_index=j, theta_deg=j * 360 / 128, x_m=row.x_m, y_m=row.y_m, abs_diff_um=row.abs_diff_um))
    with OUTPUT.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} wall nodes to {OUTPUT.name}', flush=True)
if __name__ == '__main__':
    main()
