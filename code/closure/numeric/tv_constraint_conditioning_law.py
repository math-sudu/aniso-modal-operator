from __future__ import annotations
import csv
import math
import sys
from pathlib import Path
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT = PROJECT_ROOT / 'results/data/derived/constraint_conditioning_law.csv'
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p21_exact_evaluation as p21

def main() -> int:
    rows = []
    for mat in p21.P.materials_main():
        previous = None
        for nc in p21.LADDER:
            out = p21.assemble_exact(mat, p21.HS1, p21.N_H, nc)
            residual = p21.heldout_certificate(out)
            increment = float(np.max(abs(out['S'] - previous)) / np.max(abs(out['S']))) if previous is not None else None
            previous = out['S']
            rows.append({'material': mat.name, 'n_t': nc, 'eta_min': f"{min(out['etas']):.12f}", 'eta_max': f"{max(out['etas']):.12f}", 'ratio_predicted': f"{max(out['etas']) / min(out['etas']):.12f}", 'cond': f"{out['cond']:.6e}", 'heldout_residual': f'{residual:.6e}', 'operator_increment': '' if increment is None else f'{increment:.6e}'})
            print(f"{mat.name:12s} n_c={nc:3d} residual={residual:.3e} increment={increment} condition={out['cond']:.3e}", flush=True)
        if residual >= 1e-11 or increment >= 1e-10:
            raise RuntimeError(f'{mat.name}: the final constraint operator is unresolved')
    mat = p21.P.materials_main()[1]
    ns = (64, 96, 128)
    for eta in (None, 0.95):
        conds = [p21.assemble_exact(mat, p21.HS1, p21.N_H, n, eta_override=eta)['cond'] for n in ns]
        factor = math.exp(float(np.polyfit(ns, np.log(conds), 1)[0]))
        if eta is None:
            radii = [p21.cauchy_radius(p21.HS1.zk_series(mu)) for mu in mat.mus]
            prediction = max(radii) / min(radii)
            assert abs(factor / prediction - 1) < 0.02
        else:
            assert abs(factor - 1) < 0.003
        print(f'F1 ring={eta}, conditions={conds}, per-mode factor={factor:.8f}')
    with OUT.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {OUT.relative_to(PROJECT_ROOT)}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
