from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import closed_kernel_ladder as ladder
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'results/data/derived/closed_kernel_paths.csv'
T_PATH = np.linspace(0.25, 1.0, 31)
RHO_PATH = np.linspace(1.0, 2.0, 31)

def main():
    materials = list(ladder.P.materials_main())
    with (ROOT / ladder.LADDER_CSV).open(newline='', encoding='utf-8') as stream:
        anchors = {(row['material'], float(row['rho_out']), float(row['t'])): row for row in csv.DictReader(stream)}
    rows, solutions, kernels = ([], {}, {})
    worst_drift = 0.0
    for path, states in (('shape', [(1.0, float(t)) for t in T_PATH]), ('interface', [(float(rho), 1.0) for rho in RHO_PATH])):
        for mat in materials:
            S0 = ladder.modal_blocks_density(mat, ladder.N_H)
            for rho, t in states:
                key = (mat.name, rho, t)
                if key not in solutions:
                    kernel_key = (mat.name, rho)
                    if kernel_key not in kernels:
                        R, C = ladder.graded_map_constants(rho)
                        kernels[kernel_key] = (ladder.closed_S1(mat, R, C, ladder.N_H), ladder.closed_S2(mat, R, C, ladder.N_H))
                    S1, S2 = kernels[kernel_key]
                    S = ladder.reference(mat, rho, t, ladder.N_T)
                    Sc = ladder.reference(mat, rho, t, ladder.N_T_CHECK)
                    errors = ladder.relatives(S, S0, S1, S2, t)
                    checked = ladder.relatives(Sc, S0, S1, S2, t)[2]
                    drift = abs(checked - errors[2]) / errors[2]
                    worst_drift = max(worst_drift, drift)
                    if drift > ladder.SATURATION_TOL:
                        raise ValueError(f'Unresolved reference at {key}: {drift:.3%}')
                    if key in anchors:
                        for order, error in enumerate(errors):
                            if f'{error:.3e}' != anchors[key][f'rel{order}']:
                                raise ValueError(f'Original ladder differs at {key}, order {order}')
                    solutions[key] = {'rho_out': rho, 'material': mat.name, 't': t, **{f'rel{k}': value for k, value in enumerate(errors)}, 'rel2_check': checked, 'reference_difference': float(np.linalg.norm(Sc - S) / np.linalg.norm(Sc)), 'n_h': ladder.N_H, 'n_t': ladder.N_T, 'n_t_check': ladder.N_T_CHECK}
                rows.append({'path': path, **solutions[key]})
            print(f'{path}: {mat.name}, {len(states)} states', flush=True)
    with OUTPUT.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows; worst second-order reference drift {worst_drift:.3e}')
if __name__ == '__main__':
    main()
