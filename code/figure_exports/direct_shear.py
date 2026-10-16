from __future__ import annotations
import argparse
import csv
import importlib
from pathlib import Path
import numpy as np
contact_solve = importlib.import_module('frictional_contact').contact_solve
TESTS = {'smooth': (2, 0.53, [(0.5, 0.35), (0.5, 0.33), (0.5, 0.3), (1.0, 0.55), (1.0, 0.6), (1.0, 0.55), (1.5, 0.92), (1.5, 0.8)]), 'bush_hammered': (3, 0.73, [(0.5, 0.4), (0.5, 0.55), (1.0, 0.74), (1.0, 0.8), (1.5, 1.26), (1.5, 1.15)]), 'natural': (4, 1.15, [(0.5, 0.62), (0.5, 0.8), (0.5, 0.47), (0.5, 0.62), (0.5, 0.43), (0.5, 0.49), (1.0, 0.95), (1.0, 1.12), (1.0, 0.88), (1.0, 0.9), (1.0, 1.45), (1.5, 1.5), (1.5, 1.74), (1.5, 1.9), (1.5, 1.86)])}

def compare():
    records = []
    for surface, (table, mu, observations) in TESTS.items():
        for test, (normal, measured) in enumerate(observations, 1):
            result = contact_solve(np.eye(2), np.array([normal, (mu + 1) * normal]), np.eye(2), np.zeros((2, 0)), mu)
            calculated = float(result['force'][0, 1])
            if result['state'][0] != 2 or result['jump'][0, 1] <= 0:
                raise AssertionError('The residual comparison requires positive sliding')
            np.testing.assert_allclose(result['force'][0], [normal, mu * normal], rtol=1e-12, atol=1e-14)
            records.append(dict(surface=surface, source_table=table, source_test=test, normal_stress_MPa=normal, measured_residual_MPa=measured, adopted_mu=mu, calculated_residual_MPa=calculated, difference_MPa=calculated - measured, data_role='parameter_selection_and_retrospective_assessment'))
    return records

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2] / 'results/data/derived/direct_shear.csv')
    args = parser.parse_args()
    rows = compare()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for surface, (_, _, observations) in TESTS.items():
        normal, measured = np.array(observations).T
        slope, intercept = np.linalg.lstsq(np.column_stack([normal, np.ones(len(normal))]), measured, rcond=None)[0]
        print(f'{surface}: {len(normal)} tests; table refit slope={slope:.6f}, intercept={intercept:.6f} MPa')
        for pressure in (0.5, 1.0, 1.5):
            selected = [row for row in rows if row['surface'] == surface and row['normal_stress_MPa'] == pressure]
            observed = np.array([row['measured_residual_MPa'] for row in selected])
            predicted = selected[0]['calculated_residual_MPa']
            print(f'  normal={pressure:g} MPa, n={len(observed)}, measured range=[{observed.min():.3f},{observed.max():.3f}], mean={observed.mean():.4f}, calculated={predicted:.4f} MPa, mean-relative difference={100 * (predicted / observed.mean() - 1):+.2f}%')
if __name__ == '__main__':
    main()
