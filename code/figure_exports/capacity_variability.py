from __future__ import annotations
import csv
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/figure_exports'))
import support_screening as screening

def main():
    source = ROOT / 'results/data/derived/pressure_uncertainty.csv'
    with source.open(encoding='utf-8', newline='') as stream:
        cases = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(stream)]
    rows = []
    for a, b in ((0.9, 0.9), (0.8, 1.0), (0.6, 1.4)):
        endpoints = [r for r in cases if r['lateral_pressure_coefficient'] in (a, b)]
        for fc in (16.0, 20.0, 24.0):
            bounds = [screening.release_interval(r, fc, 1.5, 24.0, 24.0) for r in endpoints]
            lo, hi = (max((v[0] for v in bounds)), min((v[1] for v in bounds)))
            row = dict(k_min=a, k_max=b, compression_limit_MPa=fc, tension_limit_MPa=1.5, lining_modulus_GPa=30.0, release_min_stress=lo, release_max_closure=hi, max_compression_at_lower_MPa=max(((1 - lo) * r['compression_zero_MPa'] for r in endpoints)), max_tension_at_lower_MPa=max(((1 - lo) * r['tension_zero_MPa'] for r in endpoints)), max_chord_at_lower_mm=max((screening.closure(r, lo, d) for r in endpoints for d in ('vertical', 'horizontal'))))
            rows.append(row)
            print(row)
    output = ROOT / 'results/data/derived/capacity_variability.csv'
    with output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
if __name__ == '__main__':
    main()
