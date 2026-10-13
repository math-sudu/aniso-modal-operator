from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DER = PROJECT_ROOT / 'results' / 'data' / 'derived'
FIG = PROJECT_ROOT / 'figures'
SUMMARY_CONSUMERS = ['stress-field-mosaic', 'interface-transparency', 'displacement-field']
HOOP_STRIDE_DEG = 5.0

def read(name: str) -> list[dict]:
    with (DER / name).open(encoding='utf-8') as fh:
        return list(csv.DictReader(fh))

def write(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {path.relative_to(PROJECT_ROOT)}  ({len(rows)} rows)')

def main() -> None:
    summary = read('near_field_case_summary.csv')
    fields = list(summary[0])
    for job in SUMMARY_CONSUMERS:
        write(FIG / job / 'data' / 'near_field_case_summary.csv', fields, summary)
    hoop = [r for r in read('wall_hoop_hs100.csv') if float(r['map_theta_deg']) % HOOP_STRIDE_DEG == 0.0]
    write(FIG / 'stress-field-mosaic' / 'data' / 'wall_hoop_profile_5deg.csv', list(hoop[0]), hoop)
    err = read('near_field_interface_error_hs100.csv')
    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in err:
        buckets[r['case'], r['rho_out'], r['node_grade']].append(float(r['abs_diff_um']))
    prof = [{'case': case, 'rho_out': rho, 'node_grade': grade, 'n_nodes': len(vals), 'mean_abs_diff_um': f'{sum(vals) / len(vals):.5f}', 'max_abs_diff_um': f'{max(vals):.5f}', 'min_abs_diff_um': f'{min(vals):.5f}'} for (case, rho, grade), vals in buckets.items()]
    write(FIG / 'interface-transparency' / 'data' / 'interface_error_radial_profile.csv', ['case', 'rho_out', 'node_grade', 'n_nodes', 'mean_abs_diff_um', 'max_abs_diff_um', 'min_abs_diff_um'], prof)
    disp = [{k: r[k] for k in ('case', 'E_ratio', 'bedding_dip_deg', 'x_m', 'y_m', 'u_h_mm', 'u_v_mm', 'u_mag_mm')} for r in read('near_field_displacement_hs100.csv') if r['on_wall'] == '1']
    write(FIG / 'displacement-field' / 'data' / 'wall_displacement_profile.csv', ['case', 'E_ratio', 'bedding_dip_deg', 'x_m', 'y_m', 'u_h_mm', 'u_v_mm', 'u_mag_mm'], disp)
if __name__ == '__main__':
    main()
