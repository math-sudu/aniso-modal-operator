from __future__ import annotations
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'results/data/derived'
COMPRESSION_LIMIT_MPA = 20.0
TENSION_LIMIT_MPA = 1.5
CLOSURE_LIMIT_MM = 24.0
CAPACITY = ROOT / 'results/data/derived/lining_capacity.csv'

def read_csv(name):
    with (DATA / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))

def cases():
    """Match lined and unlined states by load, material, mesh and stiffness."""
    with CAPACITY.open(encoding='utf-8', newline='') as stream:
        capacity = {(float(r['pressure_coefficient']), float(r['ratio']), float(r['bedding_dip_deg'])): r for r in csv.DictReader(stream)}
    keyed = {}
    for name in ('lined_demo_summary.csv', 'lined_lateral_sweep.csv'):
        for row in read_csv(name):
            if row['near_mesh'] != '24x128' or float(row['s_scale']) != 1.0:
                continue
            if row['route'] not in ('condensed', 'unlined'):
                continue
            key = (float(row.get('lateral_pressure_coefficient', 0.9)), float(row['ratio']), float(row['bedding_dip_deg']))
            pair = keyed.setdefault(key, {})
            if row['route'] in pair:
                raise ValueError(f"Duplicate state: {key}, {row['route']}")
            pair[row['route']] = row
    output = []
    for (lateral, ratio, dip), pair in sorted(keyed.items()):
        lined, unlined = (pair['condensed'], pair['unlined'])
        row = {'lateral_pressure_coefficient': lateral, 'modulus_ratio': ratio, 'bedding_dip_deg': dip, 'stress_zero_MPa': float(lined['fiber_stress_max_MPa'])}
        stress = capacity[lateral, ratio, dip]
        row.update(compression_zero_MPa=float(stress['compression_MPa']), tension_zero_MPa=float(stress['tension_MPa']))
        for direction in ('vertical', 'horizontal'):
            key = f'closure_{direction}_mm'
            row[f'{direction}_lined_mm'] = float(lined[key])
            row[f'{direction}_unlined_mm'] = float(unlined[key])
            if float(unlined[key]) <= float(lined[key]):
                raise ValueError('This interval formula requires increasing closure.')
        output.append(row)
    return output

def closure(row, release, direction):
    return (1 - release) * row[f'{direction}_lined_mm'] + release * row[f'{direction}_unlined_mm']

def release_interval(row, compression_limit, tension_limit, vertical_limit, horizontal_limit):
    lower = max([0.0] + [1.0 - limit / row[key] for key, limit in (('compression_zero_MPa', compression_limit), ('tension_zero_MPa', tension_limit)) if row[key] > 0.0])
    upper = min(1.0, *[(limit - row[f'{direction}_lined_mm']) / (row[f'{direction}_unlined_mm'] - row[f'{direction}_lined_mm']) for direction, limit in (('vertical', vertical_limit), ('horizontal', horizontal_limit))])
    return (lower, upper)

def screening_rows():
    rows = []
    for case in cases():
        lo, hi = release_interval(case, COMPRESSION_LIMIT_MPA, TENSION_LIMIT_MPA, CLOSURE_LIMIT_MM, CLOSURE_LIMIT_MM)
        vertical = closure(case, lo, 'vertical')
        horizontal = closure(case, lo, 'horizontal')
        rows.append({**case, 'compression_limit_MPa': COMPRESSION_LIMIT_MPA, 'tension_limit_MPa': TENSION_LIMIT_MPA, 'closure_limit_mm': CLOSURE_LIMIT_MM, 'release_min_stress': lo, 'release_max_closure': hi, 'stress_closure_overlap': int(lo <= hi), 'vertical_at_stress_limit_mm': vertical, 'horizontal_at_stress_limit_mm': horizontal, 'minimum_common_allowance_mm': max(vertical, horizontal), 'controlling_chord': 'vertical' if vertical >= horizontal else 'horizontal'})
    return rows

def verify_saved_partition(rows):
    """Check interpolation against the independently retained release export."""
    lookup = {(r['lateral_pressure_coefficient'], r['modulus_ratio'], r['bedding_dip_deg']): r for r in rows}
    errors = []
    for saved in read_csv('lined_release_partition.csv'):
        key = tuple((float(saved[k]) for k in ('lateral_pressure_coefficient', 'ratio', 'bedding_dip_deg')))
        row = lookup[key]
        release = float(saved['lambda_released'])
        errors.append(abs((1 - release) * row['stress_zero_MPa'] - float(saved['fiber_stress_max_MPa'])))
        for direction in ('vertical', 'horizontal'):
            errors.append(abs(closure(row, release, direction) - float(saved[f'closure_{direction}_mm'])))
    if max(errors) > 1e-10:
        raise ValueError(f'Saved release partition disagrees: {max(errors)}')
    return max(errors)

def main():
    rows = screening_rows()
    error = verify_saved_partition(rows)
    output = DATA / 'lined_support_screening.csv'
    with output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    lower = max((r['release_min_stress'] for r in rows))
    upper = min((r['release_max_closure'] for r in rows))
    critical = max(((closure(r, lower, d), r, d) for r in rows for d in ('vertical', 'horizontal')))
    print(f"{len(rows)} cases; {sum((r['stress_closure_overlap'] for r in rows))} with overlapping stress/closure bounds")
    print(f'Common stress/closure bounds before the ground-yield check: [{lower:.8f}, {upper:.8f}]')
    print(f'Minimum common allowance: {critical[0]:.8f} mm, {critical[2]} chord')
    print(f'Saved release-partition agreement: {error:.3g}')
    for lateral in sorted({r['lateral_pressure_coefficient'] for r in rows}):
        subset = [r for r in rows if r['lateral_pressure_coefficient'] == lateral]
        print(f"K={lateral:g}: minimum release {min((r['release_min_stress'] for r in subset)):.4f}--{max((r['release_min_stress'] for r in subset)):.4f}; controlling chords {sorted({r['controlling_chord'] for r in subset})}")
if __name__ == '__main__':
    main()
