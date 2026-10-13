from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('lining_capacity', Path(__file__).with_name('lining_capacity.py'))
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)
lined = capacity.lined
import support_screening as screening
from exec26_src.fe_mapped import solve_condensed

def key(row):
    return tuple((row[k] for k in ('modulus_ratio', 'bedding_dip_deg', 'lateral_pressure_coefficient')))

def solve_case(mesh, fe, ground, ratio, dip, pressure):
    load = lined.mapped_released_load(mesh, *lined.insitu(pressure))
    uu = solve_condensed(fe, ground, load)
    _, ring, ug, ur = lined._assemble_and_solve(mesh, ground, load, lined.E_LIN, None)
    samples = capacity.stress_samples(ring, ur)
    fields = np.column_stack([samples[s] for s in ('xx', 'yy', 'xy', 'zz')])
    row = dict(lateral_pressure_coefficient=pressure, modulus_ratio=ratio, bedding_dip_deg=90 - dip, compression_zero_MPa=float(samples['compression'].max()), tension_zero_MPa=float(samples['tension'].max()))
    for state, u in (('lined', ug), ('unlined', uu)):
        v, h = lined.closures(mesh, u)
        row.update({f'vertical_{state}_mm': v, f'horizontal_{state}_mm': h})
    return (row, fields)

def interval_rows(rows):
    output = []
    for a, b in ((0.9, 0.9), (0.8, 1.0), (0.6, 1.4)):
        endpoints = [r for r in rows if r['lateral_pressure_coefficient'] in (a, b)]
        lower_case = max(endpoints, key=lambda r: r['release_min_stress'])
        raw_upper, upper_case, direction = min((((24.0 - r[f'{d}_lined_mm']) / (r[f'{d}_unlined_mm'] - r[f'{d}_lined_mm']), r, d) for r in endpoints for d in ('vertical', 'horizontal')), key=lambda item: item[0])
        output.append(dict(k_min=a, k_max=b, release_min_stress=lower_case['release_min_stress'], release_max_closure=min(1.0, raw_upper), unclipped_closure_ceiling=raw_upper, max_unlined_chord_mm=max((r[f'{d}_unlined_mm'] for r in endpoints for d in ('vertical', 'horizontal'))), lower_active_k=lower_case['lateral_pressure_coefficient'], lower_active_ratio=lower_case['modulus_ratio'], lower_active_bedding_deg=lower_case['bedding_dip_deg'], closure_active_k=upper_case['lateral_pressure_coefficient'], closure_active_ratio=upper_case['modulus_ratio'], closure_active_bedding_deg=upper_case['bedding_dip_deg'], closure_active_direction=direction))
    return output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/data/derived/pressure_uncertainty.csv')
    parser.add_argument('--closure-refinement', action='store_true', help="Refine the broad interval's closure-controlling case")
    args = parser.parse_args()
    if args.closure_refinement:
        output = []
        for nr, nt, layers in ((24, 128, 16), (48, 256, 32)):
            lined.LINING_NL = layers
            mesh = lined.MappedMesh(lined.SMAP, lined.RHO, nr, nt)
            mat = lined.material_F(2.625, 0.0)
            fe = lined.AnisoAnnulusFE(mesh, mat)
            ground, _ = lined.condensed_mapped_system(fe, mat, lined.SMAP, lined.RHO, n_modes=lined.P.N_MODES_GENERAL)
            row, _ = solve_case(mesh, fe, ground, 2.625, 0.0, 1.4)
            lo, hi = screening.release_interval(row, 20.0, 1.5, 24.0, 24.0)
            row.update(n_r=nr, n_t=nt, layers=layers, release_min_stress=lo, release_max_closure=hi)
            output.append(row)
            print(row, flush=True)
        capacity.write_csv(args.output, output)
        return
    saved = {key(r): r for r in screening.cases()}
    rows = dict(saved)
    mesh = lined.MappedMesh(lined.SMAP, lined.RHO, *lined.MESH_MAIN)
    for ratio in lined.RATIOS:
        for dip in lined.MAT_DIPS:
            start = time.perf_counter()
            mat = lined.material_F(ratio, dip)
            fe = lined.AnisoAnnulusFE(mesh, mat)
            ground, _ = lined.condensed_mapped_system(fe, mat, lined.SMAP, lined.RHO, n_modes=lined.P.N_MODES_GENERAL)
            fields = []
            for pressure in (0.8, 0.9, 1.0):
                row, stress = solve_case(mesh, fe, ground, ratio, dip, pressure)
                fields.append(stress)
                if pressure == 0.9:
                    old = saved[key(row)]
                    for field in row:
                        np.testing.assert_allclose(row[field], old[field], rtol=1e-08, atol=1e-09)
                rows[key(row)] = row
            np.testing.assert_allclose(fields[1], 0.5 * (fields[0] + fields[2]), rtol=1e-08, atol=1e-08)
            for d in ('vertical', 'horizontal'):
                for state in ('lined', 'unlined'):
                    values = [rows[ratio, 90 - dip, k][f'{d}_{state}_mm'] for k in (0.8, 0.9, 1.0)]
                    np.testing.assert_allclose(values[1], 0.5 * (values[0] + values[2]), atol=1e-09)
            print(f'ratio={ratio:g}, bedding={90 - dip:g}: nominal reproduction and affine stress/closure checks passed, {time.perf_counter() - start:.1f} s', flush=True)
    output = []
    for _, case in sorted(rows.items()):
        row = {k: v for k, v in case.items() if k != 'stress_zero_MPa'}
        if any((row[f'{d}_unlined_mm'] <= row[f'{d}_lined_mm'] for d in ('vertical', 'horizontal'))):
            raise ValueError('The stated closure upper bound requires increasing closure.')
        lo, hi = screening.release_interval(row, 20.0, 1.5, 24.0, 24.0)
        row.update(release_min_stress=lo, release_max_closure=hi)
        output.append(row)
    capacity.write_csv(args.output, output)
    summaries = interval_rows(output)
    capacity.write_csv(args.output.with_name(args.output.stem + '_intervals.csv'), summaries)
    for row in summaries:
        print(row, flush=True)
if __name__ == '__main__':
    main()
