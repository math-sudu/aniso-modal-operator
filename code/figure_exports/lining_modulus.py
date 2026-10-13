from __future__ import annotations
import argparse
import importlib.util
from itertools import product
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('lining_capacity', Path(__file__).with_name('lining_capacity.py'))
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)
lined = capacity.lined

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', nargs=2, type=int, default=(24, 128))
    parser.add_argument('--layers', type=int, default=16)
    parser.add_argument('--ratios', nargs='+', type=float, default=lined.RATIOS)
    parser.add_argument('--dips', nargs='+', type=float, default=lined.MAT_DIPS)
    parser.add_argument('--pressures', nargs='+', type=float, default=(0.6, 0.9, 1.4))
    parser.add_argument('--factors', nargs='+', type=float, default=(1.0, 0.75, 0.5))
    parser.add_argument('--case', action='append', nargs=3, type=float, metavar=('RATIO', 'MATERIAL_DIP', 'PRESSURE'), help='Select explicit cases instead of the Cartesian parameter grid')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/data/derived/lining_modulus.csv')
    args = parser.parse_args()
    lined.LINING_NL = args.layers
    rows, profiles = ([], [])
    mesh = lined.MappedMesh(lined.SMAP, lined.RHO, *args.mesh)
    cases = args.case or product(args.ratios, args.dips, args.pressures)
    groups = {}
    for ratio, dip, pressure in cases:
        groups.setdefault((ratio, dip), []).append(pressure)
    for (ratio, dip), pressures in groups.items():
        start = time.perf_counter()
        mat = lined.material_F(ratio, dip)
        fe = lined.AnisoAnnulusFE(mesh, mat)
        ground, _ = lined.condensed_mapped_system(fe, mat, lined.SMAP, lined.RHO, n_modes=lined.P.N_MODES_GENERAL)
        for pressure in pressures:
            load = lined.mapped_released_load(mesh, *lined.insitu(pressure))
            for factor in args.factors:
                modulus = lined.E_LIN * factor
                _, ring, ug, ur = lined._assemble_and_solve(mesh, ground, load, modulus, pin_ring=None)
                identity = dict(ratio=ratio, material_dip_deg=dip, bedding_dip_deg=90 - dip, pressure_coefficient=pressure, n_r=args.mesh[0], n_t=args.mesh[1], layers=args.layers, lining_modulus_GPa=modulus / 1000000000.0, modulus_factor=factor, installation_release=0.0, law='bonded')
                row, profile = capacity.recover(ring, ur, identity, modulus)
                n = np.array([p['thrust_MN_per_m'] for p in profile])
                m = np.array([p['moment_kNm_per_m'] for p in profile])
                angles = np.array([p['theta_deg'] for p in profile])
                dv, dh = lined.closures(mesh, ug)
                row.update(closure_vertical_mm=dv, closure_horizontal_mm=dh, thrust_max_MN_per_m=float(n.max()), moment_absmax_kNm_per_m=float(np.abs(m).max()), thrust_max_theta_deg=float(angles[np.argmax(n)]), moment_absmax_theta_deg=float(angles[np.argmax(np.abs(m))]))
                rows.append(row)
                profiles.extend(profile)
        capacity.write_csv(args.output, rows)
        capacity.write_csv(args.output.with_name(args.output.stem + '_profiles.csv'), profiles)
        print(f'ratio={ratio:g}, material dip={dip:g}: {len(pressures) * len(args.factors)} cases, {time.perf_counter() - start:.1f} s', flush=True)
if __name__ == '__main__':
    main()
