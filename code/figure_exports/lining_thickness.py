from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('lining_capacity', Path(__file__).with_name('lining_capacity.py'))
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)
lined = capacity.lined

def thickness(theta, center, change, width):
    delta = (np.degrees(theta) - center + 180) % 360 - 180
    weight = np.where(np.abs(delta) < width, 0.5 * (1 + np.cos(np.pi * delta / width)), 0.0)
    return lined.LINING_T + change * weight

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', nargs=2, type=int, default=(24, 128))
    parser.add_argument('--layers', type=int, default=16)
    parser.add_argument('--case', action='append', nargs=3, type=float, metavar=('RATIO', 'MATERIAL_DIP', 'PRESSURE'))
    parser.add_argument('--centers', nargs='+', type=float, default=(0.0, 70.0, 112.5, 247.5))
    parser.add_argument('--changes', nargs='+', type=float, default=(-0.1, 0.1))
    parser.add_argument('--width', type=float, default=20.0)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/data/derived/lining_thickness.csv')
    args = parser.parse_args()
    cases = args.case or [(2.625, 45.0, 0.9), (1.25, 90.0, 0.6), (4.0, 0.0, 1.4)]
    variations = [(0.0, 0.0)] + [(c, dt) for c in args.centers for dt in args.changes]
    mesh = lined.MappedMesh(lined.SMAP, lined.RHO, *args.mesh)
    rows, profiles = ([], [])
    for ratio, dip, pressure in cases:
        mat = lined.material_F(ratio, dip)
        fe = lined.AnisoAnnulusFE(mesh, mat)
        ground, _ = lined.condensed_mapped_system(fe, mat, lined.SMAP, lined.RHO)
        load = lined.mapped_released_load(mesh, *lined.insitu(pressure))
        for center, change in variations:
            t = thickness(mesh.theta_nodes, center, change, args.width)
            ring = lined.LiningRing(lined.SMAP, mesh.theta_nodes, t, args.layers)
            if ring.min_detj() <= 0:
                raise ValueError('The prescribed thickness creates an inverted lining element')
            _, _, ug, ur = lined._assemble_and_solve(mesh, ground, load, lined.E_LIN, pin_ring=None, ring=ring)
            identity = dict(ratio=ratio, material_dip_deg=dip, bedding_dip_deg=90 - dip, pressure_coefficient=pressure, n_r=args.mesh[0], n_t=args.mesh[1], layers=args.layers, patch_center_deg=center, thickness_change_m=change, patch_halfwidth_deg=args.width, installation_release=0.0)
            row, profile = capacity.recover(ring, ur, identity)
            n = np.array([p['thrust_MN_per_m'] for p in profile])
            m = np.array([p['moment_kNm_per_m'] for p in profile])
            angles = lined.station_theta(ring.n_t)
            xy = lined.SMAP.boundary(np.radians(angles))
            dv, dh = lined.closures(mesh, ug)
            im = int(np.argmax(np.abs(m)))
            ic = int(np.argmax([p['compression_MPa'] for p in profile]))
            row.update(thrust_max_MN_per_m=float(n.max()), moment_absmax_kNm_per_m=float(abs(m[im])), moment_peak_theta_deg=float(angles[im]), compression_peak_theta_deg=float(angles[ic]), moment_peak_wall_x_m=float(-xy[im].imag), moment_peak_wall_y_m=float(xy[im].real), closure_vertical_mm=dv, closure_horizontal_mm=dh)
            tm = 0.5 * (t + np.roll(t, -1))
            for j, p in enumerate(profile):
                p.update(thickness_m=float(tm[j]), wall_x_m=float(-xy[j].imag), wall_y_m=float(xy[j].real))
            rows.append(row)
            profiles.extend(profile)
            print(f"{ratio:g}/{dip:g}/{pressure:g}, patch={center:g}, dt={change:g}: M={abs(m[im]):.2f} at {angles[im]:.2f} deg, C={row['compression_MPa']:.2f} MPa", flush=True)
        capacity.write_csv(args.output, rows)
        capacity.write_csv(args.output.with_name(args.output.stem + '_profiles.csv'), profiles)
if __name__ == '__main__':
    main()
