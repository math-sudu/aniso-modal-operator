import csv
import importlib.util
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'results/data/derived'
spec = importlib.util.spec_from_file_location('capacity', Path(__file__).with_name('lining_capacity.py'))
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)
lined = capacity.lined

def read(name):
    with (DATA / name).open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream))

def key(row):
    return tuple((float(row[k]) for k in ('ratio', 'material_dip_deg', 'pressure_coefficient')))

def selected_rows():
    modulus_cases = {(1.25, 0.0, 1.4), (1.25, 90.0, 0.6), (2.625, 45.0, 0.9), (2.625, 90.0, 1.4), (4.0, 0.0, 0.6), (4.0, 0.0, 1.4), (4.0, 90.0, 0.6)}
    rows = [('modulus', r) for r in read('lining_modulus.csv') if key(r) in modulus_cases and (not (key(r) == (1.25, 90.0, 0.6) and float(r['modulus_factor']) == 1.0))]
    capacity_cases = {(1.25, 90.0, 0.9), (1.25, 45.0, 0.6), (1.25, 45.0, 1.4)}
    rows += [('capacity', r) for r in read('lining_capacity.csv') if key(r) in capacity_cases]
    for r in read('lining_thickness.csv'):
        patch = (float(r['patch_center_deg']), float(r['thickness_change_m']))
        central = key(r) == (2.625, 45.0, 0.9)
        central_new = central and patch[1] != 0.0 and (patch != (247.5, -0.1))
        extrema = (key(r), patch) in {((4.0, 0.0, 1.4), (0.0, 0.1)), ((1.25, 90.0, 0.6), (247.5, -0.1)), ((1.25, 90.0, 0.6), (112.5, 0.1))}
        if central_new or extrema:
            rows.append(('thickness', r))
    return rows

def solve(mesh, ground, saved, family):
    pressure = float(saved['pressure_coefficient'])
    modulus = float(saved.get('lining_modulus_GPa', 30.0)) * 1000000000.0
    change = float(saved.get('thickness_change_m', 0.0))
    center = float(saved.get('patch_center_deg', 0.0))
    delta = (np.degrees(mesh.theta_nodes) - center + 180.0) % 360.0 - 180.0
    weight = np.where(abs(delta) < 20.0, 0.5 * (1.0 + np.cos(np.pi * delta / 20.0)), 0.0)
    thickness = lined.LINING_T + change * weight
    ring = lined.LiningRing(lined.SMAP, mesh.theta_nodes, thickness, 16)
    if ring.min_detj() <= 0.0:
        raise ValueError('The selected thickness creates an inverted element')
    force = lined.mapped_released_load(mesh, *lined.insitu(pressure))
    _, _, ug, ur = lined._assemble_and_solve(mesh, ground, force, modulus, pin_ring=None, ring=ring)
    current, profiles = capacity.recover(ring, ur, {}, modulus)
    n = np.array([r['thrust_MN_per_m'] for r in profiles])
    m = np.array([r['moment_kNm_per_m'] for r in profiles])
    angles = lined.station_theta(ring.n_t)
    dv, dh = lined.closures(mesh, ug)
    current.update(thrust_max_MN_per_m=float(n.max()), moment_absmax_kNm_per_m=float(abs(m).max()), closure_vertical_mm=dv, closure_horizontal_mm=dh, moment_peak_theta_deg=float(angles[np.argmax(abs(m))]), compression_peak_theta_deg=float(angles[np.argmax([r['compression_MPa'] for r in profiles])]))
    row = dict(family=family, ratio=float(saved['ratio']), material_dip_deg=float(saved['material_dip_deg']), bedding_dip_deg=float(saved['bedding_dip_deg']), pressure_coefficient=pressure, n_r=24, n_t=128, layers=16, lining_modulus_GPa=modulus / 1000000000.0, patch_center_deg=center, thickness_change_m=change, cauchy_resultant_order=127, work_pairing='two-sided P1')
    metrics = ('section_compression_MPa', 'compression_MPa', 'tension_MPa', 'thrust_max_MN_per_m', 'moment_absmax_kNm_per_m', 'closure_vertical_mm', 'closure_horizontal_mm', 'moment_peak_theta_deg', 'compression_peak_theta_deg')
    for metric in metrics:
        if metric not in saved:
            continue
        row['saved_' + metric] = float(saved[metric])
        row['current_' + metric] = current[metric]
        row['change_' + metric] = current[metric] - float(saved[metric])
    return row

def main():
    rows, groups = ([], {})
    for family, saved in selected_rows():
        groups.setdefault(key(saved)[:2], []).append((family, saved))
    mesh = lined.MappedMesh(lined.SMAP, lined.RHO, 24, 128)
    with threadpool_limits(limits=2):
        for (ratio, dip), cases in groups.items():
            mat = lined.material_F(ratio, dip)
            fe = lined.AnisoAnnulusFE(mesh, mat)
            ground, _ = lined.condensed_mapped_system(fe, mat, lined.SMAP, lined.RHO)
            for family, saved in cases:
                row = solve(mesh, ground, saved, family)
                rows.append(row)
                print(f"{family}: {ratio:g}/{dip:g}/{row['pressure_coefficient']:g}, E={row['lining_modulus_GPa']:g}, patch={row['patch_center_deg']:g}/{row['thickness_change_m']:g}, dC={row['change_compression_MPa']:.3g} MPa", flush=True)
    fields = list(dict.fromkeys((k for row in rows for k in row)))
    capacity.write_csv(DATA / 'support_endpoints.csv', [{k: r.get(k, '') for k in fields} for r in rows])
if __name__ == '__main__':
    main()
