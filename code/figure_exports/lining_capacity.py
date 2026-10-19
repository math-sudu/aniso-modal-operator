from __future__ import annotations
import argparse
import csv
import gc
import importlib.util
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/figure_exports'))
import lined_section_demo as lined

def load_evidence(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def stress_samples(ring, u, modulus=lined.E_LIN):
    """Return physical stresses from the Q4 strain field, tension positive."""
    edof = np.empty((len(ring.elems), 8), dtype=int)
    edof[:, 0::2], edof[:, 1::2] = (2 * ring.elems, 2 * ring.elems + 1)
    ue = u[edof]
    xy = ring.xy[ring.elems]
    d = lined.dmat_plane_strain(modulus, lined.NU_LIN) / 1000000.0
    parts = []
    for xi in (*lined._G2, -1.0, 1.0):
        for eta in lined._G2:
            b, _ = lined._bmat_all(ring, xi, eta)
            sig = np.einsum('ab,ebi,ei->ea', d, b, ue)
            shape = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
            point = np.einsum('a,eai->ei', shape, xy)
            tangent = np.einsum('a,eai->ei', 0.25 * np.array([-(1 - xi), -(1 + xi), 1 + xi, 1 - xi]), xy)
            tangent /= np.linalg.norm(tangent, axis=1)[:, None]
            hoop = sig[:, 0] * tangent[:, 0] ** 2 + sig[:, 1] * tangent[:, 1] ** 2 + 2 * sig[:, 2] * tangent[:, 0] * tangent[:, 1]
            zz = lined.NU_LIN * (sig[:, 0] + sig[:, 1])
            center = 0.5 * (sig[:, 0] + sig[:, 1])
            radius = np.hypot(0.5 * (sig[:, 0] - sig[:, 1]), sig[:, 2])
            low, high = (np.minimum(center - radius, zz), np.maximum(center + radius, zz))
            elements = np.arange(len(ring.elems))
            keep = np.ones(len(elements), dtype=bool)
            if xi == -1.0:
                keep = elements < ring.n_t
            elif xi == 1.0:
                keep = elements >= (ring.n_l - 1) * ring.n_t
            parts.append(dict(element=elements[keep], station=(elements % ring.n_t)[keep], surface=np.full(keep.sum(), int(abs(xi) == 1.0)), x=-point[keep, 1], y=point[keep, 0], hoop=hoop[keep], xx=sig[keep, 0], yy=sig[keep, 1], xy=sig[keep, 2], zz=zz[keep], compression=np.maximum(-low[keep], 0.0), tension=np.maximum(high[keep], 0.0)))
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}

def recover(ring, u, identity, modulus=lined.E_LIN):
    n, m = lined.ring_actions(ring, u, modulus, lined.NU_LIN)
    thickness = ring.thickness
    if np.ndim(thickness):
        thickness = 0.5 * (thickness + np.roll(thickness, -1))
    intrados = -n / thickness / 1000000.0 + 6 * m / thickness ** 2 / 1000000.0
    extrados = -n / thickness / 1000000.0 - 6 * m / thickness ** 2 / 1000000.0
    section_c = np.maximum(-np.minimum(intrados, extrados), 0.0)
    section_t = np.maximum(np.maximum(intrados, extrados), 0.0)
    samples = stress_samples(ring, u, modulus)
    ic, it = (int(np.argmax(samples[key])) for key in ('compression', 'tension'))
    gp = samples['surface'] == 0
    row = dict(**identity, section_compression_MPa=float(section_c.max()), section_tension_MPa=float(section_t.max()), compression_MPa=float(samples['compression'][ic]), tension_MPa=float(samples['tension'][it]), gauss_compression_MPa=float(samples['compression'][gp].max()), gauss_tension_MPa=float(samples['tension'][gp].max()), compression_x_m=float(samples['x'][ic]), compression_y_m=float(samples['y'][ic]), tension_x_m=float(samples['x'][it]), tension_y_m=float(samples['y'][it]), compression_at_surface=int(samples['surface'][ic]), tension_at_surface=int(samples['surface'][it]))
    profiles = []
    for j, angle in enumerate(lined.station_theta(ring.n_t)):
        mask = samples['station'] == j
        profiles.append(dict(**identity, theta_deg=angle, thrust_MN_per_m=n[j] / 1000000.0, moment_kNm_per_m=m[j] / 1000.0, section_intrados_MPa=intrados[j], section_extrados_MPa=extrados[j], compression_MPa=float(samples['compression'][mask].max()), tension_MPa=float(samples['tension'][mask].max())))
    return (row, profiles)

def checks():
    """Affine displacement and curved-ring Lame stress recovery controls."""
    from mapped_elasticity.geometry import circle_map
    ring = lined.LiningRing(circle_map(5.0), np.arange(128) * 2 * np.pi / 128, 0.4, 16)
    strain = np.array([-0.0001, 2e-05, 3e-05])
    u = np.column_stack((strain[0] * ring.xy[:, 0] + 0.5 * strain[2] * ring.xy[:, 1], strain[1] * ring.xy[:, 1] + 0.5 * strain[2] * ring.xy[:, 0])).ravel()
    samples = stress_samples(ring, u)
    expected = lined.dmat_plane_strain(lined.E_LIN, lined.NU_LIN) @ strain / 1000000.0
    for key, value in zip(('xx', 'yy', 'xy'), expected):
        np.testing.assert_allclose(samples[key], value, atol=1e-10)
    np.testing.assert_allclose(samples['zz'], lined.NU_LIN * (expected[0] + expected[1]), atol=1e-10)
    for nt, nl in ((128, 16), (256, 32)):
        ro, ri, pressure = (0.966, 0.566, 4.0)
        ring = lined.LiningRing(circle_map(ro), np.arange(nt) * 2 * np.pi / nt, ro - ri, nl)
        r = np.linalg.norm(ring.xy, axis=1)
        a = -pressure * ro ** 2 / (ro ** 2 - ri ** 2)
        b = a * ri ** 2
        ur = (1 + lined.NU_LIN) / (lined.E_LIN / 1000000.0) * ((1 - 2 * lined.NU_LIN) * a * r + b / r)
        u = (ur[:, None] * ring.xy / r[:, None]).ravel()
        sig = stress_samples(ring, u)
        rs = np.hypot(sig['x'], sig['y'])
        exact = a + b / rs ** 2
        error = np.max(np.abs(sig['hoop'] - exact)) / np.max(np.abs(exact))
        print(f'Lame local hoop recovery {nt}x{nl}: maximum relative difference {error:.5g}', flush=True)
        if nt == 256 and error >= previous:
            raise AssertionError('Curved-ring stress recovery does not improve with refinement')
        previous = error
    print('Affine strain and plane-strain axial stress checks passed', flush=True)

def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

def plastic_run(args):
    """Recover lining demand throughout the retained staged plastic example."""
    plasticity = load_evidence('localized_plasticity')
    model = plasticity.CoupledSection(*args.mesh, 2.0, 5.0)
    u = np.zeros(model.mesh.n_dof)
    ep = np.zeros((*model.weight.shape, 4))
    installed = None
    rows, profiles = ([], [])
    for lo, hi in ((0.0, 0.82), (0.82, 1.0)):
        if lo == 0.82:
            installed = u.copy()
        for fraction in np.linspace(lo, hi, int(np.ceil((hi - lo) / 0.02)) + 1)[1:]:
            u, ep, _, _ = model.solve_step(u, ep, fraction, installed)
            if installed is not None:
                wall = (u - installed)[model.wall]
                ur = np.concatenate([model.ring_transfer @ wall, wall])
                identity = dict(ratio=4.0, material_dip_deg=0.0, bedding_dip_deg=90.0, pressure_coefficient=0.6, n_r=args.mesh[0], n_t=args.mesh[1], layers=args.layers, law='bonded_plastic_ground', mu=0.0, installation_release=0.82, load_fraction=fraction)
                row, profile = recover(model.ring, ur, identity)
                rows.append(row)
                profiles.extend(profile)
                print(f"Supported load={fraction:.3f}: C/T={row['compression_MPa']:.4f}/{row['tension_MPa']:.4f} MPa", flush=True)
            else:
                print(f'Unsupported load={fraction:.3f}', flush=True)
    write_csv(args.output, rows)
    write_csv(args.output.with_name(args.output.stem + '_profiles.csv'), profiles)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', nargs=2, type=int, default=(24, 128))
    parser.add_argument('--layers', type=int, default=16)
    parser.add_argument('--ratios', nargs='+', type=float, default=lined.RATIOS)
    parser.add_argument('--dips', nargs='+', type=float, default=lined.MAT_DIPS, help='Material-axis dip; bedding dip is 90 minus this value')
    parser.add_argument('--pressures', nargs='+', type=float, default=(0.6, 0.9, 1.4))
    parser.add_argument('--contact', action='store_true')
    parser.add_argument('--plastic', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/data/derived/lining_capacity.csv')
    args = parser.parse_args()
    if args.check_only:
        checks()
        return
    lined.LINING_NL = args.layers
    if args.plastic:
        plastic_run(args)
        return
    rows, profiles = ([], [])
    contact = load_evidence('frictional_contact') if args.contact else None
    for ratio in args.ratios:
        for dip in args.dips:
            for pressure in args.pressures:
                start = time.perf_counter()
                identity = dict(ratio=ratio, material_dip_deg=dip, bedding_dip_deg=90 - dip, pressure_coefficient=pressure, n_r=args.mesh[0], n_t=args.mesh[1], layers=args.layers, law='bonded', mu=0.0, installation_release=0.0)
                if contact is None:
                    mat = lined.material_F(ratio, dip)
                    mesh, ring, ug, ur = lined.merged_operator_solve(mat, *args.mesh, lined.E_LIN, load=lined.insitu(pressure))
                    solutions = [(identity, ring, ur)]
                else:
                    model = contact.ContactSection(*args.mesh, ratio, dip, pressure)
                    solutions = []
                    for law, mu in [('bonded', 0.0), ('Coulomb', 0.0), ('Coulomb', 0.53), ('Coulomb', 0.73), ('Coulomb', 1.15)]:
                        result = model.solve(mu, bonded=law == 'bonded')
                        solutions.append(({**identity, 'law': law, 'mu': mu}, model.ring, result['lining']))
                for ident, ring, u in solutions:
                    row, profile = recover(ring, u, ident)
                    rows.append(row)
                    profiles.extend(profile)
                    print(f"r={ratio:g} bedding={90 - dip:g} k={pressure:g} {ident['law']} mu={ident['mu']:g}: continuum C/T={row['compression_MPa']:.4f}/{row['tension_MPa']:.4f} section C/T={row['section_compression_MPa']:.4f}/{row['section_tension_MPa']:.4f} ({time.perf_counter() - start:.1f}s)", flush=True)
                write_csv(args.output, rows)
                write_csv(args.output.with_name(args.output.stem + '_profiles.csv'), profiles)
                del solutions
                if contact is not None:
                    del model, result
                gc.collect()
if __name__ == '__main__':
    main()
