from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/figure_exports'))
import lined_section_demo as lined
import support_screening as screening
from exec26_src.fe_mapped import solve_condensed

def axial_stress(sig, mat):
    """Recover sigma_zz from the TI 3D compliance, in the raw map frame."""
    c, s = (np.cos(mat.dip), np.sin(mat.dip))
    parallel = c * c * sig[..., 0] + s * s * sig[..., 1] + 2 * c * s * sig[..., 2]
    normal = s * s * sig[..., 0] + c * c * sig[..., 1] - 2 * c * s * sig[..., 2]
    return mat.nu_hh * parallel + mat.nu_vh * mat.Eh / mat.Ev * normal

def stress_3d(sig, mat):
    """Return xx, yy, xy, zz in MPa; yz=xz=0 for this bedding orientation."""
    return np.column_stack((sig, axial_stress(sig, mat)))

def principal_extremes(sig):
    """Minimum and maximum tension-positive principal stresses."""
    center = (sig[..., 0] + sig[..., 1]) / 2
    radius = np.hypot((sig[..., 0] - sig[..., 1]) / 2, sig[..., 2])
    return (np.minimum(center - radius, sig[..., 3]), np.maximum(center + radius, sig[..., 3]))

def strength_demand(sig, friction=30.0):
    """Minimum cohesion and tensile strength needed at each stress point."""
    low, high = principal_extremes(sig)
    nphi = (1 + np.sin(np.deg2rad(friction))) / (1 - np.sin(np.deg2rad(friction)))
    cohesion = (-low + nphi * high) / (2 * np.sqrt(nphi))
    return (np.maximum(cohesion, 0), np.maximum(high, 0))

def first_yield(s0, du, cohesion, friction=30.0):
    """First shear yield on the unlined path, or 1 if no yield before 1.

    Mohr-Coulomb is convex in stress. Starting inside the elastic domain,
    the first exit along an affine stress path is the unique crossing.
    Tensile demand is reported separately, without assigning a tensile strength.
    """
    nphi = (1 + np.sin(np.deg2rad(friction))) / (1 - np.sin(np.deg2rad(friction)))

    def excess(release):
        low, high = principal_extremes(s0 + release * du)
        return float((-low + nphi * high - 2 * cohesion * np.sqrt(nphi)).max())
    if excess(0) >= 0:
        return (0.0, 'initial_yield')
    if excess(1) <= 0:
        return (1.0, 'elastic_to_full_release')
    return (brentq(excess, 0, 1), 'shear_yield')

def gauss_stress(mesh, mat, u):
    edof = np.empty((len(mesh.elems), 8), dtype=int)
    edof[:, 0::2], edof[:, 1::2] = (2 * mesh.elems, 2 * mesh.elems + 1)
    ue = u[edof]
    stresses, coordinates = ([], [])
    for xi in lined._G2:
        for eta in lined._G2:
            b, _ = lined._bmat_all(mesh, xi, eta)
            sig = np.einsum('kl,elb,eb->ek', mat.D(), b, ue) / 1000000.0
            n = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
            coordinates.append(np.einsum('a,eai->ei', n, mesh.xy[mesh.elems]))
            stresses.append(stress_3d(sig, mat))
    return (np.concatenate(stresses), np.concatenate(coordinates))

def case_rows(mat, ratio, dip, n_r, n_t, lateral_values, release):
    mesh = lined.MappedMesh(lined.SMAP, lined.RHO, n_r, n_t)
    fe = lined.AnisoAnnulusFE(mesh, mat)
    k, _ = lined.condensed_mapped_system(fe, mat, lined.SMAP, lined.RHO, n_modes=lined.P.N_MODES_GENERAL)
    rows = []
    for lateral in lateral_values:
        load = lined.insitu(lateral)
        f = lined.mapped_released_load(mesh, *load)
        uu = solve_condensed(fe, k, f)
        _, ring, ul, ur = lined._assemble_and_solve(mesh, k, f, lined.E_LIN, None)
        du, xy = gauss_stress(mesh, mat, uu)
        dl, _ = gauss_stress(mesh, mat, ul)
        s0 = stress_3d(np.array(load)[None, :] / 1000000.0, mat)
        base = dict(n_r=n_r, n_t=n_t, lateral_pressure_coefficient=lateral, modulus_ratio=ratio, bedding_dip_deg=90 - dip, friction_deg=30.0, axial_prestress_MPa=float(s0[0, 3]))
        for lam in (0.0, release, 1.0):
            states = (s0 + np.zeros_like(du), s0 + lam * du, s0 + lam * du + (1 - lam) * dl)
            cs, ts = zip(*(strength_demand(sig) for sig in states))
            carray = np.array(cs)
            stage, point = np.unravel_index(np.argmax(carray), carray.shape)
            rows.append({**base, 'quantity': 'path_strength_requirement', 'release': lam, 'cohesion_MPa': float(carray.max()), 'tension_MPa': float(np.max(ts)), 'state': ('initial', 'installation', 'final')[stage], 'x_physical_m': -xy[point, 1], 'y_physical_m': xy[point, 0]})
        for cohesion in (5.0, 10.0, 20.0):
            lam, status = first_yield(s0, du, cohesion)
            cs, ts = strength_demand(s0 + lam * du)
            point = int(np.argmax(cs))
            rows.append({**base, 'quantity': 'unlined_first_shear_yield', 'release': lam, 'cohesion_MPa': cohesion, 'tension_MPa': float(ts.max()), 'state': status, 'x_physical_m': -xy[point, 1], 'y_physical_m': xy[point, 0]})
        n, m = lined.ring_actions(ring, ur, lined.E_LIN, lined.NU_LIN)
        stress = lined.fiber_stress(n, m, lined.LINING_T).max() / 1000000.0
        print(f"mesh={n_r}x{n_t} ratio={ratio:g} dip={90 - dip:g} K={lateral:g} fiber={stress:.6f} MPa c_req={rows[-5]['cohesion_MPa']:.6f}", flush=True)
        if (n_r, n_t) == lined.MESH_MAIN:
            saved = next((r for r in screening.cases() if r['lateral_pressure_coefficient'] == lateral and r['modulus_ratio'] == ratio and (r['bedding_dip_deg'] == 90 - dip)))
            np.testing.assert_allclose(stress, saved['stress_zero_MPa'], rtol=1e-08)
    return rows

def checks():
    p, cohesion = (10.0, 2.0)
    s0 = np.array([[-p, -p, 0, -0.5 * p]])
    du = np.array([[p, -p, 0, 0]])
    lam, status = first_yield(s0, du, cohesion)
    np.testing.assert_allclose(lam, (2 * p + 2 * np.sqrt(3) * cohesion) / (4 * p), atol=1e-12)
    assert status == 'shear_yield'
    np.testing.assert_allclose(first_yield(s0, du, 0)[0], 0.5, atol=1e-12)
    from exec20_src.material import compliance_voigt_principal, _voigt_to_tensor, _rotate_z, _tensor_to_voigt
    sig = np.array([[-7.0, -15.0, 3.0]])
    for dip in (0, 31, 90):
        mat = lined.material_F(4, dip)
        a = _tensor_to_voigt(_rotate_z(_voigt_to_tensor(compliance_voigt_principal(mat.Eh, mat.Ev, mat.nu_vh, mat.nu_hh, mat.Gvh)), mat.dip))
        szz = axial_stress(sig, mat)
        np.testing.assert_allclose(szz, -(sig @ a[2, [0, 1, 5]]) / a[2, 2], atol=1e-12)
    low, high = principal_extremes(np.array([[-5, -8, 0, -20]]))
    np.testing.assert_equal(low, [-20])
    np.testing.assert_equal(high, [-5])
    print('Analytic first-yield and 3D axial-stress checks passed.', flush=True)

def unlined_reference(ratio, bedding_dip, lateral, n_modes=512, wall_stations=1440):
    """Resolve first shear yield with the independent complex-potential field.

    Sample the smooth wall and concentric mapped curves extending to the far
    field. This resolves the stress-concentration location without Q4 stress
    extrapolation. The wall traction residual is printed for interpretation.
    """
    import near_field_clouds as clouds
    clouds.REF_N_MODES = n_modes
    mat = lined.material_F(ratio, 90 - bedding_dip)
    load = lined.insitu(lateral)
    ref = clouds.RingReference(mat, lined.SMAP, *load)
    theta = np.arange(wall_stations) * (2 * np.pi / wall_stations)
    rhos = (1.0, 1.01, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0)
    z = np.concatenate([lined.SMAP.omega(rho * np.exp(1j * theta)) for rho in rhos])
    sig = np.column_stack(ref.stresses_perturbation(z.real, z.imag)) / 1000000.0
    du = stress_3d(sig, mat)
    s0 = stress_3d(np.array(load)[None, :] / 1000000.0, mat)
    rows = []
    for cohesion in (5.0, 10.0, 20.0):
        lam, status = first_yield(s0, du, cohesion)
        cs, ts = strength_demand(s0 + lam * du)
        point = int(np.argmax(cs))
        state_stress = s0[0] + lam * du[point]
        center = 0.5 * (state_stress[0] + state_stress[1])
        radius = np.hypot(0.5 * (state_stress[0] - state_stress[1]), state_stress[2])
        principal = np.sort([center - radius, center + radius, state_stress[3]])
        initial_low, initial_high = principal_extremes(s0)
        nphi = 3.0
        rows.append(dict(n_modes=n_modes, wall_stations=len(theta), lateral_pressure_coefficient=lateral, modulus_ratio=ratio, bedding_dip_deg=bedding_dip, friction_deg=30.0, axial_prestress_MPa=float(s0[0, 3]), cohesion_MPa=cohesion, release=lam, tension_MPa=float(ts.max()), state=status, rho=rhos[point // len(theta)], x_physical_m=-z[point].imag, y_physical_m=z[point].real, sigma_min_MPa=principal[0], sigma_mid_MPa=principal[1], sigma_max_MPa=principal[2], shear_yield_residual_MPa=-principal[0] + nphi * principal[2] - 2 * cohesion * np.sqrt(nphi), initial_shear_yield_MPa=float((-initial_low + nphi * initial_high - 2 * cohesion * np.sqrt(nphi))[0]), initial_tensile_yield_MPa=float(initial_high[0]), wall_traction_residual=ref.wall_residual))
        print(f"Reference n={n_modes} c={cohesion:g}: lambda_y={lam:.8f} rho={rows[-1]['rho']} tension={ts.max():.6f} MPa", flush=True)
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', nargs=2, type=int, default=(24, 128))
    parser.add_argument('--case', nargs=3, type=float, action='append', metavar=('RATIO', 'BEDDING_DIP', 'K'))
    parser.add_argument('--output', type=Path, default=ROOT / 'results/data/derived/elastic_release_admissibility.csv')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--reference-modes', nargs='+', type=int)
    parser.add_argument('--wall-stations', nargs='+', type=int, default=[1440])
    args = parser.parse_args()
    checks()
    if args.check_only:
        return
    release = max((r['release_min_stress'] for r in screening.screening_rows()))
    cases = [(r, d, (0.6, 0.9, 1.4)) for r in lined.RATIOS for d in lined.MAT_DIPS]
    if args.case:
        cases = [(r, 90 - d, (k,)) for r, d, k in args.case]
    rows = []
    start = time.perf_counter()
    for ratio, dip, lateral in cases:
        if args.reference_modes:
            for k in lateral:
                for n_modes in args.reference_modes:
                    for stations in args.wall_stations:
                        rows.extend(unlined_reference(ratio, 90 - dip, k, n_modes, stations))
        else:
            mat = lined.material_F(ratio, dip)
            rows.extend(case_rows(mat, ratio, dip, *args.mesh, lateral, release))
    with args.output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {args.output}; {time.perf_counter() - start:.1f} s', flush=True)
if __name__ == '__main__':
    main()
