from __future__ import annotations
import csv
import json
import math
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'lib'))
import numpy as np
from matplotlib.path import Path as MplPath
from exec20_src.fe_aniso import AnisoAnnulusFE
from exec20_src.material import material_F
from exec26_src import protocol as P
from exec26_src.fe_mapped import MappedMesh, condensed_mapped_system, mapped_released_load, solve_condensed, strip_translation
from exec26_src.geometry import horseshoe_map
from exec26_src.ls_ref import LSReference, _ti_bridge
from lekhnitskii_general import LekhnitskiiGeneral, solve_ls
OUT_DIR = PROJECT_ROOT / 'results' / 'data' / 'derived'
P0_VERTICAL = -15000000.0
P0_HORIZONTAL = -13500000.0
RATIOS = (1.25, 2.625, 4.0)
MAT_DIPS = (0.0, 22.5, 45.0, 67.5, 90.0)
RHO_MAIN = 2.0
RHO_COMPARE = 1.5
COMPARE_CELL = (2.625, 45.0)
MESH_NR, MESH_NT = (24, 128)
GRID_N = 151
GRID_STRIDE = 2
REF_N_MODES, REF_N_COLLOCATION, REF_ETA_RING = (512, 6144, 0.97)

class RingReference(LSReference):
    """Use the existing ring-mode basis with the LSReference frame conversion."""

    def __init__(self, mat, smap, *load):
        self.mat, self.smap, self.load = (mat, smap, load)
        self.rep = LekhnitskiiGeneral.from_load(_ti_bridge(mat), smap.R, np.asarray(smap.c, dtype=complex), *(-value for value in load), n_modes=REF_N_MODES, eta_ring=REF_ETA_RING)
        solution = solve_ls(self.rep, n_collocation=REF_N_COLLOCATION)
        self.coef_1, self.coef_2 = (solution['coef_1'], solution['coef_2'])
        self.wall_residual = solution['wall_traction_residual_max']
        self.rank = solution['rank']

def to_phys(x_raw, y_raw):
    """Rotate map coordinates by +90 deg into the physical frame."""
    return (-np.asarray(y_raw), np.asarray(x_raw))

def tensor_to_phys(sxx, syy, sxy):
    """Same +90 deg rotation applied to a symmetric 2-tensor."""
    return (np.asarray(syy), np.asarray(sxx), -np.asarray(sxy))

def insitu_raw():
    """Physical (sigma_h, sigma_v) expressed in raw map coordinates."""
    return (P0_VERTICAL, P0_HORIZONTAL, 0.0)

def verify_dip_convention(mat, dip_deg: float) -> float:
    """Return the measured physical dip and assert the 90 - dip relation."""
    from exec20_src.material import compliance_voigt_principal, _rotate_z, _voigt_to_tensor, _tensor_to_voigt
    a0 = compliance_voigt_principal(mat.Eh, mat.Ev, mat.nu_vh, mat.nu_hh, mat.Gvh)
    S0 = _voigt_to_tensor(a0)
    phis = np.linspace(0.0, math.pi, 3601)
    comp = np.array([_tensor_to_voigt(_rotate_z(S0, mat.dip - phi))[0, 0] for phi in phis])
    stiff_raw = math.degrees(phis[int(np.argmin(comp))])
    phys_dip = (90.0 - stiff_raw) % 180.0
    expected = (90.0 - dip_deg) % 180.0
    if min(abs(phys_dip - expected), 180.0 - abs(phys_dip - expected)) > 1.0:
        raise SystemExit(f'dip convention check failed: measured physical dip {phys_dip:.1f} deg, expected {expected:.1f} deg')
    return phys_dip

def case_id(ratio: float, dip: float) -> str:
    return f'R{ratio:.3f}_D{dip:.1f}'.replace('.', 'p')

def main() -> None:
    t_start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smap = horseshoe_map(1.0, P.R_HS, P.C_HS)
    sxx_r, syy_r, sxy_r = insitu_raw()
    th = np.linspace(0.0, 2.0 * math.pi, 721)[:-1]
    zb = smap.boundary(th)
    poly = MplPath(np.column_stack([np.real(zb), np.imag(zb)]))
    span, height = (np.ptp(np.imag(zb)), np.ptp(np.real(zb)))
    i_wide = int(np.argmax(np.abs(np.imag(zb))))
    print(f'section: span {span:.2f} m x height {height:.2f} m; widest at map theta {math.degrees(th[i_wide]):.0f} deg, {np.real(zb[i_wide]) - np.min(np.real(zb)):.2f} m above the invert', flush=True)
    half_w, half_h = (1.85 * span / 2, 1.85 * height / 2)
    gx_r, gy_r = np.meshgrid(np.linspace(-half_h, half_h, GRID_N), np.linspace(-half_w, half_w, GRID_N))
    outside = ~poly.contains_points(np.column_stack([gx_r.ravel(), gy_r.ravel()])).reshape(gx_r.shape)
    fx_all, fy_all = (gx_r[outside], gy_r[outside])
    fx, fy = (fx_all[::GRID_STRIDE], fy_all[::GRID_STRIDE])
    n_hat, _jac = smap.wall_normal(th)
    zo = zb + 0.0002 * P.R_HS * n_hat
    nx, ny = (np.real(n_hat), np.imag(n_hat))
    tx_, ty_ = (-ny, nx)
    wx, wy = to_phys(np.real(zb), np.imag(zb))
    stress_rows, hoop_rows, err_rows, disp_rows = ([], [], [], [])
    summary = []
    outputs = {'near_field_stress_hs100.csv': stress_rows, 'wall_hoop_hs100.csv': hoop_rows, 'near_field_interface_error_hs100.csv': err_rows, 'near_field_displacement_hs100.csv': disp_rows, 'near_field_case_summary.csv': summary}
    cells = [(r, d, RHO_MAIN) for d in MAT_DIPS for r in RATIOS]
    cells.append((COMPARE_CELL[0], COMPARE_CELL[1], RHO_COMPARE))
    for ratio, dip_deg, rho_out in cells:
        cid = case_id(ratio, dip_deg)
        mat = material_F(ratio, dip_deg, cid)
        phys_dip = verify_dip_convention(mat, dip_deg)
        tag = f'{cid} rho{rho_out}'
        t0 = time.perf_counter()
        ref = RingReference(mat, smap, sxx_r, syy_r, sxy_r)
        if rho_out == RHO_MAIN:
            pxx, pyy, pxy = ref.stresses_perturbation(fx, fy)
            s_hh, s_vv, s_hv = tensor_to_phys(pxx + sxx_r, pyy + syy_r, pxy + sxy_r)
            px, py = to_phys(fx, fy)
            mean = 0.5 * (s_hh + s_vv)
            rad = np.hypot(0.5 * (s_hh - s_vv), s_hv)
            for k in range(px.size):
                stress_rows.append({'case': cid, 'E_ratio': f'{ratio:.3f}', 'bedding_dip_deg': f'{phys_dip:.1f}', 'x_m': f'{px[k]:.3f}', 'y_m': f'{py[k]:.3f}', 'sigma_hh_MPa': f'{s_hh[k] / 1000000.0:.4f}', 'sigma_vv_MPa': f'{s_vv[k] / 1000000.0:.4f}', 'sigma_hv_MPa': f'{s_hv[k] / 1000000.0:.4f}', 'sigma_1_MPa': f'{(mean[k] + rad[k]) / 1000000.0:.4f}', 'sigma_3_MPa': f'{(mean[k] - rad[k]) / 1000000.0:.4f}', 'max_shear_MPa': f'{rad[k] / 1000000.0:.4f}'})
            pxx, pyy, pxy = ref.stresses_perturbation(np.real(zo), np.imag(zo))
            txx, tyy, txy = (pxx + sxx_r, pyy + syy_r, pxy + sxy_r)
            s_tt = txx * tx_ ** 2 + 2 * txy * tx_ * ty_ + tyy * ty_ ** 2
            i_pk = int(np.argmin(s_tt))
            for k in range(0, th.size, 2):
                hoop_rows.append({'case': cid, 'E_ratio': f'{ratio:.3f}', 'bedding_dip_deg': f'{phys_dip:.1f}', 'map_theta_deg': f'{math.degrees(th[k]):.1f}', 'x_m': f'{wx[k]:.3f}', 'y_m': f'{wy[k]:.3f}', 'hoop_sigma_MPa': f'{s_tt[k] / 1000000.0:.4f}'})
            peak_th, peak_val = (math.degrees(th[i_pk]), s_tt[i_pk] / 1000000.0)
        else:
            peak_th = peak_val = float('nan')
        mesh = MappedMesh(smap, rho_out, MESH_NR, MESH_NT)
        fe = AnisoAnnulusFE(mesh, mat)
        K_sys, _ = condensed_mapped_system(fe, mat, smap, rho_out, n_modes=P.N_MODES_GENERAL)
        f_load = mapped_released_load(mesh, sxx_r, syy_r, sxy_r)
        u_c = strip_translation(solve_condensed(fe, K_sys, f_load))
        ux, uy = ref.displacements(mesh.xy[:, 0], mesh.xy[:, 1])
        u_e = np.empty(mesh.n_dof)
        u_e[0::2], u_e[1::2] = (ux, uy)
        u_e = strip_translation(u_e)
        rel = float(np.linalg.norm(u_c - u_e) / np.linalg.norm(u_e))
        nxp, nyp = to_phys(mesh.xy[:, 0], mesh.xy[:, 1])
        ucx, ucy = to_phys(u_c[0::2], u_c[1::2])
        du = np.hypot(u_c[0::2] - u_e[0::2], u_c[1::2] - u_e[1::2])
        umag = np.hypot(u_c[0::2], u_c[1::2])
        ring = np.asarray(mesh.r_node).ravel()
        wall_node = np.abs(ring - 1.0) < 1e-09
        for k in range(nxp.size):
            err_rows.append({'case': cid, 'E_ratio': f'{ratio:.3f}', 'bedding_dip_deg': f'{phys_dip:.1f}', 'rho_out': f'{rho_out:.1f}', 'x_m': f'{nxp[k]:.3f}', 'y_m': f'{nyp[k]:.3f}', 'node_grade': f'{ring[k]:.4f}', 'abs_diff_um': f'{du[k] * 1000000.0:.5f}'})
            if rho_out == RHO_MAIN:
                disp_rows.append({'case': cid, 'E_ratio': f'{ratio:.3f}', 'bedding_dip_deg': f'{phys_dip:.1f}', 'x_m': f'{nxp[k]:.3f}', 'y_m': f'{nyp[k]:.3f}', 'on_wall': int(wall_node[k]), 'u_h_mm': f'{ucx[k] * 1000.0:.5f}', 'u_v_mm': f'{ucy[k] * 1000.0:.5f}', 'u_mag_mm': f'{umag[k] * 1000.0:.5f}'})
        summary.append({'case': cid, 'E_ratio': f'{ratio:.3f}', 'material_dip_deg': f'{dip_deg:.1f}', 'bedding_dip_deg': f'{phys_dip:.1f}', 'rho_out': f'{rho_out:.1f}', 'ref_basis': 'ring_modes', 'ref_n_modes': REF_N_MODES, 'ref_n_collocation': REF_N_COLLOCATION, 'ref_eta_ring': REF_ETA_RING, 'ls_wall_residual_rel': f'{ref.wall_residual:.3e}', 'peak_hoop_MPa': f'{peak_val:.3f}', 'peak_hoop_theta_deg': f'{peak_th:.1f}', 'max_wall_displacement_mm': f'{umag[wall_node].max() * 1000.0:.3f}', 'coupled_vs_reference_rel': f'{rel:.3e}', 'max_abs_diff_um': f'{du.max() * 1000000.0:.3f}'})
        print(f'  [{time.perf_counter() - t_start:6.0f}s] {tag}: bedding {phys_dip:g} deg, peak hoop {peak_val:.1f} MPa at {peak_th:.0f} deg, max displacement {umag[wall_node].max() * 1000.0:.2f} mm, rel {rel:.2e}  ({time.perf_counter() - t0:.0f}s)', flush=True)
    for name, rows in outputs.items():
        _write(OUT_DIR / name, rows)
    print(f'total {time.perf_counter() - t_start:.0f}s')

def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise SystemExit(f'no rows for {path.name}')
    with path.open('w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {path.relative_to(PROJECT_ROOT)}  ({len(rows)} rows)', flush=True)
if __name__ == '__main__':
    main()
