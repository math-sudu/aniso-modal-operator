from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
R_HS = 5.034
C_HS = (-0.068772, -0.0604, 0.0656, -0.0031, -0.0212, 0.006, 0.0042, -0.0051, 0.0025, 0.0034)
ROOT = Path(__file__).resolve().parents[2]
OUT_PROFILE = ROOT / 'results' / 'data' / 'derived' / 'hs_section_profile.csv'
OUT_SCALARS = ROOT / 'results' / 'data' / 'derived' / 'hs_section_geometry.csv'
FAMILY_CSV = ROOT / 'results' / 'data' / 'derived' / 'section_family_boundary.csv'
N_DENSE = 400000

def _omega(zeta: np.ndarray) -> np.ndarray:
    w = zeta.astype(complex).copy()
    for j, cj in enumerate(C_HS):
        w = w + (cj if j == 0 else cj * zeta ** (-float(j)))
    return R_HS * w

def source_polar_angle_deg(theta_deg):
    """Zeng (2024) Fig. 1 polar angle delta, from the crown axis about O.

    The image-plane parameter theta is not this physical angle. Keep c_0:
    the source measures about the map origin, without subtracting R*c_0.
    In the upright frame delta = atan2(-x_phys, y_phys), modulo 360 deg.
    """
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))
    return np.mod(np.angle(_omega(np.exp(1j * theta)), deg=True), 360.0)

def _domega(zeta: np.ndarray) -> np.ndarray:
    w = np.ones_like(zeta)
    for j, cj in enumerate(C_HS):
        if j >= 1 and cj != 0.0:
            w = w - j * cj * zeta ** (-(j + 1))
    return R_HS * w

def _ddomega(zeta: np.ndarray) -> np.ndarray:
    w = np.zeros_like(zeta)
    for j, cj in enumerate(C_HS):
        if j >= 1 and cj != 0.0:
            w = w + j * (j + 1) * cj * zeta ** (-(j + 2))
    return R_HS * w

def _curvature(zeta: np.ndarray) -> np.ndarray:
    dw, ddw = (_domega(zeta), _ddomega(zeta))
    zp = 1j * zeta * dw
    zpp = -zeta * dw - zeta ** 2 * ddw
    return np.imag(np.conj(zp) * zpp) / np.abs(zp) ** 3

def _crosscheck_family_csv(theta: np.ndarray, z: np.ndarray) -> None:
    """Tie this export to the display-plan-pinned boundary sample."""
    if not FAMILY_CSV.exists():
        return
    with FAMILY_CSV.open(encoding='utf-8') as handle:
        rows = [r for r in csv.DictReader(handle) if r['member'] == 'HS100' and r['rho'] == '1.0']
    for r in rows[::max(1, len(rows) // 24)]:
        th = float(r['theta_deg'])
        i = int(round(th / 360.0 * len(theta))) % len(theta)
        dz = abs(z[i] - complex(float(r['x_m']), float(r['y_m'])))
        if dz > 0.002:
            raise AssertionError(f'HS100 boundary mismatch vs section_family_boundary.csv at theta={th} deg: |dz| = {dz:.2e} m')

def main() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, N_DENSE, endpoint=False)
    zeta = np.exp(1j * theta)
    z = _omega(zeta)
    dw = _domega(zeta)
    kappa = _curvature(zeta)
    _crosscheck_family_csv(theta, z)
    x_map, y_map = (z.real, z.imag)
    x_phys, y_phys = (-y_map, x_map)
    span = x_phys.max() - x_phys.min()
    height = y_phys.max() - y_phys.min()
    invert_y = y_phys.min()
    i_corner = int(np.argmax(kappa[:N_DENSE // 2]))
    corner_deg = np.degrees(theta[i_corner])
    kappa_max = float(kappa[i_corner])
    i_widest = int(np.argmin(x_phys))
    widest_deg = np.degrees(theta[i_widest])
    widest_h = float(y_phys[i_widest] - invert_y)
    i_kmin = int(np.argmin(kappa))
    j_idx = np.arange(len(C_HS), dtype=float)
    sum_j_cj = float(np.sum(j_idx * np.abs(np.asarray(C_HS))))
    min_dw = float(np.abs(dw).min())
    sign_changes = int(np.sum(np.sign(kappa[1:]) * np.sign(kappa[:-1]) < 0))
    OUT_PROFILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PROFILE.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['theta_deg', 'x_map_m', 'y_map_m', 'x_phys_m', 'y_phys_m', 'height_above_invert_m', 'signed_curvature_per_m', 'local_radius_m'])
        for k in range(360):
            i = int(round(k / 360.0 * N_DENSE)) % N_DENSE
            writer.writerow([f'{k:.1f}', f'{x_map[i]:.4f}', f'{y_map[i]:.4f}', f'{x_phys[i]:.4f}', f'{y_phys[i]:.4f}', f'{y_phys[i] - invert_y:.4f}', f'{kappa[i]:.4f}', f'{1.0 / abs(kappa[i]):.3f}'])
    scalars = [('span_m', f'{span:.4f}', 'widest horizontal extent, physical frame'), ('height_m', f'{height:.4f}', 'vertical extent, physical frame'), ('widest_point_theta_deg', f'{widest_deg:.1f}', 'map angle of the widest point (mirror at 360 minus this)'), ('widest_point_height_above_invert_m', f'{widest_h:.4f}', 'the greatest width sits low on the wall'), ('widest_point_height_fraction', f'{widest_h / height:.3f}', 'fraction of the section height'), ('corner_theta_deg', f'{corner_deg:.3f}', 'wall-invert corner, curvature maximum (mirror at 360 minus this)'), ('kappa_max_per_m', f'{kappa_max:.4f}', 'sharpest boundary curvature, at the wall-invert corners'), ('corner_fillet_radius_m', f'{1.0 / kappa_max:.4f}', 'equivalent fillet radius at the corners'), ('crown_kappa_per_m', f'{kappa[0]:.4f}', 'curvature at theta = 0'), ('crown_local_radius_m', f'{1.0 / kappa[0]:.3f}', 'local radius of the crown arch'), ('invert_kappa_per_m', f'{kappa[N_DENSE // 2]:.4f}', 'curvature at theta = 180'), ('invert_local_radius_m', f'{1.0 / kappa[N_DENSE // 2]:.3f}', 'local radius of the invert'), ('kappa_min_per_m', f'{kappa[i_kmin]:.4f}', f'flattest boundary point, theta = {np.degrees(theta[i_kmin]):.1f} deg (upper side wall)'), ('kappa_sign_changes', f'{sign_changes}', 'zero means the boundary is convex at every point'), ('min_abs_domega_m', f'{min_dw:.4f}', "smoothness gauge min|omega'| on the unit circle"), ('min_abs_domega_over_R', f'{min_dw / R_HS:.4f}', 'the same gauge normalized by R'), ('sum_j_abs_cj', f'{sum_j_cj:.4f}', 'coefficient-decay gauge sum of j |c_j|; below 1 on this section'), ('R_m', f'{R_HS:.3f}', 'family radius factor, frozen upstream')]
    with OUT_SCALARS.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['quantity', 'value', 'note'])
        writer.writerows(scalars)
    print(f'wrote {OUT_PROFILE}')
    print(f'wrote {OUT_SCALARS}')
if __name__ == '__main__':
    main()
