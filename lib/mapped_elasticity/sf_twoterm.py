from __future__ import annotations
import numpy as np
from anisotropic_elasticity.material import AnisoMaterial
from .geometry import SectionMap
__all__ = ['TwoTermExterior', 'solve_dirichlet_twoterm', 'solve_neumann_twoterm', 'mode_traction_density_cf', 'modal_blocks_density', 'nodal_sf_density', 'released_resultants']

def _zeta_from_zk(gam: complex, dlt: complex, zk: np.ndarray) -> np.ndarray:
    """Invert z_k = gam zeta + dlt / zeta on |zeta| >~ 1 (exterior
branch of the quadratic; extended to arbitrary (gam, dlt) with |dlt| < |gam|)."""
    qh = dlt / gam
    zk = np.asarray(zk, dtype=complex)
    disc = np.sqrt(zk * zk - 4.0 * gam * dlt)
    zp = (zk + disc) / (2.0 * gam)
    zm = (zk - disc) / (2.0 * gam)
    zeta = np.where(np.abs(zp) >= np.abs(zm), zp, zm)
    thr = max(0.95, 1.05 * np.sqrt(abs(qh)))
    if np.any(np.abs(zeta) < thr):
        raise ValueError(f'two-term inverse reached |zeta| < {thr:.3f} (inside the mapped hole or near the critical locus sqrt(|qhat_ell|))')
    return zeta

class TwoTermExterior:
    """Truncated decaying two-branch field outside the two-term section
    boundary (wall = smap.boundary(theta), the rho = 1 curve of smap;
    interface tiers are reached by feeding smap.scaled(rho))."""

    def __init__(self, mat: AnisoMaterial, smap: SectionMap, n_modes: int):
        if smap.n_map > 1:
            raise ValueError('TwoTermExterior is the c = [c0, c1] tier')
        self.mat = mat
        self.smap = smap
        self.n_modes = int(n_modes)
        self.gd = [smap.twoterm_gamma_delta(mu) for mu in mat.mus]
        for k, (g, d) in enumerate(self.gd):
            if abs(d) >= abs(g):
                raise ValueError(f'branch {k}: |del| >= |gam| (two-term image map not univalent)')
        self.A = np.zeros((2, self.n_modes), dtype=complex)

    def _phi_dphi(self, zeta: np.ndarray, k: int):
        zinv = 1.0 / zeta
        phi = np.zeros_like(zeta)
        dphi = np.zeros_like(zeta)
        zm = np.ones_like(zeta)
        for m in range(1, self.n_modes + 1):
            zm = zm * zinv
            a = self.A[k, m - 1]
            if a != 0.0:
                phi = phi + a * zm
                dphi = dphi - m * a * zm * zinv
        return (phi, dphi)

    def fields_xy(self, x, y):
        """Tension-positive Cartesian fields at physical points (x, y)
        in the represented exterior."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        sxx = 0.0
        syy = 0.0
        sxy = 0.0
        ux = 0.0
        uy = 0.0
        for k, mu in enumerate(self.mat.mus):
            gam, dlt = self.gd[k]
            zk = x + mu * y
            c0 = self.smap.c[0]
            if c0 != 0.0:
                bm, bp = SectionMap._betas(mu)
                zk = zk - self.smap.R * (bm * c0 + bp * np.conj(c0))
            zeta = _zeta_from_zk(gam, dlt, zk)
            phi, dphi = self._phi_dphi(zeta, k)
            omp = gam - dlt * zeta ** (-2)
            Phi = dphi / omp
            p, q = self.mat.pq[k]
            sxx = sxx + 2.0 * np.real(mu * mu * Phi)
            syy = syy + 2.0 * np.real(Phi)
            sxy = sxy - 2.0 * np.real(mu * Phi)
            ux = ux + 2.0 * np.real(p * phi)
            uy = uy + 2.0 * np.real(q * phi)
        return {'s_xx': sxx, 's_yy': syy, 's_xy': sxy, 'u_x': ux, 'u_y': uy}

    def boundary_fields(self, theta):
        """Fields exactly on the wall (zeta_k = sigma, no inversion)."""
        theta = np.asarray(theta, dtype=float)
        sigma = np.exp(1j * theta)
        sxx = 0.0
        syy = 0.0
        sxy = 0.0
        ux = 0.0
        uy = 0.0
        for k, mu in enumerate(self.mat.mus):
            gam, dlt = self.gd[k]
            phi, dphi = self._phi_dphi(sigma, k)
            omp = gam - dlt * sigma ** (-2)
            Phi = dphi / omp
            p, q = self.mat.pq[k]
            sxx = sxx + 2.0 * np.real(mu * mu * Phi)
            syy = syy + 2.0 * np.real(Phi)
            sxy = sxy - 2.0 * np.real(mu * Phi)
            ux = ux + 2.0 * np.real(p * phi)
            uy = uy + 2.0 * np.real(q * phi)
        return {'s_xx': sxx, 's_yy': syy, 's_xy': sxy, 'u_x': ux, 'u_y': uy}

    def boundary_traction_density(self, theta):
        """(t_x, t_y) per-d-theta density on the wall through the FIELD
        path: t_value = sigma . n (outward normal), density = t_value *
        |dz/dtheta| -- the cross-check twin of the closed-form modal
        route (they must agree to spectral accuracy)."""
        f = self.boundary_fields(theta)
        n, jac = self.smap.wall_normal(theta)
        nx, ny = (np.real(n), np.imag(n))
        tx = f['s_xx'] * nx + f['s_xy'] * ny
        ty = f['s_xy'] * nx + f['s_yy'] * ny
        return (tx * jac, ty * jac)

def _fft_neg_coeffs(f: np.ndarray, n_modes: int) -> np.ndarray:
    """Coefficients of e^{-i m th}, m = 1..n_modes."""
    N = len(f)
    F = np.fft.fft(np.asarray(f, dtype=float)) / N
    return np.array([F[(N - m) % N] for m in range(1, n_modes + 1)])

def solve_dirichlet_twoterm(mat: AnisoMaterial, smap: SectionMap, u_x: np.ndarray, u_y: np.ndarray, n_modes: int) -> TwoTermExterior:
    """Displacement values on the wall (uniform theta grid in the MAP
    parameter) -> decaying representation.  Identical per-harmonic 2x2
    to the circle case (material matrix only); harmonic 0 projected out
    (rigid translation, zero-response bookkeeping)."""
    (p1, q1), (p2, q2) = mat.pq
    M = np.array([[p1, p2], [q1, q2]], dtype=complex)
    ext = TwoTermExterior(mat, smap, n_modes)
    cx = _fft_neg_coeffs(u_x, n_modes)
    cy = _fft_neg_coeffs(u_y, n_modes)
    for m in range(1, n_modes + 1):
        A = np.linalg.solve(M, np.array([cx[m - 1], cy[m - 1]]))
        ext.A[0, m - 1] = A[0]
        ext.A[1, m - 1] = A[1]
    return ext

def released_resultants(smap: SectionMap, sxx_inf: float, syy_inf: float, sxy_inf: float, theta: np.ndarray):
    """Closed-form resultant functions (X, Y)(theta) of the RELEASED
    uniform far-field traction on the wall (perturbation boundary
    values, tension-positive): t ds of the constant state -sigma_inf
    integrates to X = -(sxx (y - y0) - sxy (x - x0)),
    Y = -(sxy (y - y0) - syy (x - x0)) with z(theta) closed-form."""
    z = smap.boundary(theta)
    x, y = (np.real(z), np.imag(z))
    X = -(sxx_inf * (y - y[0]) - sxy_inf * (x - x[0]))
    Y = -(sxy_inf * (y - y[0]) - syy_inf * (x - x[0]))
    return (X, Y)

def solve_neumann_twoterm(mat: AnisoMaterial, smap: SectionMap, sxx_inf: float, syy_inf: float, sxy_inf: float, n_modes: int, n_grid: int=512) -> TwoTermExterior:
    """Released-load Neumann solve via the potential-level resultant
    form: X = 2Re[mu1 phi1 + mu2 phi2] + C, Y = -2Re[phi1 + phi2] + C
    (shape-independent identities); per-harmonic [[mu1, mu2], [1, 1]]."""
    th = 2.0 * np.pi * np.arange(n_grid) / n_grid
    X, Y = released_resultants(smap, sxx_inf, syy_inf, sxy_inf, th)
    mu1, mu2 = mat.mus
    M = np.array([[mu1, mu2], [1.0, 1.0]], dtype=complex)
    ext = TwoTermExterior(mat, smap, n_modes)
    cX = _fft_neg_coeffs(X, n_modes)
    cY = _fft_neg_coeffs(Y, n_modes)
    for m in range(1, n_modes + 1):
        A = np.linalg.solve(M, np.array([cX[m - 1], -cY[m - 1]]))
        ext.A[0, m - 1] = A[0]
        ext.A[1, m - 1] = A[1]
    return ext

def mode_traction_density_cf(mat: AnisoMaterial, k: int, m: int, unit: complex):
    """EXACT single-harmonic traction-density coefficients of one mode
    phi_k = unit sigma^{-m} on ANY two-term section: (c_tx, c_ty) of
    e^{-i m th} with c_tx = -i m unit mu_k, c_ty = +i m unit.  Geometry
    enters nowhere (resultant identities); equals R times the
    circle traction-VALUE coefficients (ds = R dth)."""
    mu = mat.mus[k]
    base = -1j * m * unit
    return (mu * base, -base)
_COMPS = ('Uc', 'Us', 'Vc', 'Vs')

def _basis_profiles(n: int, comp: str, theta: np.ndarray):
    c, s = (np.cos(n * theta), np.sin(n * theta))
    z = np.zeros_like(theta)
    return {'Uc': (c, z), 'Us': (s, z), 'Vc': (z, c), 'Vs': (z, s)}[comp]

def modal_blocks_density(mat: AnisoMaterial, n_h: int) -> np.ndarray:
    """Closed-form modal DtN in density pairing, block-diagonal per
harmonic: returns S_mod (4 n_h, 4 n_h) on the ordered real basis
[(n, comp)] n = 1..n_h, comps (Uc, Us, Vc, Vs).  Material-only (the two-term geometry invariance property)."""
    (p1, q1), (p2, q2) = mat.pq
    M = np.array([[p1, p2], [q1, q2]], dtype=complex)
    mu1, mu2 = mat.mus
    S = np.zeros((4 * n_h, 4 * n_h))
    for n in range(1, n_h + 1):
        blk = np.zeros((4, 4))
        for jc, comp in enumerate(_COMPS):
            cx = {'Uc': 0.5, 'Us': 0.5j, 'Vc': 0.0, 'Vs': 0.0}[comp]
            cy = {'Uc': 0.0, 'Us': 0.0, 'Vc': 0.5, 'Vs': 0.5j}[comp]
            A1, A2 = np.linalg.solve(M, np.array([cx, cy]))
            ctx = -1j * n * (mu1 * A1 + mu2 * A2)
            cty = 1j * n * (A1 + A2)
            blk[0, jc] = 2.0 * np.real(ctx)
            blk[1, jc] = 2.0 * np.imag(ctx)
            blk[2, jc] = 2.0 * np.real(cty)
            blk[3, jc] = 2.0 * np.imag(cty)
        i0 = 4 * (n - 1)
        S[i0:i0 + 4, i0:i0 + 4] = blk
    return S

def nodal_sf_density(mat: AnisoMaterial, theta_nodes: np.ndarray, n_h: int | None=None) -> np.ndarray:
    """Nodal far DtN in DENSITY pairing: interleaved Cartesian ring
    displacements (map-parameter nodes) -> interleaved Cartesian
    traction-density values.  Closed-form columns (no field solve);
    harmonic 0 zero response (decaying class)."""
    th = np.asarray(theta_nodes, dtype=float)
    n_t = len(th)
    if n_h is None:
        n_h = n_t // 2 - 1
    (p1, q1), (p2, q2) = mat.pq
    M = np.array([[p1, p2], [q1, q2]], dtype=complex)
    mu1, mu2 = mat.mus
    S = np.zeros((2 * n_t, 2 * n_t))
    w = 2.0 / n_t
    for n in range(1, n_h + 1):
        for comp in _COMPS:
            ux, uy = _basis_profiles(n, comp, th)
            cx = {'Uc': 0.5, 'Us': 0.5j, 'Vc': 0.0, 'Vs': 0.0}[comp]
            cy = {'Uc': 0.0, 'Us': 0.0, 'Vc': 0.5, 'Vs': 0.5j}[comp]
            A1, A2 = np.linalg.solve(M, np.array([cx, cy]))
            ctx = -1j * n * (mu1 * A1 + mu2 * A2)
            cty = 1j * n * (A1 + A2)
            e = np.exp(-1j * n * th)
            tx = 2.0 * np.real(ctx * e)
            ty = 2.0 * np.real(cty * e)
            col = np.empty(2 * n_t)
            col[0::2], col[1::2] = (tx, ty)
            row = np.empty(2 * n_t)
            row[0::2], row[1::2] = (ux, uy)
            S += np.outer(col, w * row)
    return S
