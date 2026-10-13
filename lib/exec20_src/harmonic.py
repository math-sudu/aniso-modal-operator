from __future__ import annotations
import numpy as np
from .material import AnisoMaterial
__all__ = ['AnisoExterior', 'solve_neumann_circle', 'solve_dirichlet_circle', 'single_hole_solution', 'released_traction_values']

def _zeta_of_zk(mu: complex, zk: np.ndarray, R: float) -> np.ndarray:
    """Invert z_k = gam R zeta + del R / zeta on |zeta| >~ 1.

    Quadratic gam*R*zeta^2 - z_k*zeta + del*R = 0; the two roots have
    product qhat (|qhat| < 1), so exactly one root lies outside; take the
    larger-|.| root.  Points slightly INSIDE |zeta| = 1 are legitimate
    analytic continuation (used by chord-midpoint Gauss evaluation on the
    polygonal FE ring, sagitta ~ 1 - cos(pi/n_t)); the representation is
    regular down to the critical locus |zeta| = sqrt(|qhat|), so the
    guard trips at a material-aware threshold
    max(0.95, 1.05 sqrt(|qhat|))."""
    gam = (1.0 - 1j * mu) / 2.0
    dlt = (1.0 + 1j * mu) / 2.0
    qh = dlt / gam
    zk = np.asarray(zk, dtype=complex)
    disc = np.sqrt(zk * zk - 4.0 * gam * dlt * R * R)
    zp = (zk + disc) / (2.0 * gam * R)
    zm = (zk - disc) / (2.0 * gam * R)
    zeta = np.where(np.abs(zp) >= np.abs(zm), zp, zm)
    thr = max(0.95, 1.05 * np.sqrt(abs(qh)))
    if np.any(np.abs(zeta) < thr):
        raise ValueError(f'inverse mapping reached |zeta| < {thr:.3f} (inside the mapped hole or near the critical locus sqrt(|qhat|))')
    return zeta

class AnisoExterior:
    """Truncated decaying two-branch Lekhnitskii field outside r = R."""

    def __init__(self, mat: AnisoMaterial, R: float, n_modes: int):
        self.mat = mat
        self.R = float(R)
        self.n_modes = int(n_modes)
        self.A = np.zeros((2, self.n_modes), dtype=complex)

    def _phi_dphi(self, zeta: np.ndarray, k: int):
        """phi_k and d phi_k / d zeta at given zeta (|zeta| >= 1)."""
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

    def fields(self, r, theta):
        """Tension-positive fields at polar points (r, theta), r >= R.

        Returns dict with s_xx, s_yy, s_xy, s_rr, s_tt, s_rt, u_x, u_y,
        u_r, u_t (broadcast shape).
        """
        r = np.asarray(r, dtype=float)
        theta = np.asarray(theta, dtype=float)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        mus = self.mat.mus
        pqs = self.mat.pq
        sxx = 0.0
        syy = 0.0
        sxy = 0.0
        ux = 0.0
        uy = 0.0
        for k, mu in enumerate(mus):
            gam = (1.0 - 1j * mu) / 2.0
            zk = x + mu * y
            zeta = _zeta_of_zk(mu, zk, self.R)
            phi, dphi = self._phi_dphi(zeta, k)
            omp = gam * self.R * (1.0 - self.mat.qhats[k] * zeta ** (-2))
            Phi = dphi / omp
            p, q = pqs[k]
            sxx = sxx + 2.0 * np.real(mu * mu * Phi)
            syy = syy + 2.0 * np.real(Phi)
            sxy = sxy - 2.0 * np.real(mu * Phi)
            ux = ux + 2.0 * np.real(p * phi)
            uy = uy + 2.0 * np.real(q * phi)
        c, s = (np.cos(theta), np.sin(theta))
        s_rr = sxx * c * c + syy * s * s + 2.0 * sxy * c * s
        s_tt = sxx * s * s + syy * c * c - 2.0 * sxy * c * s
        s_rt = (syy - sxx) * c * s + sxy * (c * c - s * s)
        u_r = ux * c + uy * s
        u_t = -ux * s + uy * c
        return {'s_xx': sxx, 's_yy': syy, 's_xy': sxy, 's_rr': s_rr, 's_tt': s_tt, 's_rt': s_rt, 'u_x': ux, 'u_y': uy, 'u_r': u_r, 'u_t': u_t}

    def boundary_fields(self, theta):
        """Fields exactly on r = R (zeta = e^{i th}; exact pointwise)."""
        return self.fields(np.full_like(np.asarray(theta, dtype=float), self.R), theta)

def _fft_neg_coeffs(f: np.ndarray, n_modes: int) -> np.ndarray:
    """Complex coefficients c_m of e^{-i m th}, m = 1..n_modes, of a REAL
    uniformly sampled profile f(th_j), th_j = 2 pi j / N."""
    N = len(f)
    F = np.fft.fft(f) / N
    return np.array([F[(N - m) % N] for m in range(1, n_modes + 1)])

def solve_dirichlet_circle(mat: AnisoMaterial, R: float, u_x: np.ndarray, u_y: np.ndarray, n_modes: int) -> AnisoExterior:
    """Closed-form per-harmonic solve: displacement values on r = R
    (uniform theta grid) -> decaying exterior representation.

    Harmonic m: u_x = 2Re[P e^{-im th}], P = p1 A1m + p2 A2m and
    u_y = 2Re[Q e^{-im th}], Q = q1 A1m + q2 A2m  ->  2x2 complex solve.
    Harmonic 0 (rigid translation) is not representable by the decaying
    class and is projected out (zero response; proto n=1 analog).
    """
    (p1, q1), (p2, q2) = mat.pq
    M = np.array([[p1, p2], [q1, q2]], dtype=complex)
    ext = AnisoExterior(mat, R, n_modes)
    cx = _fft_neg_coeffs(np.asarray(u_x, float), n_modes)
    cy = _fft_neg_coeffs(np.asarray(u_y, float), n_modes)
    for m in range(1, n_modes + 1):
        rhs = np.array([cx[m - 1], cy[m - 1]], dtype=complex)
        A = np.linalg.solve(M, rhs)
        ext.A[0, m - 1] = A[0]
        ext.A[1, m - 1] = A[1]
    return ext

def solve_neumann_circle(mat: AnisoMaterial, R: float, s_rr: np.ndarray, s_rt: np.ndarray, n_modes: int) -> AnisoExterior:
    """Closed-form per-harmonic solve: traction values (s_rr, s_rt) on
    r = R -> decaying exterior representation, via the potential-level
    resultant form (Lekhnitskii; Bobet Eq. 14 / Zeng Eqs. 11-18 family).

    X(th) = int t_x R dth = 2Re[mu1 phi1 + mu2 phi2] + C
    Y(th) = int t_y R dth = -2Re[phi1 + phi2] + C
    Harmonic m >= 1: [ [mu1, mu2], [1, 1] ] [A1m; A2m] = [X_m; -Y_m].
    The n = 0 traction content integrates into harmonic +-1 resultant
    content; net force (e^{0}-content of t) must vanish for the class
    (checked; released loads from a uniform far field comply).
    """
    s_rr = np.asarray(s_rr, float)
    s_rt = np.asarray(s_rt, float)
    N = len(s_rr)
    th = 2.0 * np.pi * np.arange(N) / N
    c, s = (np.cos(th), np.sin(th))
    t_x = s_rr * c - s_rt * s
    t_y = s_rr * s + s_rt * c

    def cum_int(f):
        F = np.fft.fft(f) / N
        if abs(F[0]) > 1e-08 * max(1.0, np.max(np.abs(f))):
            raise ValueError(f'nonzero net force content {F[0]:.3e}: profile outside the zero-net-force decaying class')
        k = np.fft.fftfreq(N, d=1.0 / N)
        with np.errstate(divide='ignore', invalid='ignore'):
            Fi = np.where(k != 0, F / (1j * k), 0.0)
        return np.fft.ifft(Fi * N).real * R
    X = cum_int(t_x)
    Y = cum_int(t_y)
    mu1, mu2 = mat.mus
    M = np.array([[mu1, mu2], [1.0, 1.0]], dtype=complex)
    ext = AnisoExterior(mat, R, n_modes)
    cX = _fft_neg_coeffs(X, n_modes)
    cY = _fft_neg_coeffs(Y, n_modes)
    for m in range(1, n_modes + 1):
        rhs = np.array([cX[m - 1], -cY[m - 1]], dtype=complex)
        A = np.linalg.solve(M, rhs)
        ext.A[0, m - 1] = A[0]
        ext.A[1, m - 1] = A[1]
    return ext

def released_traction_values(sxx_inf: float, syy_inf: float, sxy_inf: float, theta: np.ndarray):
    """Field-value profiles (s_rr, s_rt) of MINUS the uniform far-field
    state on a circle (the excavation released load, tension-positive)."""
    c, s = (np.cos(theta), np.sin(theta))
    s_rr = -(sxx_inf * c * c + syy_inf * s * s + 2.0 * sxy_inf * c * s)
    s_rt = -((syy_inf - sxx_inf) * c * s + sxy_inf * (c * c - s * s))
    return (s_rr, s_rt)

def single_hole_solution(mat: AnisoMaterial, a: float, sxx_inf: float, syy_inf: float, sxy_inf: float, n_modes: int=8, n_grid: int=512) -> AnisoExterior:
    """Disturbance field of a traction-free circular hole r = a in a
    uniform tension-positive far field: Neumann solve of the released
    traction.  Perturbation displacements are the disturbance
    displacements (Hefny-Lo case (b) reading); total stress = far field
    + disturbance."""
    th = 2.0 * np.pi * np.arange(n_grid) / n_grid
    s_rr, s_rt = released_traction_values(sxx_inf, syy_inf, sxy_inf, th)
    return solve_neumann_circle(mat, a, s_rr, s_rt, n_modes)
