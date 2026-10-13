import numpy as np
from .kirsch import shear_modulus
__all__ = ['shear_modulus', 'kappa', 'km_coefficients', 'stresses', 'displacements']
_R_TOL = 1e-09
_ZONES = ('liner', 'ground')

def kappa(nu):
    """Kolosov constant for plane strain: kappa = 3 - 4 nu."""
    return 3.0 - 4.0 * nu

def _validate_geometry(R_i, R_o):
    if not (R_i > 0.0 and R_o > R_i):
        raise ValueError('geometry requires 0 < R_i < R_o')

def _equilibrated_solve(M, b):
    """Solve M x = b after scaling each row by its max-abs entry."""
    scale = np.max(np.abs(M), axis=1)
    return np.linalg.solve(M / scale[:, None], b / scale)

def km_coefficients(R_i, R_o, sigma_v, K0, p_i, E_l, nu_l, E_g, nu_g):
    """Kolosov-Muskhelishvili coefficients of the two-zone solution.

Solves the nine boundary/interface conditions
and returns the REAL coefficients of the tension-positive-frame
potentials

* liner:  phi_l = a1 z + a3 z^3 + am1 / z,
          psi_l = b1 z + bm1 / z + bm3 / z^3,
* ground: phi_g = Gamma z + cm1 / z,
          psi_g = Gamma' z + dm1 / z + dm3 / z^3,

as the dict keys ``a1, a3, am1, b1, bm1, bm3`` (liner) and
``Gamma, cm1, Gamma_prime, dm1, dm3`` (ground; the far-field
carrying values Gamma = -(sigma_v + sigma_h)/4 and
Gamma' = -(sigma_v - sigma_h)/2 are included for completeness).

Parameters
----------
R_i, R_o : float
    Liner inner/outer radius, 0 < R_i < R_o.
sigma_v : float
    Vertical in-situ stress (compression positive).
K0 : float
    Lateral pressure coefficient; sigma_h = K0 * sigma_v.
p_i : float
    Uniform normal pressure on the inner wall (compression
    positive, zero wall shear).
E_l, nu_l : float
    Liner Young's modulus and Poisson's ratio (plane strain).
E_g, nu_g : float
    Ground Young's modulus and Poisson's ratio (plane strain)."""
    _validate_geometry(R_i, R_o)
    sigma_h = K0 * sigma_v
    m = 0.5 * (sigma_v + sigma_h)
    d = 0.5 * (sigma_v - sigma_h)
    Gamma = -0.5 * m
    Gamma_p = -d
    G_l = shear_modulus(E_l, nu_l)
    G_g = shear_modulus(E_g, nu_g)
    k_l = kappa(nu_l)
    k_g = kappa(nu_g)
    g = G_l / G_g
    rho = R_i / R_o
    rho2 = rho * rho
    rho4 = rho2 * rho2
    M0 = np.array([[2.0, 1.0 / rho2, 0.0], [2.0, 1.0, -1.0], [k_l - 1.0, -1.0, g]])
    b0 = np.array([-p_i, 2.0 * Gamma, g * (k_g - 1.0) * Gamma])
    a1, bm1_h, dm1_h = _equilibrated_solve(M0, b0)
    M2 = np.array([[0.0, -4.0 / rho2, -1.0, 3.0 / rho4, 0.0, 0.0], [6.0 * rho2, -2.0 / rho2, 1.0, 3.0 / rho4, 0.0, 0.0], [0.0, -4.0, -1.0, 3.0, 4.0, -3.0], [6.0, -2.0, 1.0, 3.0, 2.0, -3.0], [k_l - 3.0, k_l + 1.0, -1.0, -1.0, -g * (k_g + 1.0), g], [k_l + 3.0, 1.0 - k_l, 1.0, -1.0, -g * (1.0 - k_g), g]])
    b2 = np.array([0.0, 0.0, -Gamma_p, Gamma_p, -g * Gamma_p, g * Gamma_p])
    a3_h, am1_h, b1, bm3_h, cm1_h, dm3_h = _equilibrated_solve(M2, b2)
    Ro2 = R_o * R_o
    Ro4 = Ro2 * Ro2
    return {'a1': float(a1), 'a3': float(a3_h) / Ro2, 'am1': float(am1_h) * Ro2, 'b1': float(b1), 'bm1': float(bm1_h) * Ro2, 'bm3': float(bm3_h) * Ro4, 'Gamma': float(Gamma), 'cm1': float(cm1_h) * Ro2, 'Gamma_prime': float(Gamma_p), 'dm1': float(dm1_h) * Ro2, 'dm3': float(dm3_h) * Ro4}

def _zone_terms(coef, zone):
    """Generic (A, B, C, D, F, H) coefficients of the requested zone."""
    if zone == 'liner':
        return (coef['a1'], coef['a3'], coef['am1'], coef['b1'], coef['bm1'], coef['bm3'])
    return (coef['Gamma'], 0.0, coef['cm1'], coef['Gamma_prime'], coef['dm1'], coef['dm3'])

def _prepare(r, theta, R_i, R_o, zone):
    """Broadcast (r, theta) to floats and validate r against the zone."""
    if zone not in _ZONES:
        raise ValueError("zone must be 'liner' or 'ground'")
    r, theta = np.broadcast_arrays(np.asarray(r, dtype=float), np.asarray(theta, dtype=float))
    if zone == 'liner':
        if np.any(r < R_i * (1.0 - _R_TOL)) or np.any(r > R_o * (1.0 + _R_TOL)):
            raise ValueError('liner zone is defined for R_i <= r <= R_o')
    elif np.any(r < R_o * (1.0 - _R_TOL)):
        raise ValueError('ground zone is defined for r >= R_o')
    return (r, theta)

def stresses(r, theta, R_i, R_o, sigma_v, K0, p_i, E_l, nu_l, E_g, nu_g, zone):
    """Total polar stress components (compression positive).

    Parameters
    ----------
    r, theta : array_like
        Polar coordinates (broadcast together); R_i <= r <= R_o for
        ``zone="liner"``, r >= R_o for ``zone="ground"``.
    R_i, R_o, sigma_v, K0, p_i, E_l, nu_l, E_g, nu_g : float
        Geometry, loading and elastic constants as in
        :func:`km_coefficients`.
    zone : str
        "liner" or "ground".

    Returns
    -------
    (sigma_rr, sigma_tt, sigma_rt) : tuple of ndarray
        Radial, hoop and shear components at the broadcast shape of
        (r, theta); ``sigma_rr = p_i`` and ``sigma_rt = 0`` at
        ``r = R_i``, and the far field is recovered as r -> infinity.
    """
    coef = km_coefficients(R_i, R_o, sigma_v, K0, p_i, E_l, nu_l, E_g, nu_g)
    r, theta = _prepare(r, theta, R_i, R_o, zone)
    A, B, C, D, F, H = _zone_terms(coef, zone)
    c2 = np.cos(2.0 * theta)
    s2 = np.sin(2.0 * theta)
    inv2 = 1.0 / (r * r)
    inv4 = inv2 * inv2
    r2 = r * r
    s_rr = 2.0 * A + F * inv2 + (-4.0 * C * inv2 - D + 3.0 * H * inv4) * c2
    s_tt = 2.0 * A - F * inv2 + (12.0 * B * r2 + D - 3.0 * H * inv4) * c2
    s_rt = (6.0 * B * r2 - 2.0 * C * inv2 + D + 3.0 * H * inv4) * s2
    return (-s_rr, -s_tt, -s_rt)

def displacements(r, theta, R_i, R_o, sigma_v, K0, p_i, E_l, nu_l, E_g, nu_g, zone):
    """Total displacements of the wished-in-place loading (plane strain).

displacement from the stress-free unloaded composite to the
loaded equilibrium state, with no rigid-body motion (mirror
symmetries); it grows like the homogeneous far-field straining as
r -> infinity.  Sign convention: u_r > 0 toward the tunnel axis
(convergence), u_theta > 0 toward decreasing theta.

Parameters as in :func:`stresses`.

Returns
-------
(u_r, u_theta) : tuple of ndarray
    Components at the broadcast shape of (r, theta)."""
    coef = km_coefficients(R_i, R_o, sigma_v, K0, p_i, E_l, nu_l, E_g, nu_g)
    r, theta = _prepare(r, theta, R_i, R_o, zone)
    A, B, C, D, F, H = _zone_terms(coef, zone)
    if zone == 'liner':
        G, k = (shear_modulus(E_l, nu_l), kappa(nu_l))
    else:
        G, k = (shear_modulus(E_g, nu_g), kappa(nu_g))
    c2 = np.cos(2.0 * theta)
    s2 = np.sin(2.0 * theta)
    inv1 = 1.0 / r
    inv3 = inv1 / (r * r)
    r3 = r * r * r
    pref = 1.0 / (2.0 * G)
    u_r = pref * ((k - 1.0) * A * r - F * inv1 + ((k - 3.0) * B * r3 + (k + 1.0) * C * inv1 - D * r - H * inv3) * c2)
    u_t = pref * (((k + 3.0) * B * r3 + (1.0 - k) * C * inv1 + D * r - H * inv3) * s2)
    return (-u_r, -u_t)
