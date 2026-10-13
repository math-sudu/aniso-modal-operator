import numpy as np
__all__ = ['shear_modulus', 'stresses', 'displacements']
_R_MIN_FACTOR = 1.0 - 1e-09

def shear_modulus(E, nu):
    """Shear modulus G = E / (2 (1 + nu))."""
    return E / (2.0 * (1.0 + nu))

def _prepare(r, theta, a, check_r=True):
    """Broadcast (r, theta) to a common float shape and validate r >= a."""
    if not a > 0.0:
        raise ValueError('hole radius a must be positive')
    r, theta = np.broadcast_arrays(np.asarray(r, dtype=float), np.asarray(theta, dtype=float))
    if check_r and np.any(r < a * _R_MIN_FACTOR):
        raise ValueError('Kirsch solution is defined for r >= a only')
    return (r, theta)

def _mean_deviator(sigma_v, K0):
    """Mean and half-difference of the in-situ field (compression +)."""
    sigma_h = K0 * sigma_v
    return (0.5 * (sigma_v + sigma_h), 0.5 * (sigma_v - sigma_h))

def stresses(r, theta, a, sigma_v, K0, part='total'):
    """Polar stress components around the tunnel (compression positive).

    Parameters
    ----------
    r, theta : array_like
        Polar coordinates (broadcast together); r >= a for parts that
        involve the hole ("total", "excavation").
    a : float
        Tunnel radius (> 0).
    sigma_v : float
        Vertical in-situ stress (compression positive).
    K0 : float
        Lateral pressure coefficient; sigma_h = K0 * sigma_v.
    part : str
        "total"      -> in-situ + excavation-induced field (default),
        "excavation" -> perturbation caused by the excavation only,
        "insitu"     -> undisturbed in-situ field (r-independent).

    Returns
    -------
    (sigma_rr, sigma_tt, sigma_rt) : tuple of ndarray
        Radial, hoop and shear components at the broadcast shape of
        (r, theta).  ``total == insitu + excavation`` identically.
    """
    r, theta = _prepare(r, theta, a, check_r=part != 'insitu')
    m, d = _mean_deviator(sigma_v, K0)
    c2 = np.cos(2.0 * theta)
    s2 = np.sin(2.0 * theta)
    if part == 'insitu':
        return (m - d * c2, m + d * c2, d * s2)
    rho2 = (a / r) ** 2
    rho4 = rho2 * rho2
    if part == 'excavation':
        s_rr = -m * rho2 + d * (4.0 * rho2 - 3.0 * rho4) * c2
        s_tt = m * rho2 + 3.0 * d * rho4 * c2
        s_rt = d * (2.0 * rho2 - 3.0 * rho4) * s2
        return (s_rr, s_tt, s_rt)
    if part == 'total':
        s_rr = m * (1.0 - rho2) - d * (1.0 - 4.0 * rho2 + 3.0 * rho4) * c2
        s_tt = m * (1.0 + rho2) + d * (1.0 + 3.0 * rho4) * c2
        s_rt = d * (1.0 + 2.0 * rho2 - 3.0 * rho4) * s2
        return (s_rr, s_tt, s_rt)
    raise ValueError("part must be 'total', 'excavation' or 'insitu'")

def displacements(r, theta, a, sigma_v, K0, E, nu):
    """Excavation-induced displacements (plane strain).

    The displacement release caused by removing the core, i.e. the
    displacement measured relative to the pre-excavation state; it
    decays to zero as r -> infinity.  Sign convention (module
    docstring): u_r > 0 toward the tunnel axis (convergence),
    u_theta > 0 toward decreasing theta.

    Parameters
    ----------
    r, theta : array_like
        Polar coordinates (broadcast together); r >= a.
    a : float
        Tunnel radius (> 0).
    sigma_v, K0 : float
        In-situ field as in :func:`stresses`.
    E, nu : float
        Young's modulus and Poisson's ratio (plane strain).

    Returns
    -------
    (u_r, u_theta) : tuple of ndarray
        Components at the broadcast shape of (r, theta).
    """
    r, theta = _prepare(r, theta, a)
    m, d = _mean_deviator(sigma_v, K0)
    G = shear_modulus(E, nu)
    rho2 = (a / r) ** 2
    pref = a * a / (2.0 * G * r)
    u_r = pref * (m - d * (4.0 * (1.0 - nu) - rho2) * np.cos(2.0 * theta))
    u_t = pref * d * (2.0 * (1.0 - 2.0 * nu) + rho2) * np.sin(2.0 * theta)
    return (u_r, u_t)
