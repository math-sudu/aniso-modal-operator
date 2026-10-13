import numpy as np
from .kirsch import shear_modulus
__all__ = ['omega', 'omega_prime', 'stresses_cartesian', 'displacements_cartesian', 'boundary_traction']
_RHO_MIN = 1.0 - 1e-09
_PARTS = ('total', 'excavation', 'insitu')

def _validate_map(R, m):
    """Validate the conformal-map parameters."""
    if not R > 0.0:
        raise ValueError('mapping radius R must be positive')
    if not 0.0 <= m < 1.0:
        raise ValueError('mapping parameter m must satisfy 0 <= m < 1')

def omega(zeta, R, m):
    """Exterior conformal map z = omega(zeta) = R (zeta + m / zeta).

    Maps |zeta| >= 1 onto the exterior of the ellipse with semi-axes
    a_ell = R (1 + m) along x and b_ell = R (1 - m) along y.

    Parameters
    ----------
    zeta : array_like (complex)
        Points of the zeta plane (intended domain |zeta| >= 1).
    R : float
        Mapping radius (> 0).
    m : float
        Mapping parameter (0 <= m < 1); m = 0 gives z = R zeta.

    Returns
    -------
    ndarray (complex)
        Physical points z = omega(zeta).
    """
    _validate_map(R, m)
    zeta = np.asarray(zeta, dtype=complex)
    return R * (zeta + m / zeta)

def omega_prime(zeta, R, m):
    """Derivative of the map: omega'(zeta) = R (1 - m / zeta^2).

Nonzero on |zeta| >= 1 for 0 <= m < 1, with
min_{|zeta| = 1} |omega'| = R (1 - m) (the mapping non-degeneracy measure delta).

Parameters as in :func:`omega`; complex in, complex out."""
    _validate_map(R, m)
    zeta = np.asarray(zeta, dtype=complex)
    return R * (1.0 - m / (zeta * zeta))

def _prepare(rho, theta_zeta, R, m, check_rho=True):
    """Broadcast (rho, theta_zeta) to floats and validate rho >= 1."""
    _validate_map(R, m)
    rho, theta_zeta = np.broadcast_arrays(np.asarray(rho, dtype=float), np.asarray(theta_zeta, dtype=float))
    if check_rho and np.any(rho < _RHO_MIN):
        raise ValueError('elliptic solution is defined for rho >= 1 only')
    return (rho, theta_zeta)

def _far_field(sigma_v, K0):
    """Tension-positive far-field carriers (Gamma, Gamma')."""
    sigma_h = K0 * sigma_v
    return (-0.25 * (sigma_v + sigma_h), -0.5 * (sigma_v - sigma_h))

def _km_constants(R, m, sigma_v, K0):
    """Constants (g1, g2, A, B) of the docstring potentials.

    g1 = Gamma R and g2 = Gamma' R are the linear (far-field carrier)
    coefficients; A and B are the decaying-part constants
    A = -R (Gamma' + m Gamma), B = R (Gamma (1 + 2 m^2) + m Gamma').
    """
    Gamma, Gamma_p = _far_field(sigma_v, K0)
    g1 = Gamma * R
    g2 = Gamma_p * R
    A = -(g2 + m * g1)
    B = g1 * (1.0 + m * m) - m * A
    return (g1, g2, A, B)

def _potentials(zeta, R, m, sigma_v, K0, part):
    """phi, phi', phi'', psi, psi' of the requested part at zeta.

    Tension-positive frame, closed-form zeta-derivatives.  The parts
    are selected by coefficient, not by subtraction:

    * "total":       phi = g1 zeta + A / zeta,
                     psi = g2 zeta - g1 / zeta - T(zeta),
    * "excavation":  phi = (A - m g1) / zeta,
                     psi = -(g1 + m g2) / zeta - T(zeta),

    with T(zeta) = (B zeta^2 - A) / (zeta (zeta^2 - m)) shared by
    both parts (T is entirely excavation-induced).
    """
    g1, g2, A, B = _km_constants(R, m, sigma_v, K0)
    if part == 'total':
        p_lin, p_inv = (g1, A)
        q_lin, q_inv = (g2, -g1)
    else:
        p_lin, p_inv = (0.0, A - m * g1)
        q_lin, q_inv = (0.0, -(g1 + m * g2))
    inv = 1.0 / zeta
    inv2 = inv * inv
    inv3 = inv2 * inv
    phi = p_lin * zeta + p_inv * inv
    dphi = p_lin - p_inv * inv2
    ddphi = 2.0 * p_inv * inv3
    zeta2 = zeta * zeta
    den = zeta * (zeta2 - m)
    T = (B * zeta2 - A) / den
    dT = (-B * zeta2 * zeta2 + (3.0 * A - B * m) * zeta2 - A * m) / (den * den)
    psi = q_lin * zeta + q_inv * inv - T
    dpsi = q_lin - q_inv * inv2 - dT
    return (phi, dphi, ddphi, psi, dpsi)

def stresses_cartesian(rho, theta_zeta, R, m, sigma_v, K0, part='total'):
    """Cartesian stress components (compression positive).

    Evaluated at the physical points z = omega(rho exp(i theta_zeta))
    through the Kolosov-Muskhelishvili relations of the module
    docstring (closed-form zeta-derivatives, global sign flip).

    Parameters
    ----------
    rho, theta_zeta : array_like
        Mapped polar coordinates of the zeta plane (broadcast
        together); rho >= 1 for parts that involve the hole
        ("total", "excavation").
    R : float
        Mapping radius (> 0).
    m : float
        Mapping parameter (0 <= m < 1).
    sigma_v : float
        Vertical in-situ stress (compression positive).
    K0 : float
        Lateral pressure coefficient; sigma_h = K0 * sigma_v.
    part : str
        "total"      -> in-situ + excavation-induced field (default),
        "excavation" -> perturbation caused by the excavation only,
        "insitu"     -> undisturbed in-situ field (uniform).

    Returns
    -------
    (sigma_xx, sigma_yy, sigma_xy) : tuple of ndarray
        Cartesian components at the broadcast shape of
        (rho, theta_zeta).  ``total == insitu + excavation``
        identically.
    """
    if part not in _PARTS:
        raise ValueError("part must be 'total', 'excavation' or 'insitu'")
    rho, theta_zeta = _prepare(rho, theta_zeta, R, m, check_rho=part != 'insitu')
    if part == 'insitu':
        shape = rho.shape
        s_xx = np.full(shape, K0 * sigma_v, dtype=float)
        s_yy = np.full(shape, sigma_v, dtype=float)
        s_xy = np.zeros(shape)
        return (s_xx, s_yy, s_xy)
    zeta = rho * np.exp(1j * theta_zeta)
    z = omega(zeta, R, m)
    wp = omega_prime(zeta, R, m)
    wpp = 2.0 * R * m / (zeta * zeta * zeta)
    _, dphi, ddphi, _, dpsi = _potentials(zeta, R, m, sigma_v, K0, part)
    Phi = dphi / wp
    Phi_p = (ddphi * wp - dphi * wpp) / (wp * wp * wp)
    Psi = dpsi / wp
    p1 = 4.0 * np.real(Phi)
    p2 = 2.0 * (np.conj(z) * Phi_p + Psi)
    s_xx_T = 0.5 * (p1 - np.real(p2))
    s_yy_T = 0.5 * (p1 + np.real(p2))
    s_xy_T = 0.5 * np.imag(p2)
    return (-s_xx_T, -s_yy_T, -s_xy_T)

def displacements_cartesian(rho, theta_zeta, R, m, sigma_v, K0, E, nu):
    """Excavation-induced Cartesian displacements (plane strain).

    The displacement release caused by removing the core, evaluated
    from the decaying potential parts (phi_exc, psi_exc) alone; it
    decays to zero as rho -> infinity.  Sign convention (module
    docstring): (u_x, u_y) = -(tension-positive elasticity
    displacement), so the radial projection is positive toward the
    tunnel axis (convergence), as in ``kirsch.py``.

    Parameters
    ----------
    rho, theta_zeta : array_like
        Mapped polar coordinates (broadcast together); rho >= 1.
    R, m : float
        Conformal-map parameters as in :func:`stresses_cartesian`.
    sigma_v, K0 : float
        In-situ field as in :func:`stresses_cartesian`.
    E, nu : float
        Young's modulus and Poisson's ratio (plane strain).

    Returns
    -------
    (u_x, u_y) : tuple of ndarray
        Components at the broadcast shape of (rho, theta_zeta).
    """
    rho, theta_zeta = _prepare(rho, theta_zeta, R, m)
    G = shear_modulus(E, nu)
    kap = 3.0 - 4.0 * nu
    zeta = rho * np.exp(1j * theta_zeta)
    z = omega(zeta, R, m)
    wp = omega_prime(zeta, R, m)
    phi, dphi, _, psi, _ = _potentials(zeta, R, m, sigma_v, K0, 'excavation')
    Phi = dphi / wp
    disp_T = (kap * phi - z * np.conj(Phi) - np.conj(psi)) / (2.0 * G)
    return (-np.real(disp_T), -np.imag(disp_T))

def boundary_traction(theta_zeta, R, m, sigma_v, K0):
    """Physical traction of the TOTAL field on the hole wall.

    Components (t_x, t_y) of sigma . n formed with the returned
    (compression-positive) total stress tensor at rho = 1 and the
    outward unit normal of the ellipse (pointing from the hole into
    the medium, direction zeta omega'(zeta) / |zeta omega'(zeta)|).
    Identically zero for this traction-free solution; used by the
    verification tests.

    Parameters
    ----------
    theta_zeta : array_like
        Mapped boundary angle(s).
    R, m, sigma_v, K0 : float
        As in :func:`stresses_cartesian`.

    Returns
    -------
    (t_x, t_y) : tuple of ndarray
        Traction components at the shape of theta_zeta.
    """
    theta_zeta = np.asarray(theta_zeta, dtype=float)
    s_xx, s_yy, s_xy = stresses_cartesian(1.0, theta_zeta, R, m, sigma_v, K0, part='total')
    s = np.exp(1j * theta_zeta)
    n = s * omega_prime(s, R, m)
    n = n / np.abs(n)
    n_x = np.real(n)
    n_y = np.imag(n)
    return (s_xx * n_x + s_xy * n_y, s_xy * n_x + s_yy * n_y)
