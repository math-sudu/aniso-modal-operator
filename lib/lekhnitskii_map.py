import time
import numpy as np
__all__ = ['IsotropicDegeneracyError', 'TIMaterial', 'k0_load', 'solve_carriers', 'solve_carriers_K0', 'LekhnitskiiCircle', 'solve_ls', 'validate_univalence_k']
_R_MIN_FACTOR = 1.0 - 1e-09
_SHEAR_COMPAT_TOL = 1e-10
_CONJ_TOL = 1e-09

class IsotropicDegeneracyError(ValueError):
    """Raised when |mu_1 - mu_2| < eps0 (isotropic-type degeneracy).

As mu_1 -> mu_2 the two-potential
Lekhnitskii representation becomes structurally singular -- the
individual carriers Gamma_k diverge like 1/(alpha_2^2 - alpha_1^2)
although every physical combination stays bounded (the general
solution changes form at the repeated root).  Rather than
computing in the ill-conditioned regime, the material constructor
declares the parameter domain |mu_1 - mu_2| >= eps0
and raises this exception for smaller root separation; the degenerate lane is the
isotropic Kolosov-Muskhelishvili family
(``refsol/kirsch.py`` / ``mapped_net.py`` / ``general_map.py``)."""

def _compliance_voigt_principal(E, E_prime, nu, nu_prime, G_prime):
    """6x6 Voigt compliance in the material principal axes.

    Axes: x' in the isotropic (bedding) plane, y' normal to bedding,
    z' the in-plane isotropic direction (out of the tunnel section).
    Five-constant transverse isotropy: E = E_h (in-bedding modulus),
    E_prime = E_v (cross-bedding modulus), nu = nu_h (in-plane ratio),
    nu_prime = nu_vh (cross ratio), G_prime = G_vh (cross shear
    modulus); matches pid_2155399c64b8 Eq.(13) as transcribed by the
    verified ``check_carrier.compliance_voigt_principal``.
    """
    a = np.zeros((6, 6))
    a[0, 0] = 1.0 / E
    a[1, 1] = 1.0 / E_prime
    a[2, 2] = 1.0 / E
    a[0, 1] = a[1, 0] = -nu_prime / E_prime
    a[1, 2] = a[2, 1] = -nu_prime / E_prime
    a[0, 2] = a[2, 0] = -nu / E
    a[3, 3] = 1.0 / G_prime
    a[4, 4] = 2.0 * (1.0 + nu) / E
    a[5, 5] = 1.0 / G_prime
    return a

def _voigt_to_tensor(a):
    """Compliance Voigt (engineering shear) -> 4th-order S_ijkl."""
    idx = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (1, 2): 3, (2, 1): 3, (0, 2): 4, (2, 0): 4, (0, 1): 5, (1, 0): 5}
    S = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    I, J = (idx[i, j], idx[k, l])
                    fac = 1.0
                    if I >= 3:
                        fac *= 0.5
                    if J >= 3:
                        fac *= 0.5
                    S[i, j, k, l] = fac * a[I, J]
    return S

def _tensor_to_voigt(S):
    """4th-order compliance tensor -> 6x6 Voigt (engineering shear)."""
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    a = np.zeros((6, 6))
    for I, (i, j) in enumerate(pairs):
        for J, (k, l) in enumerate(pairs):
            fac = 1.0
            if I >= 3:
                fac *= 2.0
            if J >= 3:
                fac *= 2.0
            a[I, J] = fac * S[i, j, k, l]
    return a

def _rotate_z(S, phi):
    """Rotate the 4th-order tensor about the out-of-plane z axis.

    phi = angle from the global x axis to the material x' axis
    (radians); components in global axes are S_pqrs = R_pi R_qj R_rk
    R_sl S'_ijkl with R = Rz(phi) mapping material axes into global
    axes -- the plain tensor-rotation route (no Bond-matrix factors).
    """
    c, s = (np.cos(phi), np.sin(phi))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.einsum('pi,qj,rk,sl,ijkl->pqrs', R, R, R, R, S)

def _reduced_beta(a):
    """Plane-strain reduced compliances beta_ij = a_ij - a_i3 a_j3 / a_33
    for i, j in {1, 2, 6} (Voigt rows 0, 1, 5; row 2 = out-of-plane)."""
    out = {}
    for key, (I, J) in {'11': (0, 0), '12': (0, 1), '16': (0, 5), '22': (1, 1), '26': (1, 5), '66': (5, 5)}.items():
        out[key] = a[I, J] - a[I, 2] * a[J, 2] / a[2, 2]
    return out

class TIMaterial:
    """Transversely isotropic material for the plane-strain section.

Builds the GLOBAL-frame reduced compliances beta_ij by 4th-order
tensor rotation of the principal compliance, solves the
characteristic quartic (source Eq.(25))

  beta11 mu^4 - 2 beta16 mu^3 + (2 beta12 + beta66) mu^2
    - 2 beta26 mu + beta22 = 0,

keeps the two upper-half-plane roots mu_1, mu_2 and the
displacement coefficients (source Eq.(24))

  p_k = beta11 mu_k^2 + beta12 - beta16 mu_k,
  q_k = beta12 mu_k + beta22 / mu_k - beta26,

and asserts the shear-compatibility identity p_k mu_k + q_k ==
beta16 mu_k^2 + beta26 - beta66 mu_k (exactly equivalent to the characteristic equation) to 1e-10 relative.

Parameters
----------
E, E_prime : float
    In-bedding-plane and cross-bedding Young's moduli (E_h, E_v).
nu, nu_prime : float
    In-plane Poisson's ratio (nu_h) and cross ratio (nu_vh: effect
    of cross-bedding stress on in-plane strain).
G_prime : float
    Cross-bedding shear modulus (G_vh).
dip : float
    Rotation of the material axes in the cross-section plane,
    RADIANS (angle from the global x axis to the bedding x' axis).
eps0 : float
    Root-separation guard: raise
    :class:`IsotropicDegeneracyError` when |mu_1 - mu_2| < eps0.
    Default 1e-6.

Raises
------
ValueError
    On non-positive moduli or a compliance matrix that is not
    positive definite (checked via eigenvalues).
IsotropicDegeneracyError
    When |mu_1 - mu_2| < eps0 (route the case to the isotropic
    Kolosov-Muskhelishvili modules instead)."""

    def __init__(self, E, E_prime, nu, nu_prime, G_prime, dip=0.0, eps0=1e-06):
        for name, val in (('E', E), ('E_prime', E_prime), ('G_prime', G_prime)):
            if not val > 0.0:
                raise ValueError(f'{name} must be positive, got {val}')
        if not eps0 > 0.0:
            raise ValueError(f'eps0 must be positive, got {eps0}')
        self.E = float(E)
        self.E_prime = float(E_prime)
        self.nu = float(nu)
        self.nu_prime = float(nu_prime)
        self.G_prime = float(G_prime)
        self.dip = float(dip)
        self.eps0 = float(eps0)
        a_principal = _compliance_voigt_principal(self.E, self.E_prime, self.nu, self.nu_prime, self.G_prime)
        eig = np.linalg.eigvalsh(a_principal)
        self.compliance_eig_min = float(eig[0])
        if self.compliance_eig_min <= 0.0:
            raise ValueError(f'principal compliance matrix is not positive definite (min eigenvalue {self.compliance_eig_min:.3e}); the five constants (E, E_prime, nu, nu_prime, G_prime) violate thermodynamic admissibility')
        self.a_voigt_principal = a_principal
        self.a_voigt = _tensor_to_voigt(_rotate_z(_voigt_to_tensor(a_principal), self.dip))
        self.beta = _reduced_beta(self.a_voigt)
        b = self.beta
        roots = np.roots([b['11'], -2.0 * b['16'], 2.0 * b['12'] + b['66'], -2.0 * b['26'], b['22']])
        upper = sorted((complex(z) for z in roots if z.imag > 0.0), key=lambda z: (z.imag, z.real))
        if len(upper) != 2:
            raise ValueError(f'characteristic quartic did not yield two upper-half-plane roots (got {upper}); the material data are outside the elliptic regime of the plane problem')
        mu1, mu2 = upper
        if abs(mu1 - mu2) < self.eps0:
            raise IsotropicDegeneracyError(f'|mu_1 - mu_2| = {abs(mu1 - mu2):.3e} < eps0 = {self.eps0:.1e}: isotropic-type degeneracy -- the two-potential Lekhnitskii representation is structurally singular at the repeated root (the individual carriers Gamma_k diverge while physical combinations stay bounded).  Use the isotropic Kolosov-Muskhelishvili modules (refsol/kirsch.py, mapped_net.py, general_map.py) for this material, or lower eps0 deliberately.')
        self.mu1, self.mu2 = (mu1, mu2)
        self.p1, self.q1 = self._pq(mu1)
        self.p2, self.q2 = self._pq(mu2)
        worst = 0.0
        for mu, p, q in ((mu1, self.p1, self.q1), (mu2, self.p2, self.q2)):
            rhs = b['16'] * mu ** 2 + b['26'] - b['66'] * mu
            worst = max(worst, abs(p * mu + q - rhs) / (abs(p * mu) + abs(q) + 1e-300))
        if worst > _SHEAR_COMPAT_TOL:
            raise AssertionError(f'shear-compatibility identity violated: {worst:.3e} > {_SHEAR_COMPAT_TOL:.1e} (indicates an inconsistent beta/mu pipeline)')
        self.shear_compat_residual = float(worst)

    def _pq(self, mu):
        b = self.beta
        p = b['11'] * mu ** 2 + b['12'] - b['16'] * mu
        q = b['12'] * mu + b['22'] / mu - b['26']
        return (p, q)

def k0_load(sigma_v, K0):
    """(sigma_xx, sigma_yy, sigma_xy) of the K0 in-situ state.

    sigma_yy = sigma_v (vertical, compression positive), sigma_xx =
    K0 * sigma_v (horizontal), sigma_xy = 0: a K0 gravitational field
    carries no shear in the global frame.  The material dip does NOT
    rotate the load -- dip rotates the MATERIAL axes only; a nonzero
    sigma_xy enters only when the LOAD itself has shear (use
    :meth:`LekhnitskiiCircle.from_load`).
    """
    return (K0 * sigma_v, sigma_v, 0.0)

def solve_carriers(mu1, mu2, sigma_xx_inf, sigma_yy_inf, sigma_xy_inf, s_gauge=0.0):
    """Far-field carriers Gamma_1, Gamma_2 (extended Vandermonde).

Solves the three real carrier equations

  2 Re[mu_1^2 Gamma_1 + mu_2^2 Gamma_2] = sigma_xx_inf,
  2 Re[Gamma_1 + Gamma_2]               = sigma_yy_inf,
 -2 Re[mu_1 Gamma_1 + mu_2 Gamma_2]     = sigma_xy_inf,

by the extended-Vandermonde construction: treat (Gamma_1, Gamma_2,
conj Gamma_1, conj Gamma_2) as four unknowns at the four pairwise
distinct points (mu_1, mu_2, conj mu_1, conj mu_2) with the power
moments (sigma_yy, -sigma_xy, sigma_xx, s_gauge).  Conjugate
consistency is automatic (real right-hand side + solution
uniqueness) and is ASSERTED here to 1e-9 of the load
scale.  The gauge s_gauge (default 0) parametrizes the 1-real-
dimensional kernel = rigid-rotation direction; it
changes no stress and no strain.

Frame-agnostic: feed tension-positive components to get
tension-positive carriers (as :class:`LekhnitskiiCircle` does
internally) or compression-positive components for compression-positive carriers."""
    pts = np.array([mu1, mu2, np.conj(mu1), np.conj(mu2)])
    V = np.vander(pts, N=4, increasing=True).T
    d = np.array([sigma_yy_inf, -sigma_xy_inf, sigma_xx_inf, s_gauge], dtype=complex)
    g = np.linalg.solve(V, d)
    scale = max(abs(sigma_xx_inf), abs(sigma_yy_inf), abs(sigma_xy_inf), 1e-300)
    conj_err = max(abs(g[2] - np.conj(g[0])), abs(g[3] - np.conj(g[1])))
    if conj_err > _CONJ_TOL * scale:
        raise AssertionError(f'carrier conjugate consistency violated: {conj_err:.3e} > {_CONJ_TOL:.1e} * load scale (expected automatic for distinct upper-half mu_k)')
    return (complex(g[0]), complex(g[1]))

def solve_carriers_K0(mu1, mu2, p0, K0, s_gauge=0.0):
    """K0 convenience wrapper of :func:`solve_carriers`.

sigma_yy_inf = p0 (vertical), sigma_xx_inf = K0 * p0, sigma_xy_inf
= 0 (see :func:`k0_load`: the material dip does not rotate the
load).  For the orthotropic-aligned case (dip = 0, mu_k = i
alpha_k) with gauge 0 this reproduces the closed form Gamma_1 = p0 (alpha_2^2 + K0) / (2 (alpha_2^2 - alpha_1^2)),
Gamma_2 = -p0 (alpha_1^2 + K0) / (2 (alpha_2^2 - alpha_1^2))."""
    sxx, syy, sxy = k0_load(p0, K0)
    return solve_carriers(mu1, mu2, sxx, syy, sxy, s_gauge=s_gauge)

class LekhnitskiiCircle:
    """Circular-hole excavation problem in a TI medium (total field).

Case data: material (a :class:`TIMaterial`), hole radius a, and
the uniform compression-positive in-situ Cartesian stresses.  Pure
excavation release: the wall of the excavated hole is traction
free and the far field is the in-situ state, carried analytically
by Gamma_k.  The released
traction of a uniform field over the closed wall is
self-equilibrated, so A_k = 0 and the representation is log-free
(asserted numerically at construction -- log terms
are deliberately NOT implemented).

Parameters
----------
material : TIMaterial
a : float
    Hole radius (> 0); the circle map is omega(zeta) = a zeta.
sigma_v : float
    Vertical in-situ stress (compression positive).
K0 : float
    Lateral pressure coefficient (see :func:`k0_load`).
n_order : int
    Truncation order N >= 1 of the decaying series.  For the
    circle the exact solution lives at n = 1 (all higher
    coefficients vanish), so any N >= 1 closes the wall to
    rounding.

Attributes
----------
gamma_1, gamma_2 : complex
    Carried far-field coefficients (tension-positive internal
    frame; data, never unknowns).
load_scale : float
    max |in-situ component| (> 0), the residual scale.
coef_scale : float
    Column/unknown scale load_scale * a (the potentials have the
    dimension stress x length)."""

    def __init__(self, material, a, sigma_v, K0, n_order=8):
        sxx, syy, sxy = k0_load(sigma_v, K0)
        self._init_from_load(material, a, sxx, syy, sxy, n_order)

    @classmethod
    def from_load(cls, material, a, sigma_xx_inf, sigma_yy_inf, sigma_xy_inf, n_order=8):
        """General uniform in-situ load (compression positive)."""
        self = cls.__new__(cls)
        self._init_from_load(material, a, sigma_xx_inf, sigma_yy_inf, sigma_xy_inf, n_order)
        return self

    def _init_from_load(self, material, a, sxx, syy, sxy, n_order):
        if not isinstance(material, TIMaterial):
            raise ValueError('material must be a TIMaterial instance')
        if not a > 0.0:
            raise ValueError('hole radius a must be positive')
        n_order = int(n_order)
        if n_order < 1:
            raise ValueError('n_order must be >= 1')
        self.material = material
        self.a = float(a)
        self.n_order = n_order
        self.sigma_xx_inf = float(sxx)
        self.sigma_yy_inf = float(syy)
        self.sigma_xy_inf = float(sxy)
        self.load_scale = max(abs(sxx), abs(syy), abs(sxy))
        if self.load_scale == 0.0:
            raise ValueError('in-situ load must be nonzero (it sets the residual and coefficient scales)')
        self.coef_scale = self.load_scale * self.a
        m = material
        self._mu = (m.mu1, m.mu2)
        self._p = (m.p1, m.p2)
        self._q = (m.q1, m.q2)
        self._gam = tuple(((1.0 - 1j * mu) / 2.0 for mu in self._mu))
        self._del = tuple(((1.0 + 1j * mu) / 2.0 for mu in self._mu))
        g1, g2 = solve_carriers(m.mu1, m.mu2, -sxx, -syy, -sxy)
        self.gamma_1, self.gamma_2 = (g1, g2)
        self._assert_log_free()

    def _assert_log_free(self):
        """Assert that the released load is self-equilibrated.

For uniform in-situ stress, the released traction has zero resultant over any closed wall by the divergence theorem. Integrate it over the mapped wall as a constructor check. The representation omits logarithmic terms and does not support net-resultant loading."""
        n = 256
        th = 2.0 * np.pi / n * np.arange(n)
        nx, ny = (np.cos(th), np.sin(th))
        tx = self.sigma_xx_inf * nx + self.sigma_xy_inf * ny
        ty = self.sigma_xy_inf * nx + self.sigma_yy_inf * ny
        ds = self.a * (2.0 * np.pi / n)
        X = float(np.sum(-tx) * ds)
        Y = float(np.sum(-ty) * ds)
        tol = 1e-12 * self.load_scale * self.a
        if max(abs(X), abs(Y)) > tol:
            raise AssertionError(f'released traction resultant ({X:.3e}, {Y:.3e}) exceeds the self-equilibration tolerance -- the log-free ansatz would be invalid')

    def omega_k(self, zeta, k):
        """omega_k(zeta) = gamma_k a zeta + delta_k a / zeta (k = 1, 2)."""
        zeta = np.asarray(zeta, dtype=complex)
        i = k - 1
        return self._gam[i] * self.a * zeta + self._del[i] * self.a / zeta

    def omega_k_prime(self, zeta, k):
        """omega_k'(zeta) = gamma_k a - delta_k a / zeta^2."""
        zeta = np.asarray(zeta, dtype=complex)
        i = k - 1
        return self._gam[i] * self.a - self._del[i] * self.a / (zeta * zeta)

    def zeta_k_from_z(self, z, k):
        """Exterior branch zeta_k of omega_k(zeta_k) = z_k(z).

        z_k = gamma_k z + delta_k conj(z); the inverse is the
        quadratic gamma_k a zeta^2 - z_k zeta + delta_k a = 0, whose
        root product delta_k/gamma_k has modulus < 1 (because
        |gamma_k|^2 - |delta_k|^2 = Im mu_k > 0), so the max-modulus
        root is the exterior branch |zeta_k| >= 1, strictly separated
        from the interior one for z outside the hole (selection is
        continuous; no branch-cut jumps around the hole).
        """
        z = np.asarray(z, dtype=complex)
        i = k - 1
        gk, dk = (self._gam[i], self._del[i])
        zk = gk * z + dk * np.conj(z)
        disc = np.sqrt(zk * zk - 4.0 * gk * dk * self.a * self.a)
        r1 = (zk + disc) / (2.0 * gk * self.a)
        r2 = (zk - disc) / (2.0 * gk * self.a)
        return np.where(np.abs(r1) >= np.abs(r2), r1, r2)

    def _as_coefficients(self, coef_1, coef_2):
        out = []
        for name, c in (('coef_1', coef_1), ('coef_2', coef_2)):
            c = np.asarray(c, dtype=complex)
            if c.shape != (self.n_order,):
                raise ValueError(f'{name} must have shape ({self.n_order},), got {c.shape}')
            out.append(c)
        return (out[0], out[1])

    def coefficients_to_vector(self, coef_1, coef_2):
        """Pack (coef_1, coef_2) into the real unknown vector.

        Ordering (fixed, mirrors the family)::

          x = [Re a^(1)_1..N | Im a^(1)_1..N |
               Re a^(2)_1..N | Im a^(2)_1..N]

        (the two free boundary constants of :func:`solve_ls` are NOT
        part of this vector; they are appended by the solve channel).
        """
        coef_1, coef_2 = self._as_coefficients(coef_1, coef_2)
        return np.concatenate([coef_1.real, coef_1.imag, coef_2.real, coef_2.imag])

    def vector_to_coefficients(self, x):
        """Inverse of :meth:`coefficients_to_vector`."""
        x = np.asarray(x, dtype=float)
        n = self.n_order
        if x.shape != (4 * n,):
            raise ValueError(f'x must have shape ({4 * n},), got {x.shape}')
        coef_1 = x[0:n] + 1j * x[n:2 * n]
        coef_2 = x[2 * n:3 * n] + 1j * x[3 * n:4 * n]
        return (coef_1, coef_2)

    def _series(self, zeta, coef):
        """phi_k0(zeta) = sum a_n zeta^{-n} and its zeta-derivative."""
        s = np.zeros(zeta.shape, dtype=complex)
        ds = np.zeros(zeta.shape, dtype=complex)
        inv = 1.0 / zeta
        p = inv
        for n in range(1, self.n_order + 1):
            s = s + coef[n - 1] * p
            ds = ds - n * coef[n - 1] * p * inv
            p = p * inv
        return (s, ds)

    def _parts(self, part):
        table = {'total': (True, True), 'excavation': (False, True), 'insitu': (True, False)}
        if part not in table:
            raise ValueError("part must be 'total', 'excavation' or 'insitu'")
        return table[part]

    def boundary_values(self, theta, coef_1, coef_2, part='total'):
        """The two Eq.(11)/(12) functionals on the wall (internal frame).

B_1(theta) = 2 Re[phi_1 + phi_2] and B_2(theta) = 2 Re[mu_1
phi_1 + mu_2 phi_2] of the requested part, evaluated on
|sigma| = 1 where zeta_1 = zeta_2 = sigma (fixed-domain boundary identity).  Tension-positive internal
frame (immaterial for the zero-traction condition); this is
the single physics source consumed by both the collocation
assembly (:meth:`boundary_system` unit-coefficient columns)
and residual evaluation."""
        theta = np.asarray(theta, dtype=float)
        carriers, decaying = self._parts(part)
        coef_1, coef_2 = self._as_coefficients(coef_1, coef_2)
        sig = np.exp(1j * theta)
        b1 = np.zeros(theta.shape, dtype=complex)
        b2 = np.zeros(theta.shape, dtype=complex)
        for k, (coef, gamma) in enumerate(((coef_1, self.gamma_1), (coef_2, self.gamma_2)), 1):
            phi = np.zeros(theta.shape, dtype=complex)
            if carriers:
                phi = phi + gamma * self.omega_k(sig, k)
            if decaying:
                s, _ = self._series(sig, coef)
                phi = phi + s
            b1 = b1 + phi
            b2 = b2 + self._mu[k - 1] * phi
        return (2.0 * np.real(b1), 2.0 * np.real(b2))

    def boundary_system(self, theta_wall):
        """Collocated functional boundary conditions, G x = h.

        Total-field zero-traction wall in the Eq.(11)/(12) functional
        form: B_1(theta_j) - C_1 = 0 and B_2(theta_j) - C_2 = 0, with
        C_1, C_2 the free additive constants of the traction
        integrals.  Unknown vector: the coefficient packing of
        :meth:`coefficients_to_vector` (4 N reals) plus [C_1, C_2] --
        columns 4N and 4N+1.  Rows: the B_1 block (M rows) stacked
        over the B_2 block (M rows); coefficient columns are built by
        unit-coefficient synthesis through :meth:`boundary_values`
        (part="excavation"); right-hand side h = -(carrier boundary
        values).

        Returns
        -------
        (G, h) : (ndarray (2 M, 4 N + 2), ndarray (2 M,))
        """
        theta = np.asarray(theta_wall, dtype=float).ravel()
        if theta.size == 0:
            raise ValueError('theta_wall must contain at least one angle')
        M = theta.size
        zero = np.zeros(self.n_order, dtype=complex)
        b1_in, b2_in = self.boundary_values(theta, zero, zero, part='insitu')
        h = -np.concatenate([b1_in, b2_in])
        n_coef = 4 * self.n_order
        G = np.zeros((2 * M, n_coef + 2))
        for kcol in range(n_coef):
            e = np.zeros(n_coef)
            e[kcol] = 1.0
            c1, c2 = self.vector_to_coefficients(e)
            b1, b2 = self.boundary_values(theta, c1, c2, part='excavation')
            G[:, kcol] = np.concatenate([b1, b2])
        G[:M, n_coef] = -1.0
        G[M:, n_coef + 1] = -1.0
        return (G, h)

    def _prepare(self, r, theta, check_r=True):
        r, theta = np.broadcast_arrays(np.asarray(r, dtype=float), np.asarray(theta, dtype=float))
        if check_r and np.any(r < self.a * _R_MIN_FACTOR):
            raise ValueError('exterior representation is defined for r >= a only')
        return (r, theta)

    def stresses_cartesian(self, r, theta, coef_1, coef_2, part='total'):
        """Cartesian stresses (COMPRESSION POSITIVE) at physical (r, theta).

Synthesis (source Eqs.(19)-(21)) with
phi_k'(z_k) = Gamma_k + phi_k0'(zeta_k) / omega_k'(zeta_k)
(the carrier derivative is exactly Gamma_k -- analytic
carrying, nothing approximated); the tension-positive result
is negated at return (the one stress sign conversion).

Parameters
----------
r, theta : array_like
    Physical polar coordinates about the hole centre
    (broadcast together); r >= a for parts that involve the
    hole ("total", "excavation").
coef_1, coef_2 : array_like, complex, shape (N,)
    Decaying-series coefficients of the two planes.
part : str
    "total" (default), "excavation" (decaying only) or
    "insitu" (carriers only; exact uniform field, any r).

Returns
-------
(sigma_xx, sigma_yy, sigma_xy) : tuple of ndarray
    ``total == insitu + excavation`` identically."""
        carriers, decaying = self._parts(part)
        coef_1, coef_2 = self._as_coefficients(coef_1, coef_2)
        r, theta = self._prepare(r, theta, check_r=decaying)
        if not decaying:
            shape = r.shape
            return (np.full(shape, self.sigma_xx_inf), np.full(shape, self.sigma_yy_inf), np.full(shape, self.sigma_xy_inf))
        z = r * np.exp(1j * theta)
        dphi = []
        for k, (coef, gamma) in enumerate(((coef_1, self.gamma_1), (coef_2, self.gamma_2)), 1):
            zk = self.zeta_k_from_z(z, k)
            _, ds = self._series(zk, coef)
            d = ds / self.omega_k_prime(zk, k)
            if carriers:
                d = d + gamma
            dphi.append(d)
        mu1, mu2 = self._mu
        s_xx_T = 2.0 * np.real(mu1 ** 2 * dphi[0] + mu2 ** 2 * dphi[1])
        s_yy_T = 2.0 * np.real(dphi[0] + dphi[1])
        s_xy_T = -2.0 * np.real(mu1 * dphi[0] + mu2 * dphi[1])
        return (-s_xx_T, -s_yy_T, -s_xy_T)

    def displacements_cartesian(self, r, theta, coef_1, coef_2):
        """Excavation-induced Cartesian displacements (plane strain).

        u = 2 Re[p_1 phi_10 + p_2 phi_20], v = 2 Re[q_1 phi_10 + q_2
        phi_20] on the decaying parts alone (the carriers hold the
        unbounded in-situ straining), then globally sign-flipped:
        (u_x, u_y) = -(tension-positive displacement), exactly the
        family convention of ``refsol/kirsch.py`` /
        ``general_map.displacements_cartesian`` (the returned strains
        map onto the returned compression-positive excavation stresses
        through the verbatim constitutive law -- both fields carry the
        same global flip).

        Parameters
        ----------
        r, theta : array_like
            Physical polar coordinates (broadcast together); r >= a.
        coef_1, coef_2 : array_like, complex, shape (N,)

        Returns
        -------
        (u_x, u_y) : tuple of ndarray
        """
        coef_1, coef_2 = self._as_coefficients(coef_1, coef_2)
        r, theta = self._prepare(r, theta)
        z = r * np.exp(1j * theta)
        u = np.zeros(z.shape, dtype=complex)
        v = np.zeros(z.shape, dtype=complex)
        for k, coef in enumerate((coef_1, coef_2), 1):
            zk = self.zeta_k_from_z(z, k)
            s, _ = self._series(zk, coef)
            u = u + self._p[k - 1] * s
            v = v + self._q[k - 1] * s
        return (-2.0 * np.real(u), -2.0 * np.real(v))

    def boundary_traction(self, theta, coef_1, coef_2, part='total'):
        """Physical wall traction (t_x, t_y) of the requested part.

        sigma . n with the outward radial normal n = (cos theta, sin
        theta) of the circle, formed from the public
        (compression-positive) stresses at r = a; the global sign flip
        is immaterial for the zero-traction condition.  Mirrors the
        family's ``boundary_traction`` API shape.
        """
        theta = np.asarray(theta, dtype=float)
        s_xx, s_yy, s_xy = self.stresses_cartesian(self.a, theta, coef_1, coef_2, part=part)
        n_x, n_y = (np.cos(theta), np.sin(theta))
        return (s_xx * n_x + s_xy * n_y, s_xy * n_x + s_yy * n_y)

def _uniform_wall_angles(n_collocation):
    if n_collocation < 1:
        raise ValueError('n_collocation must be >= 1')
    return 2.0 * np.pi / n_collocation * np.arange(n_collocation)

def solve_ls(rep, n_collocation=200):
    """Collocation least-squares solve of the traction-free wall.

Assembles :meth:`LekhnitskiiCircle.boundary_system` at
``n_collocation`` uniform angles and solves by
``numpy.linalg.lstsq`` in uniformly scaled unknowns (columns
multiplied by ``coef_scale = load_scale * a``; both the series
coefficients and the free constants C_1, C_2 carry the dimension
stress x length, so one uniform scale makes the unknown vector
O(1) and leaves the least-squares solution unchanged up to the
factor).  Mirrors the family ``solve_ls`` shape.

Returns
-------
dict
    ``coef_1`` / ``coef_2`` (complex (N,)), ``bc_constants``
    (C_1, C_2), ``coef_1_scaled`` / ``coef_2_scaled``, ``wall_s``,
    ``n_params`` (0 -- classical channel), ``rank``,
    ``wall_residual_max`` (max functional residual |G x - h| /
    (load_scale * a) -- the Eq.(11)/(12) functionals have the
    dimension stress x length), ``wall_traction_residual_max``
    (max |wall traction of the solved total field| / load_scale
    at ``n_collocation`` OFFSET probe angles -- a solve
    independent check through the stress evaluation),
    and ``n_collocation``."""
    theta = _uniform_wall_angles(n_collocation)
    t0 = time.perf_counter()
    G, h = rep.boundary_system(theta)
    g_scaled = G * rep.coef_scale
    x_scaled, _, rank, _ = np.linalg.lstsq(g_scaled, h, rcond=None)
    wall_s = time.perf_counter() - t0
    x = rep.coef_scale * x_scaled
    coef_1, coef_2 = rep.vector_to_coefficients(x[:4 * rep.n_order])
    residual = g_scaled @ x_scaled - h
    theta_probe = theta + np.pi / n_collocation
    t_x, t_y = rep.boundary_traction(theta_probe, coef_1, coef_2, part='total')
    return {'channel': 'ls', 'coef_1': coef_1, 'coef_2': coef_2, 'bc_constants': (float(x[-2]), float(x[-1])), 'coef_1_scaled': coef_1 / rep.coef_scale, 'coef_2_scaled': coef_2 / rep.coef_scale, 'wall_s': wall_s, 'n_params': 0, 'rank': int(rank), 'wall_residual_max': float(np.max(np.abs(residual)) / rep.coef_scale), 'wall_traction_residual_max': float(np.max(np.abs(np.concatenate([t_x, t_y]))) / rep.load_scale), 'n_collocation': int(n_collocation)}

def _omega_k_general(zeta, R, c, mu):
    """General affine double map (source Eq.(10) + c_0 constant).

omega_k(zeta) = gamma_k R (zeta + sum_{j=0..n} c_j zeta^{-j})
              + delta_k conj(R) (zeta^{-1} + sum_{j=0..n}
                conj(c_j) zeta^{j}),

the reflection form gamma_k omega(zeta) + delta_k
conj-coefficient-omega(1/zeta)
(Eq.(10) of the source omits the constant c_0 block; the constant
translation changes no derivative and no simplicity, and is kept
here so the boundary image is the exact affine image of the
physical boundary)."""
    zeta = np.asarray(zeta, dtype=complex)
    gk = (1.0 - 1j * mu) / 2.0
    dk = (1.0 + 1j * mu) / 2.0
    s_dir = zeta.astype(complex).copy()
    s_ref = 1.0 / zeta
    p_inv = np.ones(zeta.shape, dtype=complex)
    p_pos = np.ones(zeta.shape, dtype=complex)
    for j in range(c.size):
        s_dir = s_dir + c[j] * p_inv
        s_ref = s_ref + np.conj(c[j]) * p_pos
        p_inv = p_inv / zeta
        p_pos = p_pos * zeta
    return gk * R * s_dir + dk * np.conj(R) * s_ref

def _omega_k_prime_general(zeta, R, c, mu):
    """Derivative of :func:`_omega_k_general`::

      omega_k'(zeta) = gamma_k R (1 - sum_{j>=1} j c_j zeta^{-(j+1)})
                     + delta_k conj(R) (-zeta^{-2} + sum_{j>=1} j
                       conj(c_j) zeta^{j-1}).
    """
    zeta = np.asarray(zeta, dtype=complex)
    gk = (1.0 - 1j * mu) / 2.0
    dk = (1.0 + 1j * mu) / 2.0
    s_dir = np.ones(zeta.shape, dtype=complex)
    s_ref = -1.0 / (zeta * zeta)
    p_inv = 1.0 / (zeta * zeta)
    p_pos = np.ones(zeta.shape, dtype=complex)
    for j in range(1, c.size):
        s_dir = s_dir - j * c[j] * p_inv
        s_ref = s_ref + j * np.conj(c[j]) * p_pos
        p_inv = p_inv / zeta
        p_pos = p_pos * zeta
    return gk * R * s_dir + dk * np.conj(R) * s_ref

def _closed_polyline_simple(points, chunk=256):
    """True when the closed polygon through ``points`` is simple.

    Two detectors: (i) the strict-orientation proper-crossing test of
    the family's ``general_map._polygon_is_simple`` (adjacent segments
    excluded by index; O(K^2) pairwise in row chunks); (ii) a
    degeneracy detector for curves the strict test cannot see --
    exactly collinear back-and-forth traversals (e.g. an omega_k image
    collapsed onto a segment) have zero proper crossings but also
    (near-)zero enclosed area, so |shoelace area| <
    1e-9 * perimeter^2 is flagged non-simple (a genuinely simple
    closed curve of perimeter P encloses O(P^2) area unless it is
    degenerate; the circle has A = P^2 / (4 pi) ~ 0.08 P^2).
    """
    pts = np.asarray(points, dtype=complex).ravel()
    K = pts.size
    if K < 4:
        return True
    x1, y1 = (np.real(pts), np.imag(pts))
    x2, y2 = (np.roll(x1, -1), np.roll(y1, -1))
    per = float(np.sum(np.hypot(x2 - x1, y2 - y1)))
    area = 0.5 * float(np.sum(x1 * y2 - x2 * y1))
    if abs(area) < 1e-09 * per * per:
        return False
    jj = np.arange(K)[None, :]
    cx, cy = (x1[None, :], y1[None, :])
    dx, dy = (x2[None, :], y2[None, :])
    for i0 in range(0, K, chunk):
        i1 = min(i0 + chunk, K)
        ii = np.arange(i0, i1)[:, None]
        ax, ay = (x1[i0:i1][:, None], y1[i0:i1][:, None])
        bx, by = (x2[i0:i1][:, None], y2[i0:i1][:, None])
        d1 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        d2 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax)
        d3 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
        d4 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)
        crossing = (d1 * d2 < 0.0) & (d3 * d4 < 0.0)
        nonadjacent = (jj >= ii + 2) & ~((ii == 0) & (jj == K - 1))
        if np.any(crossing & nonadjacent):
            return False
    return True

def validate_univalence_k(R, c, mu, rho_work=1.5, n_scan_boundary=2048, n_rho=17, n_scan_annulus=1024):
    """Work-domain univalence acceptance of one double map (R, c; mu_k).

Evaluate the double map

  omega_k(zeta) = gamma_k R (zeta + sum_j c_j zeta^{-j})
                + delta_k conj(R) (zeta^{-1} + sum_j conj(c_j)
                  zeta^{j}),
  gamma_k = (1 - i mu)/2,  delta_k = (1 + i mu)/2.

For truncation order n >= 2 the far field of omega_k is
an n-fold cover (omega_k ~ delta_k conj(R) conj(c_n) zeta^n at
infinity whenever delta_k != 0, i.e. mu != i, and c_n != 0), so
GLOBAL exterior univalence is impossible and no far guard exists;
univalence is claimed and checked on the work annulus
1 <= |zeta| <= rho_work only.  The readouts are acceptance indicators at scan resolution (they certify the sampled grid and
polygon, not the continuum):

* min |omega_k'| over a dense |zeta| = 1 scan and over the
  n_rho x n_scan_annulus work-annulus grid;
* simplicity of the closed boundary image polygon (proper
  crossings + collinear-degeneracy detector);
* the sufficient boundary bound (analytic companion):
  |omega_k'| >= R [|gamma_k| (1 - s) - |delta_k| (1 + s)] on
  |zeta| = 1 with s = sum_{j>=1} j |c_j|, which is positive iff
  s < Im(mu) / (|gamma_k| + |delta_k|)^2; the threshold reduces to
  the isotropic value 1 EXACTLY at mu = i (gamma = 1,
  delta = 0).  The bound is SUFFICIENT, NOT NECESSARY: maps
  violating it can still pass the measured gate.

Parameters
----------
R : float
    Mapping radius (> 0) of the family convention.
c : array_like, complex, shape (n_map + 1,)
    Map coefficients (c[0] = translation, family-shaped; c[0]
    affects neither derivative nor simplicity).
mu : complex
    Upper-half-plane characteristic root of the material plane.
rho_work : float
    Outer radius of the work annulus (> 1); it must cover the
    intended computational domain of the field reconstruction.
n_scan_boundary, n_rho, n_scan_annulus : int
    Scan resolutions.

Returns
-------
dict
    ``min_abs_omega_k_prime_boundary``,
    ``min_abs_omega_k_prime_work`` (annulus grid incl. the
    boundary row), ``boundary_simple``, ``boundary_lower_bound``
    (the analytic boundary bound, as-is, possibly <= 0),
    ``sum_j_abs_jcj``, ``d14_bound_value`` (the threshold
    Im(mu) / (|gamma_k| + |delta_k|)^2), ``d14_bound_ok``
    (strict inequality sum < threshold), and ``univalent_ok`` =
    (both scan minima > 0) AND boundary_simple."""
    if not R > 0.0:
        raise ValueError('mapping radius R must be positive')
    c = np.asarray(c, dtype=complex).ravel()
    if c.size < 1:
        raise ValueError('c must contain at least the translation c_0')
    mu = complex(mu)
    if not mu.imag > 0.0:
        raise ValueError('mu must lie in the upper half plane')
    if not rho_work > 1.0:
        raise ValueError('rho_work must be > 1')
    if n_scan_boundary < 8:
        raise ValueError('n_scan_boundary must be >= 8')
    if n_rho < 2:
        raise ValueError('n_rho must be >= 2')
    if n_scan_annulus < 8:
        raise ValueError('n_scan_annulus must be >= 8')
    gk = (1.0 - 1j * mu) / 2.0
    dk = (1.0 + 1j * mu) / 2.0
    theta_b = 2.0 * np.pi / n_scan_boundary * np.arange(n_scan_boundary)
    sig = np.exp(1j * theta_b)
    min_boundary = float(np.min(np.abs(_omega_k_prime_general(sig, R, c, mu))))
    rho_g = np.linspace(1.0, rho_work, n_rho)[:, None]
    theta_a = 2.0 * np.pi / n_scan_annulus * np.arange(n_scan_annulus)[None, :]
    min_work = float(np.min(np.abs(_omega_k_prime_general(rho_g * np.exp(1j * theta_a), R, c, mu))))
    j = np.arange(1, c.size)
    s = float(np.sum(j * np.abs(c[1:])))
    d14_bound_value = mu.imag / (abs(gk) + abs(dk)) ** 2
    boundary_lower_bound = R * (abs(gk) * (1.0 - s) - abs(dk) * (1.0 + s))
    boundary_simple = bool(_closed_polyline_simple(_omega_k_general(sig, R, c, mu)))
    univalent_ok = bool(min_boundary > 0.0 and min_work > 0.0 and boundary_simple)
    return {'min_abs_omega_k_prime_boundary': min_boundary, 'min_abs_omega_k_prime_work': min_work, 'boundary_simple': boundary_simple, 'boundary_lower_bound': boundary_lower_bound, 'sum_j_abs_jcj': s, 'd14_bound_value': d14_bound_value, 'd14_bound_ok': bool(s < d14_bound_value), 'univalent_ok': univalent_ok}
