import os
import sys
import time
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import lekhnitskii_map as lm
__all__ = ['LekhnitskiiGeneral', 'solve_ls']
_RHO_MIN = 1.0 - 1e-09
_CLUSTER_ZERO_MIN = 0.35

def _winding_inside(pts, poly):
    """True per point when strictly inside the closed polygon
    ``poly`` (complex vertices; even-odd crossing rule)."""
    pts = np.asarray(pts, dtype=complex).ravel()
    x, y = (np.real(poly), np.imag(poly))
    x2, y2 = (np.roll(x, -1), np.roll(y, -1))
    dy = np.where(y2 == y, 1e-300, y2 - y)
    inside = np.zeros(pts.shape, dtype=bool)
    for i, z in enumerate(pts):
        px, py = (z.real, z.imag)
        cond = (y > py) != (y2 > py)
        xin = x + (py - y) * (x2 - x) / dy
        inside[i] = bool(np.sum(cond & (px < xin)) % 2)
    return inside

class LekhnitskiiGeneral:
    """General-Laurent-map anisotropic excavation problem (total
field, completed globally-valid decaying basis).

Case data: material (a :class:`lekhnitskii_map.TIMaterial`), the
GIVEN iso map (R, c), and the uniform compression-positive
in-situ Cartesian stresses.  Pure excavation release:
traction-free wall, far field carried analytically by Gamma_k
with no logarithmic terms (self-equilibration asserted at construction).  Gate the iso map through
``general_map.validate_univalence`` first;
the double-map work-domain validator
``lekhnitskii_map.validate_univalence_k`` remains the family
map-quality check.

Parameters
----------
material : lekhnitskii_map.TIMaterial
R : float
    Mapping radius (> 0); sets ``coef_scale = load_scale * R``.
c : array_like, complex, shape (n_map + 1,)
    Iso-map coefficients (c[0] = translation), GIVEN data.
sigma_v : float
    Vertical in-situ stress (compression positive).
K0 : float
    Lateral pressure coefficient (:func:`lekhnitskii_map.k0_load`;
    the material dip never rotates the load).
n_order : int
    Order N >= 1 of the ztilde-Laurent backbone.  For
    n_map <= 1 this is the whole basis (the classically
    complete case); for n_map >= 2 the pole block (ring of
    ``ring_multiple * N`` + lightning clusters) is added.
ring_multiple : int
    Uniform MFS ring size as a multiple of N (default 2; the
    ring carries the smooth shape content).
eta_ring : float or None
    Iso level of the ring poles; default (None) uses
    (1 + rho_crit)/2 with rho_crit the largest interior
    critical-zero modulus of omega.
cluster_len, cluster_ratio : int, float
    Lightning cluster length L and geometric ratio q per
    interior critical value (defaults 12 and 0.5; poles at
    distances 0.5 |w_c - z0_k| q^j from the critical value
    toward the hole centre image).
cluster_zero_min : float or None
    Modulus floor for the interior critical zeros that receive
    clusters (default None = the module constant 0.35);
    cluster_zero_min = 1.0 disables the clusters entirely --
    the full-rank ring-only channel used by the equivariance
    control leg.
mfs_offset : float
    Angular offset of the uniform ring generator (rotation-covariance parameter).
n_modes : int or None
    None (default) selects the classical completed dictionary
    (ztilde backbone + poles).  An int selects the RING-MODE
    channel: the decaying
    basis per plane is ``n_modes`` DFT ring modes, built for any
    n_map; ``n_order`` plays no basis role there and the
    cluster knobs are ignored.
ring_quad_multiple : int
    Ring-mode quadrature density: J = max(ring_quad_multiple *
    n_modes, alias floor) poles per plane (>= 2; default 4).

Attributes
----------
gamma_1, gamma_2 : complex
    Carried far-field coefficients (tension-positive internal
    frame; data, never unknowns).
load_scale, coef_scale : float
    max |in-situ component| (> 0) and load_scale * R.
n_map, n_pole, n_basis : int
    Highest inverse map power; poles per plane (ring-mode
    channel: quadrature poles); basis size per plane
    (n_order + n_pole classically, n_modes on the ring-mode
    channel)."""

    def __init__(self, material, R, c, sigma_v, K0, n_order=16, ring_multiple=2, eta_ring=None, cluster_len=12, cluster_ratio=0.5, cluster_zero_min=None, mfs_offset=0.0, n_modes=None, ring_quad_multiple=4):
        sxx, syy, sxy = lm.k0_load(sigma_v, K0)
        self._init_from_load(material, R, c, sxx, syy, sxy, n_order, ring_multiple, eta_ring, cluster_len, cluster_ratio, cluster_zero_min, mfs_offset, n_modes, ring_quad_multiple)

    @classmethod
    def from_load(cls, material, R, c, sigma_xx_inf, sigma_yy_inf, sigma_xy_inf, n_order=16, ring_multiple=2, eta_ring=None, cluster_len=12, cluster_ratio=0.5, cluster_zero_min=None, mfs_offset=0.0, n_modes=None, ring_quad_multiple=4):
        """General uniform in-situ load (compression positive)."""
        self = cls.__new__(cls)
        self._init_from_load(material, R, c, sigma_xx_inf, sigma_yy_inf, sigma_xy_inf, n_order, ring_multiple, eta_ring, cluster_len, cluster_ratio, cluster_zero_min, mfs_offset, n_modes, ring_quad_multiple)
        return self

    def _init_from_load(self, material, R, c, sxx, syy, sxy, n_order, ring_multiple, eta_ring, cluster_len, cluster_ratio, cluster_zero_min, mfs_offset, n_modes=None, ring_quad_multiple=4):
        if not isinstance(material, lm.TIMaterial):
            raise ValueError('material must be a TIMaterial instance')
        if not R > 0.0:
            raise ValueError('mapping radius R must be positive')
        c = np.array(c, dtype=complex, copy=True)
        if c.ndim != 1 or c.size < 1:
            raise ValueError(f'c must be a 1-D array (c_0, .., c_{{n_map}}) with at least the translation entry c_0, got shape {c.shape}')
        n_order = int(n_order)
        if n_order < 1:
            raise ValueError('n_order must be >= 1')
        if int(ring_multiple) < 1:
            raise ValueError('ring_multiple must be >= 1')
        if int(cluster_len) < 1:
            raise ValueError('cluster_len must be >= 1')
        if not 0.0 < float(cluster_ratio) < 1.0:
            raise ValueError('cluster_ratio must be in (0, 1)')
        if cluster_zero_min is None:
            cluster_zero_min = _CLUSTER_ZERO_MIN
        if not 0.0 < float(cluster_zero_min) <= 1.0:
            raise ValueError('cluster_zero_min must be in (0, 1] (1.0 disables the lightning clusters -- the ring-only channel)')
        if n_modes is not None:
            n_modes = int(n_modes)
            if n_modes < 1:
                raise ValueError('n_modes must be >= 1')
            if int(ring_quad_multiple) < 2:
                raise ValueError('ring_quad_multiple must be >= 2')
        self.n_modes = n_modes
        self.ring_quad_multiple = int(ring_quad_multiple)
        self.material = material
        self.R = float(R)
        self.c = c
        self.n_map = c.size - 1
        self.n_order = n_order
        self.sigma_xx_inf = float(sxx)
        self.sigma_yy_inf = float(syy)
        self.sigma_xy_inf = float(sxy)
        self.load_scale = max(abs(sxx), abs(syy), abs(sxy))
        if self.load_scale == 0.0:
            raise ValueError('in-situ load must be nonzero (it sets the residual and coefficient scales)')
        self.coef_scale = self.load_scale * self.R
        m = material
        self._mu = (m.mu1, m.mu2)
        self._p = (m.p1, m.p2)
        self._q = (m.q1, m.q2)
        self._gam = tuple(((1.0 - 1j * mu) / 2.0 for mu in self._mu))
        self._del = tuple(((1.0 + 1j * mu) / 2.0 for mu in self._mu))
        g1, g2 = lm.solve_carriers(m.mu1, m.mu2, -sxx, -syy, -sxy)
        self.gamma_1, self.gamma_2 = (g1, g2)
        self._assert_log_free()
        self._build_basis(int(ring_multiple), eta_ring, int(cluster_len), float(cluster_ratio), float(cluster_zero_min), float(mfs_offset))

    def _assert_log_free(self):
        """Assert that the released load is self-equilibrated.

For uniform in-situ stress, the released traction has zero resultant over any closed wall by the divergence theorem. Integrate it over the mapped wall as a constructor check. The representation omits logarithmic terms and does not support net-resultant loading."""
        n = 512
        th = 2.0 * np.pi / n * np.arange(n)
        sig = np.exp(1j * th)
        dz = 1j * sig * self.omega_prime(sig) * (2.0 * np.pi / n)
        dx, dy = (np.real(dz), np.imag(dz))
        X = float(np.sum(-(self.sigma_xx_inf * dy - self.sigma_xy_inf * dx)))
        Y = float(np.sum(-(self.sigma_xy_inf * dy - self.sigma_yy_inf * dx)))
        tol = 1e-12 * self.load_scale * self.R * (1.0 + self.n_map)
        if max(abs(X), abs(Y)) > tol:
            raise AssertionError(f'released traction resultant ({X:.3e}, {Y:.3e}) exceeds the self-equilibration tolerance -- the log-free ansatz would be invalid')

    def _interior_crit_zeros(self, zero_min):
        """Interior zeros of omega' (singularity anchors): roots of
        zeta^{n+1} - sum_j j c_j zeta^{n-j} with modulus in
        (zero_min, 1)."""
        n = self.n_map
        coeff = np.zeros(n + 2, dtype=complex)
        coeff[0] = 1.0
        for j in range(1, n + 1):
            coeff[j + 1] = -j * self.c[j]
        r = np.roots(coeff)
        return r[(np.abs(r) > zero_min) & (np.abs(r) < 1.0)]

    def _build_basis(self, ring_multiple, eta_ring, cluster_len, cluster_ratio, cluster_zero_min, mfs_offset):
        """Ellipse-truncation constants, pole sets, validity asserts."""
        c0 = self.c[0]
        c1 = self.c[1] if self.n_map >= 1 else 0.0 + 0j
        th512 = 2.0 * np.pi / 512 * np.arange(512)
        wall_iso = self.omega(np.exp(1j * th512))
        self._ell = []
        for k in range(2):
            gk, dk = (self._gam[k], self._del[k])
            A = gk * self.R * c0 + dk * np.conj(self.R * c0)
            B = gk * self.R + dk * np.conj(self.R * c1)
            C = gk * self.R * c1 + dk * np.conj(self.R)
            if not abs(C) < abs(B):
                raise ValueError(f'ellipse truncation of the double map is not exterior-univalent for plane {k + 1} (|C| = {abs(C):.3e} >= |B| = {abs(B):.3e}); the completed basis requires c_1 inside the elliptic univalence range')
            self._ell.append((A, B, C))
        self.eta_ring = None
        self._w_dft = None
        self._poles = [np.zeros(0, dtype=complex), np.zeros(0, dtype=complex)]
        if self.n_modes is not None:
            zc_all = self._interior_crit_zeros(0.0)
            rho_crit = float(np.max(np.abs(zc_all))) if zc_all.size else 0.5
            eta = float(eta_ring) if eta_ring is not None else 0.5 * (1.0 + rho_crit)
            if not 0.0 < eta < 1.0:
                raise ValueError('eta_ring must be in (0, 1)')
            self.eta_ring = eta
            j_alias = int(np.ceil(np.log(1e-15) / np.log(eta)))
            J = max(self.ring_quad_multiple * self.n_modes, j_alias)
            thj = mfs_offset + 2.0 * np.pi / J * np.arange(J)
            ring_iso = self.omega(eta * np.exp(1j * thj))
            for k in range(2):
                gk, dk = (self._gam[k], self._del[k])
                wall_k = gk * wall_iso + dk * np.conj(wall_iso)
                ring_k = gk * ring_iso + dk * np.conj(ring_iso)
                ok = _winding_inside(ring_k, wall_k)
                if not ok.all():
                    raise ValueError(f'{int(np.sum(~ok))} ring-mode quadrature poles of plane {k + 1} fall outside the mapped hole; the map geometry is outside the validity domain of the ring layout (adjust eta_ring)')
                self._poles[k] = ring_k
            m = np.arange(self.n_modes)
            self._w_dft = np.exp(-1j * np.outer(thj, m)) / (J * eta ** m)[None, :]
        elif self.n_map >= 2:
            zc_all = self._interior_crit_zeros(0.0)
            rho_crit = float(np.max(np.abs(zc_all))) if zc_all.size else 0.5
            zc = zc_all[np.abs(zc_all) > cluster_zero_min]
            eta = float(eta_ring) if eta_ring is not None else 0.5 * (1.0 + rho_crit)
            if not 0.0 < eta < 1.0:
                raise ValueError('eta_ring must be in (0, 1)')
            self.eta_ring = eta
            J = ring_multiple * self.n_order
            thj = mfs_offset + 2.0 * np.pi / J * np.arange(J)
            ring_iso = self.omega(eta * np.exp(1j * thj))
            wc_iso = self.omega(zc) if zc.size else np.zeros(0, dtype=complex)
            z0_iso = self.R * c0
            for k in range(2):
                gk, dk = (self._gam[k], self._del[k])
                wall_k = gk * wall_iso + dk * np.conj(wall_iso)
                ring_k = gk * ring_iso + dk * np.conj(ring_iso)
                wc_k = gk * wc_iso + dk * np.conj(wc_iso)
                z0_k = gk * z0_iso + dk * np.conj(z0_iso)
                cl = []
                for w in wc_k:
                    d0 = 0.5 * abs(w - z0_k)
                    if d0 == 0.0:
                        continue
                    dirn = (z0_k - w) / abs(z0_k - w)
                    for jj in range(1, cluster_len + 1):
                        cl.append(w + dirn * d0 * cluster_ratio ** jj)
                pk = np.concatenate([ring_k, np.asarray(cl, dtype=complex)])
                ok = _winding_inside(pk, wall_k)
                if not ok.all():
                    raise ValueError(f'{int(np.sum(~ok))} basis poles of plane {k + 1} fall outside the mapped hole; the map geometry is outside the validity domain of the default pole layout (adjust eta_ring / cluster parameters)')
                A, B, C = self._ell[k]
                rr = np.sqrt(abs(C / B))
                amb = A + B * rr * np.exp(1j * th512[::8]) + C / (rr * np.exp(1j * th512[::8]))
                if not _winding_inside(amb, wall_k).all():
                    raise ValueError(f'ztilde branch-ambiguity circle leaves the mapped hole for plane {k + 1}; the ellipse-truncated backbone is invalid for this map')
                zcrit = np.sqrt(C / B)
                for s in (zcrit, -zcrit):
                    w = A + B * s + C / s
                    if not _winding_inside(np.array([w]), wall_k)[0]:
                        raise ValueError(f'ztilde critical value outside the mapped hole for plane {k + 1}')
                self._poles[k] = pk
        self.n_pole = int(self._poles[0].size)
        self.n_basis = self.n_modes if self.n_modes is not None else self.n_order + self.n_pole
        self._s_mfs = tuple((0.1 * self.R * abs(g) for g in self._gam))

    def omega(self, zeta):
        """Iso map z = omega(zeta) = R (zeta + sum c_j zeta^{-j})."""
        zeta = np.asarray(zeta, dtype=complex)
        s = np.zeros(zeta.shape, dtype=complex)
        p = np.ones(zeta.shape, dtype=complex)
        for j in range(self.n_map + 1):
            s = s + self.c[j] * p
            p = p / zeta
        return self.R * (zeta + s)

    def omega_prime(self, zeta):
        """omega'(zeta) = R (1 - sum_{j>=1} j c_j zeta^{-(j+1)})."""
        zeta = np.asarray(zeta, dtype=complex)
        s = np.zeros(zeta.shape, dtype=complex)
        p = 1.0 / (zeta * zeta)
        for j in range(1, self.n_map + 1):
            s = s + j * self.c[j] * p
            p = p / zeta
        return self.R * (1.0 - s)

    def omega_k(self, zeta, k):
        """Double map omega_k = gamma_k omega(zeta) + delta_k
omega-bar(1/zeta) (k = 1, 2).

Map-analysis layer: the boundary identity omega_k(sigma) =
gamma_k omega(sigma) + delta_k conj(omega(sigma)) anchors
the wall rows, and
``lekhnitskii_map.validate_univalence_k`` carries the same
double-map form.  The FIELD path of this module
never inverts omega_k."""
        zeta = np.asarray(zeta, dtype=complex)
        i = k - 1
        s_ref = 1.0 / zeta
        p = np.ones(zeta.shape, dtype=complex)
        for j in range(self.n_map + 1):
            s_ref = s_ref + np.conj(self.c[j]) * p
            p = p * zeta
        return self._gam[i] * self.omega(zeta) + self._del[i] * np.conj(self.R) * s_ref

    def omega_k_prime(self, zeta, k):
        """omega_k'(zeta) = gamma_k omega'(zeta) + delta_k conj(R)
        (-zeta^{-2} + sum_{j>=1} j conj(c_j) zeta^{j-1})."""
        zeta = np.asarray(zeta, dtype=complex)
        i = k - 1
        s_ref = -1.0 / (zeta * zeta)
        p = np.ones(zeta.shape, dtype=complex)
        for j in range(1, self.n_map + 1):
            s_ref = s_ref + j * np.conj(self.c[j]) * p
            p = p * zeta
        return self._gam[i] * self.omega_prime(zeta) + self._del[i] * np.conj(self.R) * s_ref

    def boundary_points(self, theta):
        """Physical wall points (x, y) = omega(e^{i theta})."""
        z = self.omega(np.exp(1j * np.asarray(theta, dtype=float)))
        return (np.real(z), np.imag(z))

    def _z_k(self, z, k):
        """z_k = gamma_k z + delta_k conj(z) (physical-point conjugation)."""
        i = k - 1
        return self._gam[i] * z + self._del[i] * np.conj(z)

    def zeta_tilde(self, z, k):
        """Exterior branch of the ellipse-truncated double-map
        inverse, ztilde_k(z_k(z)).

        omega_k^ell(zeta) = A + B zeta + C / zeta with the
        construction-time constants; the exterior root of
        B zeta^2 + (A - z_k) zeta + C = 0 is the max-modulus one
        (root product C/B has modulus < 1, and the branch-ambiguity
        circle |r_1| = |r_2| is asserted inside the hole, so the
        selection is continuous on the exterior domain).  For
        n_map <= 1, omega_k^ell == omega_k and this is the
        classical image variable zeta_k.
        """
        z = np.asarray(z, dtype=complex)
        zk = self._z_k(z, k)
        A, B, C = self._ell[k - 1]
        b = A - zk
        disc = np.sqrt(b * b - 4.0 * B * C)
        r1 = (-b + disc) / (2.0 * B)
        r2 = (-b - disc) / (2.0 * B)
        return np.where(np.abs(r1) >= np.abs(r2), r1, r2)

    def zeta_iso_from_z(self, z):
        """Unique exterior iso-map preimage zeta of omega(zeta) = z.

Exterior-domain gate of the field APIs (degree n_map + 1
polynomial per point; the GIVEN iso map is required globally
exterior-univalent, so at most one root
has |root| >= 1).  Zero exterior roots => the point is not
in the represented exterior domain; more than one => the
univalence precondition is violated -- both raise
ValueError."""
        z = np.asarray(z, dtype=complex)
        flat = z.ravel()
        n = self.n_map
        base = np.zeros(n + 2, dtype=complex)
        base[0] = self.R
        for j in range(n + 1):
            base[j + 1] += self.R * self.c[j]
        out = np.empty(flat.shape, dtype=complex)
        for i, zi in enumerate(flat):
            coeff = base.copy()
            coeff[1] -= zi
            roots = np.roots(coeff)
            cand = roots[np.abs(roots) >= _RHO_MIN]
            if cand.size == 0:
                raise ValueError(f'point is not in the represented exterior domain (no |zeta| >= 1 preimage under the iso map): z = {zi:.6g}')
            if cand.size > 1:
                raise ValueError(f'iso map is not exterior-univalent at z = {zi:.6g} ({cand.size} exterior preimages); gate the map through general_map.validate_univalence first')
            out[i] = cand[0]
        return out.reshape(z.shape)

    def _phi_basis(self, z, k, derivative=False):
        """(val, der) matrices of the decaying basis at points z.

Shape z.shape + (n_basis,): column i is basis function i of
plane k (ztilde-Laurent block, then ring poles, then
lightning clusters) evaluated at z_k(z); ``der`` is the
d/dz_k derivative (chain rule through the QUADRATIC
ellipse-truncation inverse for the Laurent block; direct
closed form for the poles).  Every column is holomorphic on
E_k and decaying."""
        z = np.asarray(z, dtype=complex)
        zk = self._z_k(z, k)
        if self.n_modes is not None:
            s_k = self._s_mfs[k - 1]
            d = zk[..., None] - self._poles[k - 1]
            val = s_k / d @ self._w_dft
            der = -s_k / (d * d) @ self._w_dft if derivative else None
            return (val, der)
        A, B, C = self._ell[k - 1]
        zt = self.zeta_tilde(z, k)
        nb = self.n_basis
        val = np.zeros(z.shape + (nb,), dtype=complex)
        der = np.zeros(z.shape + (nb,), dtype=complex) if derivative else None
        if derivative:
            dom = B - C / (zt * zt)
        invz = 1.0 / zt
        p = invz.copy()
        for n in range(1, self.n_order + 1):
            val[..., n - 1] = p
            if derivative:
                der[..., n - 1] = -n * p * invz / dom
            p = p * invz
        s_k = self._s_mfs[k - 1]
        for j in range(self.n_pole):
            d = zk - self._poles[k - 1][j]
            val[..., self.n_order + j] = s_k / d
            if derivative:
                der[..., self.n_order + j] = -s_k / (d * d)
        return (val, der)

    def _as_coefficients(self, coef_1, coef_2):
        out = []
        for name, cc in (('coef_1', coef_1), ('coef_2', coef_2)):
            cc = np.asarray(cc, dtype=complex)
            if cc.shape != (self.n_basis,):
                raise ValueError(f'{name} must have shape ({self.n_basis},), got {cc.shape}')
            out.append(cc)
        return (out[0], out[1])

    def coefficients_to_vector(self, coef_1, coef_2):
        """Pack (coef_1, coef_2) into the real unknown vector
        [Re c^(1) | Im c^(1) | Re c^(2) | Im c^(2)] over the
        n_basis = n_order + n_pole columns per plane (Laurent
        block, ring poles, lightning clusters); the two free
        boundary constants are appended by the solve channel."""
        coef_1, coef_2 = self._as_coefficients(coef_1, coef_2)
        return np.concatenate([coef_1.real, coef_1.imag, coef_2.real, coef_2.imag])

    def vector_to_coefficients(self, x):
        """Inverse of :meth:`coefficients_to_vector`."""
        x = np.asarray(x, dtype=float)
        n = self.n_basis
        if x.shape != (4 * n,):
            raise ValueError(f'x must have shape ({4 * n},), got {x.shape}')
        return (x[0:n] + 1j * x[n:2 * n], x[2 * n:3 * n] + 1j * x[3 * n:4 * n])

    def _parts(self, part):
        table = {'total': (True, True), 'excavation': (False, True), 'insitu': (True, False)}
        if part not in table:
            raise ValueError("part must be 'total', 'excavation' or 'insitu'")
        return table[part]

    def boundary_values(self, theta, coef_1, coef_2, part='total'):
        """The two Eq.(11)/(12) functionals on the wall (internal
        frame).

        B_1(theta) = 2 Re[phi_1 + phi_2], B_2(theta) = 2 Re[mu_1
        phi_1 + mu_2 phi_2] of the requested part at the shared
        wall parameter sigma = e^{i theta} (fixed-domain result:
        both planes evaluate at z_k = gamma_k omega(sigma) +
        delta_k conj(omega(sigma)) of the SAME sigma).
        Tension-positive internal frame; both the collocation
        assembly and residual evaluation consume this single
        physics source (via :meth:`_phi_basis`).
        """
        theta = np.asarray(theta, dtype=float)
        carriers, decaying = self._parts(part)
        coef_1, coef_2 = self._as_coefficients(coef_1, coef_2)
        wb = self.omega(np.exp(1j * theta))
        b1 = np.zeros(theta.shape, dtype=complex)
        b2 = np.zeros(theta.shape, dtype=complex)
        for k, (coef, gamma) in enumerate(((coef_1, self.gamma_1), (coef_2, self.gamma_2)), 1):
            phi = np.zeros(theta.shape, dtype=complex)
            if carriers:
                phi = phi + gamma * self._z_k(wb, k)
            if decaying:
                val, _ = self._phi_basis(wb, k)
                phi = phi + val @ coef
            b1 = b1 + phi
            b2 = b2 + self._mu[k - 1] * phi
        return (2.0 * np.real(b1), 2.0 * np.real(b2))

    def boundary_system(self, theta_wall):
        """Collocated functional boundary conditions, G x = h.

        Total-field zero-traction wall in the Eq.(11)/(12)
        functional form: B_1(theta_j) - C_1 = 0 and B_2(theta_j) -
        C_2 = 0 with C_1, C_2 the free additive constants.  Unknown
        vector: the packing of :meth:`coefficients_to_vector`
        (4 n_basis reals) plus [C_1, C_2] -- columns 4 n_basis and
        4 n_basis + 1.  Rows: the B_1 block (M rows) stacked over
        the B_2 block; coefficient columns from the same
        :meth:`_phi_basis` traces consumed by
        :meth:`boundary_values` (single physics source).

        Returns
        -------
        (G, h) : (ndarray (2 M, 4 n_basis + 2), ndarray (2 M,))
        """
        theta = np.asarray(theta_wall, dtype=float).ravel()
        if theta.size == 0:
            raise ValueError('theta_wall must contain at least one angle')
        M = theta.size
        wb = self.omega(np.exp(1j * theta))
        zero = np.zeros(self.n_basis, dtype=complex)
        b1_in, b2_in = self.boundary_values(theta, zero, zero, part='insitu')
        h = -np.concatenate([b1_in, b2_in])
        nb = self.n_basis
        G = np.zeros((2 * M, 4 * nb + 2))
        for k in (1, 2):
            val, _ = self._phi_basis(wb, k)
            mu = self._mu[k - 1]
            base = 2 * nb * (k - 1)
            G[:M, base:base + nb] = 2.0 * np.real(val)
            G[M:, base:base + nb] = 2.0 * np.real(mu * val)
            G[:M, base + nb:base + 2 * nb] = -2.0 * np.imag(val)
            G[M:, base + nb:base + 2 * nb] = -2.0 * np.imag(mu * val)
        G[:M, 4 * nb] = -1.0
        G[M:, 4 * nb + 1] = -1.0
        return (G, h)

    def _prepare_points(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return x + 1j * y

    def stresses_cartesian(self, x, y, coef_1, coef_2, part='total'):
        """Cartesian stresses (COMPRESSION POSITIVE) at physical
(x, y).

Synthesis (source Eqs.(19)-(21)) with
phi_k'(z_k) = Gamma_k + phi_k0'(z_k) (analytic carrying --
the carrier derivative is exactly Gamma_k, nothing
approximated; phi_k0' in closed form from the completed
basis, no image-variable inversion); the tension-positive
result is negated at return (the one stress sign
conversion).

Parameters
----------
x, y : array_like
    Physical Cartesian coordinates (broadcast together).
    Points must lie in the represented exterior domain for
    parts involving the hole ("total", "excavation"); the
    gate is :meth:`zeta_iso_from_z`.
coef_1, coef_2 : array_like, complex, shape (n_basis,)
    Decaying-basis coefficients of the two planes.
part : str
    "total" (default), "excavation" (decaying only) or
    "insitu" (carriers only; exact uniform field, any
    point).

Returns
-------
(sigma_xx, sigma_yy, sigma_xy) : tuple of ndarray
    ``total == insitu + excavation`` identically."""
        carriers, decaying = self._parts(part)
        coef_1, coef_2 = self._as_coefficients(coef_1, coef_2)
        z = self._prepare_points(x, y)
        if not decaying:
            shape = z.shape
            return (np.full(shape, self.sigma_xx_inf), np.full(shape, self.sigma_yy_inf), np.full(shape, self.sigma_xy_inf))
        self.zeta_iso_from_z(z)
        dphi = []
        for k, (coef, gamma) in enumerate(((coef_1, self.gamma_1), (coef_2, self.gamma_2)), 1):
            _, der = self._phi_basis(z, k, derivative=True)
            d = der @ coef
            if carriers:
                d = d + gamma
            dphi.append(d)
        mu1, mu2 = self._mu
        s_xx_T = 2.0 * np.real(mu1 ** 2 * dphi[0] + mu2 ** 2 * dphi[1])
        s_yy_T = 2.0 * np.real(dphi[0] + dphi[1])
        s_xy_T = -2.0 * np.real(mu1 * dphi[0] + mu2 * dphi[1])
        return (-s_xx_T, -s_yy_T, -s_xy_T)

    def displacements_cartesian(self, x, y, coef_1, coef_2):
        """Excavation-induced Cartesian displacements (plane
        strain).

        u = 2 Re[p_1 phi_10 + p_2 phi_20], v = 2 Re[q_1 phi_10 +
        q_2 phi_20] on the decaying parts alone (the carriers hold
        the unbounded in-situ straining), then globally
        sign-flipped -- the family convention shared with
        ``lekhnitskii_map.displacements_cartesian``.

        Parameters
        ----------
        x, y : array_like
            Physical Cartesian coordinates in the exterior domain.
        coef_1, coef_2 : array_like, complex, shape (n_basis,)

        Returns
        -------
        (u_x, u_y) : tuple of ndarray
        """
        coef_1, coef_2 = self._as_coefficients(coef_1, coef_2)
        z = self._prepare_points(x, y)
        self.zeta_iso_from_z(z)
        u = np.zeros(z.shape, dtype=complex)
        v = np.zeros(z.shape, dtype=complex)
        for k, coef in enumerate((coef_1, coef_2), 1):
            val, _ = self._phi_basis(z, k)
            s = val @ coef
            u = u + self._p[k - 1] * s
            v = v + self._q[k - 1] * s
        return (-2.0 * np.real(u), -2.0 * np.real(v))

    def boundary_traction(self, theta, coef_1, coef_2, part='total'):
        """Physical wall traction (t_x, t_y) of the requested part.

sigma . n at the wall points omega(e^{i theta}) with the
outward unit normal n = sigma omega'(sigma) / |sigma
omega'(sigma)| (the image of the radial direction, exactly
as ``general_map.boundary_traction``), formed from the
public (compression-positive) stresses through the same
physical-point synthesis path -- an independent stress-based traction check."""
        theta = np.asarray(theta, dtype=float)
        xw, yw = self.boundary_points(theta)
        s_xx, s_yy, s_xy = self.stresses_cartesian(xw, yw, coef_1, coef_2, part=part)
        sig = np.exp(1j * theta)
        nvec = sig * self.omega_prime(sig)
        nvec = nvec / np.abs(nvec)
        n_x, n_y = (np.real(nvec), np.imag(nvec))
        return (s_xx * n_x + s_xy * n_y, s_xy * n_x + s_yy * n_y)

def _uniform_wall_angles(n_collocation, theta_offset):
    if n_collocation < 1:
        raise ValueError('n_collocation must be >= 1')
    return theta_offset + 2.0 * np.pi / n_collocation * np.arange(n_collocation)

def solve_ls(rep, n_collocation=400, theta_offset=0.0):
    """Solve the traction-free wall conditions by scaled least squares.

Assemble boundary_system at n_collocation uniform angles with theta_offset. Scale columns by coef_scale and solve using numpy.linalg.lstsq. The minimum-norm solution resolves redundant columns; the returned rank records numerical rank. Evaluate wall traction independently at offset probe angles.

Returns
-------
dict
    ``coef_1`` / ``coef_2`` (complex (n_basis,)),
    ``bc_constants`` (C_1, C_2), ``coef_1_scaled`` /
    ``coef_2_scaled``, ``wall_s``, ``n_params`` (0 -- classical
    channel), ``rank``, ``n_columns``, ``wall_residual_max``
    (max functional residual |G x - h| / coef_scale),
    ``wall_traction_residual_max`` (max |wall traction of the
    solved total field| / load_scale at ``n_collocation``
    OFFSET probe angles -- an independent check through the stress evaluation), ``n_collocation`` and
    ``theta_offset``."""
    theta = _uniform_wall_angles(n_collocation, theta_offset)
    t0 = time.perf_counter()
    G, h = rep.boundary_system(theta)
    g_scaled = G * rep.coef_scale
    x_scaled, _, rank, _ = np.linalg.lstsq(g_scaled, h, rcond=None)
    wall_s = time.perf_counter() - t0
    x = rep.coef_scale * x_scaled
    coef_1, coef_2 = rep.vector_to_coefficients(x[:4 * rep.n_basis])
    residual = g_scaled @ x_scaled - h
    theta_probe = theta + np.pi / n_collocation
    t_x, t_y = rep.boundary_traction(theta_probe, coef_1, coef_2, part='total')
    return {'channel': 'ls', 'coef_1': coef_1, 'coef_2': coef_2, 'bc_constants': (float(x[-2]), float(x[-1])), 'coef_1_scaled': coef_1 / rep.coef_scale, 'coef_2_scaled': coef_2 / rep.coef_scale, 'wall_s': wall_s, 'n_params': 0, 'rank': int(rank), 'n_columns': int(4 * rep.n_basis + 2), 'wall_residual_max': float(np.max(np.abs(residual)) / rep.coef_scale), 'wall_traction_residual_max': float(np.max(np.abs(np.concatenate([t_x, t_y]))) / rep.load_scale), 'n_collocation': int(n_collocation), 'theta_offset': float(theta_offset)}
