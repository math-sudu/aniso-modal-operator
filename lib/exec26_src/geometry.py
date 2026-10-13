from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
__all__ = ['SectionMap', 'circle_map', 'ellipse_map', 'horseshoe_map']

@dataclass
class SectionMap:
    """Container for one family Laurent map (R, c); c[0] = c_0."""
    R: float
    c: np.ndarray

    def __post_init__(self):
        self.c = np.atleast_1d(np.asarray(self.c, dtype=complex))
        if self.c.size == 0:
            self.c = np.zeros(1, dtype=complex)
        self.R = float(self.R)

    @property
    def n_map(self) -> int:
        return len(self.c) - 1

    def omega(self, zeta):
        zeta = np.asarray(zeta, dtype=complex)
        w = zeta.astype(complex).copy()
        zinv = np.ones_like(zeta)
        for j, cj in enumerate(self.c):
            if j > 0:
                zinv = zinv / zeta
            else:
                zinv = np.ones_like(zeta)
            if cj != 0.0:
                w = w + cj * zinv
        return self.R * w

    def domega(self, zeta):
        zeta = np.asarray(zeta, dtype=complex)
        w = np.ones_like(zeta)
        for j, cj in enumerate(self.c):
            if j >= 1 and cj != 0.0:
                w = w - j * cj * zeta ** (-(j + 1))
        return self.R * w

    def boundary(self, theta, rho: float=1.0):
        """Physical boundary points z = omega(rho e^{i theta})."""
        return self.omega(rho * np.exp(1j * np.asarray(theta, dtype=float)))

    def wall_normal(self, theta, rho: float=1.0):
        """Outward unit normal n = zeta omega'/|zeta omega'| (family
        convention: image of the radial direction) + parameter jacobian
        |d z / d theta| = rho |omega'|."""
        zeta = rho * np.exp(1j * np.asarray(theta, dtype=float))
        dw = self.domega(zeta)
        n = zeta * dw
        jac = np.abs(n)
        return (n / np.abs(n), jac)

    def scaled(self, rho: float) -> 'SectionMap':
        """Gamma_rho tier as a family map (R', c'_j) -- see docstring."""
        j = np.arange(len(self.c))
        return SectionMap(self.R * rho, self.c * rho ** (-(j + 1.0)))

    def delta_bound(self) -> float:
        """D4 convexity bound R (1 - sum j |c_j|) (sufficient, not
        necessary; reported as-is, may be negative)."""
        j = np.arange(len(self.c), dtype=float)
        return self.R * (1.0 - float(np.sum(j * np.abs(self.c))))

    @staticmethod
    def _betas(mu: complex):
        return ((1.0 - 1j * mu) / 2.0, (1.0 + 1j * mu) / 2.0)

    def zk_boundary(self, mu: complex, theta, rho: float=1.0):
        """z_k = betam z + betap conj(z) on the rho-tier boundary
        (actual conjugate; exact at any rho)."""
        z = self.boundary(theta, rho)
        bm, bp = self._betas(mu)
        return bm * z + bp * np.conj(z)

    def zk_series(self, mu: complex) -> dict[int, complex]:
        """FINITE Laurent coefficients of z_k(sigma) on |sigma| = 1:
        power +1: R(betam + betap conj(c1));  power -1: R(betam c1 + betap);
        power +j (j>=2): R betap conj(c_j);   power -j: R betam c_j;
        power 0: R(betam c0 + betap conj(c0))."""
        bm, bp = self._betas(mu)
        c = self.c
        out: dict[int, complex] = {}
        out[1] = self.R * (bm + (bp * np.conj(c[1]) if len(c) > 1 else 0.0))
        out[-1] = self.R * ((bm * c[1] if len(c) > 1 else 0.0) + bp)
        if len(c) > 0 and c[0] != 0.0:
            out[0] = self.R * (bm * c[0] + bp * np.conj(c[0]))
        for j in range(2, len(c)):
            if c[j] != 0.0:
                out[j] = self.R * bp * np.conj(c[j])
                out[-j] = self.R * bm * c[j]
        return out

    def twoterm_gamma_delta(self, mu: complex):
        """(gam_k, del_k) of the two-term tier (ellipse truncation of
        the boundary correspondence; exact for n_map <= 1)."""
        bm, bp = self._betas(mu)
        c1 = self.c[1] if len(self.c) > 1 else 0.0
        gam = self.R * (bm + bp * np.conj(c1))
        dlt = self.R * (bm * c1 + bp)
        return (gam, dlt)

    def qhat_ell(self, mu: complex) -> complex:
        gam, dlt = self.twoterm_gamma_delta(mu)
        return dlt / gam

def circle_map(R: float=1.0) -> SectionMap:
    return SectionMap(R, [0.0])

def ellipse_map(c1: float, R: float=1.0) -> SectionMap:
    return SectionMap(R, [0.0, c1])

def horseshoe_map(t: float, R: float, c_hs: np.ndarray) -> SectionMap:
    """Literature horseshoe ray member c(t) = t * c_hs."""
    return SectionMap(R, t * np.asarray(c_hs, dtype=complex))
