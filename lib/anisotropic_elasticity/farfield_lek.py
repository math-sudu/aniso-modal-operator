from __future__ import annotations
import numpy as np
from isotropic_elasticity.farfield import _rows_for
from .material import AnisoMaterial
from .harmonic import AnisoExterior
__all__ = ['FarLekhnitskiiLS']

class FarLekhnitskiiLS:
    name = 'lekhnitskii_ls'

    def __init__(self, R: float, theta_nodes: np.ndarray, mat: AnisoMaterial, n_coeff: int=8):
        self.R = float(R)
        self.theta = np.asarray(theta_nodes, dtype=float)
        self.mat = mat
        self.n_coeff = int(n_coeff)
        self.p = np.zeros(4 * self.n_coeff)
        self.misfit = np.nan
        self._designs = {}

    def _ext_from_params(self, p: np.ndarray) -> AnisoExterior:
        ext = AnisoExterior(self.mat, self.R, self.n_coeff)
        ext.A[0, :] = p[0::4] + 1j * p[1::4]
        ext.A[1, :] = p[2::4] + 1j * p[3::4]
        return ext

    def _unit_fields(self, k: int, m: int, unit: complex, r, theta):
        ext = AnisoExterior(self.mat, self.R, self.n_coeff)
        ext.A[k, m - 1] = unit
        return ext.fields(r, theta)

    def _design(self, kind, beta=0.0):
        key = (kind, float(beta))
        if key not in self._designs:
            cols = []
            for m in range(1, self.n_coeff + 1):
                for k in (0, 1):
                    for unit in (1.0, 1j):
                        f = self._unit_fields(k, m, unit, self.R, self.theta)
                        cols.append(_rows_for(kind, f, beta))
            D = np.column_stack(cols)
            s = np.linalg.norm(D, axis=0)
            s[s == 0.0] = 1.0
            self._designs[key] = (D / s, s)
        return self._designs[key]

    def fit(self, kind, data, beta=0.0):
        Ds, s = self._design(kind, beta)
        data = np.asarray(data, dtype=float)
        q, *_ = np.linalg.lstsq(Ds, data, rcond=None)
        self.p = q / s
        resid = Ds @ q - data
        self.misfit = np.linalg.norm(resid) / max(np.linalg.norm(data), 1e-300)
        return self

    def coeffs(self):
        return (self.p[0::4] + 1j * self.p[1::4], self.p[2::4] + 1j * self.p[3::4])

    def fields(self, r, theta):
        return self._ext_from_params(self.p).fields(r, theta)

    def eval_kind(self, kind, beta=0.0):
        Ds, s = self._design(kind, beta)
        return Ds @ (s * self.p)
