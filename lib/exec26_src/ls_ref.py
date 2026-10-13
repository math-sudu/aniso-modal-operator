from __future__ import annotations
import numpy as np
from exec20_src.material import AnisoMaterial
from .geometry import SectionMap
__all__ = ['LSReference']

def _ti_bridge(mat: AnisoMaterial):
    import lekhnitskii_map as lm
    return lm.TIMaterial(E=mat.Eh, E_prime=mat.Ev, nu=mat.nu_hh, nu_prime=mat.nu_vh, G_prime=mat.Gvh, dip=mat.dip)

class LSReference:
    """Solved completed-basis reference for one (material, section,
    tension-positive uniform load) case."""

    def __init__(self, mat: AnisoMaterial, smap: SectionMap, sxx_inf: float, syy_inf: float, sxy_inf: float, n_order: int=16, n_collocation: int=400):
        from lekhnitskii_general import LekhnitskiiGeneral, solve_ls
        self.mat = mat
        self.smap = smap
        self.load = (sxx_inf, syy_inf, sxy_inf)
        self.rep = LekhnitskiiGeneral.from_load(_ti_bridge(mat), smap.R, np.asarray(smap.c, dtype=complex), -sxx_inf, -syy_inf, -sxy_inf, n_order=n_order)
        sol = solve_ls(self.rep, n_collocation=n_collocation)
        self.coef_1, self.coef_2 = (sol['coef_1'], sol['coef_2'])
        self.wall_residual = sol['wall_traction_residual_max']
        self.rank = sol['rank']

    def displacements(self, x, y):
        """Tension-positive perturbation displacements at (x, y)."""
        ux, uy = self.rep.displacements_cartesian(x, y, self.coef_1, self.coef_2)
        return (-np.asarray(ux), -np.asarray(uy))

    def stresses_perturbation(self, x, y):
        """Tension-positive PERTURBATION stresses at (x, y)."""
        sxx, syy, sxy = self.rep.stresses_cartesian(x, y, self.coef_1, self.coef_2, part='total')
        return (-np.asarray(sxx) - self.load[0], -np.asarray(syy) - self.load[1], -np.asarray(sxy) - self.load[2])
