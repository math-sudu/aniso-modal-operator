from __future__ import annotations
import numpy as np
from anisotropic_elasticity.material import AnisoMaterial
from .geometry import SectionMap, horseshoe_map
from .sf_general import assemble_modal_dtn
from .sf_twoterm import modal_blocks_density
__all__ = ['perturbation_kernel', 'perturbation_ladder']

def _S_of_t(mat: AnisoMaterial, t: float, R: float, c_hs: np.ndarray, rho_out: float, n_h: int, n_modes: int, n_grid: int=512) -> np.ndarray:
    smap = horseshoe_map(t, R, c_hs).scaled(rho_out) if rho_out != 1.0 else horseshoe_map(t, R, c_hs)
    out = assemble_modal_dtn(mat, smap, n_h, n_modes, n_grid=n_grid)
    return out['S'][2:, :]
T_FIT = (-0.2, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.2)

def perturbation_kernels(mat: AnisoMaterial, R: float, c_hs: np.ndarray, rho_out: float, n_h: int, n_modes: int, t_fit=T_FIT, n_grid: int=512):
    """First- and second-order kernels (S1, S2) by a multi-point
    entrywise least-squares fit S(t) - S0 = t S1 + t^2 S2 + t^3 S3
    over the small-t stencil (noise-averaged; the cubic column absorbs
    the next-order bias).  The scaling locks rel1 ~ t^2, rel2 ~ t^3
    of the ladder are the machine evidence that the extracted kernels
    are the true Taylor coefficients."""
    S0 = modal_blocks_density(mat, n_h)
    ts = np.asarray(t_fit, dtype=float)
    Vm = np.column_stack([ts, ts ** 2, ts ** 3])
    rhs = np.stack([(_S_of_t(mat, float(t), R, c_hs, rho_out, n_h, n_modes, n_grid) - S0).ravel() for t in ts])
    coef, *_ = np.linalg.lstsq(Vm, rhs, rcond=None)
    S1 = coef[0].reshape(S0.shape)
    S2 = coef[1].reshape(S0.shape)
    return (S0, S1, S2)

def perturbation_ladder(mat: AnisoMaterial, R: float, c_hs: np.ndarray, rho_out: float, t_ladder, n_h: int, n_modes: int, n_grid: int=512):
    """J2 readout rows: for each t the relative remainders of the
    zeroth / first / second-order principal parts against the spectral
    assembly,

        rel0 = ||S(t) - S0|| / ||S(t)||,
        rel1 = ||S(t) - S0 - t S1|| / ||S(t)||,
        rel2 = ||S(t) - S0 - t S1 - t^2 S2|| / ||S(t)||   (Frobenius)."""
    S0, S1, S2 = perturbation_kernels(mat, R, c_hs, rho_out, n_h, n_modes, n_grid=n_grid)
    rows = []
    for t in t_ladder:
        t = float(t)
        St = _S_of_t(mat, t, R, c_hs, rho_out, n_h, n_modes, n_grid)
        nrm = np.linalg.norm(St)
        rows.append({'t': t, 'rel0': np.linalg.norm(St - S0) / nrm, 'rel1': np.linalg.norm(St - S0 - t * S1) / nrm, 'rel2': np.linalg.norm(St - S0 - t * S1 - t * t * S2) / nrm})
    return {'S0': S0, 'S1': S1, 'S2': S2, 'rows': rows}
