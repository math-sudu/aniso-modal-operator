from __future__ import annotations
import numpy as np
from .material import AnisoMaterial
from .assembly import modal_response_matrix, modal_impedance_near, modal_index
__all__ = ['modal_sf_square', 'dn_spr', 'rr_spr', 'beta_ladder', 'predicted_sweeps']

def modal_sf_square(mat: AnisoMaterial, R: float, n_h: int, n_grid: int=512) -> np.ndarray:
    """Square far modal response on the (n=1..n_h) x (Uc,Us,Vc,Vs) basis
    (Cartesian encoding; exactly block-diagonal per harmonic)."""
    rows, idx, M = modal_response_matrix(mat, R, n_h, n_grid)
    return M[2:, :]

def dn_spr(Sf: np.ndarray, Sn: np.ndarray, rho: float) -> float:
    T = (1.0 - rho) * np.eye(Sf.shape[0]) + rho * np.linalg.solve(Sf, Sn)
    return float(np.max(np.abs(np.linalg.eigvals(T))))

def rr_spr(Sf: np.ndarray, Sn: np.ndarray, beta: float, rho: float=1.0) -> float:
    I = np.eye(Sf.shape[0])
    M = (Sf + beta * I) @ np.linalg.inv(Sf - beta * I) @ (Sn - beta * I) @ np.linalg.inv(Sn + beta * I)
    if rho != 1.0:
        M = (1.0 - rho) * I + rho * M
    return float(np.max(np.abs(np.linalg.eigvals(M))))

def beta_ladder(mat: AnisoMaterial, R: float) -> dict:
    """Frozen three-tier Robin parameter ladder (documented equivalent-
    shear choice G_eq = 1/beta66 of the ROTATED reduced compliance)."""
    G_eq = 1.0 / mat.beta['66']
    b0 = 2.0 * G_eq / R
    return {'beta_mid': b0, 'beta_lo': 0.5 * b0, 'beta_hi': 2.0 * b0, 'G_eq': G_eq}

def predicted_sweeps(spr: float, tol: float=1e-06) -> float:
    """Geometric contraction to the displacement tolerance (ch04:919
    reading; the measured count carries one extra confirmation sweep)."""
    if spr >= 1.0:
        return np.inf
    return np.ceil(np.log(tol) / np.log(spr))
