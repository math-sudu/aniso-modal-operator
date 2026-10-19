from __future__ import annotations
import numpy as np
from anisotropic_elasticity.material import AnisoMaterial
from .fe_mapped import MappedMesh, mapped_released_load
from .geometry import SectionMap
from .sf_general import GeneralExterior, _real_design, _spectral_dtheta
__all__ = ['schwarz_dn']

def schwarz_dn(fe, mat: AnisoMaterial, smap: SectionMap, rho_out: float, sxx_inf: float, syy_inf: float, sxy_inf: float, n_modes: int, relax: float=0.5, aitken: bool=True, max_sweeps: int=200, tol: float=1e-10):
    """DN alternating solve; returns dict(u, sweeps, history, u_trace)."""
    mesh: MappedMesh = fe.mesh
    th = mesh.theta_nodes
    n_t = mesh.n_t
    tier = smap.scaled(rho_out)
    ext = GeneralExterior(mat, tier, n_modes)
    n_fit = max(512, 4 * 4 * ext.n_cols)
    th_fit = 2.0 * np.pi * np.arange(n_fit) / n_fit
    tr = ext.wall_traces(th_fit)
    TX = _spectral_dtheta(_real_design(tr['X']))
    TY = _spectral_dtheta(_real_design(tr['Y']))
    A = np.concatenate([TX, TY], axis=0)
    U_svd, sv, Vt = np.linalg.svd(A, full_matrices=False)
    tol_sv = sv[0] * max(A.shape) * np.finfo(float).eps
    inv_s = np.where(sv > tol_sv, 1.0 / sv, 0.0)
    tr_n = ext.wall_traces(th)
    Ux = _real_design(tr_n['Ux'])
    Uy = _real_design(tr_n['Uy'])
    _, jac = tier.wall_normal(th)

    def _fft_interp(f: np.ndarray, n_out: int) -> np.ndarray:
        F = np.fft.rfft(f)
        Fo = np.zeros(n_out // 2 + 1, dtype=complex)
        Fo[:len(F)] = F
        return np.fft.irfft(Fo, n=n_out) * (n_out / len(f))
    f_rel = mapped_released_load(mesh, sxx_inf, syy_inf, sxy_inf)
    u_tr = np.zeros((n_t, 2))
    u_prev_update = None
    rho = relax
    hist = []
    u = np.zeros(mesh.n_dof)
    for it in range(1, max_sweeps + 1):
        u = fe.solve_dirichlet({'outer': u_tr}, f_rel)
        t_vals = fe.outer_traction_from_reactions(u, f_rel)
        t_dens = t_vals * jac[:, None]
        b = np.concatenate([_fft_interp(t_dens[:, 0], n_fit), _fft_interp(t_dens[:, 1], n_fit)])
        coef = Vt.T @ (inv_s * (U_svd.T @ b))
        u_far = np.column_stack([Ux @ coef, Uy @ coef])
        upd = u_far - u_tr
        scale = max(np.max(np.abs(u_far)), 1e-300)
        hist.append(np.max(np.abs(upd)) / scale)
        if aitken and u_prev_update is not None:
            d = upd - u_prev_update
            denom = float(np.sum(d * d))
            if denom > 0.0:
                rho = -float(np.sum(u_prev_update * d)) / denom * rho
                rho = min(max(rho, 0.05), 1.5)
        u_tr = u_tr + rho * upd
        u_prev_update = upd
        if hist[-1] < tol:
            break
    return {'u': u, 'u_trace': u_tr, 'sweeps': it, 'history': hist}
