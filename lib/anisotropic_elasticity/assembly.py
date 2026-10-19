from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from .material import AnisoMaterial
from .harmonic import AnisoExterior, solve_dirichlet_circle
__all__ = ['harmonic_columns', 'nodal_sf', 'modal_response_matrix', 'mode_traction_closed_form', 'condensed_aniso_system', 'modal_impedance_near', 'modal_index']

def _basis_profiles(n: int, comp: str, theta: np.ndarray):
    c, s = (np.cos(n * theta), np.sin(n * theta))
    z = np.zeros_like(theta)
    if comp == 'Uc':
        return (c, z)
    if comp == 'Us':
        return (s, z)
    if comp == 'Vc':
        return (z, c)
    if comp == 'Vs':
        return (z, s)
    raise ValueError(comp)
_COMPS = ('Uc', 'Us', 'Vc', 'Vs')

def modal_index(n_h: int):
    """Ordered basis [(n, comp)] for n = 1..n_h, comps Uc,Us,Vc,Vs."""
    return [(n, comp) for n in range(1, n_h + 1) for comp in _COMPS]

def harmonic_columns(mat: AnisoMaterial, R: float, n_h: int, theta: np.ndarray):
    """For each basis entry: solved exterior + exact traction-value
    profiles (t_rr, t_rt) on the given theta grid.  Returns
    (index, U_cols, T_cols) with U/T stacked [comp1; comp2] rows
    (2*len(theta), 4*n_h)."""
    idx = modal_index(n_h)
    n_g = len(theta)
    U = np.zeros((2 * n_g, len(idx)))
    T = np.zeros((2 * n_g, len(idx)))
    for j, (n, comp) in enumerate(idx):
        ux, uy = _basis_profiles(n, comp, theta)
        ext = solve_dirichlet_circle(mat, R, ux, uy, n_modes=n)
        f = ext.boundary_fields(theta)
        U[:n_g, j], U[n_g:, j] = (ux, uy)
        T[:n_g, j] = f['s_rr']
        T[n_g:, j] = f['s_rt']
    return (idx, U, T)

def nodal_sf(mat: AnisoMaterial, R: float, theta_nodes: np.ndarray, n_h: int | None=None) -> np.ndarray:
    """Nodal far DtN: interleaved Cartesian ring displacements ->
    interleaved Cartesian nodal traction values (2 n_t, 2 n_t)."""
    th = np.asarray(theta_nodes, dtype=float)
    n_t = len(th)
    if n_h is None:
        n_h = n_t // 2 - 1
    S = np.zeros((2 * n_t, 2 * n_t))
    for n in range(1, n_h + 1):
        w = 2.0 / n_t
        for comp in _COMPS:
            ux, uy = _basis_profiles(n, comp, th)
            ext = solve_dirichlet_circle(mat, R, ux, uy, n_modes=n)
            f = ext.boundary_fields(th)
            c, s = (np.cos(th), np.sin(th))
            t_x = f['s_rr'] * c - f['s_rt'] * s
            t_y = f['s_rr'] * s + f['s_rt'] * c
            col = np.empty(2 * n_t)
            col[0::2], col[1::2] = (t_x, t_y)
            row = np.empty(2 * n_t)
            row[0::2], row[1::2] = (ux, uy)
            S += np.outer(col, w * row)
    return S

def modal_response_matrix(mat: AnisoMaterial, R: float, n_h: int, n_grid: int=512):
    """Modal (Fourier-space) response matrix T_mod[(n', c'), (n, c)]:
    harmonic content of the traction response to each displacement
    basis entry.  Content rows include n' = 0..n_h (row block 0 is the
    mean/translation-force row, expected zero-force class)."""
    th = 2.0 * np.pi * np.arange(n_grid) / n_grid
    idx, U, T = harmonic_columns(mat, R, n_h, th)
    rows = [(0, 'Uc'), (0, 'Vc')]
    rows += [(n, comp) for n in range(1, n_h + 1) for comp in _COMPS]
    M = np.zeros((len(rows), len(idx)))
    c_all = {n: np.cos(n * th) for n in range(n_h + 1)}
    s_all = {n: np.sin(n * th) for n in range(n_h + 1)}
    for j in range(len(idx)):
        t_rr, t_rt = (T[:n_grid, j], T[n_grid:, j])
        c, s = (np.cos(th), np.sin(th))
        t_x = t_rr * c - t_rt * s
        t_y = t_rr * s + t_rt * c
        for i, (n, comp) in enumerate(rows):
            w = (1.0 if n == 0 else 2.0) / n_grid
            if comp == 'Uc':
                M[i, j] = w * (t_x @ c_all[n])
            elif comp == 'Us':
                M[i, j] = w * (t_x @ s_all[n])
            elif comp == 'Vc':
                M[i, j] = w * (t_y @ c_all[n])
            else:
                M[i, j] = w * (t_y @ s_all[n])
    return (rows, idx, M)

def mode_traction_closed_form(mat: AnisoMaterial, R: float, k: int, m: int, unit: complex):
    """EXACT on-circle Cartesian traction of a single branch mode
phi_k = unit * zeta_k^{-m}: SINGLE harmonic, closed form.

Traction components are arc-length derivatives
of the potential-level resultants,

    t_x = 2Re[sum_k mu_k dphi_k/ds],   t_y = -2Re[sum_k dphi_k/ds],

and on the mapping circle dphi_k/ds = (-i m unit / R) e^{-i m th} --
the 1/omega_k' factor cancels EXACTLY, the geometric |qhat| ladder
does not reach the on-circle traction map.  Returns the one-sided
complex coefficients (c_tx, c_ty) of e^{-i m th}."""
    mu = mat.mus[k]
    base = -1j * m * unit / R
    return (mu * base, -base)

def condensed_aniso_system(fe, mat: AnisoMaterial):
    """K_sys = K_FE - B_out * S_cart (proto condensed_system pattern,
    Cartesian throughout).  Returns (K_sys csr, S_cart)."""
    mesh = fe.mesh
    th = mesh.theta_nodes
    n_t = mesh.n_t
    S_cart = nodal_sf(mat, mesh.R, th)
    M = fe._m_out
    Bm = np.zeros((2 * n_t, 2 * n_t))
    Bm[0::2, 0::2] = M
    Bm[1::2, 1::2] = M
    A_loc = Bm @ S_cart
    dofs = mesh.ring_dofs('outer')
    rows = np.repeat(dofs, 2 * n_t)
    cols = np.tile(dofs, 2 * n_t)
    A_glob = sp.coo_matrix((A_loc.ravel(), (rows, cols)), shape=(mesh.n_dof, mesh.n_dof)).tocsr()
    return ((fe.K - A_glob).tocsr(), S_cart)

def modal_impedance_near(fe, n_h: int) -> np.ndarray:
    """Near-zone modal impedance probe (single-harmonic Dirichlet solves
    on the outer ring, inner boundary traction-free): modal content of
    the recovered outer traction values.  Same (n, comp) layout as
    modal_response_matrix columns; rows n' = 1..n_h."""
    mesh = fe.mesh
    th = mesh.theta_nodes
    n_t = mesh.n_t
    f0 = np.zeros(mesh.n_dof)
    idx = modal_index(n_h)
    M = np.zeros((4 * n_h, len(idx)))
    c_all = {n: np.cos(n * th) for n in range(n_h + 1)}
    s_all = {n: np.sin(n * th) for n in range(n_h + 1)}
    for j, (n, comp) in enumerate(idx):
        ux, uy = _basis_profiles(n, comp, th)
        u = fe.solve_dirichlet({'outer': np.column_stack([ux, uy])}, f0)
        t = fe.outer_traction_from_reactions(u, f0)
        for i, (npr, cpr) in enumerate(idx):
            w = 2.0 / n_t
            if cpr == 'Uc':
                M[i, j] = w * (t[:, 0] @ c_all[npr])
            elif cpr == 'Us':
                M[i, j] = w * (t[:, 0] @ s_all[npr])
            elif cpr == 'Vc':
                M[i, j] = w * (t[:, 1] @ c_all[npr])
            else:
                M[i, j] = w * (t[:, 1] @ s_all[npr])
    return M
