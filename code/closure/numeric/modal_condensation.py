from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from exec26_src.fe_mapped import AnisoAnnulusFE, _basis_profiles, _COMPS, theta_ring_mass
__all__ = ['nodal_from_modal', 'condense_with', 'condense_work']

def nodal_from_modal(S_mod: np.ndarray, theta_nodes: np.ndarray, n_h: int) -> np.ndarray:
    """Nodal traction-density DtN synthesized from an arbitrary modal matrix.

    The synthesis half of ``fe_mapped.nodal_dtn_general`` with the modal
    matrix supplied rather than assembled, which is what lets the same
    condensation carry a composed or a band-limited operator.
    """
    th = np.asarray(theta_nodes, dtype=float)
    n_t = len(th)
    idx = [(n, c) for n in range(1, n_h + 1) for c in _COMPS]
    S = np.zeros((2 * n_t, 2 * n_t))
    w = 2.0 / n_t
    for jj, (n, comp) in enumerate(idx):
        ux, uy = _basis_profiles(n, comp, th)
        row = np.empty(2 * n_t)
        row[0::2], row[1::2] = (ux, uy)
        txp = np.zeros(n_t)
        typ = np.zeros(n_t)
        for npr in range(1, n_h + 1):
            i0 = 4 * (npr - 1)
            txp += S_mod[i0 + 0, jj] * np.cos(npr * th) + S_mod[i0 + 1, jj] * np.sin(npr * th)
            typ += S_mod[i0 + 2, jj] * np.cos(npr * th) + S_mod[i0 + 3, jj] * np.sin(npr * th)
        S_col = np.empty(2 * n_t)
        S_col[0::2], S_col[1::2] = (txp, typ)
        S += np.outer(S_col, w * row)
    return S

def condense_with(fe: AnisoAnnulusFE, S_dens: np.ndarray):
    """K_sys = K_FE - B_theta S_dens on the outer ring, the condensation of
    ``fe_mapped.condensed_mapped_system`` with the density supplied."""
    mesh = fe.mesh
    n_t = mesh.n_t
    M = theta_ring_mass(n_t)
    Bm = np.zeros((2 * n_t, 2 * n_t))
    Bm[0::2, 0::2] = M
    Bm[1::2, 1::2] = M
    A_loc = Bm @ S_dens
    return condense_work(fe, A_loc)

def condense_work(fe: AnisoAnnulusFE, work: np.ndarray):
    """Subtract the exterior boundary-work matrix on the outer-ring DOFs."""
    mesh = fe.mesh
    n_t = mesh.n_t
    dofs = mesh.ring_dofs('outer')
    rows = np.repeat(dofs, 2 * n_t)
    cols = np.tile(dofs, 2 * n_t)
    A_glob = sp.coo_matrix((work.ravel(), (rows, cols)), shape=(mesh.n_dof, mesh.n_dof)).tocsr()
    return (fe.K - A_glob).tocsr()
