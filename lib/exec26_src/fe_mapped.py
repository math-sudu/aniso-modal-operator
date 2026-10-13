from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from exec20_src.fe_aniso import AnisoAnnulusFE
from exec20_src.material import AnisoMaterial
from proto_src.fe_annulus import Mesh
from .geometry import SectionMap
from .sf_general import assemble_modal_dtn
from .sf_twoterm import nodal_sf_density, _basis_profiles, _COMPS
__all__ = ['MappedMesh', 'mapped_released_load', 'theta_ring_mass', 'nodal_dtn_general', 'condensed_mapped_system', 'solve_condensed', 'strip_translation']

class MappedMesh(Mesh):
    """proto structured annulus with coordinates pushed through the
    section map: nodes z = omega(rho_i e^{i theta_j})."""

    def __init__(self, smap: SectionMap, rho_out: float, n_r: int, n_t: int):
        super().__init__(1.0, float(rho_out), n_r, n_t)
        self.smap = smap
        self.rho_out = float(rho_out)
        zeta = self.r_node * np.exp(1j * self.th_node)
        z = smap.omega(zeta)
        self.xy = np.column_stack([np.real(z), np.imag(z)])

def mapped_released_load(mesh: MappedMesh, sxx_inf: float, syy_inf: float, sxy_inf: float, n_gauss: int=3) -> np.ndarray:
    """Consistent nodal loads of the RELEASED far-field traction on the
mapped wall (inner ring).

The perturbation field's wall BOUNDARY VALUES are
-sigma_inf . n (n = hole-outward normal, the wall_normal
direction), but the LOAD applied to the near-zone body acts across
the body's outward wall normal -n, so the load density is
t = sigma_pert . (-n) = +sigma_inf . n, integrated in the map
parameter: f_i = int N_i(th) t(th) |dz/dth| dth.
Self-equilibration (closed wall, uniform state) is asserted."""
    smap = mesh.smap
    n_t = mesh.n_t
    th_nodes = mesh.theta_nodes
    h = 2.0 * np.pi / n_t
    gp, gw = np.polynomial.legendre.leggauss(n_gauss)
    f = np.zeros(mesh.n_dof)
    nodes = mesh.ring_nodes('inner')
    for j in range(n_t):
        jp = (j + 1) % n_t
        th0 = th_nodes[j]
        for p, w in zip(gp, gw):
            th = th0 + 0.5 * (p + 1.0) * h
            n_hat, jac = smap.wall_normal(np.array([th]))
            nx, ny = (float(np.real(n_hat[0])), float(np.imag(n_hat[0])))
            tx = (sxx_inf * nx + sxy_inf * ny) * float(jac[0])
            ty = (sxy_inf * nx + syy_inf * ny) * float(jac[0])
            N0 = 0.5 * (1.0 - p)
            N1 = 0.5 * (1.0 + p)
            c = w * 0.5 * h
            f[2 * nodes[j]] += c * N0 * tx
            f[2 * nodes[j] + 1] += c * N0 * ty
            f[2 * nodes[jp]] += c * N1 * tx
            f[2 * nodes[jp] + 1] += c * N1 * ty
    res = np.array([np.sum(f[0::2]), np.sum(f[1::2])])
    scale = np.max(np.abs(f)) if np.max(np.abs(f)) > 0 else 1.0
    if np.max(np.abs(res)) > 1e-09 * scale * n_t:
        raise ValueError(f'released load not self-equilibrated: {res}')
    return f

def theta_ring_mass(n_t: int) -> np.ndarray:
    """Periodic P1 mass in the map parameter theta (n_t, n_t)."""
    h = 2.0 * np.pi / n_t
    M = np.zeros((n_t, n_t))
    idx = np.arange(n_t)
    M[idx, idx] = 2.0 * h / 3.0
    M[idx, (idx + 1) % n_t] = h / 6.0
    M[idx, (idx - 1) % n_t] = h / 6.0
    return M

def nodal_dtn_general(mat: AnisoMaterial, smap_tier: SectionMap, theta_nodes: np.ndarray, n_h: int, n_modes: int, n_grid: int=512) -> np.ndarray:
    """Nodal traction-density DtN on a general tier: harmonic synthesis
of the spectral modal matrix (outer-product construction with the assembled S_mod in place of the closed-form blocks)."""
    out = assemble_modal_dtn(mat, smap_tier, n_h, n_modes, n_grid=n_grid)
    S_mod = out['S'][2:, :]
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

def condensed_mapped_system(fe: AnisoAnnulusFE, mat: AnisoMaterial, smap: SectionMap, rho_out: float, n_h: int | None=None, n_modes: int | None=None):
    """K_sys = K_FE - B_theta * S_dens on the outer ring.

Two-term sections use the closed-form material-only density
operator; general sections the spectral assembly on the scaled
tier map.  n_h defaults to the mesh Nyquist
n_t // 2 - 1 (the condensed operator is not artificially truncated; N_H_MAIN is the MODAL
structure-spectrum tier, not the condensed tier)."""
    mesh = fe.mesh
    th = mesh.theta_nodes
    n_t = mesh.n_t
    if n_h is None:
        n_h = n_t // 2 - 1
    if smap.n_map <= 1:
        S_dens = nodal_sf_density(mat, th, n_h=n_h)
    else:
        if n_modes is None:
            raise ValueError('general tier needs n_modes')
        tier = smap.scaled(rho_out)
        S_dens = nodal_dtn_general(mat, tier, th, n_h, n_modes)
    M = theta_ring_mass(n_t)
    Bm = np.zeros((2 * n_t, 2 * n_t))
    Bm[0::2, 0::2] = M
    Bm[1::2, 1::2] = M
    A_loc = Bm @ S_dens
    dofs = mesh.ring_dofs('outer')
    rows = np.repeat(dofs, 2 * n_t)
    cols = np.tile(dofs, 2 * n_t)
    A_glob = sp.coo_matrix((A_loc.ravel(), (rows, cols)), shape=(mesh.n_dof, mesh.n_dof)).tocsr()
    return ((fe.K - A_glob).tocsr(), S_dens)

def solve_condensed(fe: AnisoAnnulusFE, K_sys, f: np.ndarray) -> np.ndarray:
    """Condensed solve with the translation nullspace pinned.

    The decaying far class carries zero response to rigid translations
    (harmonic-0 bookkeeping), so K_sys = K - B S keeps the two
    translation vectors in its nullspace (rotation IS locked: the
    rotation trace lies in the decaying trace space with nonzero
    response).  Two Lagrange constraints pin the zero-mean-translation
    representative; compare against exact fields through
    :func:`strip_translation` (quotient-space comparison)."""
    import scipy.sparse.linalg as spla
    C = sp.csc_matrix(fe.rigid[:, :2])
    Ka = sp.bmat([[K_sys, C], [C.T, None]], format='csc')
    rhs = np.concatenate([f, np.zeros(2)])
    return spla.splu(Ka).solve(rhs)[:fe.mesh.n_dof]

def strip_translation(u: np.ndarray) -> np.ndarray:
    """Remove the mean translation (quotient-space representative)."""
    v = u.copy()
    v[0::2] -= np.mean(v[0::2])
    v[1::2] -= np.mean(v[1::2])
    return v
