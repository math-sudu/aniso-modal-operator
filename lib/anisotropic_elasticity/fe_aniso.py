from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from isotropic_elasticity.fe_annulus import AnnulusFE, Mesh, _bmat_all, _G2, ring_mass
from .material import AnisoMaterial
__all__ = ['AnisoAnnulusFE', 'assemble_stiffness_D', 'Mesh']

def assemble_stiffness_D(mesh: Mesh, D: np.ndarray):
    """proto assemble_stiffness with an explicit D matrix."""
    ne = mesh.elems.shape[0]
    Ke = np.zeros((ne, 8, 8))
    for xi in _G2:
        for eta in _G2:
            B, detJ = _bmat_all(mesh, xi, eta)
            Ke += np.einsum('eka,kl,elb->eab', B, D, B) * detJ[:, None, None]
    edof = np.empty((ne, 8), dtype=np.int64)
    edof[:, 0::2] = 2 * mesh.elems
    edof[:, 1::2] = 2 * mesh.elems + 1
    rows = np.repeat(edof, 8, axis=1).ravel()
    cols = np.tile(edof, (1, 8)).ravel()
    return sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(mesh.n_dof, mesh.n_dof)).tocsr()

class AnisoAnnulusFE(AnnulusFE):

    def __init__(self, mesh: Mesh, mat: AnisoMaterial):
        self.mesh = mesh
        self.mat = mat
        self.E, self.nu = (np.nan, np.nan)
        self.K = assemble_stiffness_D(mesh, mat.D())
        self._fact = {}
        self._m_out = ring_mass(mesh, 'outer')
        self._m_out_lu = None
        tx = np.zeros(mesh.n_dof)
        tx[0::2] = 1.0
        ty = np.zeros(mesh.n_dof)
        ty[1::2] = 1.0
        rot = np.zeros(mesh.n_dof)
        rot[0::2] = -mesh.xy[:, 1]
        rot[1::2] = mesh.xy[:, 0]
        self.rigid = np.column_stack([v / np.linalg.norm(v) for v in (tx, ty, rot)])

    def stress_at(self, u, points):
        """Parent stress recovery with the anisotropic D."""
        from isotropic_elasticity.kirsch_ref import stress_cart_to_polar
        D = self.mat.D()
        edof = np.empty((self.mesh.elems.shape[0], 8), dtype=np.int64)
        edof[:, 0::2] = 2 * self.mesh.elems
        edof[:, 1::2] = 2 * self.mesh.elems + 1
        ue = u[edof]
        xy = self.mesh.xy[self.mesh.elems]
        rs, ths, srr, stt, srt = ([], [], [], [], [])
        for xi, eta in points:
            B, _ = _bmat_all(self.mesh, xi, eta)
            sig = np.einsum('kl,elb,eb->ek', D, B, ue)
            N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
            xp = np.einsum('a,eai->ei', N, xy)
            r = np.hypot(xp[:, 0], xp[:, 1])
            th = np.arctan2(xp[:, 1], xp[:, 0])
            s_rr, s_tt, s_rt = stress_cart_to_polar(sig[:, 0], sig[:, 1], sig[:, 2], th)
            rs.append(r)
            ths.append(th)
            srr.append(s_rr)
            stt.append(s_tt)
            srt.append(s_rt)
        return (np.concatenate(rs), np.concatenate(ths), np.concatenate(srr), np.concatenate(stt), np.concatenate(srt))
