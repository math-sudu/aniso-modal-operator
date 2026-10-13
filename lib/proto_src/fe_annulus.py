import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .kirsch_ref import vec_cart_to_polar, stress_cart_to_polar
_G2 = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
_G3_P = (-np.sqrt(0.6), 0.0, np.sqrt(0.6))
_G3_W = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)

def dmat_plane_strain(E, nu):
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array([[1.0 - nu, nu, 0.0], [nu, 1.0 - nu, 0.0], [0.0, 0.0, 0.5 * (1.0 - 2.0 * nu)]])

class Mesh:
    """Structured annulus mesh; node id = i_r * n_t + j_theta."""

    def __init__(self, a, R, n_r, n_t):
        self.a, self.R, self.n_r, self.n_t = (float(a), float(R), int(n_r), int(n_t))
        ratio = (self.R / self.a) ** (1.0 / self.n_r)
        self.radii = self.a * ratio ** np.arange(self.n_r + 1)
        self.radii[-1] = self.R
        self.theta_nodes = 2.0 * np.pi * np.arange(self.n_t) / self.n_t
        rr = np.repeat(self.radii, self.n_t)
        tt = np.tile(self.theta_nodes, self.n_r + 1)
        self.r_node, self.th_node = (rr, tt)
        self.xy = np.column_stack([rr * np.cos(tt), rr * np.sin(tt)])
        self.n_nodes = (self.n_r + 1) * self.n_t
        self.n_dof = 2 * self.n_nodes
        i = np.repeat(np.arange(self.n_r), self.n_t)
        j = np.tile(np.arange(self.n_t), self.n_r)
        jp = (j + 1) % self.n_t
        self.elems = np.column_stack([i * self.n_t + j, (i + 1) * self.n_t + j, (i + 1) * self.n_t + jp, i * self.n_t + jp])
        self.inner_nodes = np.arange(self.n_t)
        self.outer_nodes = self.n_r * self.n_t + np.arange(self.n_t)

    def ring_nodes(self, ring):
        return self.inner_nodes if ring == 'inner' else self.outer_nodes

    def ring_dofs(self, ring):
        n = self.ring_nodes(ring)
        return np.column_stack([2 * n, 2 * n + 1]).ravel()

def _shape_derivs(xi, eta):
    return 0.25 * np.array([[-(1.0 - eta), -(1.0 - xi)], [1.0 - eta, -(1.0 + xi)], [1.0 + eta, 1.0 + xi], [-(1.0 + eta), 1.0 - xi]])

def _bmat_all(mesh, xi, eta):
    """B matrices (ne,3,8) and det(J) (ne,) at one Gauss point."""
    dN = _shape_derivs(xi, eta)
    xy = mesh.xy[mesh.elems]
    J = np.einsum('ak,eai->eki', dN, xy)
    detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    Jinv = np.empty_like(J)
    Jinv[:, 0, 0] = J[:, 1, 1]
    Jinv[:, 0, 1] = -J[:, 0, 1]
    Jinv[:, 1, 0] = -J[:, 1, 0]
    Jinv[:, 1, 1] = J[:, 0, 0]
    Jinv /= detJ[:, None, None]
    dNxy = np.einsum('eik,ak->eai', Jinv, dN)
    ne = mesh.elems.shape[0]
    B = np.zeros((ne, 3, 8))
    B[:, 0, 0::2] = dNxy[:, :, 0]
    B[:, 1, 1::2] = dNxy[:, :, 1]
    B[:, 2, 0::2] = dNxy[:, :, 1]
    B[:, 2, 1::2] = dNxy[:, :, 0]
    return (B, detJ)

def assemble_stiffness(mesh, E, nu):
    D = dmat_plane_strain(E, nu)
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
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(mesh.n_dof, mesh.n_dof)).tocsr()
    return K

def ring_edges(mesh, ring):
    """Consecutive node pairs (n0, n1) around a boundary ring (CCW)."""
    nodes = mesh.ring_nodes(ring)
    return [(nodes[j], nodes[(j + 1) % mesh.n_t]) for j in range(mesh.n_t)]

def ring_load(mesh, ring, traction_xy_fn):
    """Consistent nodal loads of a traction FIELD t(x, y) on a ring.

    Vectorized: all edge Gauss points are evaluated in one call of
    ``traction_xy_fn`` (which must broadcast over arrays).
    """
    nodes = mesh.ring_nodes(ring)
    n0 = nodes
    n1 = nodes[np.r_[1:mesh.n_t, 0]]
    x0, x1 = (mesh.xy[n0], mesh.xy[n1])
    L = np.hypot(*(x1 - x0).T)
    f = np.zeros(mesh.n_dof)
    for p, w in zip(_G3_P, _G3_W):
        N0, N1 = (0.5 * (1.0 - p), 0.5 * (1.0 + p))
        xp = N0 * x0 + N1 * x1
        tx, ty = traction_xy_fn(xp[:, 0], xp[:, 1])
        c = w * 0.5 * L
        np.add.at(f, 2 * n0, c * N0 * tx)
        np.add.at(f, 2 * n0 + 1, c * N0 * ty)
        np.add.at(f, 2 * n1, c * N1 * tx)
        np.add.at(f, 2 * n1 + 1, c * N1 * ty)
    return f

def ring_mass(mesh, ring):
    """Scalar boundary mass matrix (n_t, n_t) on a ring (P1 edges)."""
    M = np.zeros((mesh.n_t, mesh.n_t))
    nodes = list(mesh.ring_nodes(ring))
    index = {n: i for i, n in enumerate(nodes)}
    for n0, n1 in ring_edges(mesh, ring):
        i, j = (index[n0], index[n1])
        L = np.hypot(*mesh.xy[n1] - mesh.xy[n0])
        M[i, i] += L / 3.0
        M[j, j] += L / 3.0
        M[i, j] += L / 6.0
        M[j, i] += L / 6.0
    return M

class AnnulusFE:

    def __init__(self, mesh, E, nu):
        self.mesh = mesh
        self.E, self.nu = (float(E), float(nu))
        self.K = assemble_stiffness(mesh, E, nu)
        self._fact = {}
        self._m_out = ring_mass(mesh, 'outer')
        self._m_out_lu = None
        n = mesh.n_nodes
        tx = np.zeros(mesh.n_dof)
        tx[0::2] = 1.0
        ty = np.zeros(mesh.n_dof)
        ty[1::2] = 1.0
        rot = np.zeros(mesh.n_dof)
        rot[0::2] = -mesh.xy[:, 1]
        rot[1::2] = mesh.xy[:, 0]
        self.rigid = np.column_stack([v / np.linalg.norm(v) for v in (tx, ty, rot)])

    def solve_dirichlet(self, bc, f):
        """bc: dict ring -> (n_t, 2) Cartesian nodal values."""
        rings = tuple(sorted(bc.keys()))
        fixed = np.concatenate([self.mesh.ring_dofs(r) for r in rings])
        key = ('dir', rings)
        if key not in self._fact:
            free = np.setdiff1d(np.arange(self.mesh.n_dof), fixed)
            K_ff = self.K[free][:, free].tocsc()
            K_fc = self.K[free][:, fixed].tocsr()
            self._fact[key] = (free, fixed, spla.splu(K_ff), K_fc)
        free, fixed, lu, K_fc = self._fact[key]
        u_fix = np.concatenate([np.asarray(bc[r], dtype=float).ravel() for r in rings])
        u = np.zeros(self.mesh.n_dof)
        u[fixed] = u_fix
        u[free] = lu.solve(f[free] - K_fc @ u_fix)
        return u

    def solve_neumann(self, f):
        """Pure-traction solve with zero-mean rigid-body constraints."""
        key = ('neu',)
        if key not in self._fact:
            C = sp.csc_matrix(self.rigid)
            Ka = sp.bmat([[self.K, C], [C.T, None]], format='csc')
            self._fact[key] = spla.splu(Ka)
        lu = self._fact[key]
        rhs = np.concatenate([f, np.zeros(3)])
        return lu.solve(rhs)[:self.mesh.n_dof]

    def _b_out_global(self):
        key = ('b_out',)
        if key not in self._fact:
            nodes = self.mesh.outer_nodes
            n_t = self.mesh.n_t
            rows, cols, vals = ([], [], [])
            for c in (0, 1):
                for i in range(n_t):
                    for j in range(n_t):
                        if self._m_out[i, j] != 0.0:
                            rows.append(2 * nodes[i] + c)
                            cols.append(2 * nodes[j] + c)
                            vals.append(self._m_out[i, j])
            B = sp.coo_matrix((vals, (rows, cols)), shape=(self.mesh.n_dof, self.mesh.n_dof)).tocsr()
            self._fact[key] = B
        return self._fact[key]

    def solve_robin(self, beta, mu_cart_nodal, f_extra):
        """(K + beta*B_out) u = f_extra + B_out * mu  (mu nodal values)."""
        B = self._b_out_global()
        key = ('rob', float(beta))
        if key not in self._fact:
            self._fact[key] = spla.splu((self.K + beta * B).tocsc())
        lu = self._fact[key]
        mu_glob = np.zeros(self.mesh.n_dof)
        mu_glob[self.mesh.ring_dofs('outer')] = np.asarray(mu_cart_nodal, dtype=float).ravel()
        return lu.solve(f_extra + B @ mu_glob)

    def solve_condensed(self, K_sys, f):
        """Solve (K - DtN) u = f with translation constraints (bordered)."""
        C = sp.csc_matrix(self.rigid[:, :2])
        Ka = sp.bmat([[K_sys, C], [C.T, None]], format='csc')
        lu = spla.splu(Ka)
        rhs = np.concatenate([f, np.zeros(2)])
        return (lu.solve(rhs)[:self.mesh.n_dof], lu)

    def outer_traction_from_reactions(self, u, f_applied):
        """Nodal traction values (n_t, 2) on Gamma from consistent reactions."""
        r_full = self.K @ u - f_applied
        r = r_full[self.mesh.ring_dofs('outer')].reshape(self.mesh.n_t, 2)
        if self._m_out_lu is None:
            self._m_out_lu = np.linalg.inv(self._m_out)
        return self._m_out_lu @ r

    def trace_outer(self, u):
        """Nodal Cartesian displacements (n_t, 2) on Gamma."""
        return u[self.mesh.ring_dofs('outer')].reshape(self.mesh.n_t, 2)

    def trace_outer_polar(self, u):
        th = self.mesh.theta_nodes
        ux, uy = self.trace_outer(u).T
        return vec_cart_to_polar(ux, uy, th)

    def stress_at(self, u, points):
        """Polar stresses at given parent-element points [(xi, eta), ...].

        Returns (r, theta, s_rr, s_tt, s_rt) flat arrays over all elements
        and all requested points.
        """
        D = dmat_plane_strain(self.E, self.nu)
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

    def stress_centers(self, u):
        """Polar stresses at element centers (the optimal/Barlow stress
        sampling points of the bilinear quad, superconvergent O(h^2))."""
        return self.stress_at(u, [(0.0, 0.0)])

    def stress_gauss(self, u):
        """Polar stresses at all 2x2 Gauss points (raw O(h) sampling)."""
        return self.stress_at(u, [(xi, eta) for xi in _G2 for eta in _G2])

    def disp_nodes_polar(self, u):
        """Polar displacements at all nodes: (r, theta, u_r, u_t)."""
        ux, uy = (u[0::2], u[1::2])
        th = self.mesh.th_node
        u_r, u_t = vec_cart_to_polar(ux, uy, th)
        return (self.mesh.r_node, th, u_r, u_t)
