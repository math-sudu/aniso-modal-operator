import numpy as np
import scipy.sparse as sp
from .farfield import eval_fields

def _proj(f, trig):
    return 2.0 * (f @ trig) / len(f) if trig.any() else 0.0

def modal_blocks(R, G, kappa, n_max, n_sample=1024):
    """List over n = 0..n_max of {'S': 2x2, 'A': 2x2} modal DtN blocks."""
    th = 2.0 * np.pi * np.arange(n_sample) / n_sample
    blocks = []
    for n in range(n_max + 1):
        if n <= 1:
            branches = [('B', n + 1, 1.0), ('B', n + 1, 1j)]
        else:
            branches = [('A', n - 1, 1.0), ('B', n + 1, 1.0), ('A', n - 1, 1j), ('B', n + 1, 1j)]
        nb = len(branches)
        US, TS = (np.zeros((2, nb)), np.zeros((2, nb)))
        UA, TA = (np.zeros((2, nb)), np.zeros((2, nb)))
        cn, sn = (np.cos(n * th), np.sin(n * th))
        wc = 1.0 if n == 0 else 2.0
        for idx, (which, k, unit) in enumerate(branches):
            A = np.zeros(k, dtype=complex)
            B = np.zeros(k, dtype=complex)
            (A if which == 'A' else B)[k - 1] = unit
            f = eval_fields(A, B, R, th, G, kappa)
            pc = lambda g: wc * (g @ cn) / n_sample
            ps = lambda g: 2.0 * (g @ sn) / n_sample
            US[:, idx] = [pc(f['u_r']), ps(f['u_t'])]
            TS[:, idx] = [pc(f['s_rr']), ps(f['s_rt'])]
            UA[:, idx] = [ps(f['u_r']), pc(f['u_t'])]
            TA[:, idx] = [ps(f['s_rr']), pc(f['s_rt'])]
            scale = max(np.max(np.abs(f['u_r'])), np.max(np.abs(f['u_t'])))
            rec_ur = US[0, idx] * cn + UA[0, idx] * sn
            rec_ut = US[1, idx] * sn + UA[1, idx] * cn
            err = max(np.max(np.abs(f['u_r'] - rec_ur)), np.max(np.abs(f['u_t'] - rec_ut)))
            if err > 1e-09 * max(scale, 1e-300):
                raise RuntimeError(f'branch (n={n}, {which}_{k}, {unit}) not a pure harmonic')
        blocks.append({'S': TS @ np.linalg.pinv(US), 'A': TA @ np.linalg.pinv(UA)})
    return blocks

def nodal_dtn(theta_nodes, R, G, kappa, n_max=None):
    """Nodal DtN matrix on the interface ring.

    Acts on stacked polar displacements [u_r_j; u_t_j] (2 n_t,) and
    returns stacked polar tractions [s_rr_j; s_rt_j].
    """
    th = np.asarray(theta_nodes, dtype=float)
    n_t = len(th)
    if n_max is None:
        n_max = n_t // 2 - 1
    blocks = modal_blocks(R, G, kappa, n_max)
    S = np.zeros((2 * n_t, 2 * n_t))
    for n in range(n_max + 1):
        c, s = (np.cos(n * th), np.sin(n * th))
        w = (1.0 if n == 0 else 2.0) / n_t
        An = np.zeros((2, 2 * n_t))
        An[0, :n_t] = w * c
        An[1, n_t:] = w * s
        Sy = np.zeros((2 * n_t, 2))
        Sy[:n_t, 0] = c
        Sy[n_t:, 1] = s
        S += Sy @ blocks[n]['S'] @ An
        An = np.zeros((2, 2 * n_t))
        An[0, :n_t] = w * s
        An[1, n_t:] = w * c
        Sy = np.zeros((2 * n_t, 2))
        Sy[:n_t, 0] = s
        Sy[n_t:, 1] = c
        S += Sy @ blocks[n]['A'] @ An
    return S

def apply_nodal_dtn(S_pol, u_r, u_t):
    n_t = len(u_r)
    v = S_pol @ np.concatenate([u_r, u_t])
    return (v[:n_t], v[n_t:])

def condensed_system(fe, prm):
    """K_sys = K_FE - (boundary DtN stiffness), assembled once.

    Weak form: a(u, v) - int_Gamma v . S_far(u) ds = inner loads, with
    the boundary integral discretized through the outer-ring boundary
    mass matrix acting on nodal traction values.
    Returns (K_sys sparse, S_pol nodal DtN matrix).
    """
    mesh = fe.mesh
    n_t = mesh.n_t
    th = mesh.theta_nodes
    S_pol = nodal_dtn(th, mesh.R, prm.G, prm.kappa)
    P = np.zeros((2 * n_t, 2 * n_t))
    c, s = (np.cos(th), np.sin(th))
    j = np.arange(n_t)
    P[j, 2 * j] = c
    P[j, 2 * j + 1] = s
    P[n_t + j, 2 * j] = -s
    P[n_t + j, 2 * j + 1] = c
    M = fe._m_out
    Bm = np.zeros((2 * n_t, 2 * n_t))
    Bm[0::2, 0::2] = M
    Bm[1::2, 1::2] = M
    A_loc = Bm @ P.T @ S_pol @ P
    dofs = mesh.ring_dofs('outer')
    rows = np.repeat(dofs, 2 * n_t)
    cols = np.tile(dofs, 2 * n_t)
    A_glob = sp.coo_matrix((A_loc.ravel(), (rows, cols)), shape=(mesh.n_dof, mesh.n_dof)).tocsr()
    return ((fe.K - A_glob).tocsr(), S_pol)
