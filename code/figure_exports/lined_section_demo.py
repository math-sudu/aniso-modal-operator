from __future__ import annotations
import csv
import gc
import sys
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code' / 'closure'))
from closure_env import bridge_pinned
bridge_pinned()
from anisotropic_elasticity.fe_aniso import assemble_stiffness_D
from anisotropic_elasticity.material import material_F
from mapped_elasticity import protocol as P
from mapped_elasticity.fe_mapped import AnisoAnnulusFE, MappedMesh, mapped_released_load
from exterior_operator import condensed_mapped_system
from mapped_elasticity.geometry import horseshoe_map
from isotropic_elasticity.fe_annulus import _bmat_all, dmat_plane_strain
OUT = ROOT / 'results/data/derived'
P0_V = -15000000.0
K_BASE = 0.9
K_SWEEP = (0.6, 1.4)

def insitu(k: float) -> tuple[float, float, float]:
    """Released in-situ state at lateral pressure coefficient ``k``."""
    return (P0_V, k * P0_V, 0.0)
LOAD_BASE = insitu(K_BASE)
RHO = 2.0
SMAP = horseshoe_map(1.0, P.R_HS, P.C_HS)
MESH_MAIN = (24, 128)
MESH_LADDER = ((12, 64), (24, 128), (48, 256))
MESH_FAR_SKIP = (48, 256)
RATIOS = (1.25, 2.625, 4.0)
MAT_DIPS = (0.0, 45.0, 90.0)
CASES_DEMO = tuple(((r, d) for d in MAT_DIPS for r in RATIOS))
CASES = ((1.25, 0.0), (2.625, 45.0), (4.0, 90.0))
LINING_T = 0.4
LINING_NL = 16
LINING_NL_LADDER = (4, 8, 16)
LAME_LADDER = ((128, 4), (128, 8), (128, 16), (128, 32))
E_LIN, NU_LIN = (30000000000.0, 0.2)
K_FAR = 14
LAM_LADDER = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
FIBER_LIMIT_MPA = 20.0
LAMBDA_REPORT = 0.82
S_LADDER = (1.01, 1.05, 1.2)
PIN_WALL_MM = {(1.25, 0.0): 9.941, (2.625, 0.0): 8.502, (4.0, 0.0): 8.511, (1.25, 45.0): 10.872, (2.625, 45.0): 10.014, (4.0, 45.0): 9.739, (1.25, 90.0): 11.609, (2.625, 90.0): 11.595, (4.0, 90.0): 11.349}
_G2 = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))

class LiningRing:
    """Structured Q4 ring inside the excavation boundary.

    Node ``k * n_t + j`` sits on layer ``k`` at boundary parameter ``theta_j``,
    layer 0 the intrados and layer ``n_l`` the extrados.  The extrados layer is
    the ground's wall ring, so the two meshes share those nodes rather than
    being tied by a constraint; layer index increases outward, which keeps the
    element orientation -- and so the Jacobian sign -- the same as the ground
    mesh's.
    """

    def __init__(self, smap, theta: np.ndarray, thickness: float | np.ndarray, n_l: int):
        self.n_t = int(len(theta))
        self.n_l = int(n_l)
        self.thickness = float(thickness) if np.ndim(thickness) == 0 else np.asarray(thickness, dtype=float)
        z = smap.boundary(theta)
        n_hat, _ = smap.wall_normal(theta)
        xy = []
        for k in range(n_l + 1):
            off = self.thickness * (1.0 - k / n_l)
            p = z - n_hat * off
            xy.append(np.column_stack([np.real(p), np.imag(p)]))
        self.xy = np.vstack(xy)
        self.n_nodes = (n_l + 1) * self.n_t
        self.n_dof = 2 * self.n_nodes
        k_idx = np.repeat(np.arange(n_l), self.n_t)
        j_idx = np.tile(np.arange(self.n_t), n_l)
        jp = (j_idx + 1) % self.n_t
        self.elems = np.column_stack([k_idx * self.n_t + j_idx, (k_idx + 1) * self.n_t + j_idx, (k_idx + 1) * self.n_t + jp, k_idx * self.n_t + jp])
        self.extrados_nodes = n_l * self.n_t + np.arange(self.n_t)
        self.intrados_nodes = np.arange(self.n_t)

    def min_detj(self) -> float:
        m = np.inf
        for xi in _G2:
            for eta in _G2:
                _, detj = _bmat_all(self, xi, eta)
                m = min(m, float(np.min(detj)))
        return m

    def global_map(self, n_ground_nodes: int) -> np.ndarray:
        """Local node -> merged node: the extrados IS the ground's wall ring."""
        g = np.empty(self.n_nodes, dtype=np.int64)
        interior = self.n_l * self.n_t
        g[:interior] = n_ground_nodes + np.arange(interior)
        g[interior:] = np.arange(self.n_t)
        return g

def lining_stiffness(ring: LiningRing, e_mod: float, nu: float):
    return assemble_stiffness_D(ring, dmat_plane_strain(e_mod, nu))

def scatter(K_local, gmap: np.ndarray, n_dof_merged: int):
    """Lift a ring-local operator onto the merged degrees of freedom."""
    dof = np.empty(2 * len(gmap), dtype=np.int64)
    dof[0::2], dof[1::2] = (2 * gmap, 2 * gmap + 1)
    C = K_local.tocoo()
    return sp.coo_matrix((C.data, (dof[C.row], dof[C.col])), shape=(n_dof_merged, n_dof_merged)).tocsr()

def ring_actions(ring: LiningRing, u_ring: np.ndarray, e_mod: float, nu: float):
    """Thrust and bending moment per angular station of the ring.

    The element's first local axis runs through the thickness and its second
    runs along the ring, so the hoop stress at a Gauss point is the stress
    resolved on the normalized ``dx/deta``.  Integrating it through the
    thickness with the same Gauss rule that assembled the element gives the
    thrust; taking its first moment about the mid-thickness gives the bending
    moment.  Thrust is returned compression-positive (the tunnelling sign) and
    the moment positive when it puts the intrados in tension.
    """
    D = dmat_plane_strain(e_mod, nu)
    n_t, n_l = (ring.n_t, ring.n_l)
    edof = np.empty((ring.elems.shape[0], 8), dtype=np.int64)
    edof[:, 0::2] = 2 * ring.elems
    edof[:, 1::2] = 2 * ring.elems + 1
    ue = u_ring[edof]
    xy_e = ring.xy[ring.elems]
    thrust = np.zeros(n_t)
    moment = np.zeros(n_t)
    dN_dxi = lambda eta: 0.25 * np.array([-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)])
    dN_deta = lambda xi: 0.25 * np.array([-(1 - xi), -(1 + xi), 1 + xi, 1 - xi])
    for xi in _G2:
        for eta in _G2:
            B, _ = _bmat_all(ring, xi, eta)
            sig = np.einsum('kl,elb,eb->ek', D, B, ue)
            dxi = np.einsum('a,eai->ei', dN_dxi(eta), xy_e)
            deta = np.einsum('a,eai->ei', dN_deta(xi), xy_e)
            tvec = deta / np.linalg.norm(deta, axis=1)[:, None]
            s_hoop = sig[:, 0] * tvec[:, 0] ** 2 + sig[:, 1] * tvec[:, 1] ** 2 + 2.0 * sig[:, 2] * tvec[:, 0] * tvec[:, 1]
            dl = np.linalg.norm(dxi, axis=1)
            layer = np.repeat(np.arange(n_l), n_t)
            thickness = ring.thickness
            if np.ndim(thickness):
                thickness = np.tile(0.5 * ((1 - eta) * thickness + (1 + eta) * np.roll(thickness, -1)), n_l)
            h = thickness / n_l
            s_gp = (layer + 0.5 * (xi + 1.0)) * h
            arm = s_gp - 0.5 * thickness
            w = 0.5
            contrib_n = w * -s_hoop * dl
            contrib_m = w * -s_hoop * arm * dl
            np.add.at(thrust, np.tile(np.arange(n_t), n_l), contrib_n)
            np.add.at(moment, np.tile(np.arange(n_t), n_l), contrib_m)
    return (thrust, moment)

def fiber_stress(thrust: np.ndarray, moment: np.ndarray, t: float):
    """Maximum absolute linear-section fiber stress, per metre of tunnel."""
    return np.abs(thrust) / t + np.abs(moment) * 6.0 / (t * t)

def merged_operator_solve(mat, n_r: int, n_t: int, ring_e: float, s_scale: float=1.0, load=None, contact='bonded'):
    """Ground closed by ``S_f``, ring at the wall, released load."""
    mesh = MappedMesh(SMAP, RHO, n_r, n_t)
    fe = AnisoAnnulusFE(mesh, mat)
    K_sys, _ = condensed_mapped_system(fe, mat, SMAP, RHO)
    if s_scale != 1.0:
        K_sys = (fe.K - (fe.K - K_sys) * s_scale).tocsr()
    f_g = mapped_released_load(mesh, *(load or LOAD_BASE))
    return _assemble_and_solve(mesh, K_sys, f_g, ring_e, pin_ring=None, contact=contact)

def merged_far_solve(mat, n_r: int, n_t: int, k: int, bracket: str, ring_e: float, load=None):
    """Ground meshed to ``rho_far = 2^k``, ring bonded at the wall.

    ``n_r`` layers per octave keeps the far model's first ``n_r + 1`` rings
    identical to the near mesh of the condensed route, so the two share their
    discretization and the ring attaches to the same nodes in both.
    """
    mesh = MappedMesh(SMAP, float(2 ** k), n_r * k, n_t)
    fe = AnisoAnnulusFE(mesh, mat)
    f_g = mapped_released_load(mesh, *(load or LOAD_BASE))
    return _assemble_and_solve(mesh, fe.K.tocsr(), f_g, ring_e, pin_ring=bracket, mesh_obj=mesh)

def _slip_constraints(mesh, ring, gmap, n_dof: int):
    """Enforce normal continuity at the extrados with free tangential motion.

Separate ground and lining nodes are coupled by one constraint per station, n . (u_ring - u_ground) = 0. Tangential components are independent."""
    n_hat, _ = SMAP.wall_normal(mesh.theta_nodes)
    nx, ny = (np.real(n_hat), np.imag(n_hat))
    wall = np.asarray(mesh.ring_nodes('inner')).ravel()
    rows, cols, vals = ([], [], [])
    for j in range(ring.n_t):
        g_ring = int(gmap[ring.extrados_nodes[j]])
        g_grd = int(wall[j])
        for dof, v in ((2 * g_ring, nx[j]), (2 * g_ring + 1, ny[j]), (2 * g_grd, -nx[j]), (2 * g_grd + 1, -ny[j])):
            rows.append(j)
            cols.append(dof)
            vals.append(v)
    return sp.coo_matrix((vals, (rows, cols)), shape=(ring.n_t, n_dof)).tocsc()

def _assemble_and_solve(mesh, K_ground, f_ground, ring_e, pin_ring, mesh_obj=None, contact='bonded', ring=None):
    if ring is None:
        ring = LiningRing(SMAP, mesh.theta_nodes, LINING_T, LINING_NL)
    if contact == 'bonded':
        gmap = ring.global_map(mesh.n_nodes)
        n_ring_own = ring.n_l * ring.n_t
    elif contact == 'slip':
        gmap = mesh.n_nodes + np.arange(ring.n_nodes, dtype=np.int64)
        n_ring_own = ring.n_nodes
    else:
        raise ValueError(f'unknown contact condition {contact!r}')
    n_dof = mesh.n_dof + 2 * n_ring_own
    K = sp.csr_matrix((n_dof, n_dof))
    K = K + sp.coo_matrix((K_ground.tocoo().data, (K_ground.tocoo().row, K_ground.tocoo().col)), shape=(n_dof, n_dof)).tocsr()
    K = K + scatter(lining_stiffness(ring, ring_e, NU_LIN), gmap, n_dof)
    f = np.zeros(n_dof)
    f[:mesh.n_dof] = f_ground
    if pin_ring is None:
        tx = np.zeros(n_dof)
        tx[0::2] = 1.0
        ty = np.zeros(n_dof)
        ty[1::2] = 1.0
        cols = [sp.csc_matrix(np.column_stack([tx / np.linalg.norm(tx), ty / np.linalg.norm(ty)]))]
        if contact == 'slip':
            cols.append(_slip_constraints(mesh, ring, gmap, n_dof).T.tocsc())
        C = sp.hstack(cols, format='csc') * float(np.abs(K.diagonal()).mean())
        Ka = sp.bmat([[K, C], [C.T, None]], format='csc')
        u = spla.splu(Ka).solve(np.concatenate([f, np.zeros(C.shape[1])]))[:n_dof]
    elif contact != 'bonded':
        raise ValueError('the slip bond is built on the condensed route only')
    elif pin_ring == 'free':
        outer = mesh_obj.ring_nodes('outer')
        a, b = (int(outer[0]), int(outer[mesh.n_t // 4]))
        fixed = np.array([2 * a, 2 * a + 1, 2 * b])
        u = _pinned_solve(K, f, fixed, n_dof)
    else:
        fixed = mesh_obj.ring_dofs('outer')
        u = _pinned_solve(K, f, fixed, n_dof)
    u_ring = np.empty(ring.n_dof)
    u_ring[0::2], u_ring[1::2] = (u[2 * gmap], u[2 * gmap + 1])
    return (mesh, ring, u, u_ring)

def _pinned_solve(K, f, fixed, n_dof):
    free = np.setdiff1d(np.arange(n_dof), fixed)
    lu = spla.splu(K[free][:, free].tocsc(), permc_spec='MMD_AT_PLUS_A')
    u = np.zeros(n_dof)
    u[free] = lu.solve(f[free])
    del lu
    gc.collect()
    return u

def unlined_operator_solve(mat, n_r: int, n_t: int, s_scale: float=1.0, load=None):
    """The pinned unlined machinery, untouched, for the pin and lam ladders."""
    from mapped_elasticity.fe_mapped import solve_condensed, strip_translation
    mesh = MappedMesh(SMAP, RHO, n_r, n_t)
    fe = AnisoAnnulusFE(mesh, mat)
    K_sys, _ = condensed_mapped_system(fe, mat, SMAP, RHO)
    if s_scale != 1.0:
        K_sys = (fe.K - (fe.K - K_sys) * s_scale).tocsr()
    f = mapped_released_load(mesh, *(load or LOAD_BASE))
    return (mesh, strip_translation(solve_condensed(fe, K_sys, f)))

def interface_net_force(mat, n_r: int, n_t: int, ring_e: float) -> float:
    """Evaluate the net exterior-interface force from (K - K_sys) u.

A self-equilibrated excavation load and internal support interaction give zero net force across the exterior interface."""
    mesh = MappedMesh(SMAP, RHO, n_r, n_t)
    fe = AnisoAnnulusFE(mesh, mat)
    K_sys, _ = condensed_mapped_system(fe, mat, SMAP, RHO)
    A = (fe.K - K_sys).tocsr()
    f_g = mapped_released_load(mesh, *LOAD_BASE)
    _, _, u, _ = _assemble_and_solve(mesh, K_sys, f_g, ring_e, pin_ring=None)
    t = A @ u[:mesh.n_dof]
    res = np.hypot(float(np.sum(t[0::2])), float(np.sum(t[1::2])))
    return res / float(np.sum(np.abs(f_g)))

def wall_station_indices(mesh) -> dict[str, int]:
    """Crown, invert and the two widest wall nodes, read off the mesh.

    The section stands up through the map: the map's x axis is the physical
    vertical.  Crown and invert are the extreme map-x wall nodes, the widest
    pair the extreme map-y ones.
    """
    w = mesh.xy[mesh.ring_nodes('inner')]
    return {'crown': int(np.argmax(w[:, 0])), 'invert': int(np.argmin(w[:, 0])), 'wide_pos': int(np.argmax(w[:, 1])), 'wide_neg': int(np.argmin(w[:, 1]))}

def closures(mesh, u_full) -> tuple[float, float]:
    """Vertical and horizontal closure of the excavation, in millimetres.

    A chord-length change is free of the rigid gauge the disturbance field is
    determined up to, which an absolute crown displacement is not.
    """
    idx = wall_station_indices(mesh)
    nodes = mesh.ring_nodes('inner')
    d = lambda i: u_full[2 * nodes[i]:2 * nodes[i] + 2]
    dv = (d(idx['invert'])[0] - d(idx['crown'])[0]) * 1000.0
    dh = (d(idx['wide_neg'])[1] - d(idx['wide_pos'])[1]) * 1000.0
    return (float(dv), float(dh))

def station_theta(n_t: int) -> np.ndarray:
    """Angular station of each element column: its own mid-parameter."""
    h = 2.0 * np.pi / n_t
    return np.degrees(np.arange(n_t) * h + 0.5 * h)

def gate_mesh() -> tuple[float, float]:
    n_r, n_t = MESH_MAIN
    mesh = MappedMesh(SMAP, RHO, n_r, n_t)
    ring = LiningRing(SMAP, mesh.theta_nodes, LINING_T, LINING_NL)
    detj = ring.min_detj()
    wall = mesh.xy[mesh.ring_nodes('inner')]
    dev = float(np.max(np.abs(ring.xy[ring.extrados_nodes] - wall)) / np.max(np.abs(wall)))
    zb = SMAP.boundary(mesh.theta_nodes)
    seg = np.abs(np.diff(np.concatenate([zb, zb[:1]])))
    aspect = seg / (LINING_T / LINING_NL)
    lo, hi = (float(aspect.min()), float(aspect.max()))
    print(f'[gate] ring min det(J) = {detj:.4e}; extrados vs wall ring deviation {dev:.3e}; {LINING_NL} layers, element aspect ratio {lo:.1f} to {hi:.1f}')
    if detj <= 0.0:
        raise SystemExit('ring mesh folds; nothing written')
    if dev >= 1e-14:
        raise SystemExit('ring extrados is not the ground wall ring')
    return (lo, hi)

def gate_lame() -> list[dict]:
    """Check recovered thrust and moment against the exact Lame ring.

Under uniform extrados pressure, the hoop stress is sigma_tt = -p Ro^2 (1 + Ri^2/r^2) / (Ro^2 - Ri^2). Its thickness integrals give thrust and moment; N = p Ro supplies an independent thrust check. Refine the layer count at fixed circumferential resolution to measure thickness convergence and possible locking as element aspect ratios increase."""
    from mapped_elasticity.geometry import circle_map
    r_o, t, p = (5.0, 0.5, 4000000.0)
    r_i = r_o - t
    smap = circle_map(r_o)
    rows: list[dict] = []
    for n_t, n_l in LAME_LADDER:
        theta = 2.0 * np.pi * np.arange(n_t) / n_t
        ring = LiningRing(smap, theta, t, n_l)
        K = lining_stiffness(ring, E_LIN, NU_LIN).tolil()
        f = np.zeros(ring.n_dof)
        h = 2.0 * np.pi / n_t
        gp, gw = np.polynomial.legendre.leggauss(3)
        nodes = ring.extrados_nodes
        for j in range(n_t):
            jp = (j + 1) % n_t
            for q, w in zip(gp, gw):
                th = theta[j] + 0.5 * (q + 1.0) * h
                nx, ny = (np.cos(th), np.sin(th))
                c = w * 0.5 * h * r_o
                for nd, N in ((nodes[j], 0.5 * (1 - q)), (nodes[jp], 0.5 * (1 + q))):
                    f[2 * nd] += c * N * (-p * nx)
                    f[2 * nd + 1] += c * N * (-p * ny)
        fixed = np.array([2 * nodes[0], 2 * nodes[0] + 1, 2 * nodes[n_t // 4]])
        u = _pinned_solve(K.tocsr(), f, fixed, ring.n_dof)
        n_meas, m_meas = ring_actions(ring, u, E_LIN, NU_LIN)
        rr = np.linspace(r_i, r_o, 20001)
        s_tt = -p * r_o ** 2 * (1.0 + r_i ** 2 / rr ** 2) / (r_o ** 2 - r_i ** 2)
        n_exact = float(-np.trapezoid(s_tt, rr))
        m_exact = float(-np.trapezoid(s_tt * (rr - 0.5 * (r_i + r_o)), rr))
        dn = abs(float(np.mean(n_meas)) - n_exact) / n_exact
        dm = abs(float(np.mean(m_meas)) - m_exact) / abs(m_exact)
        dn2 = abs(float(np.mean(n_meas)) - p * r_o) / (p * r_o)
        spread = float(np.ptp(n_meas)) / n_exact
        aspect = 2.0 * np.pi * r_o / n_t / (t / n_l)
        prev = rows[-1] if rows and rows[-1]['stations'] == n_t else None
        rate = '' if prev is None else f", falling by {prev['moment_rel_err'] / dm:.2f}"
        rows.append({'stations': n_t, 'layers': n_l, 'element_aspect_ratio': aspect, 'thrust_rel_err_integral': dn, 'thrust_rel_err_pRo': dn2, 'moment_rel_err': dm, 'thrust_station_spread': spread, 'moment_err_ratio_to_previous': '' if prev is None else prev['moment_rel_err'] / dm})
        print(f'[gate] Lame {n_t}x{n_l} (element aspect {aspect:.1f}): N {dn:.2e} vs integral, {dn2:.2e} vs p*Ro, M {dm:.2e}{rate}, station spread {spread:.2e}')
        if (n_t, n_l) == (MESH_MAIN[1], LINING_NL) and (dn > 0.0002 or dn2 > 0.0002 or dm > 0.002):
            raise SystemExit('ring action recovery failed the Lame gate')
    return rows

def gate_pin() -> dict:
    """Re-measure the published wall convergences before anything is written.

    The solves are kept: they are the unlined baseline every later stage reads
    against, so the demonstration superposes against the published field and
    not against a second computation of it.
    """
    n_r, n_t = MESH_MAIN
    unlined = {}
    worst = 0.0
    for ratio, dip in CASES_DEMO:
        mat = material_F(ratio, dip, f'pin{ratio}_{dip}')
        mesh, u = unlined_operator_solve(mat, n_r, n_t)
        wall = mesh.ring_nodes('inner')
        umag = np.hypot(u[2 * wall], u[2 * wall + 1])
        meas = float(umag.max()) * 1000.0
        pinned = PIN_WALL_MM[ratio, dip]
        dev = abs(meas - pinned) / pinned
        worst = max(worst, dev)
        if dev >= 0.0005:
            raise SystemExit(f'published wall convergence not reproduced at r={ratio} d={dip}: {meas:.3f} vs {pinned:.3f}')
        unlined[ratio, dip] = (mesh, u)
    print(f'[gate] nine published wall convergences reproduced, worst deviation {worst:.2e}')
    return unlined

def gate_degenerate(unlined: dict) -> None:
    """A vanishing ring must return the lined solve to the unlined one."""
    n_r, n_t = MESH_MAIN
    ratio, dip = CASES[1]
    mat = material_F(ratio, dip, 'degen')
    u0 = unlined[ratio, dip][1]
    prev = None
    for frac in (0.01, 0.001, 0.0001):
        mesh, _, u, _ = merged_operator_solve(mat, n_r, n_t, E_LIN * frac)
        dv0, dh0 = closures(mesh, u0)
        dv, dh = closures(mesh, u)
        dev = max(abs(dv - dv0) / abs(dv0), abs(dh - dh0) / abs(dh0))
        rate = '' if prev is None else f', falling by {prev / dev:.1f}'
        print(f'[gate] ring modulus x{frac:g}: closure deviation from the unlined solve {dev:.3e}{rate}')
        prev = dev
    if prev is None or prev > 0.001:
        raise SystemExit('vanishing ring does not return the unlined solve')

def _summary_row(ratio, dip, n_r, n_t, route, s_scale, mesh, u, ring, u_ring, solve_s, net_force=''):
    th = station_theta(n_t)
    dv, dh = closures(mesh, u)
    if ring is None:
        n_op = m_op = np.zeros(n_t)
    else:
        n_op, m_op = ring_actions(ring, u_ring, E_LIN, NU_LIN)
    return ({'ratio': ratio, 'material_dip_deg': dip, 'bedding_dip_deg': 90.0 - dip, 'near_mesh': f'{n_r}x{n_t}', 'route': route, 's_scale': s_scale, 'closure_vertical_mm': dv, 'closure_horizontal_mm': dh, 'thrust_max_kN_per_m': float(n_op.max()) * 0.001, 'thrust_max_theta_deg': float(th[int(np.argmax(n_op))]), 'moment_absmax_kNm_per_m': float(np.abs(m_op).max()) * 0.001, 'moment_absmax_theta_deg': float(th[int(np.argmax(np.abs(m_op)))]), 'fiber_stress_max_MPa': float(fiber_stress(n_op, m_op, LINING_T).max()) * 1e-06, 'interface_net_force_rel': net_force, 'solve_s': round(solve_s, 3)}, n_op, m_op)

def _action_rows(ratio, dip, n_r, n_t, route, n_op, m_op):
    th = station_theta(n_t)
    return [{'ratio': ratio, 'material_dip_deg': dip, 'near_mesh': f'{n_r}x{n_t}', 'route': route, 'station_theta_deg': round(float(th[j]), 4), 'thrust_kN_per_m': float(n_op[j]) * 0.001, 'moment_kNm_per_m': float(m_op[j]) * 0.001} for j in range(n_t)]

def stage_thickness() -> list[dict]:
    """Measure thickness refinement of peak thrust and bending moment.

Keep the circumferential station count fixed and refine the ring layers at all nine operating points to track changes in the reported actions."""
    n_r, n_t = MESH_MAIN
    shipped = LINING_NL
    rows = []
    try:
        for ratio, dip in CASES_DEMO:
            mat = material_F(ratio, dip, f't{ratio}_{dip}')
            per_nl = {}
            for n_l in LINING_NL_LADDER:
                globals()['LINING_NL'] = n_l
                mesh, ring, u, u_ring = merged_operator_solve(mat, n_r, n_t, E_LIN)
                n_op, m_op = ring_actions(ring, u_ring, E_LIN, NU_LIN)
                dv, dh = closures(mesh, u)
                per_nl[n_l] = (float(n_op.max()) * 0.001, float(np.abs(m_op).max()) * 0.001, dv, dh)
                del mesh, ring, u, u_ring
                gc.collect()
            base = per_nl[LINING_NL_LADDER[0]]
            for n_l in LINING_NL_LADDER:
                n_max, m_max, dv, dh = per_nl[n_l]
                rows.append({'ratio': ratio, 'material_dip_deg': dip, 'bedding_dip_deg': 90.0 - dip, 'near_mesh': f'{n_r}x{n_t}', 'ring_layers': n_l, 'thrust_max_kN_per_m': n_max, 'moment_absmax_kNm_per_m': m_max, 'closure_vertical_mm': dv, 'closure_horizontal_mm': dh, 'thrust_shift_pct': 100.0 * (n_max - base[0]) / base[0], 'moment_shift_pct': 100.0 * (m_max - base[1]) / base[1]})
            print(f'[T] r={ratio} bedding={90.0 - dip:.0f}deg: Mmax ' + ' -> '.join((f'{per_nl[k][1]:.1f}' for k in LINING_NL_LADDER)) + f" kNm/m ({rows[-1]['moment_shift_pct']:+.2f}%), Nmax {rows[-1]['thrust_shift_pct']:+.3f}%", flush=True)
    finally:
        globals()['LINING_NL'] = shipped
    worst_m = max((abs(r['moment_shift_pct']) for r in rows))
    worst_n = max((abs(r['thrust_shift_pct']) for r in rows))
    print(f'[T] over the nine, {LINING_NL_LADDER[0]} -> {LINING_NL_LADDER[-1]} layers moves the peak moment by at most {worst_m:.2f} per cent and the peak thrust by at most {worst_n:.3f} per cent', flush=True)
    return rows

def stage_lateral() -> list[dict]:
    """Evaluate all nine material cases at each lateral pressure coefficient.

Solve bonded and unlined end members with the same inputs, retaining their actions and closures for the elastic release partition."""
    n_r, n_t = MESH_MAIN
    rows = []
    for k in K_SWEEP:
        load = insitu(k)
        for ratio, dip in CASES_DEMO:
            mat = material_F(ratio, dip, f'k{k}_{ratio}_{dip}')
            t0 = time.perf_counter()
            mesh, ring, u, u_ring = merged_operator_solve(mat, n_r, n_t, E_LIN, load=load)
            row, _, _ = _summary_row(ratio, dip, n_r, n_t, 'condensed', 1.0, mesh, u, ring, u_ring, time.perf_counter() - t0)
            row['lateral_pressure_coefficient'] = k
            rows.append(row)
            mesh_u, u_un = unlined_operator_solve(mat, n_r, n_t, load=load)
            row_u, _, _ = _summary_row(ratio, dip, n_r, n_t, 'unlined', 1.0, mesh_u, u_un, None, None, 0.0)
            row_u['lateral_pressure_coefficient'] = k
            rows.append(row_u)
            print(f"[E] k={k} r={ratio} bedding={90.0 - dip:.0f}deg: lined dv={row['closure_vertical_mm']:.3f} dh={row['closure_horizontal_mm']:.3f} mm, Nmax={row['thrust_max_kN_per_m']:.1f} kN/m, Mmax={row['moment_absmax_kNm_per_m']:.1f} kNm/m", flush=True)
            del mesh, ring, u, u_ring, mesh_u, u_un
            gc.collect()
    return rows

def stage_slip() -> list[dict]:
    """What the bond condition is worth, at all nine reported points.

    Both cited lined-tunnel sources carry a full-slip and a no-slip case and
    this demonstration bonds the ring silently.  The bonded route merges the
    ring's extrados into the ground's wall ring; the slip route gives it its
    own nodes and couples them through the wall normal alone, so the interface
    transmits pressure and no shear.  The pair brackets the contact condition
    from both sides, which is what the reported actions need attached to them.
    """
    n_r, n_t = MESH_MAIN
    rows = []
    for ratio, dip in CASES_DEMO:
        mat = material_F(ratio, dip, f's{ratio}_{dip}')
        out = {}
        slip_mm = float('nan')
        for contact in ('bonded', 'slip'):
            mesh, ring, u, u_ring = merged_operator_solve(mat, n_r, n_t, E_LIN, contact=contact)
            n_op, m_op = ring_actions(ring, u_ring, E_LIN, NU_LIN)
            dv, dh = closures(mesh, u)
            out[contact] = (float(n_op.max()) * 0.001, float(np.abs(m_op).max()) * 0.001, float(fiber_stress(n_op, m_op, LINING_T).max()) * 1e-06, dv, dh)
            if contact == 'slip':
                n_hat, _ = SMAP.wall_normal(mesh.theta_nodes)
                wall = np.asarray(mesh.ring_nodes('inner')).ravel()
                du = np.column_stack([u_ring[2 * ring.extrados_nodes] - u[2 * wall], u_ring[2 * ring.extrados_nodes + 1] - u[2 * wall + 1]])
                nx, ny = (np.real(n_hat), np.imag(n_hat))
                gap_n = np.abs(du[:, 0] * nx + du[:, 1] * ny)
                slip_t = np.abs(-du[:, 0] * ny + du[:, 1] * nx)
                scale = float(np.hypot(u[0::2], u[1::2]).max())
                gap_rel, slip_mm = (float(gap_n.max()) / scale, float(slip_t.max()) * 1000.0)
                if gap_rel > 1e-10:
                    raise SystemExit('slip bond leaves a normal gap')
                if slip_mm < 0.001:
                    raise SystemExit('slip bond transmits no relative motion')
            del mesh, ring, u, u_ring
            gc.collect()
        b, s = (out['bonded'], out['slip'])
        rows.append({'ratio': ratio, 'material_dip_deg': dip, 'bedding_dip_deg': 90.0 - dip, 'near_mesh': f'{n_r}x{n_t}', 'thrust_bonded_kN_per_m': b[0], 'thrust_slip_kN_per_m': s[0], 'moment_bonded_kNm_per_m': b[1], 'moment_slip_kNm_per_m': s[1], 'fiber_bonded_MPa': b[2], 'fiber_slip_MPa': s[2], 'closure_vertical_bonded_mm': b[3], 'closure_vertical_slip_mm': s[3], 'closure_horizontal_bonded_mm': b[4], 'closure_horizontal_slip_mm': s[4], 'thrust_shift_pct': 100.0 * (s[0] - b[0]) / b[0], 'moment_shift_pct': 100.0 * (s[1] - b[1]) / b[1], 'fiber_shift_pct': 100.0 * (s[2] - b[2]) / b[2], 'closure_vertical_shift_pct': 100.0 * (s[3] - b[3]) / b[3], 'max_tangential_slip_mm': slip_mm})
        print(f"[F] r={ratio} bedding={90.0 - dip:.0f}deg: slip moves Nmax by {rows[-1]['thrust_shift_pct']:+.1f}%, Mmax by {rows[-1]['moment_shift_pct']:+.1f}%, dv by {rows[-1]['closure_vertical_shift_pct']:+.1f}%, sliding {slip_mm:.2f} mm", flush=True)
    for q in ('thrust', 'moment', 'fiber', 'closure_vertical'):
        w = max(rows, key=lambda r: abs(r[f'{q}_shift_pct']))
        print(f"[F] over the nine, full slip moves {q} by at most {w[f'{q}_shift_pct']:+.1f} per cent", flush=True)
    return rows

def stage_demo(unlined: dict) -> tuple[list[dict], list[dict]]:
    """The demonstration proper: nine operating points at the main mesh."""
    n_r, n_t = MESH_MAIN
    summary, actions = ([], [])
    for ratio, dip in CASES_DEMO:
        mat = material_F(ratio, dip, f'd{ratio}_{dip}')
        t0 = time.perf_counter()
        mesh, ring, u, u_ring = merged_operator_solve(mat, n_r, n_t, E_LIN)
        nf = interface_net_force(mat, n_r, n_t, E_LIN)
        row, n_op, m_op = _summary_row(ratio, dip, n_r, n_t, 'condensed', 1.0, mesh, u, ring, u_ring, time.perf_counter() - t0, nf)
        summary.append(row)
        actions += _action_rows(ratio, dip, n_r, n_t, 'condensed', n_op, m_op)
        mesh_u, u_un = unlined[ratio, dip]
        row_u, _, _ = _summary_row(ratio, dip, n_r, n_t, 'unlined', 1.0, mesh_u, u_un, None, None, 0.0)
        summary.append(row_u)
        print(f"[A] r={ratio} bedding={90.0 - dip:.0f}deg: lined dv={row['closure_vertical_mm']:.3f} dh={row['closure_horizontal_mm']:.3f} mm (unlined {row_u['closure_vertical_mm']:.3f} / {row_u['closure_horizontal_mm']:.3f}), Nmax={row['thrust_max_kN_per_m']:.1f} kN/m, Mmax={row['moment_absmax_kNm_per_m']:.1f} kNm/m", flush=True)
        del mesh, ring, u, u_ring
        gc.collect()
    return (summary, actions)

def stage_independence() -> tuple[list[dict], list[dict]]:
    """The exterior read twice, and what size of operator error would show."""
    summary, actions = ([], [])
    for ratio, dip in CASES:
        mat = material_F(ratio, dip, f'i{ratio}')
        for n_r, n_t in MESH_LADDER:
            t0 = time.perf_counter()
            mesh, ring, u, u_ring = merged_operator_solve(mat, n_r, n_t, E_LIN)
            row, n_op, m_op = _summary_row(ratio, dip, n_r, n_t, 'condensed', 1.0, mesh, u, ring, u_ring, time.perf_counter() - t0)
            summary.append(row)
            print(f"[B] r={ratio} {n_r}x{n_t} condensed: dv={row['closure_vertical_mm']:.4f} mm, Nmax={row['thrust_max_kN_per_m']:.1f} kN/m", flush=True)
            if (n_r, n_t) != MESH_FAR_SKIP:
                for bracket in ('free', 'fixed'):
                    t0 = time.perf_counter()
                    mesh_f, ring_f, u_f, ur_f = merged_far_solve(mat, n_r, n_t, K_FAR, bracket, E_LIN)
                    rowf, n_f, m_f = _summary_row(ratio, dip, n_r, n_t, f'meshed_{bracket}', 1.0, mesh_f, u_f, ring_f, ur_f, time.perf_counter() - t0)
                    summary.append(rowf)
                    if (n_r, n_t) == MESH_MAIN:
                        actions += _action_rows(ratio, dip, n_r, n_t, f'meshed_{bracket}', n_f, m_f)
                    print(f"[C] r={ratio} {n_r}x{n_t} meshed {bracket}: dv={rowf['closure_vertical_mm']:.4f} mm, Nmax={rowf['thrust_max_kN_per_m']:.1f} kN/m", flush=True)
                    del mesh_f, ring_f, u_f, ur_f
                    gc.collect()
            del mesh, ring, u, u_ring
            gc.collect()
        n_r, n_t = MESH_MAIN
        for s in S_LADDER:
            mp, rp, up, urp = merged_operator_solve(mat, n_r, n_t, E_LIN, s_scale=s)
            rowp, _, _ = _summary_row(ratio, dip, n_r, n_t, 'condensed', s, mp, up, rp, urp, 0.0)
            summary.append(rowp)
            mu2, uu2 = unlined_operator_solve(mat, n_r, n_t, s_scale=s)
            rowu, _, _ = _summary_row(ratio, dip, n_r, n_t, 'unlined', s, mu2, uu2, None, None, 0.0)
            summary.append(rowu)
            print(f"[B] r={ratio} s_scale={s}: lined dv={rowp['closure_vertical_mm']:.4f}, unlined dv={rowu['closure_vertical_mm']:.4f}", flush=True)
            del mp, rp, up, urp, mu2, uu2
            gc.collect()
        mu, uu = unlined_operator_solve(mat, n_r, n_t)
        row0, _, _ = _summary_row(ratio, dip, n_r, n_t, 'unlined', 1.0, mu, uu, None, None, 0.0)
        summary.append(row0)
        del mu, uu
        gc.collect()
    n_r, n_t = MESH_MAIN
    for ratio, dip in CASES_DEMO:
        if (ratio, dip) in CASES:
            continue
        mat = material_F(ratio, dip, f'i{ratio}_{dip}')
        for bracket in ('free', 'fixed'):
            t0 = time.perf_counter()
            mesh_f, ring_f, u_f, ur_f = merged_far_solve(mat, n_r, n_t, K_FAR, bracket, E_LIN)
            rowf, n_f, m_f = _summary_row(ratio, dip, n_r, n_t, f'meshed_{bracket}', 1.0, mesh_f, u_f, ring_f, ur_f, time.perf_counter() - t0)
            summary.append(rowf)
            actions += _action_rows(ratio, dip, n_r, n_t, f'meshed_{bracket}', n_f, m_f)
            print(f"[C] r={ratio} d={dip} {n_r}x{n_t} meshed {bracket}: dv={rowf['closure_vertical_mm']:.4f} mm, Nmax={rowf['thrust_max_kN_per_m']:.1f} kN/m", flush=True)
            del mesh_f, ring_f, u_f, ur_f
            gc.collect()
    return (summary, actions)

def stage_release(unlined: dict) -> list[dict]:
    """The staging ladder, by superposition of the two end members."""
    n_r, n_t = MESH_MAIN
    rows = []
    for ratio, dip in CASES_DEMO:
        mat = material_F(ratio, dip, f'l{ratio}_{dip}')
        mesh, ring, u, u_ring = merged_operator_solve(mat, n_r, n_t, E_LIN)
        n_op, m_op = ring_actions(ring, u_ring, E_LIN, NU_LIN)
        dv_l, dh_l = closures(mesh, u)
        dv_u, dh_u = closures(mesh, unlined[ratio, dip][1])
        for lam in LAM_LADDER:
            n_lam = n_op * (1.0 - lam)
            m_lam = m_op * (1.0 - lam)
            rows.append({'ratio': ratio, 'material_dip_deg': dip, 'bedding_dip_deg': 90.0 - dip, 'lambda_released': lam, 'closure_vertical_mm': lam * dv_u + (1.0 - lam) * dv_l, 'closure_horizontal_mm': lam * dh_u + (1.0 - lam) * dh_l, 'thrust_max_kN_per_m': float(n_lam.max()) * 0.001, 'moment_absmax_kNm_per_m': float(np.abs(m_lam).max()) * 0.001, 'fiber_stress_max_MPa': float(fiber_stress(n_lam, m_lam, LINING_T).max()) * 1e-06})
            print(f"[D] r={ratio} d={dip} lam={lam:.1f}: sigma_max={rows[-1]['fiber_stress_max_MPa']:.1f} MPa, dv={rows[-1]['closure_vertical_mm']:.3f} mm", flush=True)
        del mesh, ring, u, u_ring
        gc.collect()
    return rows

def release_threshold(rows: list[dict], limit_mpa: float) -> list[dict]:
    """Find the smallest release fraction keeping every case below the stress limit.

Actions scale with 1 - lambda, so the largest zero-release fibre stress determines the threshold. Use every case written by release_partition, including all lateral pressure coefficients."""
    at_zero = [r for r in rows if r['route'] == 'condensed' and float(r['s_scale']) == 1.0 and (r['near_mesh'] == f'{MESH_MAIN[0]}x{MESH_MAIN[1]}')]
    worst = max(at_zero, key=lambda r: float(r['fiber_stress_max_MPa']))
    sigma0 = float(worst['fiber_stress_max_MPa'])
    lam = 1.0 - limit_mpa / sigma0
    row = {'fiber_stress_limit_MPa': limit_mpa, 'binding_lateral_pressure_coefficient': float(worst.get('lateral_pressure_coefficient') or K_BASE), 'binding_ratio': worst['ratio'], 'binding_bedding_dip_deg': worst['bedding_dip_deg'], 'fiber_stress_at_zero_release_MPa': sigma0, 'lambda_required': lam, 'lambda_reported': LAMBDA_REPORT, 'fiber_stress_at_reported_MPa': (1.0 - LAMBDA_REPORT) * sigma0}
    print(f"[D] release threshold: lambda >= {lam:.4f} holds all {len(at_zero)} cases under {limit_mpa:.0f} MPa (binding case {sigma0:.1f} MPa at zero release, coefficient {row['binding_lateral_pressure_coefficient']}); reported {LAMBDA_REPORT} leaves {row['fiber_stress_at_reported_MPa']:.1f} MPa", flush=True)
    if len(at_zero) != 27:
        raise SystemExit(f'release_threshold read {len(at_zero)} cases, not the twenty-seven the manuscript quotes over')
    if not lam <= LAMBDA_REPORT < lam + 0.01:
        raise SystemExit(f'LAMBDA_REPORT {LAMBDA_REPORT} is no longer the smallest two-decimal fraction above {lam:.4f}')
    return [row]

def release_partition(rows: list[dict], lam: float) -> list[dict]:
    """The lined table's released-fraction columns, at every lateral ratio.

    The ring carries ``1 - lam`` of the actions exactly, so thrust, moment and
    extreme fibre stress at the reported fraction are that factor times their
    own zero-release entries and are not re-solved.  The closure is the one
    quantity that is not a scaling of a single column: the ground converges by
    ``lam`` of the unlined amount plus ``1 - lam`` of the lined one, so both
    end members are read.
    """
    keyed = {}
    for r in rows:
        if float(r['s_scale']) != 1.0 or r['near_mesh'] != f'{MESH_MAIN[0]}x{MESH_MAIN[1]}':
            continue
        k = r.get('lateral_pressure_coefficient', K_BASE)
        keyed.setdefault((float(k), r['ratio'], r['material_dip_deg']), {})[r['route']] = r
    out = []
    for (k, ratio, dip), pair in sorted(keyed.items()):
        if 'condensed' not in pair or 'unlined' not in pair:
            continue
        lined, unl = (pair['condensed'], pair['unlined'])
        f = 1.0 - lam
        out.append({'lateral_pressure_coefficient': k, 'ratio': ratio, 'material_dip_deg': dip, 'bedding_dip_deg': 90.0 - float(dip), 'lambda_released': lam, 'thrust_max_kN_per_m': f * float(lined['thrust_max_kN_per_m']), 'moment_absmax_kNm_per_m': f * float(lined['moment_absmax_kNm_per_m']), 'closure_vertical_mm': lam * float(unl['closure_vertical_mm']) + f * float(lined['closure_vertical_mm']), 'closure_horizontal_mm': lam * float(unl['closure_horizontal_mm']) + f * float(lined['closure_horizontal_mm']), 'fiber_stress_max_MPa': f * float(lined['fiber_stress_max_MPa'])})
    print(f"[D] release partition at lambda = {lam}: {len(out)} points, fibre stress {min((r['fiber_stress_max_MPa'] for r in out)):.1f} to {max((r['fiber_stress_max_MPa'] for r in out)):.1f} MPa", flush=True)
    return out

def write_csv(name: str, rows: list[dict]) -> None:
    path = OUT / name
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open('w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'  wrote {path.relative_to(ROOT)} ({len(rows)} rows)')

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gate_mesh()
    lame = gate_lame()
    unlined = gate_pin()
    gate_degenerate(unlined)
    print()
    demo_summary, demo_actions = stage_demo(unlined)
    print()
    ind_summary, ind_actions = stage_independence()
    print()
    ladder = stage_release(unlined)
    print()
    thickness = stage_thickness()
    print()
    lateral = stage_lateral()
    print()
    slip = stage_slip()
    print()
    write_csv('lined_demo_summary.csv', demo_summary)
    write_csv('lined_demo_actions.csv', demo_actions + ind_actions)
    write_csv('lined_exterior_check.csv', ind_summary)
    write_csv('lined_ring_thickness_ladder.csv', thickness)
    write_csv('lined_lame_gate_ladder.csv', lame)
    write_csv('lined_lateral_sweep.csv', lateral)
    write_csv('lined_contact_sensitivity.csv', slip)
    write_csv('lined_release_ladder.csv', ladder)
    write_csv('lined_release_threshold.csv', release_threshold(demo_summary + lateral, FIBER_LIMIT_MPA))
    write_csv('lined_release_partition.csv', release_partition(demo_summary + lateral, LAMBDA_REPORT))
if __name__ == '__main__':
    main()
