import time
from dataclasses import dataclass, field
import numpy as np
from . import kirsch_ref as kr
from .fe_annulus import ring_load

@dataclass
class SchwarzResult:
    status: str
    n_iter: int
    resid_hist: list
    u: np.ndarray | None
    far: object
    wall_s: float
    errors: dict = field(default_factory=dict)
    final_resid: float = np.nan

def inner_load(fe, prm):
    """Consistent nodal loads of the excavation perturbation traction at r=a."""

    def t_fn(x, y):
        th = np.arctan2(y, x)
        t_r, t_t = kr.inner_wall_traction_polar(th, prm)
        return kr.vec_polar_to_cart(t_r, t_t, th)
    return ring_load(fe.mesh, 'inner', t_fn)

def _polar_stacked_to_cart_nodal(x, theta):
    n_t = len(theta)
    vx, vy = kr.vec_polar_to_cart(x[:n_t], x[n_t:], theta)
    return np.column_stack([vx, vy])

def _cart_nodal_to_polar_stacked(v, theta):
    vr, vt = kr.vec_cart_to_polar(v[:, 0], v[:, 1], theta)
    return np.concatenate([vr, vt])

def run_schwarz(fe, far, transmission, relax, prm, f_in=None, tol=1e-06, k_max=100, beta=None, div_guard=1000000.0):
    """Run one Schwarz alternating iteration to tolerance.

    relax: float rho, or the string "aitken".
    beta: absolute Robin parameter (required for transmission == "RR").
    """
    mesh = fe.mesh
    theta = mesh.theta_nodes
    n_t = mesh.n_t
    if f_in is None:
        f_in = inner_load(fe, prm)
    if transmission == 'DN-FEdir':

        def sweep(x):
            g_cart = _polar_stacked_to_cart_nodal(x, theta)
            u = fe.solve_dirichlet({'outer': g_cart}, f_in)
            t_nod = fe.outer_traction_from_reactions(u, f_in)
            t_pol = _cart_nodal_to_polar_stacked(t_nod, theta)
            far.fit('traction', t_pol)
            x_hat = far.eval_kind('displacement')
            return (x_hat, u, x.copy())
    elif transmission == 'DN-FEneu':

        def sweep(x):
            far.fit('displacement', x)

            def t_fn(xp, yp):
                r = np.hypot(xp, yp)
                th = np.arctan2(yp, xp)
                f = far.fields(r, th)
                return kr.vec_polar_to_cart(f['s_rr'], f['s_rt'], th)
            f_out = ring_load(mesh, 'outer', t_fn)
            u = fe.solve_neumann(f_in + f_out)
            u_r, u_t = fe.trace_outer_polar(u)
            y = np.concatenate([u_r, u_t])
            return (y, u, y)
    elif transmission == 'RR':
        if beta is None or beta <= 0.0:
            raise ValueError('RR needs beta > 0')

        def sweep(x):
            mu_cart = _polar_stacked_to_cart_nodal(x, theta)
            u = fe.solve_robin(beta, mu_cart, f_in)
            u_r, u_t = fe.trace_outer_polar(u)
            y = np.concatenate([u_r, u_t])
            nu_vec = x - 2.0 * beta * y
            far.fit('robin_minus', nu_vec, beta=beta)
            x_hat = far.eval_kind('robin_plus', beta=beta)
            return (x_hat, u, y)
    else:
        raise ValueError(f'unknown transmission {transmission!r}')
    x = np.zeros(2 * n_t)
    aitken = isinstance(relax, str) and relax == 'aitken'
    rho_prev = 0.5
    r_prev = None
    y_prev = None
    hist = []
    status = 'max_iter'
    n_iter = k_max
    u_final = None
    t0 = time.perf_counter()
    for k in range(1, k_max + 1):
        x_hat, u, y = sweep(x)
        u_final = u
        if k >= 2:
            res = np.linalg.norm(y - y_prev) / max(np.linalg.norm(y), 1e-300)
            hist.append(res)
            if not np.isfinite(res) or res > div_guard:
                status, n_iter = ('diverged', k)
                break
            if res <= tol:
                status, n_iter = ('converged', k)
                break
        y_prev = y
        r = x_hat - x
        if aitken:
            if r_prev is None:
                rho = 0.5
            else:
                dr = r - r_prev
                den = float(dr @ dr)
                rho = -rho_prev * float(r_prev @ dr) / den if den > 0.0 else rho_prev
                rho = float(np.clip(rho, 0.05, 2.0))
            rho_prev = rho
            r_prev = r
        else:
            rho = float(relax)
        x = x + rho * r
        if not np.all(np.isfinite(x)):
            status, n_iter = ('diverged', k)
            break
    wall = time.perf_counter() - t0
    return SchwarzResult(status=status, n_iter=n_iter, resid_hist=hist, u=u_final, far=far, wall_s=wall, final_resid=hist[-1] if hist else np.nan)

def far_grid(prm, R, n_r=24, n_th=64, r_max_factor=10.0):
    r = np.geomspace(R, r_max_factor * prm.a, n_r)
    th = 2.0 * np.pi * np.arange(n_th) / n_th
    rr, tt = np.meshgrid(r, th, indexing='ij')
    return (rr.ravel(), tt.ravel())

def _rel(diff_list, ref_list):
    d = np.concatenate([np.ravel(x) for x in diff_list])
    r = np.concatenate([np.ravel(x) for x in ref_list])
    return float(np.linalg.norm(d) / np.linalg.norm(r))

def global_errors(fe, u, far, prm):
    """Relative L2 errors vs the Kirsch reference over both zones.

    Near-zone sampling: displacements at FE nodes; stresses at element
    centers -- the optimal (Barlow) stress points of the bilinear quad,
    where the recovered stress superconverges at O(h^2).
    """
    r_g, th_g, srr, stt, srt = fe.stress_centers(u)
    ref_s = kr.pert_stress_polar(r_g, th_g, prm)
    ds_near = [srr - ref_s[0], stt - ref_s[1], srt - ref_s[2]]
    r_n, th_n, ur, ut = fe.disp_nodes_polar(u)
    ref_u = kr.disp_polar(r_n, th_n, prm)
    du_near = [ur - ref_u[0], ut - ref_u[1]]
    r_f, th_f = far_grid(prm, fe.mesh.R)
    f = far.fields(r_f, th_f)
    ref_sf = kr.pert_stress_polar(r_f, th_f, prm)
    ref_uf = kr.disp_polar(r_f, th_f, prm)
    ds_far = [f['s_rr'] - ref_sf[0], f['s_tt'] - ref_sf[1], f['s_rt'] - ref_sf[2]]
    du_far = [f['u_r'] - ref_uf[0], f['u_t'] - ref_uf[1]]
    return {'stress_near': _rel(ds_near, ref_s), 'stress_far': _rel(ds_far, ref_sf), 'stress_global': _rel(ds_near + ds_far, list(ref_s) + list(ref_sf)), 'disp_near': _rel(du_near, ref_u), 'disp_far': _rel(du_far, ref_uf), 'disp_global': _rel(du_near + du_far, list(ref_u) + list(ref_uf))}

def condensed_errors(fe, u, S_pol, prm):
    """Errors for the condensed variant: near = FE field; far = the
    Laurent field reconstructed from the interface displacement trace
    (Dirichlet fit -- the far zone's own representation of the solution)."""
    from .farfield import FarLaurentLS
    u_r, u_t = fe.trace_outer_polar(u)
    far = FarLaurentLS(fe.mesh.R, fe.mesh.theta_nodes, prm)
    far.fit('displacement', np.concatenate([u_r, u_t]))
    return (global_errors(fe, u, far, prm), far)
