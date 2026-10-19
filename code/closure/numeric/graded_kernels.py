from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from closure_env import bridge_pinned
cfg = bridge_pinned()
from anisotropic_elasticity.material import AnisoMaterial
from mapped_elasticity import protocol as P
from mapped_elasticity.geometry import SectionMap, horseshoe_map
from mapped_elasticity.sf_twoterm import modal_blocks_density
from mapped_elasticity.perturb import perturbation_kernels
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cauchy_constraints import lprime
from cauchy_evaluation import assemble_exact, branch_rho_crit, e_row_mirror, _wall_ring, _eval_series, HS1, N_H
FAILURES: list[str] = []
E03_C = np.array([0.0, 0.2, 0.15], dtype=complex)
PC2_C = np.array([0.0, 0.0, 0.15], dtype=complex)

def _report(label: str, ok: bool, detail: str) -> None:
    print(f"{('PASS' if ok else 'FAIL')}  {label}: {detail}")
    if not ok:
        FAILURES.append(label)

def velocity_series(R: float, C: np.ndarray, mu: complex) -> dict:
    """v_k = d/dt zk_series(t C)|_t: the finite-Laurent boundary
    velocity (zk_series of the full member minus the circle part)."""
    full = SectionMap(R, C).zk_series(mu)
    circ = SectionMap(R, [0.0]).zk_series(mu)
    v = {}
    for j in set(full) | set(circ):
        val = full.get(j, 0.0) - circ.get(j, 0.0)
        if val != 0.0:
            v[j] = val
    return v
_CXY = {'Uc': (0.5, 0.0), 'Us': (0.5j, 0.0), 'Vc': (0.0, 0.5), 'Vs': (0.0, 0.5j)}

def _branch_data(mat: AnisoMaterial, R: float, C: np.ndarray):
    out = []
    for k, mu in enumerate(mat.mus):
        bm = (1.0 - 1j * mu) / 2.0
        bp = (1.0 + 1j * mu) / 2.0
        out.append({'mu': mu, 'gam0': R * bm, 'qhat': bp / bm, 'lp': lprime(mat.beta, mu), 'v': velocity_series(R, C, mu)})
    return out

def _gdot_modes(br: dict, n: int, A_k: complex, n_out: int) -> np.ndarray:
    """(GDOT): gdot_k(m), m = 1..n_out (finite qhat ladder)."""
    g = np.zeros(n_out + 1, dtype=complex)
    for m in range(1, n_out + 1):
        s = 0.0
        j = 0
        while True:
            l = n + 1 + m + 2 * j
            vl = br['v'].get(l, 0.0)
            if l > max(br['v'], default=0):
                break
            s += br['qhat'] ** j * vl
            j += 1
        g[m] = -(n * A_k / br['gam0']) * s
    return g

def closed_S1(mat: AnisoMaterial, R: float, C: np.ndarray, n_h: int, conj_bug: bool=False, ladder_j0: bool=False) -> np.ndarray:
    """Literal S1 on the ray c(t) = t C."""
    Cv = np.asarray(C, dtype=complex)
    if conj_bug:
        Cv = np.conj(Cv)
    brs = _branch_data(mat, R, Cv)
    (p1, q1), (p2, q2) = mat.pq
    M = np.array([[p1, p2], [q1, q2]], dtype=complex)
    MT = M.T
    S1 = np.zeros((4 * n_h, 4 * n_h))
    for jin, (n, comp) in enumerate([(n, c) for n in range(1, n_h + 1) for c in ('Uc', 'Us', 'Vc', 'Vs')]):
        cx, cy = _CXY[comp]
        A = np.linalg.solve(M, np.array([cx, cy]))
        gd = []
        for k, br in enumerate(brs):
            g = _gdot_modes(br, n, A[k], n_h)
            if ladder_j0:
                g2 = np.zeros_like(g)
                for m in range(1, n_h + 1):
                    l = n + 1 + m
                    g2[m] = -(n * A[k] / br['gam0']) * br['v'].get(l, 0.0)
                g = g2
            gd.append(g)
        for npr in range(1, n_h + 1):
            rhs = np.array([brs[0]['lp'] * gd[0][npr], brs[1]['lp'] * gd[1][npr]])
            Xd, Yd = np.linalg.solve(MT, rhs)
            ctx = -1j * npr * np.conj(Xd)
            cty = -1j * npr * np.conj(Yd)
            i0 = 4 * (npr - 1)
            S1[i0 + 0, jin] = 2.0 * np.real(ctx)
            S1[i0 + 1, jin] = 2.0 * np.imag(ctx)
            S1[i0 + 2, jin] = 2.0 * np.real(cty)
            S1[i0 + 3, jin] = 2.0 * np.imag(cty)
    return S1

class PRep:
    """Polynomial in the {1, p_m} basis: coeffs[0] = constant slot,
    coeffs[m] = coefficient of p_m (branch-local; gamma, qhat
    fixed)."""

    def __init__(self, n_max: int):
        self.c = np.zeros(n_max + 1, dtype=complex)

    def add(self, other, fac=1.0):
        self.c += fac * other.c
        return self

def _rep_U(j: int, qhat: complex, n_max: int) -> PRep:
    """(QGEO): U_j = sum_i qhat^i p_{j-2i} (+ constant qhat^{j/2})."""
    r = PRep(n_max)
    if j < 0:
        return r
    i = 0
    while j - 2 * i >= 1:
        r.c[j - 2 * i] += qhat ** i
        i += 1
    if j % 2 == 0:
        r.c[0] += qhat ** (j // 2)
    return r

def _rep_Uprime(j: int, gam: complex, qhat: complex, n_max: int) -> PRep:
    """U'_j = (1/gamma) sum_i qhat^i (j - 2i) U_{j-2i-1}."""
    r = PRep(n_max)
    i = 0
    while j - 2 * i - 1 >= 0:
        fac = (j - 2 * i) / gam
        r.add(_rep_U(j - 2 * i - 1, qhat, n_max), fac * qhat ** i)
        i += 1
    return r

def _gdot_full_laurent(br: dict, gd_pos: np.ndarray, Xd: np.ndarray, Yd: np.ndarray, p_k: complex, q_k: complex, n_band: int) -> dict:
    """First-order trace gdot_fn as a Laurent dict: positive modes
    from (GDOT), zero parameter mean, negative modes reconstructed
    via (TDOT): gdot(-m) = [p conj(Xd(m)) + q conj(Yd(m))]/l'."""
    g = {}
    for m in range(1, n_band + 1):
        if m < len(gd_pos) and gd_pos[m] != 0.0:
            g[m] = gd_pos[m]
        gm = (p_k * np.conj(Xd[m - 1] if m - 1 < len(Xd) else 0.0) + q_k * np.conj(Yd[m - 1] if m - 1 < len(Yd) else 0.0)) / br['lp']
        if gm != 0.0:
            g[-m] = gm
    return g

def closed_S2(mat: AnisoMaterial, R: float, C: np.ndarray, n_h: int) -> np.ndarray:
    """Literal S2 on the ray (Section 6): source = 2 x Lemma A on
    gdot_fn + double-pole term via K2 = U'/gamma, expanded in the
    {1, p_m} basis, then (TDOT) on g2 and readout / 2."""
    Cv = np.asarray(C, dtype=complex)
    brs = _branch_data(mat, R, Cv)
    (p1, q1), (p2, q2) = mat.pq
    pq = ((p1, q1), (p2, q2))
    M = np.array([[p1, p2], [q1, q2]], dtype=complex)
    MT = M.T
    N_top = max((max((abs(j) for j in br['v'])) if br['v'] else 0 for br in brs))
    n_band = 2 * N_top + n_h + 2
    S2 = np.zeros((4 * n_h, 4 * n_h))
    for jin, (n, comp) in enumerate([(n, c) for n in range(1, n_h + 1) for c in ('Uc', 'Us', 'Vc', 'Vs')]):
        cx, cy = _CXY[comp]
        A = np.linalg.solve(M, np.array([cx, cy]))
        gjs = []
        for k, br in enumerate(brs):
            g = _gdot_modes(br, n, A[k], n_band)
            gjs.append(g)
        Xd = np.zeros(n_band, dtype=complex)
        Yd = np.zeros(n_band, dtype=complex)
        for m in range(1, n_band + 1):
            rhs = np.array([brs[0]['lp'] * gjs[0][m], brs[1]['lp'] * gjs[1][m]])
            Xd[m - 1], Yd[m - 1] = np.linalg.solve(MT, rhs)
        g2s = []
        for k, br in enumerate(brs):
            gam, qhat = (br['gam0'], br['qhat'])
            v = br['v']
            gfull = _gdot_full_laurent(br, gjs[k], Xd, Yd, *pq[k], n_band)
            conv = {}
            for mm, gv in gfull.items():
                if gv == 0.0 or mm == 0:
                    continue
                for l, vl in v.items():
                    jj = mm - 1 + l
                    conv[jj] = conv.get(jj, 0.0) + mm * gv * vl
            rep = PRep(n_band)
            for jj, w in conv.items():
                if jj >= 0 and w != 0.0:
                    rep.add(_rep_U(jj, qhat, n_band), 2.0 * w / gam)
            for l, vl in v.items():
                for lp_, vlp in v.items():
                    jj = -n + 1 + l + lp_
                    if jj >= 3 and vl != 0.0 and (vlp != 0.0):
                        rep.add(_rep_Uprime(jj - 2, gam, qhat, n_band), n * A[k] * vl * vlp / gam)
            g2s.append(rep)
        for npr in range(1, n_h + 1):
            rhs = np.array([brs[0]['lp'] * g2s[0].c[npr], brs[1]['lp'] * g2s[1].c[npr]])
            X2, Y2 = np.linalg.solve(MT, rhs)
            ctx = -1j * npr * np.conj(X2) / 2.0
            cty = -1j * npr * np.conj(Y2) / 2.0
            i0 = 4 * (npr - 1)
            S2[i0 + 0, jin] = 2.0 * np.real(ctx)
            S2[i0 + 1, jin] = 2.0 * np.imag(ctx)
            S2[i0 + 2, jin] = 2.0 * np.real(cty)
            S2[i0 + 3, jin] = 2.0 * np.imag(cty)
    return S2

def _S_of_t(mat, R, C, t: float, n_h: int, n_t: int) -> np.ndarray:
    return assemble_exact(mat, SectionMap(R, t * np.asarray(C)), n_h, n_t)['S']

def richardson_S1(mat, R, C, n_h: int, n_t: int=96, h: float=0.02) -> np.ndarray:
    D1 = (_S_of_t(mat, R, C, +h, n_h, n_t) - _S_of_t(mat, R, C, -h, n_h, n_t)) / (2 * h)
    D2 = (_S_of_t(mat, R, C, +h / 2, n_h, n_t) - _S_of_t(mat, R, C, -h / 2, n_h, n_t)) / h
    return (4.0 * D2 - D1) / 3.0

def richardson_S2(mat, R, C, n_h: int, n_t: int=96, h: float=0.05) -> np.ndarray:
    S0 = modal_blocks_density(mat, n_h)

    def D2(hh):
        return (_S_of_t(mat, R, C, +hh, n_h, n_t) + _S_of_t(mat, R, C, -hh, n_h, n_t) - 2.0 * S0) / (hh * hh)
    return (4.0 * D2(h / 2) - D2(h)) / 3.0 / 2.0

def check_kn1(mats) -> None:
    worst = 0.0
    for mat in (mats[1], mats[4]):
        mu = mat.mus[0]
        ser0 = SectionMap(P.R_HS, [0.0]).zk_series(mu)
        w0 = complex(_eval_series(ser0, 0.87 * np.exp(0.9j)))
        v = velocity_series(P.R_HS, P.C_HS, mu)
        h = 1e-05
        for t0 in (0.0, 0.1):
            for m in (-3, -1, 1, 2, 5):
                sp = horseshoe_map(t0 + h, P.R_HS, P.C_HS)
                sm = horseshoe_map(t0 - h, P.R_HS, P.C_HS)
                fd = (e_row_mirror(sp.zk_series(mu), w0, abs(m))[m] - e_row_mirror(sm.zk_series(mu), w0, abs(m))[m]) / (2 * h)
                st = horseshoe_map(t0, P.R_HS, P.C_HS)
                ser_t = st.zk_series(mu)
                n_g = 8192
                th = 2.0 * np.pi * np.arange(n_g) / n_g
                sig = np.exp(1j * th)
                om = np.zeros(n_g, dtype=complex)
                for j, a in ser_t.items():
                    if a != 0.0:
                        om += a * sig ** j
                vv = np.zeros(n_g, dtype=complex)
                for j, a in v.items():
                    vv += a * sig ** j
                fo = complex(np.mean(-m * sig ** (m - 1) * vv / (om - w0) * sig))
                worst = max(worst, abs(fd - fo))
    _report('KN1 Lemma-A derivative formula', worst <= 1e-08, f'formula vs central differences of exact E: max |dev| {worst:.3e} (<= 1e-8; F1/WAI branches, t0 = 0 and 0.1)')

def check_kn2(mats):
    f1 = mats[1]
    S1fd = richardson_S1(f1, P.R_HS, P.C_HS, 8)
    scale = float(np.max(np.abs(S1fd)))
    N = len(P.C_HS) - 1
    inb, outb = (0.0, 0.0)
    for n in range(1, 9):
        for npr in range(1, 9):
            blk = np.max(np.abs(S1fd[4 * (npr - 1):4 * npr, 4 * (n - 1):4 * n])) / scale
            if n + npr <= N - 1:
                inb = max(inb, blk)
            else:
                outb = max(outb, blk)
    _report('KN2 anti-triangular band of S1', inb >= 0.001 and outb <= 1e-09, f"HS(t=1), F1, n_h = 8: max rel block inside n + n' <= {N - 1}: {inb:.2e}; outside: {outb:.2e} (difference floor) — an anti-diagonal support, not a difference band")
    return S1fd

def check_kn3(mats) -> None:
    f1 = mats[1]
    S1_e03 = richardson_S1(f1, 2.0, E03_C, 4, n_t=48)
    s0 = float(np.max(np.abs(modal_blocks_density(f1, 4))))
    r_e03 = float(np.max(np.abs(S1_e03))) / s0
    S1_pc2 = richardson_S1(f1, 2.0, PC2_C, 4, n_t=48)
    r_pc2 = float(np.max(np.abs(S1_pc2))) / s0
    S2_pc2 = richardson_S2(f1, 2.0, PC2_C, 4, n_t=48)
    r2_pc2 = float(np.max(np.abs(S2_pc2))) / s0
    C_mod = np.array(P.C_HS, dtype=complex)
    C_mod[2] += 0.05 + 0.03j
    d_c2 = float(np.max(np.abs(closed_S1(f1, P.R_HS, P.C_HS, 6) - closed_S1(f1, P.R_HS, C_mod, 6))))
    _report('KN3 first-order silences', r_e03 <= 1e-09 and r_pc2 <= 1e-09 and (r2_pc2 >= 0.0001) and (d_c2 == 0.0), f'E0.3 ray |S1|/|S0| = {r_e03:.2e}, pure-c2 ray {r_pc2:.2e} (both at the difference floor: first-order invariant); pure-c2 |S2|/|S0| = {r2_pc2:.2e} != 0 (defect starts at t^2); closed S1 c_2-shift = {d_c2:.1e} (exactly independent of c_2)')

def check_kn4(mats, S1fd_f1) -> None:
    worst = 0.0
    for mat in mats:
        S1fd = S1fd_f1 if mat.name == mats[1].name else richardson_S1(mat, P.R_HS, P.C_HS, 8)
        S1c = closed_S1(mat, P.R_HS, P.C_HS, 8)
        worst = max(worst, float(np.max(np.abs(S1c - S1fd)) / np.max(np.abs(S1fd))))
    ls_rows = []
    ok_ls = True
    for mat in (mats[0], mats[1], mats[4]):
        _, S1_ls, _ = perturbation_kernels(mat, P.R_HS, P.C_HS, 1.0, 6, 96)
        S1c = closed_S1(mat, P.R_HS, P.C_HS, 6)
        d = float(np.linalg.norm(S1c - S1_ls) / np.linalg.norm(S1c))
        ls_rows.append(f'{mat.name}: {d:.1e}')
        ok_ls &= d <= 0.05
    _report('KN4 closed S1 locks', worst <= 1e-06 and ok_ls, f'vs exact-assembly Richardson: max rel dev {worst:.3e} (<= 1e-6, five materials, n_h = 8); vs pinned LS kernels (fit tier): ' + ', '.join(ls_rows))

def check_kn5(mats) -> None:
    f1 = mats[1]
    worst = 0.0
    for R, C, n_t in ((P.R_HS, P.C_HS, 96), (2.0, PC2_C, 48)):
        S2fd = richardson_S2(f1, R, C, 4, n_t=n_t)
        S2c = closed_S2(f1, R, C, 4)
        worst = max(worst, float(np.max(np.abs(S2c - S2fd)) / max(np.max(np.abs(S2fd)), 1e-300)))
    t_star = None
    for t in np.arange(0.1, 3.01, 0.1):
        rcs = [branch_rho_crit(SectionMap(P.R_HS, t * np.asarray(P.C_HS)).zk_series(mu)) for mat in mats for mu in mat.mus]
        if max(rcs) >= 0.97:
            t_star = t
            break
    _report('KN5 closed S2 locks + radius probe', worst <= 0.001 and (t_star is None or t_star > 1.0), f"closed S2 vs second-Richardson: max rel dev {worst:.3e} (<= 1e-3; HS + pure-c2 rays); fold shells reach 0.97 first at t* = {t_star} (> 1: the pinned t-ladder range is inside the graded expansion's real-axis validity)")

def negative_controls(mats, S1fd_f1) -> None:
    f1 = mats[1]
    C_cx = np.asarray(P.C_HS, dtype=complex) * (1.0 + 0.25j)
    S1fd_cx = richardson_S1(f1, P.R_HS, C_cx, 6)
    S1c_cx = closed_S1(f1, P.R_HS, C_cx, 6)
    d_ok = float(np.max(np.abs(S1c_cx - S1fd_cx)) / np.max(np.abs(S1fd_cx)))
    S1_bug = closed_S1(f1, P.R_HS, C_cx, 6, conj_bug=True)
    d1 = float(np.max(np.abs(S1_bug - S1fd_cx)) / np.max(np.abs(S1fd_cx)))
    _report('Negative control (conj(c) -> c corruption)', d_ok <= 1e-06 and d1 >= 0.01, f'complex ray: closed S1 matches Richardson at {d_ok:.2e} while the corrupted (GDOT) deviates {d1:.2e} — the conjugate channel is required')
    S1_j0 = closed_S1(f1, P.R_HS, P.C_HS, 8, ladder_j0=True)
    d2 = float(np.max(np.abs(S1_j0 - S1fd_f1)) / np.max(np.abs(S1fd_f1)))
    _report('Negative control (qhat ladder truncated)', d2 >= 0.001, f'j = 0 term only vs Richardson: rel dev {d2:.2e} — the geometric ladder tail is required')
    scale = float(np.max(np.abs(S1fd_f1)))
    N = len(P.C_HS) - 1
    occ = max((np.max(np.abs(S1fd_f1[4 * (npr - 1):4 * npr, 4 * (n - 1):4 * n])) / scale for n in range(1, 9) for npr in range(1, 9) if n + npr == N - 1))
    _report('Negative control (anti-band moved to s = N - 1)', occ > 1e-09, f'claiming vanishing on the occupied anti-diagonal s = {N - 1} fails: max rel block {occ:.2e}')

def main() -> int:
    mats = P.materials_main()
    print('materials:', ', '.join((m.name for m in mats)))
    print('rays     : HS (N = 9), E0.3 c=[0,.2,.15] (N = 2), pure-c2 c=[0,0,.15] (N = 2)')
    print()
    check_kn1(mats)
    S1fd_f1 = check_kn2(mats)
    check_kn3(mats)
    check_kn4(mats, S1fd_f1)
    check_kn5(mats)
    print()
    negative_controls(mats, S1fd_f1)
    print()
    if FAILURES:
        print(f'Graded kernels checks failed: {FAILURES}')
        return 1
    print('Graded kernels checks passed: the graded kernels are closed-form —')
    print('  S1 from the conjugate-channel geometric ladder with the')
    print("  anti-triangular band n + n' <= N - 1 (c_2 silent at")
    print('  first order, E0.5 explanandum (b) resolved), S2 from')
    print("  the U'-ladder with the quadratic channels; both lock")
    print('  against exact-assembly differences, and the pinned LS')
    print('  kernel is reproduced at its own fit tier; controls fire.')
    return 0
if __name__ == '__main__':
    sys.exit(main())
