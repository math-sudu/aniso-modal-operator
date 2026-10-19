from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from closure_env import bridge_pinned
cfg = bridge_pinned()
from mapped_elasticity import protocol as P
from mapped_elasticity.geometry import SectionMap, horseshoe_map
from mapped_elasticity.sf_twoterm import modal_blocks_density
FAILURES: list[str] = []
CIRC = SectionMap(1.0, [0.0])
ELL = SectionMap(1.0, [0.0, P.ELL_C1])
CPLX = SectionMap(2.0, [0.3 - 0.1j, 0.2 + 0.15j])
E03 = SectionMap(2.0, [0.0, 0.2, 0.15])
HS1 = horseshoe_map(1.0, P.R_HS, P.C_HS)
TWOTERM = (('CIRC', CIRC), ('ELL', ELL), ('CPLX', CPLX))
GENERAL = (('E03', E03), ('HS1', HS1))

def _report(label: str, ok: bool, detail: str) -> None:
    print(f"{('PASS' if ok else 'FAIL')}  {label}: {detail}")
    if not ok:
        FAILURES.append(label)

def lprime(beta: dict, mu: complex) -> complex:
    return 4.0 * beta['11'] * mu ** 3 - 6.0 * beta['16'] * mu ** 2 + 2.0 * (2.0 * beta['12'] + beta['66']) * mu - 2.0 * beta['26']

def _NP_of(ser: dict[int, complex]) -> tuple[int, int]:
    N = max((-j for j, a in ser.items() if j < 0 and a != 0.0), default=0)
    Np = max((j for j, a in ser.items() if j > 0 and a != 0.0), default=0)
    return (N, Np)

def _poly_PQ(ser: dict[int, complex], w0: complex):
    """Ascending coefficients of P = sigma^N (omega_k - w0) and
    Q = sigma^{N+1} omega_k'."""
    N, Np = _NP_of(ser)
    Pc = np.zeros(N + Np + 1, dtype=complex)
    Qc = np.zeros(N + Np + 1, dtype=complex)
    for j, a in ser.items():
        if a == 0.0:
            continue
        Pc[j + N] += a
        if j != 0:
            Qc[j + N] += j * a
    Pc[N] -= w0
    return (Pc, Qc, N, Np)

def _taylor_div(Qc: np.ndarray, Pc: np.ndarray, n_terms: int) -> np.ndarray:
    """Taylor coefficients of Q/P at sigma = 0 (P[0] != 0)."""
    t = np.zeros(n_terms, dtype=complex)
    for i in range(n_terms):
        s = Qc[i] if i < len(Qc) else 0.0
        for l in range(1, min(i, len(Pc) - 1) + 1):
            s -= Pc[l] * t[i - l]
        t[i] = s / Pc[0]
    return t

def _corr(ser: dict, Pc: np.ndarray, Qc: np.ndarray, m: int) -> complex:
    """sigma = 0 residue correction r(m; w0) (zero for m >= 1)."""
    if m >= 1:
        return 0.0
    return _taylor_div(Qc, Pc, abs(m) + 1)[abs(m)]

def e_entry_root(ser: dict, m: int, w0: complex):
    """E(m; w0) by interior preimage-root power sums; returns
    (value, n_inside, N, Np, scale) — scale is the cancellation
    magnitude sum(|sigma_r^m|) + |r(m; w0)| (floored at 1), the
    natural yardstick for the structural cancellations at deep
    negative m."""
    Pc, Qc, N, Np = _poly_PQ(ser, w0)
    roots = np.roots(Pc[::-1])
    inside = roots[np.abs(roots) < 1.0]
    ps = inside ** m if inside.size else np.zeros(0, dtype=complex)
    corr = _corr(ser, Pc, Qc, m)
    val = complex(np.sum(ps) + corr)
    scale = max(1.0, float(np.sum(np.abs(ps))) + abs(corr))
    return (val, int(inside.size), N, Np, scale)

def _newton_psums(monic_desc: np.ndarray, mmax: int) -> np.ndarray:
    """Power sums p_1..p_mmax of a monic polynomial with descending
    coefficients [1, c1, ..., cd] (Newton identities, no roots)."""
    c = monic_desc
    d = len(c) - 1
    p = np.zeros(mmax + 1, dtype=complex)
    for i in range(1, mmax + 1):
        s = -i * c[i] if i <= d else 0.0
        for l in range(1, min(i - 1, d) + 1):
            s -= c[l] * p[i - l]
        p[i] = s
    return p[1:]

def e_entry_newton(ser: dict, m: int, w0: complex) -> complex:
    """E(m; w0) by FULL-root Newton recurrences — valid only when
    Np <= 1 (all roots interior; two-term / isotropic tiers)."""
    Pc, Qc, N, Np = _poly_PQ(ser, w0)
    if Np > 1:
        raise ValueError('Newton path is the Np <= 1 evaluation')
    d = len(Pc) - 1
    if m == 0:
        val = complex(d)
    elif m >= 1:
        val = _newton_psums(Pc[::-1] / Pc[-1], m)[m - 1]
    else:
        val = _newton_psums(Pc / Pc[0], -m)[-m - 1]
    return val + _corr(ser, Pc, Qc, m)

def e_entry_mirror(ser: dict, m: int, w0: complex) -> complex:
    """Evaluate E(m; w0) using the mirror-residue identity.

For positive m, sum the m-th powers of interior roots; for m = 0, return the winding number one; for negative m, negate the sum over exterior roots. The negative-mode expression follows from vanishing residue at infinity and avoids cancellation of large terms. The two-term map has no exterior roots, so its negative entries vanish."""
    if m == 0:
        return 1.0
    Pc, _, _, _ = _poly_PQ(ser, w0)
    roots = np.roots(Pc[::-1])
    if m >= 1:
        inside = roots[np.abs(roots) < 1.0]
        return complex(np.sum(inside ** m)) if inside.size else 0.0
    outside = roots[np.abs(roots) >= 1.0]
    return complex(-np.sum(outside ** m)) if outside.size else 0.0

def e_entry_fft(ser: dict, m: int, w0: complex, n_grid: int=8192, slope_drop: float=0.0) -> complex:
    """(1/2 pi i) oint sigma^m omega_k'/(omega_k - w0) dsigma by
    trapezoid FFT quadrature; slope_drop corrupts the kernel
    derivative."""
    th = 2.0 * np.pi * np.arange(n_grid) / n_grid
    sig = np.exp(1j * th)
    om = np.zeros(n_grid, dtype=complex)
    dom = np.zeros(n_grid, dtype=complex)
    for j, a in ser.items():
        om += a * sig ** j
        if j != 0:
            dom += j * a * sig ** (j - 1)
    if slope_drop:
        dom = dom - slope_drop * ser[1]
    f = sig ** m * dom / (om - w0)
    return complex(np.mean(f * sig))

def _test_ring(ser: dict, n_pts: int, fac: float) -> np.ndarray:
    a0 = ser.get(0, 0.0)
    a1 = ser[1]
    ang = 2.0 * np.pi * (np.arange(n_pts) + 0.31) / n_pts
    return a0 + fac * abs(a1) * np.exp(1j * ang)

def check_kernel_entries(mats) -> None:
    worst2, worstg = (0.0, 0.0)
    ok_count = True
    for mat in mats:
        for mu in mat.mus:
            for name, smap in TWOTERM + GENERAL:
                two = smap.n_map <= 1
                ser = smap.zk_series(mu)
                fac = 0.25 if two else 0.15
                for w0 in _test_ring(ser, 3, fac):
                    for m in range(-6, 7):
                        er, n_in, N, Np, sc = e_entry_root(ser, m, w0)
                        ef = e_entry_fft(ser, m, w0)
                        ok_count &= n_in == N + 1
                        if m == 0:
                            ok_count &= abs(er - 1.0) <= 1e-10
                        if two:
                            en = e_entry_newton(ser, m, w0)
                            worst2 = max(worst2, abs(er - ef) / sc, abs(er - en) / sc, abs(en - ef) / sc)
                        else:
                            worstg = max(worstg, abs(er - ef) / sc)
    _report('kernel-entry agreement', worst2 <= 1e-13 and worstg <= 1e-11 and ok_count, f'two-term three-way max rel dev {worst2:.3e} (<= 1e-13); general root-vs-FFT max rel dev {worstg:.3e} (<= 1e-11); interior count == N + 1 and E(0) == 1 everywhere')

def _neg_col_max(smap: SectionMap, mu: complex, fac: float=0.25, mmin: int=-6) -> float:
    """max |E(m; w0)| / scale over the negative modes (relative to
    the cancellation magnitude)."""
    ser = smap.zk_series(mu)
    worst = 0.0
    for w0 in _test_ring(ser, 5, fac):
        for m in range(mmin, 0):
            er, _, _, _, sc = e_entry_root(ser, m, w0)
            worst = max(worst, abs(er) / sc)
    return worst

def check_two_term_columns(mats) -> None:
    worst = 0.0
    for mat in mats:
        for mu in mat.mus:
            for name, smap in TWOTERM:
                worst = max(worst, _neg_col_max(smap, mu))
    _report('two-term negative columns vanish', worst <= 1e-13, f'max |E(m; w0)|/scale, m = -1..-6, over CIRC/ELL/CPLX x 5 materials x branches: {worst:.3e} (Riesz degeneration)')

def project_constant_rows(values: np.ndarray) -> np.ndarray:
    """Orthonormal Helmert projection of sampled rows modulo constants.

    For L sample points this returns L-1 rows. Its null space is exactly
    the constant vector, so it eliminates a free interior Cauchy value.
    """
    values = np.asarray(values)
    n = np.arange(1, len(values), dtype=float)
    shape = (len(n),) + (1,) * (values.ndim - 1)
    n = n.reshape(shape)
    return (np.cumsum(values, axis=0)[:-1] - n * values[1:]) / np.sqrt(n * (n + 1))

def assemble_p20(mat, smap: SectionMap, n_h: int, ring_fac: float=0.35, flip_uy: bool=False) -> np.ndarray:
    """Exact-constraint DtN assembly on a two-term member: unknowns
    (X_hat(n), Y_hat(n)) n = 1..n_h; n_h+1 interior samples per
    branch modulo constants; square LU; density in the e01 packing."""
    sgn_uy = 1.0 if flip_uy else -1.0
    mus = mat.mus
    pqs = mat.pq
    lps = [lprime(mat.beta, mu) for mu in mus]
    sers = [smap.zk_series(mu) for mu in mus]
    rings = [_test_ring(ser, n_h + 1, ring_fac) for ser in sers]
    Et = {}
    for k in range(2):
        for m in list(range(-n_h, 0)) + list(range(1, n_h + 1)):
            projected = project_constant_rows(np.array([e_entry_mirror(sers[k], m, w0) for w0 in rings[k]]))
            for i, value in enumerate(projected):
                Et[k, i, m] = value
    nunk = 4 * n_h

    def col(n, comp):
        return 4 * (n - 1) + {'xr': 0, 'xi': 1, 'yr': 2, 'ys': 3}[comp]
    A = np.zeros((nunk, nunk))
    loads = {}
    for k in range(2):
        mu, (p, q), lp = (mus[k], pqs[k], lps[k])
        for i in range(n_h):
            r = 2 * (k * n_h + i)
            for n in range(1, n_h + 1):
                Ep, Em = (Et[k, i, n], Et[k, i, -n])
                for cc, cr, ci in (('X', 'xr', 'xi'), ('Y', 'yr', 'ys')):
                    coef = p if cc == 'X' else q
                    al = Ep * coef / lp
                    be = Em * coef / lp
                    A[r, col(n, cr)] += np.real(al + be)
                    A[r, col(n, ci)] += -np.imag(al - be)
                    A[r + 1, col(n, cr)] += np.imag(al + be)
                    A[r + 1, col(n, ci)] += np.real(al - be)
    S = np.zeros((4 * n_h, 4 * n_h))
    cxy = {'Uc': (0.5, 0.0), 'Us': (0.5j, 0.0), 'Vc': (0.0, 0.5), 'Vs': (0.0, 0.5j)}
    for jin, (n_in, comp) in enumerate([(n, c) for n in range(1, n_h + 1) for c in ('Uc', 'Us', 'Vc', 'Vs')]):
        cx, cy = cxy[comp]
        b = np.zeros(nunk)
        for k in range(2):
            mu, lp = (mus[k], lps[k])
            for i in range(n_h):
                r = 2 * (k * n_h + i)
                Ep, Em = (Et[k, i, n_in], Et[k, i, -n_in])
                load = Ep * (mu * np.conj(cx) + sgn_uy * np.conj(cy)) / lp + Em * (mu * cx + sgn_uy * cy) / lp
                b[r] -= np.real(load)
                b[r + 1] -= np.imag(load)
        v = np.linalg.solve(A, b)
        for n in range(1, n_h + 1):
            Xh = v[col(n, 'xr')] + 1j * v[col(n, 'xi')]
            Yh = v[col(n, 'yr')] + 1j * v[col(n, 'ys')]
            ctx = -1j * n * np.conj(Xh)
            cty = -1j * n * np.conj(Yh)
            i0 = 4 * (n - 1)
            S[i0 + 0, jin] = 2.0 * np.real(ctx)
            S[i0 + 1, jin] = 2.0 * np.imag(ctx)
            S[i0 + 2, jin] = 2.0 * np.real(cty)
            S[i0 + 3, jin] = 2.0 * np.imag(cty)
    return S

def check_exact_assembly(mats, n_h: int=4) -> None:
    worst, spread = (0.0, 0.0)
    for mat in mats:
        S0 = modal_blocks_density(mat, n_h)
        scale = float(np.max(np.abs(S0)))
        Ss = []
        for name, smap in TWOTERM:
            S = assemble_p20(mat, smap, n_h)
            Ss.append(S)
            worst = max(worst, float(np.max(np.abs(S - S0))) / scale)
        for S in Ss[1:]:
            spread = max(spread, float(np.max(np.abs(S - Ss[0]))) / scale)
    _report('gate: exact assembly == pinned closed form', worst <= 5e-12 and spread <= 5e-12, f'max rel dev vs modal_blocks_density over CIRC/ELL/CPLX x 5 materials (n_h = {n_h}): {worst:.3e}; cross-geometry spread {spread:.3e} — invariance recovered instantly')

def check_higher_order_mapping(mats) -> None:
    f1 = mats[1]
    nz_e03 = _neg_col_max(E03, f1.mus[0], fac=0.15, mmin=-3)
    nz_hs = max((_neg_col_max(HS1, mu, fac=0.15, mmin=-3) for mat in mats for mu in mat.mus))
    iso = max(_neg_col_max(E03, 1j, fac=0.15, mmin=-3), _neg_col_max(HS1, 1j, fac=0.15, mmin=-3))
    ser = E03.zk_series(f1.mus[0])
    Pc, _, N, Np = _poly_PQ(ser, complex(_test_ring(ser, 1, 0.15)[0]))
    n_ext = int(np.sum(np.abs(np.roots(Pc[::-1])) >= 1.0))
    _report('non-degeneration on j >= 2 carriers', nz_e03 > 0.0001 and nz_hs > 0.0001 and (iso <= 1e-12) and (n_ext == Np - 1), f'E0.3 F1-branch max |E(m<0)| = {nz_e03:.3e}, HS(t=1) {nz_hs:.3e} (nonzero: higher-order mapping correction); isotropic branch on the same carriers {iso:.3e} (escape route); exterior roots {n_ext} == Np - 1 = {Np - 1}')

def negative_controls(mats) -> None:
    f2 = mats[2]
    ser = ELL.zk_series(f2.mus[0])
    w0 = complex(_test_ring(ser, 1, 0.25)[0])
    er = e_entry_root(ser, 2, w0)[0]
    ef_bad = e_entry_fft(ser, 2, w0, slope_drop=0.5)
    _report('Negative control (corrupted kernel derivative)', abs(er - ef_bad) > 0.001, f'half the a_1 slope dropped: |root - FFT| = {abs(er - ef_bad):.3e} > 1e-3')
    S0 = modal_blocks_density(f2, 3)
    S_bad = assemble_p20(f2, ELL, 3, flip_uy=True)
    dev = float(np.max(np.abs(S_bad - S0))) / float(np.max(np.abs(S0)))
    _report('Negative control (flipped u_y RECON row)', dev > 0.01, f'gate deviation {dev:.3e} > 1e-2: the -u_y sign is required')
    nz = _neg_col_max(E03, mats[1].mus[0], fac=0.15, mmin=-3)
    _report('Negative control (two-term assertion on E0.3)', nz > 0.0001, f'The two-term vanishing bound fails on the three-term carrier: max |E(m<0)| = {nz:.3e}')

def main() -> int:
    mats = P.materials_main()
    print('materials:', ', '.join((m.name for m in mats)))
    print('carriers : CIRC, ELL, CPLX (two-term), E0.3 c=[0,.2,.15], HS(t=1) ten-term')
    print()
    check_kernel_entries(mats)
    check_two_term_columns(mats)
    check_exact_assembly(mats)
    check_higher_order_mapping(mats)
    print()
    negative_controls(mats)
    print()
    if FAILURES:
        print(f'Constraint formulation checks failed: {FAILURES}')
        return 1
    print('Constraint formulation checks passed: kernel entries agree three ways at')
    print('  machine tier; two-term negative columns vanish; the exact')
    print('  square assembly reproduces the pinned closed form with')
    print('  zero cross-geometry spread (the gate); j >= 2 carriers do')
    print('  not degenerate while the isotropic branch does; controls')
    print('  fire.')
    return 0
if __name__ == '__main__':
    sys.exit(main())
