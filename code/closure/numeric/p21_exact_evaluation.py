from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from closure_env import bridge_pinned
cfg = bridge_pinned()
from exec26_src import protocol as P
from exec26_src.geometry import SectionMap, horseshoe_map
from exec26_src.sf_twoterm import modal_blocks_density
from exec26_src.sf_general import assemble_modal_dtn
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p20_exact_constraint_formulation import e_entry_root, e_entry_fft, e_entry_mirror, lprime, _poly_PQ, _NP_of, project_constant_rows
FAILURES: list[str] = []
CIRC = SectionMap(1.0, [0.0])
ELL = SectionMap(1.0, [0.0, P.ELL_C1])
CPLX = SectionMap(2.0, [0.3 - 0.1j, 0.2 + 0.15j])
E03 = SectionMap(2.0, [0.0, 0.2, 0.15])
HS1 = horseshoe_map(1.0, P.R_HS, P.C_HS)
N_H = 4
LADDER = (8, 16, 24, 32, 64)

def _report(label: str, ok: bool, detail: str) -> None:
    print(f"{('PASS' if ok else 'FAIL')}  {label}: {detail}")
    if not ok:
        FAILURES.append(label)

def branch_rho_crit(ser: dict) -> float:
    """Return the maximum modulus of nonzero interior zeros of omega_k'."""
    N, Np = _NP_of(ser)
    Qc = np.zeros(N + Np + 1, dtype=complex)
    for j, a in ser.items():
        if j != 0 and a != 0.0:
            Qc[j + N] += j * a
    roots = np.roots(Qc[::-1])
    inside = np.abs(roots)[np.abs(roots) < 1.0]
    return float(inside.max()) if inside.size else 0.0

def cauchy_radius(ser: dict, override: float | None=None, minimum: float=0.85) -> float:
    """Choose a test radius strictly between the branch fold shell and one.

    The image points still require the winding check in e_row_mirror.
    """
    critical = branch_rho_crit(ser)
    if override is None:
        eta = max(minimum, critical + 0.02) if critical + 0.02 < 1.0 else (1.0 + critical) / 2.0
    else:
        eta = float(override)
    if not critical < eta < 1.0:
        raise ValueError(f'Cauchy radius must satisfy {critical:.16g} < eta < 1; received {eta:.16g}')
    return eta

def _eval_series(ser: dict, zeta: np.ndarray) -> np.ndarray:
    out = np.zeros_like(np.asarray(zeta, dtype=complex))
    for j, a in ser.items():
        if a != 0.0:
            out = out + a * np.asarray(zeta, dtype=complex) ** j
    return out

def _wall_ring(ser: dict, n_pts: int, eta: float, offset: float=0.31) -> np.ndarray:
    """Interior points hugging the wall: w0 = omega_k(eta sigma) on
    the eta ring of the Laurent extension (inward at first order);
    interiorness certified by the winding anchor at use sites."""
    ang = 2.0 * np.pi * (np.arange(n_pts) + offset) / n_pts
    return _eval_series(ser, eta * np.exp(1j * ang))

def e_row_mirror(ser: dict, w0: complex, mmax: int):
    """E(m; w0) for all m in [-mmax..mmax] with ONE root solve
    (mirror residue identity). Reject points outside the branch interior
    and preimages numerically on the unit circle before using the sums.
    """
    Pc, _, N, Np = _poly_PQ(ser, w0)
    roots = np.roots(Pc[::-1])
    radii = np.abs(roots)
    if np.any(np.abs(radii - 1.0) <= 64 * np.finfo(float).eps):
        raise ValueError('Cauchy point has a preimage numerically on the unit circle')
    inside = roots[radii < 1.0]
    outside = roots[radii > 1.0]
    if len(inside) - N != 1:
        raise ValueError(f'Cauchy point is outside the branch interior: winding number {len(inside) - N}, expected 1')
    out = {0: complex(len(inside) - N)}
    for m in range(1, mmax + 1):
        out[m] = complex(np.sum(inside ** m)) if inside.size else 0.0
        out[-m] = complex(-np.sum(outside ** (-m))) if outside.size else 0.0
    return out

def require_simple_roots(mat) -> None:
    """Reject overlapping root-uncertainty discs before dividing by ell'.

    The disc radius is the first-order polynomial residual bound divided
    by |ell'|. The 32-operation roundoff bound includes complex quartic
    evaluation and the stored coefficient precision.
    """
    b = mat.beta
    coefficients = np.array([b['11'], -2 * b['16'], 2 * b['12'] + b['66'], -2 * b['26'], b['22']])
    eps = np.finfo(float).eps
    gamma = 32 * eps / (1 - 32 * eps)
    radii = []
    for mu in mat.mus:
        derivative = abs(lprime(b, mu))
        bound = abs(np.polyval(coefficients, mu)) + gamma * np.polyval(abs(coefficients), abs(mu))
        radii.append(bound / derivative if derivative else np.inf)
    if abs(mat.mus[0] - mat.mus[1]) <= sum(radii):
        raise ValueError('General-map Cauchy assembly requires numerically separated simple characteristic roots; the root uncertainty discs overlap. A confluent general-map construction is required for this material.')

def assemble_exact(mat, smap: SectionMap, n_h: int, n_t: int, eta_override: float | None=None, conj_channel_bug: bool=False, radius_floor: float=0.85):
    """Assemble the Cauchy-constraint DtN operator.

Project n_t+1 samples per branch modulo constants, leaving 4*n_t real equations for the nonzero resultant harmonics. Boundary displacement and resultant have zero parameter mean. constants[(n_in, comp)] holds the potential values at infinity. Invalid radii or image points raise ValueError before the solve. Return S, cond, sols, constants, sers, lps and etas."""
    if not 1 <= n_h <= n_t:
        raise ValueError('Require 1 <= n_h <= n_t for the retained harmonic band')
    require_simple_roots(mat)
    mus = mat.mus
    pqs = mat.pq
    lps = [lprime(mat.beta, mu) for mu in mus]
    sers = [smap.zk_series(mu) for mu in mus]
    etas = [cauchy_radius(ser, eta_override, radius_floor) for ser in sers]
    rings = [_wall_ring(ser, n_t + 1, eta) for ser, eta in zip(sers, etas)]
    Et = {}
    means = []
    for k in range(2):
        rows = []
        for i, w0 in enumerate(rings[k]):
            row = e_row_mirror(sers[k], w0, n_t)
            rows.append([row[m] for m in range(-n_t, n_t + 1)])
        entries = np.asarray(rows)
        means.append(dict(zip(range(-n_t, n_t + 1), entries.mean(axis=0))))
        for i, projected in enumerate(project_constant_rows(entries)):
            Et[k, i] = dict(zip(range(-n_t, n_t + 1), projected))
    nunk = 4 * n_t

    def col(n, comp):
        return 4 * (n - 1) + {'xr': 0, 'xi': 1, 'yr': 2, 'ys': 3}[comp]
    A = np.zeros((nunk, nunk))
    for k in range(2):
        mu, (p, q), lp = (mus[k], pqs[k], lps[k])
        for i in range(n_t):
            r = 2 * (k * n_t + i)
            row = Et[k, i]
            for n in range(1, n_t + 1):
                Ep, Em = (row[n], row[-n])
                for cc, cr, ci in (('X', 'xr', 'xi'), ('Y', 'yr', 'ys')):
                    coef = p if cc == 'X' else q
                    al = Ep * coef / lp
                    be = Em * (np.conj(coef / lp) if conj_channel_bug else coef / lp)
                    A[r, col(n, cr)] += np.real(al + be)
                    A[r, col(n, ci)] += -np.imag(al - be)
                    A[r + 1, col(n, cr)] += np.imag(al + be)
                    A[r + 1, col(n, ci)] += np.real(al - be)
    eta_s = float(np.mean(etas))
    s = np.array([eta_s ** (-(1 + j // 4)) for j in range(nunk)])
    A_s = A * s[None, :]
    cond = float(np.linalg.cond(A_s))
    basis = [(n, c) for n in range(1, n_h + 1) for c in ('Uc', 'Us', 'Vc', 'Vs')]
    cxy = {'Uc': (0.5, 0.0), 'Us': (0.5j, 0.0), 'Vc': (0.0, 0.5), 'Vs': (0.0, 0.5j)}
    B = np.zeros((nunk, len(basis)))
    for jin, (n_in, comp) in enumerate(basis):
        cx, cy = cxy[comp]
        for k in range(2):
            mu, lp = (mus[k], lps[k])
            for i in range(n_t):
                r = 2 * (k * n_t + i)
                row = Et[k, i]
                Ep, Em = (row[n_in], row[-n_in])
                load = Ep * (mu * np.conj(cx) - np.conj(cy)) / lp + Em * (mu * cx - cy) / lp
                B[r, jin] -= np.real(load)
                B[r + 1, jin] -= np.imag(load)
    V = s[:, None] * np.linalg.solve(A_s, B)
    S = np.zeros((4 * n_h, 4 * n_h))
    sols = {}
    constants = {}
    for jin, (n_in, comp) in enumerate(basis):
        v = V[:, jin]
        Xh = np.array([v[col(n, 'xr')] + 1j * v[col(n, 'xi')] for n in range(1, n_t + 1)])
        Yh = np.array([v[col(n, 'yr')] + 1j * v[col(n, 'ys')] for n in range(1, n_t + 1)])
        sols[n_in, comp] = (Xh, Yh)
        constants[n_in, comp] = tuple((sum((means[k][m] * gm for m, gm in potential_modes(mat, lps, n_in, comp, Xh, Yh, k).items())) for k in range(2)))
        for n in range(1, n_h + 1):
            ctx = -1j * n * np.conj(Xh[n - 1])
            cty = -1j * n * np.conj(Yh[n - 1])
            i0 = 4 * (n - 1)
            S[i0 + 0, jin] = 2.0 * np.real(ctx)
            S[i0 + 1, jin] = 2.0 * np.imag(ctx)
            S[i0 + 2, jin] = 2.0 * np.real(cty)
            S[i0 + 3, jin] = 2.0 * np.imag(cty)
    return {'S': S, 'cond': cond, 'sols': sols, 'sers': sers, 'lps': lps, 'etas': etas, 'mat': mat, 'constants': constants}

def potential_modes(mat, lps, n_in, comp, Xh, Yh, k):
    """Reconstruct the zero-mean potential trace from real boundary data."""
    cx, cy = {'Uc': (0.5, 0.0), 'Us': (0.5j, 0.0), 'Vc': (0.0, 0.5), 'Vs': (0.0, 0.5j)}[comp]
    mu, (p, q), lp = (mat.mus[k], mat.pq[k], lps[k])
    gm = {}
    for n in range(1, len(Xh) + 1):
        ux = np.conj(cx) if n == n_in else 0.0
        uy = np.conj(cy) if n == n_in else 0.0
        gm[n] = (mu * ux - uy + p * Xh[n - 1] + q * Yh[n - 1]) / lp
        gm[-n] = (mu * np.conj(ux) - np.conj(uy) + p * np.conj(Xh[n - 1]) + q * np.conj(Yh[n - 1])) / lp
    return gm

def heldout_certificate(out: dict, n_pts: int=24) -> float:
    """Independent-ring residual C[g]-a, normalized by the prescribed datum.

    The constant a is recovered on the assembly ring, never refitted on
    the validation ring. The scale is the circular potential amplitude of
    the input, independent of solved coefficients and truncation.
    Every validation point must have winding one.
    """
    mat = out['mat']
    cxy = {'Uc': (0.5, 0.0), 'Us': (0.5j, 0.0), 'Vc': (0.0, 0.5), 'Vs': (0.0, 0.5j)}
    worst = 0.0
    n_used = 0
    validation_rows = {}
    for (n_in, comp), (Xh, Yh) in out['sols'].items():
        cx, cy = cxy[comp]
        n_t = len(Xh)
        M = np.asarray(mat.pq, dtype=complex).T
        scale = max(float(np.max(np.abs(np.linalg.solve(M, [cx, cy])))), 1e-300)
        for k in range(2):
            ser = out['sers'][k]
            gm = potential_modes(mat, out['lps'], n_in, comp, Xh, Yh, k)
            constant = out['constants'][n_in, comp][k]
            if (k, n_t) not in validation_rows:
                critical = branch_rho_crit(ser)
                eta_h = min(0.97, max(0.85, critical) + 0.06)
                if eta_h <= critical:
                    eta_h = (1.0 + critical) / 2.0
                eta_h = cauchy_radius(ser, eta_h)
                validation_rows[k, n_t] = [e_row_mirror(ser, w0, n_t) for w0 in _wall_ring(ser, n_pts, eta_h, offset=0.77)]
            for row in validation_rows[k, n_t]:
                n_used += 1
                c = sum((row[m] * g for m, g in gm.items())) - constant
                worst = max(worst, abs(c) / scale)
    return worst if n_used else float('inf')

def check_qn1(mats) -> None:
    worst = 0.0
    ok = True
    for mat in mats:
        for mu in mat.mus:
            for name, smap in (('E03', E03), ('HS1', HS1)):
                ser = smap.zk_series(mu)
                eta = cauchy_radius(ser)
                for w0 in _wall_ring(ser, 3, eta):
                    row = e_row_mirror(ser, w0, 8)
                    ok &= abs(row[0] - 1.0) <= 1e-10
                    for m in list(range(-8, 0)) + list(range(1, 9)):
                        er, _, _, _, sc = e_entry_root(ser, m, w0)
                        ef = e_entry_fft(ser, m, w0)
                        em = row[m]
                        worst = max(worst, abs(em - er) / sc, abs(em - ef) / sc)
    _report('near-wall kernel agreement', worst <= 1e-11 and ok, f'mirror vs interior-side vs FFT max rel dev {worst:.3e} (<= 1e-11) on E0.3 + HS(t=1) at the production rings; winding anchors pass')

def check_qn2(mats) -> None:
    worst, tail_w = (0.0, 0.0)
    for mat in mats:
        S0 = modal_blocks_density(mat, N_H)
        scale = float(np.max(np.abs(S0)))
        for name, smap in (('CIRC', CIRC), ('ELL', ELL), ('CPLX', CPLX)):
            out = assemble_exact(mat, smap, N_H, 2 * N_H)
            worst = max(worst, float(np.max(np.abs(out['S'] - S0))) / scale)
            for Xh, Yh in out['sols'].values():
                tmax = max(float(np.max(np.abs(Xh))), float(np.max(np.abs(Yh))), 1e-300)
                tail = max(float(np.max(np.abs(Xh[N_H:]))), float(np.max(np.abs(Yh[N_H:]))))
                tail_w = max(tail_w, tail / tmax)
    _report('two-term digit regression (general path)', worst <= 5e-13 and tail_w <= 1e-10, f'max rel dev vs modal_blocks_density {worst:.3e} (<= 5e-13) over CIRC/ELL/CPLX x 5 materials at n_t = 2 n_h; inflated tail modes {tail_w:.3e} rel (<= 1e-10)')

def check_qn3(mats):
    """Resolve the independent Cauchy residual and retained operator together."""
    finals = {}
    okay = True
    for mat in mats:
        previous = None
        residual_previous = float('inf')
        for n_t in LADDER:
            out = assemble_exact(mat, HS1, N_H, n_t)
            res = heldout_certificate(out)
            dS = float(np.max(np.abs(out['S'] - previous)) / np.max(np.abs(out['S']))) if previous is not None else None
            print(f"        {mat.name:12s} n_t {n_t:3d}  cond {out['cond']:.6e}  heldout {res:.6e}  dS " + (f'{dS:.6e}' if dS is not None else '--'), flush=True)
            okay &= res < max(residual_previous, 1e-11)
            residual_previous, previous = (res, out['S'])
        finals[mat.name] = (res, dS, out)
        okay &= res < 1e-11 and dS < 1e-10
    _report('projected horseshoe convergence', okay, 'independent residual falls to the floating-point floor; final residual < 1e-11 and operator increment < 1e-10')
    return finals

def check_qn4(mats, finals) -> None:
    ok = True
    rows = []
    for mat in mats:
        residual, _, converged = finals[mat.name]
        S_ex = converged['S']
        diffs = []
        for n_modes in (96, 192):
            t = assemble_modal_dtn(mat, HS1, N_H, n_modes)
            St = t['S'][2:, :]
            diffs.append(float(np.linalg.norm(St - S_ex) / np.linalg.norm(S_ex)))
        d96, d192 = diffs
        ok &= d192 <= 0.05 and d192 <= 1.2 * d96 and (residual <= 1e-11)
        rows.append(f'{mat.name}: {d96:.1e}->{d192:.1e}')
    _report('teacher agreement at its own floor', ok, 'teacher (ring-pole LS) vs exact operator, rel Frobenius 96->192 modes: ' + ', '.join(rows) + ' — within the teacher tier and non-increasing; differences at floating-point accuracy are not resolved operator-error measurements')

def negative_controls(mats) -> None:
    f0 = mats[0]
    good = assemble_exact(f0, HS1, N_H, 64)
    bad = assemble_exact(f0, HS1, N_H, 64, conj_channel_bug=True)
    r_good = heldout_certificate(good)
    r_bad = heldout_certificate(bad)
    _report('Negative control (conj-channel corruption)', r_bad >= 100.0 * r_good, f'held-out {r_good:.3e} -> {r_bad:.3e} ({r_bad / max(r_good, 1e-300):.0f}x): the conjugate-channel rule is required on the general carrier')
    member = SectionMap(1.0, [0.0, 0.0, 0.15])
    out = assemble_exact(f0, member, N_H, 32)
    constant = max((abs(a) for aa in out['constants'].values() for a in aa))
    M = np.asarray(f0.pq, dtype=complex).T
    scale = np.max(np.abs(np.linalg.inv(M)))
    res = heldout_certificate(out)
    _report('NNQ2 constant exterior value is required', constant / scale > 0.001 and res < 1e-11, f'|a|/scale={constant / scale:.3e}, projected residual={res:.3e}')
    h = 0.001
    a_plus = assemble_exact(f0, SectionMap(1.0, [0, 0, 0.15 * h]), 1, 16)
    a_minus = assemble_exact(f0, SectionMap(1.0, [0, 0, -0.15 * h]), 1, 16)
    derivative = (np.asarray(a_plus['constants'][1, 'Uc']) - np.asarray(a_minus['constants'][1, 'Uc'])) / (2 * h)
    amplitude = np.linalg.solve(M, [0.5, 0.0])
    circle = SectionMap(1.0, [0.0])
    qhat = np.array([circle.zk_series(mu)[-1] / circle.zk_series(mu)[1] for mu in f0.mus])
    expected = amplitude * qhat * 0.15
    deviation = np.linalg.norm(derivative - expected) / np.linalg.norm(expected)
    _report('NNQ2 pure-c2 constant derivative', deviation < 1e-06, f'relative difference from A*qhat*conj(C2): {deviation:.3e}')
    ser = HS1.zk_series(f0.mus[0])
    eta = cauchy_radius(ser)
    w0 = complex(_wall_ring(ser, 1, eta)[0])
    Pc, _, _, _ = _poly_PQ(ser, w0)
    roots = np.roots(Pc[::-1])
    flip = complex(np.sum(roots[np.abs(roots) >= 1.0] ** 2))
    ef = e_entry_fft(ser, 2, w0)
    _report('Negative control (mirror classification flip)', abs(flip - ef) > 0.01, f'exterior-side sum used at m = +2: |dev vs FFT| = {abs(flip - ef):.3e} > 1e-2')

def check_sampling() -> None:
    """Exercise the thin-ellipse counterexample and strict interior checks."""
    from exec20_src.material import AnisoMaterial
    mat = AnisoMaterial('thin_ellipse', 1000000000.0, 250000000.0, 0.0, 0.0, 200000000.0)
    smap = SectionMap(1.0, [0.0, 0.99])
    assert np.allclose(sorted((mu.imag for mu in mat.mus)), [1.0, 2.0])
    assert np.linalg.eigvalsh(mat.D()).min() > 0
    for mu in mat.mus:
        ser = smap.zk_series(mu)
        critical = branch_rho_crit(ser)
        old_eta = max(0.85, critical + 0.02)
        eta = cauchy_radius(ser)
        assert old_eta > 1 and critical < eta < 1
        points = _wall_ring(ser, 65, eta, offset=0.0)
        a, b = (1.99, 0.01 * mu.imag)
        ellipse_level = (points.real / a) ** 2 + (points.imag / b) ** 2
        assert np.max(ellipse_level) < 1
        assert all((e_row_mirror(ser, w0, 4)[0] == 1 for w0 in points))
        old_point = complex(_wall_ring(ser, 1, old_eta, offset=0.0)[0])
        assert old_point.real > a
        for w0 in (old_point, complex(_eval_series(ser, 1.0))):
            try:
                e_row_mirror(ser, w0, 4)
            except ValueError:
                pass
            else:
                raise AssertionError('An exterior or boundary Cauchy point was accepted')
        for invalid in (critical, old_eta, 1.0, float('nan')):
            try:
                assemble_exact(mat, smap, 2, 4, eta_override=invalid)
            except ValueError:
                pass
            else:
                raise AssertionError('An invalid Cauchy radius was accepted')
        print(f'thin ellipse mu={mu.imag:.6g}i: r={critical:.10f}, old eta={old_eta:.10f}, eta={eta:.10f}, max ellipse level={max(ellipse_level):.10f}')
    circle = {1: 1.0}
    assert branch_rho_crit(circle) == 0 and cauchy_radius(circle) == 0.85
    for critical in (0.979, 0.98, 0.999):
        ser = {1: 1.0, -1: critical ** 2}
        eta = cauchy_radius(ser)
        assert branch_rho_crit(ser) < eta < 1
        assert all((e_row_mirror(ser, w0, 2)[0] == 1 for w0 in _wall_ring(ser, 17, eta)))
    out = assemble_exact(mat, smap, 4, 8)
    exact = modal_blocks_density(mat, 4)
    error = np.linalg.norm(out['S'] - exact) / np.linalg.norm(exact)
    residual = heldout_certificate(out)
    _report('Cauchy sampling and thin-ellipse operator', error < 1e-10 and residual < 1e-10, f'relative operator difference={error:.3e}; independent-ring residual={residual:.3e}; invalid layouts rejected')

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sampling-only', action='store_true', help='Run the Cauchy sampling regression only')
    args = parser.parse_args()
    check_sampling()
    if args.sampling_only:
        return int(bool(FAILURES))
    mats = P.materials_main()
    print('materials:', ', '.join((m.name for m in mats)))
    print('carriers : CIRC, ELL, CPLX (two-term), E0.3, HS(t=1) ten-term')
    rcs = {mat.name: [branch_rho_crit(HS1.zk_series(mu)) for mu in mat.mus] for mat in mats}
    print('fold shells on HS1:', ', '.join((f'{k}: {v[0]:.3f}/{v[1]:.3f}' for k, v in rcs.items())))
    print()
    check_qn1(mats)
    check_qn2(mats)
    finals = check_qn3(mats)
    check_qn4(mats, finals)
    print()
    negative_controls(mats)
    print()
    if FAILURES:
        print(f'Cauchy-constraint checks failed: {FAILURES}')
        return 1
    print('Cauchy-constraint checks passed: projected square assembly, two-term identity,')
    print('  horseshoe convergence and constant-mode controls passed.')
    return 0
if __name__ == '__main__':
    sys.exit(main())
