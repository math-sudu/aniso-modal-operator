from __future__ import annotations
import csv
import math
from pathlib import Path
import mpmath as mp
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results' / 'data' / 'derived' / 'degenerate_switch.csv'
EV, NU_VH, NU_HH = (10.5, 0.2, 0.25)
G_REF = EV / 2.5
UNIT_ROUNDOFF = 2.0 ** (-53)
C_NU = 2.0 / (3.0 - 4.0 * NU_HH)
FAILURES: list[str] = []

def _report(label: str, ok: bool, detail: str) -> None:
    print(f"{('PASS' if ok else 'FAIL')}  {label}: {detail}")
    if not ok:
        FAILURES.append(label)

def beta(ratio: float, g: float, ctx):
    """Bedding-frame reduced compliances, where beta_16 = beta_26 = 0."""
    if ctx is mp:
        e_h, e_v = (mp.mpf(str(ratio)) * mp.mpf(str(EV)), mp.mpf(str(EV)))
        nu_vh, nu_hh = (mp.mpf(str(NU_VH)), mp.mpf(str(NU_HH)))
        g_vh = mp.mpf(repr(g)) * mp.mpf(str(G_REF))
    else:
        e_h, e_v, nu_vh, nu_hh = (ratio * EV, EV, NU_VH, NU_HH)
        g_vh = g * G_REF
    return {'11': (1 - nu_hh ** 2) / e_h, '22': 1 / e_v - nu_vh ** 2 * e_h / e_v ** 2, '12': -nu_vh * (1 + nu_hh) / e_v, '66': 1 / g_vh, '16': 0 * e_h, '26': 0 * e_h}

def roots(b, ctx):
    """Upper-half-plane roots of the biquadratic b11 z^2 + (2 b12 + b66) z +
    b22 = 0 in z = mu^2. Solving the biquadratic rather than the quartic keeps
    the 60-digit reference free of a general root finder."""
    a, bb, c = (b['11'], 2 * b['12'] + b['66'], b['22'])
    disc = bb * bb - 4 * a * c
    if ctx is mp:
        sq = mp.sqrt(mp.mpc(disc))
        zs = [(-bb + sq) / (2 * a), (-bb - sq) / (2 * a)]
        out = [mp.sqrt(mp.mpc(z)) for z in zs]
        return [-m if mp.im(m) < 0 else m for m in out]
    sq = np.emath.sqrt(complex(disc))
    zs = [(-bb + sq) / (2 * a), (-bb - sq) / (2 * a)]
    out = [np.emath.sqrt(complex(z)) for z in zs]
    return [-m if m.imag < 0 else m for m in out]

def blocks(ratio: float, g: float, ctx):
    """(two-root Lambda or None if M is numerically singular, Lambda_conf,
    mu_gap, kappa_2(M))."""
    b = beta(ratio, g, ctx)
    mu1, mu2 = roots(b, ctx)

    def pq(mu):
        return (b['11'] * mu ** 2 + b['12'] - b['16'] * mu, b['12'] * mu + b['22'] / mu - b['26'])
    p1, q1 = pq(mu1)
    p2, q2 = pq(mu2)
    dp = 2 * b['11'] * mu1 - b['16']
    dq = b['12'] - b['22'] / mu1 ** 2
    den = p1 * dq - dp * q1
    if ctx is mp:
        M = mp.matrix([[p1, p2], [q1, q2]])
        B = mp.matrix([[mu1, mu2], [-1, -1]])
        lam = mp.mpc(0, -1) * (B * M ** (-1))
        conf = mp.mpc(0, -1) / den * mp.matrix([[mu1 * dq - q1, p1 - mu1 * dp], [-dq, dp]])
        return (lam, conf, float(abs(mu1 - mu2)), None)
    M = np.array([[p1, p2], [q1, q2]], dtype=complex)
    B = np.array([[mu1, mu2], [-1.0, -1.0]], dtype=complex)
    try:
        lam = -1j * (B @ np.linalg.inv(M))
    except np.linalg.LinAlgError:
        lam = None
    conf = -1j / den * np.array([[mu1 * dq - q1, p1 - mu1 * dp], [-dq, dp]], dtype=complex)
    return (lam, conf, float(abs(mu1 - mu2)), float(np.linalg.cond(M)))

def rel(a_np, ref_mp) -> float:
    """Max-entry relative difference of a float64 block against the reference."""
    num = max((abs(complex(ref_mp[i, j]) - a_np[i, j]) for i in range(2) for j in range(2)))
    den = max((abs(complex(ref_mp[i, j])) for i in range(2) for j in range(2)))
    return float(num / den)

def g_star(ratio: float) -> float:
    b = beta(ratio, 1.0, None)
    return 1.0 / (2.0 * math.sqrt(b['11'] * b['22']) + 2.0 * NU_VH * (1.0 + NU_HH) / EV) / G_REF
RATIOS = (1.25, 4.0)
STEPS = tuple((10.0 ** (-k / 2.0) for k in range(2, 33)))

def walk(ratio: float) -> list[dict]:
    mp.mp.dps = 60
    gs = g_star(ratio)
    rows = []
    for t in STEPS:
        g = gs * (1.0 - t)
        lam_ref, _cr, _gr, _ = blocks(ratio, g, mp)
        lam, conf, gap, kappa = blocks(ratio, g, None)
        rows.append({'ratio': ratio, 't': t, 'g': g, 'mu_gap': gap, 'kappa_M': kappa, 'err_2root': rel(lam, lam_ref) if lam is not None else float('nan'), 'err_conf': rel(conf, lam_ref)})
    return rows

def neighbourhood(ratio: float) -> list[dict]:
    """The float64 neighbours of g*: the closest a double-precision material
    specification can be placed to the locus."""
    mp.mp.dps = 60
    gs = g_star(ratio)
    out = []
    for label, g in (('g_star', gs), ('one_ulp_below', math.nextafter(gs, 0.0)), ('one_ulp_above', math.nextafter(gs, 10.0))):
        lam_ref, _cr, _gr, _ = blocks(ratio, g, mp)
        lam, conf, gap, kappa = blocks(ratio, g, None)
        out.append({'ratio': ratio, 'label': label, 'g': g, 'mu_gap': gap, 'kappa_M': kappa, 'err_2root': rel(lam, lam_ref) if lam is not None else float('nan'), 'err_conf': rel(conf, lam_ref), 'singular': lam is None})
    return out

def main() -> None:
    per_ratio = {r: walk(r) for r in RATIOS}
    near = {r: neighbourhood(r) for r in RATIOS}
    spread = 0.0
    for rows in per_ratio.values():
        live = [w for w in rows if w['err_2root'] > 1e-13]
        vals = [w['err_2root'] / (UNIT_ROUNDOFF * w['kappa_M']) for w in live]
        spread = max(spread, max(vals) / min(vals))
    grows = all((rows[-1]['err_2root'] > rows[0]['err_2root'] * 100000.0 for rows in per_ratio.values()))
    _report('two-root loss follows u kappa_2(M)', spread <= 10.0 and grows, f'err / (u kappa_2) spans a factor {spread:.2f} over both ratios (<= 10), and grows by over five decades as the roots close')
    coef = {}
    settled = 0.0
    for r, rows in per_ratio.items():
        tail = [w['err_conf'] / w['mu_gap'] for w in rows[-6:-2]]
        coef[r] = tail[-1]
        settled = max(settled, (max(tail) - min(tail)) / max(tail))
    _report('degenerate block departs at the O(mu_gap) rate', settled <= 0.001, '; '.join((f'ratio {r}: err_conf / mu_gap -> {coef[r]:.4f}' for r in RATIOS)) + f"; settled to {settled:.1e} over the last four rungs (<= 1e-3). The coefficient is the medium's, not a universal 1/2")
    predicted = {r: math.sqrt(2.0 * C_NU * UNIT_ROUNDOFF / coef[r]) for r in RATIOS}
    floor = {r: min((w['mu_gap'] for w in near[r] if w['mu_gap'] > 0.0)) for r in RATIOS}
    collapsed = {r: [w['label'] for w in near[r] if w['mu_gap'] == 0.0] for r in RATIOS}
    ok = all((0.5 <= predicted[r] / floor[r] <= 2.0 for r in RATIOS))
    _report('the crossing lands at the float64 floor on the separation', ok, '; '.join((f'ratio {r}: crossing {predicted[r]:.2e} against a float64 floor of {floor[r]:.2e}' for r in RATIOS)) + (f"; and at ratio 4.0 the float64 neighbours {', '.join(collapsed[4.0])} put the two roots at the SAME value" if collapsed[4.0] else ''))
    cost = {r: coef[r] * floor[r] for r in RATIOS}
    singular = {r: [w['label'] for w in near[r] if w['singular'] or w['mu_gap'] == 0.0] for r in RATIOS}
    box = {}
    for r in RATIOS:
        lam, conf, gap, kappa = blocks(r, 1.0, None)
        lam_ref, _c, _g, _k = blocks(r, 1.0, mp)
        box[r] = (gap, rel(lam, lam_ref), rel(conf, lam_ref))
    ok = max(cost.values()) <= 1e-07 and bool(singular[4.0]) and all((box[r][1] < 1e-14 for r in RATIOS))
    _report('the switch is free, and the box never needs it', ok, '; '.join((f'ratio {r}: switch costs {cost[r]:.1e} in the block' for r in RATIOS)) + f"; at ratio 4.0 the two-root route is singular at {', '.join(singular[4.0])}; " + '; '.join((f'box slice ratio {r}: mu_gap {box[r][0]:.3f}, two-root {box[r][1]:.1e}' for r in RATIOS)))
    dsn1 = any((w['err_conf'] == 0.0 for rows in per_ratio.values() for w in rows))
    _report('Negative control (degenerate block exact at finite gap)', not dsn1, 'no rung returns err_conf == 0, so the finite-gap check measures a real departure')
    dsn2 = all((predicted[r] < 1e-06 * box[r][0] for r in RATIOS))
    _report('Negative control (crossing at the box slice)', dsn2, '; '.join((f"ratio {r}: the crossing at {predicted[r]:.1e} is seven decades below the box's own {box[r][0]:.3f}" for r in RATIOS)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['record', 'ratio', 'label', 'g', 'mu_gap', 'kappa_M', 'err_2root', 'err_conf'])

        def num(x):
            return '' if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, '.12g')
        for r in RATIOS:
            for row in per_ratio[r]:
                w.writerow(['walk', format(r, '.12g'), '', num(row['g']), num(row['mu_gap']), num(row['kappa_M']), num(row['err_2root']), num(row['err_conf'])])
            for row in near[r]:
                w.writerow(['float64_neighbour', format(r, '.12g'), row['label'] + ('_singular' if row['singular'] else ''), num(row['g']), num(row['mu_gap']), num(row['kappa_M']), num(row['err_2root']), num(row['err_conf'])])
            w.writerow(['box_slice', format(r, '.12g'), '', '1', num(box[r][0]), '', num(box[r][1]), num(box[r][2])])
            w.writerow(['switch', format(r, '.12g'), '', '', num(predicted[r]), '', '', num(cost[r])])
            w.writerow(['departure_coefficient', format(r, '.12g'), '', '', '', '', '', num(coef[r])])
    print(f'\nwrote {OUT.relative_to(ROOT)}')
    for r in RATIOS:
        print(f'  ratio {r}: switch at mu_gap {predicted[r]:.2e} (kappa_2(M) {2 * C_NU / predicted[r]:.1e}), costing {cost[r]:.1e} in the block; box slice sits at mu_gap {box[r][0]:.3f}')
    raise SystemExit(1 if FAILURES else 0)
if __name__ == '__main__':
    main()
