from __future__ import annotations
import csv
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from closure_env import bridge_pinned
cfg = bridge_pinned()
from exec20_src import harmonic as hm
from exec20_src import material as mt
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/data/derived/anchor_mode_content.csv'
FAILURES: list[str] = []
RAN: list[str] = []
PINNED = dict(seed=20260717, n_cases=32, ratio=(1.3, 3.0), dip=(0.0, 45.0))
BOX = dict(seed=20260812, n_cases=128, ratio=(1.25, 4.0), dip=(0.0, 90.0))
N_MODES = 6
ALIGNED = dict(seed=20260815, n_cases=64, ratio=(1.25, 4.0), dips=(0.0, 90.0))
PINNED_COEFF_ERR = 8.96e-16
CORNERS = ((1.25, 0.0), (1.25, 90.0), (4.0, 0.0), (4.0, 90.0), (4.0, 45.0))
N_CORNER_LOADINGS = 32
EPS = float(np.finfo(float).eps)
ULP_BAND = (8.0, 40.0)
MODE_BAND = (1e-18, 1e-15)
IMAG_CEILING = 1e-14
BRACKET_ULP_BAND = (0.5, 20.0)
OPERATOR_ULP_BAND = (0.5, 20.0)
OFF_FRAME_MEDIAN_FLOOR = 0.1
CORNER_ULP_BAND = (1.0, 40.0)
EDGE_FRACTION = 0.05
HARD_CORNER = (3.5, 70.0)
HARD_CORNER_MIN = 3
CTRL_HARMONIC = 3
CTRL_AMPLITUDE = 0.1

def _report(label: str, ok: bool, detail: str) -> None:
    RAN.append(label)
    print(f"{('PASS' if ok else 'FAIL')}  {label}: {detail}")
    if not ok:
        FAILURES.append(label)

def _draw(rng, fam: dict):
    """One case of an anchor family, in the pinned draw order."""
    ratio = rng.uniform(*fam['ratio'])
    dip = rng.uniform(*fam['dip'])
    mat = mt.material_F(ratio, dip)
    sh, sv, tvh = (rng.uniform(-3000000.0, 3000000.0), rng.uniform(-3000000.0, 3000000.0), rng.uniform(-1000000.0, 1000000.0))
    r0 = rng.uniform(1.0, 4.0)
    return (mat, ratio, dip, sh, sv, tvh, r0)

def _derived_pair(mat, sh, sv, tvh, r0):
    """Bobet's own derived resultant form for the m = 1 pair."""
    mu1, mu2 = mat.mus
    X1 = r0 * (tvh - 1j * sh) / 2.0
    Y1 = r0 * (sv - 1j * tvh) / 2.0
    return ((X1 + mu2 * Y1) / (mu1 - mu2), -(X1 + mu1 * Y1) / (mu1 - mu2))

def _printed_bracket(m1, m2, sh, sv, tvh, r0):
    """Bobet's eq. (23): the zeta_k^{-1} coefficients of the Problem II pair.

    ``m1`` and ``m2`` are whatever the caller reads the printed symbols as --
    the root moduli, which is what the source's own sentence below its eq. (24)
    states, or the roots themselves, which is AMN4's substitution.
    """
    pre = 0.5j * r0 / (m1 - m2)
    return (-pre * ((1 + m2) * tvh + (m2 * sv - sh) * 1j), +pre * ((1 + m1) * tvh + (m1 * sv - sh) * 1j))

def _bracket_gap(mus, ref, sh, sv, tvh, r0, moduli: bool) -> float:
    """Eq. (23), read one way or the other, against a reference pair."""
    mu1, mu2 = mus
    m1, m2 = (abs(mu1), abs(mu2)) if moduli else (mu1, mu2)
    B1, B2 = _printed_bracket(m1, m2, sh, sv, tvh, r0)
    R1, R2 = ref
    return max(abs(B1 - R1) / max(abs(R1), 1e-30), abs(B2 - R2) / max(abs(R2), 1e-30))

def _one_case(mat, ratio, dip, sh, sv, tvh, r0, perturb=0.0) -> dict:
    ext = hm.single_hole_solution(mat, r0, sh, sv, tvh, n_modes=N_MODES)
    A1, A2 = _derived_pair(mat, sh, sv, tvh, r0)
    off = _bracket_gap(mat.mus, (A1, A2), sh, sv, tvh, r0, moduli=True)
    A1, A2 = (A1 * (1.0 + perturb), A2 * (1.0 + perturb))
    err = max(abs(ext.A[0, 0] - A1) / max(abs(A1), 1.0), abs(ext.A[1, 0] - A2) / max(abs(A2), 1.0))
    m1 = float(np.max(np.abs(ext.A[:, 0])))
    ge2 = float(np.max(np.abs(ext.A[:, 1:])))
    return {'ratio': ratio, 'dip_deg': dip, 'coeff_err_rel': err, 'm1_magnitude': m1, 'mode_ge2_magnitude': ge2, 'mode_ge2_over_m1': ge2 / m1, 'bracket_modulus_gap': off}

def _aligned_case(ratio, dip, sh, sv, tvh, r0, bracket_perturb=0.0) -> dict:
    """Evaluate the aligned-frame reference and operator solution.

Measure the roots' real components, agreement between the root-modulus and resultant formulas, and agreement of the operator solution with the reference coefficients."""
    mat = mt.material_F(ratio, dip)
    mu1, mu2 = mat.mus
    ref = _derived_pair(mat, sh, sv, tvh, r0)
    B1, B2 = _printed_bracket(abs(mu1), abs(mu2), sh, sv, tvh, r0)
    B1, B2 = (B1 * (1.0 + bracket_perturb), B2 * (1.0 + bracket_perturb))
    ext = hm.single_hole_solution(mat, r0, sh, sv, tvh, n_modes=N_MODES)
    return {'family': 'aligned64', 'ratio': ratio, 'dip_deg': dip, 'root_real_fraction': max(abs(mu1.real) / abs(mu1), abs(mu2.real) / abs(mu2)), 'bracket_modulus_gap': _bracket_gap(mat.mus, ref, sh, sv, tvh, r0, moduli=True), 'bracket_literal_gap': _bracket_gap(mat.mus, ref, sh, sv, tvh, r0, moduli=False), 'operator_vs_bracket': max(abs(ext.A[0, 0] - B1) / max(abs(B1), 1.0), abs(ext.A[1, 0] - B2) / max(abs(B2), 1.0))}

def measure_aligned(bracket_perturb: float=0.0) -> list[dict]:
    """The aligned family, in the box's own draw order with the dip taken
    from the two endpoints in turn rather than drawn."""
    rng = np.random.default_rng(ALIGNED['seed'])
    rows = []
    for case in range(ALIGNED['n_cases']):
        ratio = rng.uniform(*ALIGNED['ratio'])
        dip = ALIGNED['dips'][case % len(ALIGNED['dips'])]
        sh, sv, tvh = (rng.uniform(-3000000.0, 3000000.0), rng.uniform(-3000000.0, 3000000.0), rng.uniform(-1000000.0, 1000000.0))
        r0 = rng.uniform(1.0, 4.0)
        rows.append({'case': case, **_aligned_case(ratio, dip, sh, sv, tvh, r0, bracket_perturb=bracket_perturb)})
    return rows

def measure(fam: dict, perturb: float=0.0) -> list[dict]:
    """One row per case: the m = 1 agreement, the m >= 2 content and the
    printed-bracket deviation.

    ``perturb`` scales the derived reference pair by ``1 + perturb`` and is
    AMN2's entry; the measurement itself runs at 0.
    """
    rng = np.random.default_rng(fam['seed'])
    rows = []
    for case in range(fam['n_cases']):
        row = _one_case(*_draw(rng, fam), perturb=perturb)
        rows.append({'family': fam['name'], 'case': case, **row})
    return rows

def measure_corners(fam: dict) -> list[dict]:
    """Evaluate parameter-box corners using the random stream after the interior cases, preserving reproducibility from one seed."""
    rng = np.random.default_rng(fam['seed'])
    for _ in range(fam['n_cases']):
        _draw(rng, fam)
    rows = []
    for ratio, dip in CORNERS:
        mat = mt.material_F(ratio, dip)
        for k in range(N_CORNER_LOADINGS):
            sh, sv, tvh = (rng.uniform(-3000000.0, 3000000.0), rng.uniform(-3000000.0, 3000000.0), rng.uniform(-1000000.0, 1000000.0))
            r0 = rng.uniform(1.0, 4.0)
            row = _one_case(mat, ratio, dip, sh, sv, tvh, r0)
            rows.append({'family': 'corners', 'case': f'r{ratio:g}d{dip:g}_{k}', **row})
    return rows

def measure_control(fam: dict) -> list[dict]:
    """AMN1: the same family under a traction that is not a uniform far
    field's, so the higher-mode coefficients carry real content.

    The released profile keeps the uniform part and adds a harmonic-``k``
    component to both traction components.  Net force is untouched -- the
    added content lives at k = 3, and the zero-net-force class the solve
    requires is a statement about k = 0 alone -- so the solve accepts it and
    the higher modes fill.
    """
    rng = np.random.default_rng(fam['seed'])
    out = []
    for case in range(fam['n_cases']):
        mat, _ratio, _dip, sh, sv, tvh, r0 = _draw(rng, fam)
        n_grid = 512
        th = 2.0 * np.pi * np.arange(n_grid) / n_grid
        s_rr, s_rt = hm.released_traction_values(sh, sv, tvh, th)
        scale = CTRL_AMPLITUDE * float(np.max(np.abs(s_rr)))
        s_rr = s_rr + scale * np.cos(CTRL_HARMONIC * th)
        s_rt = s_rt + scale * np.sin(CTRL_HARMONIC * th)
        ext = hm.solve_neumann_circle(mat, r0, s_rr, s_rt, N_MODES)
        m1 = float(np.max(np.abs(ext.A[:, 0])))
        ge2 = float(np.max(np.abs(ext.A[:, 1:])))
        out.append({'case': case, 'mode_ge2_over_m1': ge2 / m1})
    return out

def coverage(rows, fam: dict) -> dict:
    ratios = [r['ratio'] for r in rows]
    dips = [r['dip_deg'] for r in rows]
    r_lo, r_hi = fam['ratio']
    d_lo, d_hi = fam['dip']
    r_span, d_span = (r_hi - r_lo, d_hi - d_lo)
    return {'ratio_min': min(ratios), 'ratio_max': max(ratios), 'dip_min': min(dips), 'dip_max': max(dips), 'reaches_edges': min(ratios) - r_lo <= EDGE_FRACTION * r_span and r_hi - max(ratios) <= EDGE_FRACTION * r_span and (min(dips) - d_lo <= EDGE_FRACTION * d_span) and (d_hi - max(dips) <= EDGE_FRACTION * d_span), 'hard_corner': sum((1 for r in rows if r['ratio'] >= HARD_CORNER[0] and r['dip_deg'] >= HARD_CORNER[1]))}

def check_am1(rows) -> None:
    worst = max((r['coeff_err_rel'] for r in rows))
    _report('the pinned family reproduces the pinned anchor reading', f'{worst:.2e}' == f'{PINNED_COEFF_ERR:.2e}', f'measured {worst:.4e} against pinned {PINNED_COEFF_ERR:.2e} over {len(rows)} cases')

def check_am2(rows) -> None:
    worst = max((r['coeff_err_rel'] for r in rows))
    ulp = worst / EPS
    lo, hi = ULP_BAND
    _report('the widened family agrees at the arithmetic floor', lo <= ulp <= hi, f'{worst:.4e} = {ulp:.2f} ulp of double precision (eps {EPS:.4e}); band {lo}-{hi}')

def check_am3(rows) -> None:
    worst = max((r['mode_ge2_over_m1'] for r in rows))
    lo, hi = MODE_BAND
    _report("the higher-mode content is at the transform's rounding floor", lo <= worst <= hi, f'max m>=2 over m=1 {worst:.3e} over {len(rows)} cases; band {lo:.0e}-{hi:.0e}')

def check_am4(rows, fam: dict) -> None:
    cov = coverage(rows, fam)
    _report('the widened draw reaches every edge of the family box', cov['reaches_edges'] and cov['hard_corner'] >= HARD_CORNER_MIN, f"ratio [{cov['ratio_min']:.3f}, {cov['ratio_max']:.3f}] of {fam['ratio']}, dip [{cov['dip_min']:.2f}, {cov['dip_max']:.2f}] of {fam['dip']}, {cov['hard_corner']} cases at ratio >= {HARD_CORNER[0]} and dip >= {HARD_CORNER[1]} (floor {HARD_CORNER_MIN})")

def check_am5(rows) -> None:
    worst = max((r['coeff_err_rel'] for r in rows))
    ulp = worst / EPS
    lo, hi = CORNER_ULP_BAND
    _report('the box corners agree at the same arithmetic floor', lo <= ulp <= hi, f'{worst:.4e} = {ulp:.2f} ulp over {len(rows)} corner loadings at {len(CORNERS)} points; band {lo}-{hi} ulp')

def check_am6(rows) -> None:
    worst = max((r['root_real_fraction'] for r in rows))
    _report('the aligned orientations carry pure imaginary roots', worst <= IMAG_CEILING, f"max |Re mu| / |mu| {worst:.3e} over {len(rows)} cases at dips {ALIGNED['dips']}; ceiling {IMAG_CEILING:.0e}")

def check_am7(rows) -> None:
    worst = max((r['bracket_modulus_gap'] for r in rows))
    ulp = worst / EPS
    lo, hi = BRACKET_ULP_BAND
    _report('eq. (23) with the root moduli is the eq. (14) form there', lo <= ulp <= hi, f'{worst:.4e} = {ulp:.2f} ulp over {len(rows)} loadings; band {lo}-{hi} ulp')

def check_am8(rows) -> None:
    worst = max((r['operator_vs_bracket'] for r in rows))
    ulp = worst / EPS
    lo, hi = OPERATOR_ULP_BAND
    _report('the operator construction agrees with eq. (23) there', lo <= ulp <= hi, f'{worst:.4e} = {ulp:.2f} ulp over {len(rows)} loadings; band {lo}-{hi} ulp')

def check_am9(box_rows) -> None:
    gaps = [r['bracket_modulus_gap'] for r in box_rows]
    med = float(np.median(gaps))
    _report('the frame carries it: off those orientations eq. (23) parts from the form the anchor evaluates', med >= OFF_FRAME_MEDIAN_FLOOR, f'median {med:.4f}, minimum {min(gaps):.4f} over {len(box_rows)} cases; floor {OFF_FRAME_MEDIAN_FLOOR}')

def negative_controls(box_rows, aligned_rows) -> None:
    ctrl = measure_control(BOX)
    worst_ctrl = max((r['mode_ge2_over_m1'] for r in ctrl))
    hi = MODE_BAND[1]
    _report('Negative control: real higher-harmonic content moves the ratio', worst_ctrl > 1000.0 * hi, f'{worst_ctrl:.3e} against the specified ceiling {hi:.0e} ({worst_ctrl / hi:.1e}x)')
    pert = measure(PINNED, perturb=1e-12)
    worst_pert = max((r['coeff_err_rel'] for r in pert))
    _report('Negative control: a 1e-12 shift breaks the pinned agreement', f'{worst_pert:.2e}' != f'{PINNED_COEFF_ERR:.2e}' and worst_pert > ULP_BAND[1] * EPS, f'perturbed {worst_pert:.3e} against pinned {PINNED_COEFF_ERR:.2e}')
    retired = dict(BOX, ratio=PINNED['ratio'], dip=PINNED['dip'], name='retired')
    rows = measure(retired)
    cov = coverage(rows, BOX)
    _report('Negative control: the restricted parameter box fails the coverage row', not (cov['reaches_edges'] and cov['hard_corner'] >= HARD_CORNER_MIN), f"ratio [{cov['ratio_min']:.3f}, {cov['ratio_max']:.3f}], dip [{cov['dip_min']:.2f}, {cov['dip_max']:.2f}], {cov['hard_corner']} cases in the hard corner")
    lit = max((r['bracket_literal_gap'] for r in aligned_rows))
    lo, hi = BRACKET_ULP_BAND
    _report('Negative control: the roots themselves break the identity', lit / EPS > hi, f'{lit:.4e} = {lit / EPS:.3e} ulp against the specified ceiling {hi} ulp')
    pert = measure_aligned(bracket_perturb=1e-12)
    worst = max((r['operator_vs_bracket'] for r in pert))
    lo, hi = OPERATOR_ULP_BAND
    _report("Negative control: a 1e-12 shift in the bracket breaks the operator's agreement with it", worst / EPS > hi, f'perturbed {worst:.3e} = {worst / EPS:.3e} ulp against the specified ceiling {hi} ulp')
    aligned_gaps = [r['bracket_modulus_gap'] for r in aligned_rows]
    med_aligned = float(np.median(aligned_gaps))
    _report('Negative control: The off-frame statistic fails its floor on the aligned family', med_aligned < OFF_FRAME_MEDIAN_FLOOR, f'aligned median {med_aligned:.3e} against the specified floor {OFF_FRAME_MEDIAN_FLOOR}')

def export(rows) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ['family', 'case', 'ratio', 'dip_deg', 'coeff_err_rel', 'm1_magnitude', 'mode_ge2_magnitude', 'mode_ge2_over_m1', 'bracket_modulus_gap', 'bracket_literal_gap', 'operator_vs_bracket', 'root_real_fraction']
    with open(OUT, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'wrote {OUT.relative_to(ROOT)} ({len(rows)} rows)')

def main() -> int:
    PINNED['name'], BOX['name'] = ('pinned32', 'box128')
    print('T-O anchor family: the coefficient anchor of Section 5.1, measured')
    for fam in (PINNED, BOX):
        print(f"  {fam['name']:9s}: {fam['n_cases']} cases, PCG64 seed {fam['seed']}, modulus ratio {fam['ratio']}, dip {fam['dip']} deg, {N_MODES} harmonics retained")
    print(f'  corners  : {N_CORNER_LOADINGS} loadings at each of {CORNERS}, from the box stream')
    print(f"  aligned64: {ALIGNED['n_cases']} cases, PCG64 seed {ALIGNED['seed']}, modulus ratio {ALIGNED['ratio']}, dips {ALIGNED['dips']} deg -- the frame eq. (23) is written in")
    print(f'  control  : harmonic {CTRL_HARMONIC} released traction at {CTRL_AMPLITUDE:g} of the uniform amplitude')
    print()
    pinned_rows = measure(PINNED)
    box_rows = measure(BOX)
    corner_rows = measure_corners(BOX)
    aligned_rows = measure_aligned()
    worst_err = max((r['coeff_err_rel'] for r in box_rows))
    worst_ge2 = max((r['mode_ge2_over_m1'] for r in box_rows))
    print(f'  m = 1 agreement : {worst_err:.4e} ({worst_err / EPS:.2f} ulp) over the widened family')
    print(f'  m >= 2 content  : {worst_ge2:.3e} of the m = 1 magnitude')
    print()
    check_am1(pinned_rows)
    check_am2(box_rows)
    check_am3(box_rows)
    check_am4(box_rows, BOX)
    check_am5(corner_rows)
    check_am6(aligned_rows)
    check_am7(aligned_rows)
    check_am8(aligned_rows)
    check_am9(box_rows)
    print()
    negative_controls(box_rows, aligned_rows)
    print()
    export([*pinned_rows, *box_rows, *corner_rows, *aligned_rows])
    print()
    if len(RAN) != 15:
        print(f'T-O NUMERIC RED: {len(RAN)} checks ran, expected 15 -- the suite did not complete: {RAN}')
        return 1
    if FAILURES:
        print(f'T-O NUMERIC RED: {FAILURES}')
        return 1
    print('T-O NUMERIC GREEN: the anchor family spans the whole box, agrees')
    print('  at a few tens of ulp of double precision with no degradation at')
    print("  the box corners, and the higher modes carry the transform's")
    print('  rounding alone; in the frame the printed bracket is written in it')
    print('  and the form the anchor evaluates are one function; controls fire.')
    return 0
if __name__ == '__main__':
    sys.exit(main())
