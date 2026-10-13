from __future__ import annotations
import csv
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code' / 'closure'))
sys.path.insert(0, str(ROOT / 'code' / 'closure' / 'numeric'))
from closure_env import bridge_pinned
bridge_pinned()
from exec26_src import protocol as P
from exec26_src.geometry import ellipse_map, horseshoe_map
from exec26_src.sf_general import structure_spectra
from exec26_src.sf_twoterm import modal_blocks_density
from p21_exact_evaluation import assemble_exact
N_H_ELL = 8
N_H_HS = 16
N_T_ELL = 16
N_T_ELL_CHECK = 32
N_T_HS = 64
N_T_HS_CHECK = 96
ELL_FLOOR = 1e-12
HS_DRIFT_TOL = 0.01
ELL_R = (1.0, 2.0, 3.0)
ELL_C1 = (0.1, 0.3, 0.5)
TABLE_MEMBER = (1.0, P.ELL_C1)
ELLIPSE_CSV = 'results/data/derived/exact_ellipse_invariance.csv'
STRUCTURE_CSV = 'results/data/derived/exact_general_structure.csv'

def readouts(S: np.ndarray, n_h: int) -> dict:
    """Pinned ``structure_spectra`` on an operator that has no n' = 0 rows."""
    padded = np.vstack([np.zeros((2, S.shape[1])), S])
    return structure_spectra(padded, n_h)

def ellipse_row(mat, R: float, c1: float, n_t: int) -> tuple[float, float, float]:
    """(invariance vs the closed form, off-block ratio, condition)."""
    out = assemble_exact(mat, ellipse_map(c1, R), N_H_ELL, n_t)
    S0 = modal_blocks_density(mat, N_H_ELL)
    inv = float(np.max(np.abs(out['S'] - S0)) / np.max(np.abs(S0)))
    return (inv, float(readouts(out['S'], N_H_ELL)['offblock_ratio']), out['cond'])

def horseshoe_row(mat, n_t: int) -> tuple[float, np.ndarray, float]:
    """(off-block ratio, decay normalized to dn = 0, condition) at t = 1."""
    smap = horseshoe_map(1.0, P.R_HS, np.asarray(P.C_HS))
    out = assemble_exact(mat, smap, N_H_HS, n_t)
    sp = readouts(out['S'], N_H_HS)
    return (float(sp['offblock_ratio']), sp['decay'] / sp['decay'][0], out['cond'])

def main() -> int:
    mats = list(P.materials_main())
    ell_rows: list[dict] = []
    ell_worst = 0.0
    for R in ELL_R:
        for c1 in ELL_C1:
            for mat in mats:
                inv, offb, cond = ellipse_row(mat, R, c1, N_T_ELL)
                inv_c, offb_c, _ = ellipse_row(mat, R, c1, N_T_ELL_CHECK)
                for label, value in (('invariance', max(inv, inv_c)), ('off-block', max(offb, offb_c))):
                    if value > ELL_FLOOR:
                        print(f'REFUSING TO WRITE: {mat.name} R={R} c1={c1} {label} reads {value:.3e}, above the {ELL_FLOOR:.0e} floating-point floor')
                        return 1
                ell_worst = max(ell_worst, inv, inv_c, offb, offb_c)
                ell_rows.append({'R': R, 'c1': c1, 'material': mat.name, 'n_t': N_T_ELL, 'invariance_rel': f'{inv:.3e}', 'offblock_ratio': f'{offb:.3e}', 'cond': f'{cond:.2e}'})
            print(f"  R={R} c1={c1}  five materials at or under {max((float(r['invariance_rel']) for r in ell_rows[-5:])):.2e}")
    hs_rows: list[dict] = []
    hs_drift = 0.0
    for mat in mats:
        offb, decay, cond = horseshoe_row(mat, N_T_HS)
        offb_c, decay_c, _ = horseshoe_row(mat, N_T_HS_CHECK)
        for label, a, b in (('off-block', offb, offb_c), ('dn7 decay', decay[7], decay_c[7])):
            drift = abs(b - a) / a
            hs_drift = max(hs_drift, drift)
            if drift > HS_DRIFT_TOL:
                print(f'REFUSING TO WRITE: {mat.name} {label} moves {drift:.2%} between n_t = {N_T_HS} and {N_T_HS_CHECK} ({a:.4e} -> {b:.4e})')
                return 1
        hs_rows.append({'section': 'HS100', 'rho_out': 1.0, 'material': mat.name, 'n_t': N_T_HS, 'offblock_ratio': f'{offb:.3e}', **{f'decay_dn{i}': f'{decay[i]:.3e}' for i in range(8)}, 'cond': f'{cond:.2e}'})
        span = np.log10(decay[0] / decay[7])
        print(f'  HS100 wall {mat.name:<14} offblock={offb:.3e} decay spans {span:.2f} decades')
    for rel, rows in ((ELLIPSE_CSV, ell_rows), (STRUCTURE_CSV, hs_rows)):
        path = ROOT / rel
        with open(path, 'w', encoding='utf-8', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator='\n')
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {rel}  ({len(rows)} rows)')
    table = [r for r in ell_rows if (float(r['R']), float(r['c1'])) == TABLE_MEMBER]
    offb = [float(r['offblock_ratio']) for r in hs_rows]
    spans = [np.log10(1.0 / float(r['decay_dn7'])) for r in hs_rows]
    print(f"\nellipse, table member R = {TABLE_MEMBER[0]}, c1 = {TABLE_MEMBER[1]}: invariance at or under {max((float(r['invariance_rel']) for r in table)):.2e}, off-block at or under {max((float(r['offblock_ratio']) for r in table)):.2e}")
    print(f'ellipse, worst over the {len(ELL_R)} x {len(ELL_C1)} (R, c1) grid            : {ell_worst:.2e}')
    print(f'horseshoe wall grade, off-block span         : {min(offb):.3e} to {max(offb):.3e}')
    print(f'horseshoe wall grade, decay over eight steps : {min(spans):.2f} to {max(spans):.2f} decades')
    print(f'worst n_t {N_T_HS} -> {N_T_HS_CHECK} drift on the horseshoe : {hs_drift:.2%}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
