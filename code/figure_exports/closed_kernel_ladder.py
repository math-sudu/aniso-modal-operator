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
from mapped_elasticity import protocol as P
from mapped_elasticity.geometry import horseshoe_map
from mapped_elasticity.sf_twoterm import modal_blocks_density
from cauchy_evaluation import assemble_exact
from graded_kernels import closed_S1, closed_S2
N_H = 8
N_T = 64
N_T_CHECK = 96
SATURATION_TOL = 0.01
RHO_TIERS = (1.0, 1.5, 2.0)
T_LADDER = (0.25, 0.5, 1.0)
CONVERGENCE_N_T = (16, 24, 32, 64)
LADDER_CSV = 'results/data/derived/closed_kernel_ladder.csv'
CONVERGENCE_CSV = 'results/data/derived/closed_kernel_convergence.csv'

def graded_map_constants(rho: float) -> tuple[float, np.ndarray]:
    """(R', c') of the grade-rho tier, for kernels that take them directly.

    ``horseshoe_map(t, R, C).scaled(rho)`` is ``SectionMap(R rho,
    t C rho^-(j+1))`` and the scaling is linear in t, so the grade tier of
    the ray is the ray of the graded constants -- which is what lets the
    closed kernels, whose signature carries no grade, cover every tier.
    """
    C = np.asarray(P.C_HS, dtype=complex)
    j = np.arange(len(C))
    return (P.R_HS * rho, C * rho ** (-(j + 1.0)))

def relatives(S_t: np.ndarray, S0: np.ndarray, S1: np.ndarray, S2: np.ndarray, t: float) -> tuple[float, float, float]:
    """Relative remainders of the three principal parts, Frobenius."""
    n = np.linalg.norm(S_t)
    return (float(np.linalg.norm(S_t - S0) / n), float(np.linalg.norm(S_t - S0 - t * S1) / n), float(np.linalg.norm(S_t - S0 - t * S1 - t * t * S2) / n))

def covering_order(rel0: float, rel1: float, rel2: float) -> int:
    """Lowest principal part that covers the 1e-3 gauge (-1 if none)."""
    if rel2 <= 0.001:
        return 2
    if rel1 <= 0.001:
        return 1
    return 0 if rel0 <= 0.001 else -1

def reference(mat, rho: float, t: float, n_t: int) -> np.ndarray:
    smap = horseshoe_map(t, P.R_HS, np.asarray(P.C_HS))
    if rho != 1.0:
        smap = smap.scaled(rho)
    return assemble_exact(mat, smap, N_H, n_t)['S']

def main() -> int:
    mats = list(P.materials_main())
    ladder_rows: list[dict] = []
    drift_worst = 0.0
    for rho in RHO_TIERS:
        R_g, C_g = graded_map_constants(rho)
        for mat in mats:
            S0 = modal_blocks_density(mat, N_H)
            S1 = closed_S1(mat, R_g, C_g, N_H)
            S2 = closed_S2(mat, R_g, C_g, N_H)
            k1 = float(np.linalg.norm(S1) / np.linalg.norm(S0))
            k2 = float(np.linalg.norm(S2) / np.linalg.norm(S0))
            for t in T_LADDER:
                r0, r1, r2 = relatives(reference(mat, rho, t, N_T), S0, S1, S2, t)
                _, _, r2c = relatives(reference(mat, rho, t, N_T_CHECK), S0, S1, S2, t)
                drift = abs(r2c - r2) / r2
                drift_worst = max(drift_worst, drift)
                if drift > SATURATION_TOL:
                    print(f'REFUSING TO WRITE: {mat.name} rho={rho} t={t} moves {drift:.2%} between n_t = {N_T} and {N_T_CHECK} ({r2:.4e} -> {r2c:.4e})')
                    return 1
                ladder_rows.append({'rho_out': rho, 'material': mat.name, 't': t, 'rel0': f'{r0:.3e}', 'rel1': f'{r1:.3e}', 'rel2': f'{r2:.3e}', 'S1_over_S0': f'{k1:.3e}', 'S2_over_S0': f'{k2:.3e}', 'lowest_covering_order': covering_order(r0, r1, r2)})
            print(f'  rho={rho} {mat.name:<14} k1={k1:.3e} k2={k2:.3e}')
    conv_rows: list[dict] = []
    for mat in mats:
        S0 = modal_blocks_density(mat, N_H)
        S1 = closed_S1(mat, P.R_HS, np.asarray(P.C_HS), N_H)
        S2 = closed_S2(mat, P.R_HS, np.asarray(P.C_HS), N_H)
        for n_t in CONVERGENCE_N_T:
            smap = horseshoe_map(1.0, P.R_HS, np.asarray(P.C_HS))
            out = assemble_exact(mat, smap, N_H, n_t)
            r0, r1, r2 = relatives(out['S'], S0, S1, S2, 1.0)
            conv_rows.append({'material': mat.name, 'rho_out': 1.0, 't': 1.0, 'n_t': n_t, 'rel0': f'{r0:.3e}', 'rel1': f'{r1:.3e}', 'rel2': f'{r2:.3e}', 'cond': f"{out['cond']:.2e}"})
    for rel, rows in ((LADDER_CSV, ladder_rows), (CONVERGENCE_CSV, conv_rows)):
        path = ROOT / rel
        with open(path, 'w', encoding='utf-8', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator='\n')
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {rel}  ({len(rows)} rows)')
    wall = max((float(r['rel2']) for r in ladder_rows if float(r['rho_out']) == 1.0 and float(r['t']) == 1.0))
    wide = max((float(r['rel2']) for r in ladder_rows if float(r['rho_out']) >= 1.5))
    print(f'\nfamily bound, wall grade, full strength : {wall:.4e}')
    print(f'worst over the two wider grades          : {wide:.4e}')
    print(f'worst n_t {N_T} -> {N_T_CHECK} drift        : {drift_worst:.2%}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
