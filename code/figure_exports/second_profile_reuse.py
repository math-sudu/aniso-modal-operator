from __future__ import annotations
import csv
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code' / 'closure'))
sys.path.insert(0, str(ROOT / 'code' / 'closure' / 'numeric'))
sys.path.insert(0, str(ROOT / 'code' / 'figure_exports'))
from closure_env import bridge_pinned
bridge_pinned()
from exec26_src import protocol as P
from exec26_src.geometry import SectionMap, horseshoe_map
from exec26_src.sf_twoterm import modal_blocks_density
from p21_exact_evaluation import assemble_exact
from p22_graded_kernels import closed_S1, closed_S2
from closed_kernel_ladder import N_H, N_T, N_T_CHECK, RHO_TIERS, SATURATION_TOL, T_LADDER, covering_order, relatives
LADDER_CSV = ROOT / 'results/data/derived/second_profile_ladder.csv'
COST_CSV = ROOT / 'results/data/derived/second_profile_cost.csv'
SHIPPED_LADDER = ROOT / 'results/data/derived/closed_kernel_ladder.csv'
EX_R = 0.9945
EX_ALPHA = (complex(0.0, 0.2836), complex(0.092, 0.092), complex(0.0389, 0.0))
EX_PHASE = 3.0 * np.pi / 4.0
PROFILE = 'EX2002-semicircular'
COST_N_H = 31
COST_N_T = 64
COST_REPEATS = 3

def ex_coefficients() -> np.ndarray:
    c = [0.0 + 0j]
    for k, a in enumerate(EX_ALPHA, start=1):
        c.append(a * np.exp(-1j * (k + 1) * EX_PHASE))
    return np.asarray(c, dtype=complex)

def convention_gate(c: np.ndarray) -> float:
    """Our coefficients draw the source's own boundary, as a point set."""
    th = np.linspace(0.0, 2.0 * np.pi, 20001)
    src = EX_R * (np.exp(-1j * th) + sum((a * np.exp(1j * (k + 1) * th) for k, a in enumerate(EX_ALPHA))))
    src = src * np.exp(-1j * EX_PHASE)
    ours = SectionMap(EX_R, c).boundary(th)
    dev = max((float(np.min(np.abs(ours[i] - src))) for i in range(0, len(th), 37)))
    imag = float(np.max(np.abs(np.imag(c))))
    print(f'[gate] convention: boundary point sets agree to {dev:.3e}; largest imaginary part of the converted coefficients {imag:.3e}')
    if dev >= 1e-12 or imag >= 1e-12:
        raise SystemExit('convention gate failed; nothing written')
    return dev

def admissibility(smap: SectionMap) -> dict:
    th = np.linspace(0.0, 2.0 * np.pi, 20001)
    dom = np.abs(smap.domega(np.exp(1j * th)))
    z = smap.boundary(th)
    decay = float(sum((j * abs(smap.c[j]) for j in range(1, len(smap.c)))))
    out = {'min_domega': float(dom.min()), 'min_domega_over_R': float(dom.min()) / smap.R, 'coefficient_decay_sum': decay, 'extent_x': float(z.real.max() - z.real.min()), 'extent_y': float(z.imag.max() - z.imag.min()), 'crown_x': float(z.real.max()), 'invert_x': float(z.real.min())}
    print(f"[gate] admissibility: min |omega'| = {out['min_domega']:.4f} ({out['min_domega_over_R']:.3f} R), sum j|c_j| = {decay:.4f}")
    if not (out['min_domega'] > 0.0 and decay < 1.0):
        raise SystemExit('admissibility gate failed; nothing written')
    return out

def ladder_rows(tag: str, R: float, C: np.ndarray) -> list[dict]:
    """The perturbation ladder on one ray, at the three interface radii."""
    rows: list[dict] = []
    j = np.arange(len(C))
    for rho in RHO_TIERS:
        R_g, C_g = (R * rho, C * rho ** (-(j + 1.0)))
        for mat in P.materials_main():
            S0 = modal_blocks_density(mat, N_H)
            S1 = closed_S1(mat, R_g, C_g, N_H)
            S2 = closed_S2(mat, R_g, C_g, N_H)
            k1 = float(np.linalg.norm(S1) / np.linalg.norm(S0))
            k2 = float(np.linalg.norm(S2) / np.linalg.norm(S0))
            for t in T_LADDER:
                smap = SectionMap(R * rho, t * C * rho ** (-(j + 1.0)))
                r0, r1, r2 = relatives(assemble_exact(mat, smap, N_H, N_T)['S'], S0, S1, S2, t)
                _, _, r2c = relatives(assemble_exact(mat, smap, N_H, N_T_CHECK)['S'], S0, S1, S2, t)
                drift = abs(r2c - r2) / r2
                if drift > SATURATION_TOL:
                    raise SystemExit(f'saturation gate failed: {tag} {mat.name} rho={rho} t={t} moves {drift:.2%} between n_t = {N_T} and {N_T_CHECK}; nothing written')
                rows.append({'profile': tag, 'rho_out': rho, 'material': mat.name, 't': t, 'rel0': f'{r0:.3e}', 'rel1': f'{r1:.3e}', 'rel2': f'{r2:.3e}', 'rel2_check_n_t': f'{r2c:.3e}', 'saturation_drift': f'{drift:.2e}', 'S1_over_S0': f'{k1:.3e}', 'S2_over_S0': f'{k2:.3e}', 'lowest_covering_order': covering_order(r0, r1, r2)})
            print(f"  {tag} rho={rho} {mat.name:<14} rel2(t=1)={rows[-1]['rel2']} k1={k1:.3e} k2={k2:.3e}", flush=True)
    return rows

def replication_gate(rows: list[dict]) -> float:
    """The horseshoe rows this file computes are the supplied refinement results."""
    shipped = {}
    with SHIPPED_LADDER.open(encoding='utf-8', newline='') as fh:
        for r in csv.DictReader(fh):
            shipped[float(r['rho_out']), r['material'], float(r['t'])] = r
    worst, n = (0.0, 0)
    for r in rows:
        if r['profile'] != 'HS':
            continue
        s = shipped[float(r['rho_out']), r['material'], float(r['t'])]
        for col in ('rel0', 'rel1', 'rel2', 'S1_over_S0', 'S2_over_S0'):
            a, b = (float(r[col]), float(s[col]))
            worst = max(worst, abs(a - b) / abs(b))
            n += 1
    print(f'[gate] replication: {n} horseshoe cells against {SHIPPED_LADDER.name}, worst relative deviation {worst:.3e}')
    if worst >= 0.001:
        raise SystemExit('replication gate failed; nothing written')
    return worst

def scale_gate(C: np.ndarray) -> float:
    """A relative remainder must not move with the map's scale."""
    mat = next((m for m in P.materials_main()))
    S0 = modal_blocks_density(mat, N_H)
    vals = []
    for R in (EX_R, 5.0 * EX_R):
        S1 = closed_S1(mat, R, C, N_H)
        S2 = closed_S2(mat, R, C, N_H)
        S_t = assemble_exact(mat, SectionMap(R, C), N_H, N_T)['S']
        vals.append(relatives(S_t, S0, S1, S2, 1.0)[2])
    dev = abs(vals[1] - vals[0]) / vals[0]
    print(f'[gate] scale: rel2 at R and 5R reads {vals[0]:.6e} and {vals[1]:.6e}, moving {dev:.2e}')
    if dev >= 1e-06:
        raise SystemExit('scale gate failed; nothing written')
    return dev

def _time(fn, reps: int=COST_REPEATS):
    best, out = (float('inf'), None)
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return (best, out)

def cost_rows(C_ex: np.ndarray) -> list[dict]:
    """Material-side base, ray-side kernels, and one constraint solve.

    Timed as minima over repeats: a loaded machine can only make a run slower,
    so the minimum is the least contaminated estimate available.  Ratios are
    what is published -- the three times come from one run, so a uniform change
    in machine speed leaves them where they are.

    The ray-side figure is an UPPER bound on what a second profile has to redo.
    ``closed_S1`` and ``closed_S2`` rebuild the branch data and the qhat ladders
    on every call rather than caching them per material, so the timing here
    charges the new ray with material-side work a caching implementation would
    carry over.  What the split measures without qualification is the other
    side: ``S_0`` is reused entry for entry, and it is a small share of the pair
    -- 0.9 to 1.1 per cent on this profile -- so the reuse that makes a second
    profile cheap is that the kernel contraction scales with the coefficient set,
    three terms against ten, and not that an expensive material-side half carries
    over.
    """
    rays = (('HS', P.R_HS, np.asarray(P.C_HS, dtype=complex)), (PROFILE, EX_R, C_ex))
    rows: list[dict] = []
    for mat in P.materials_main():
        t_base, _ = _time(lambda: modal_blocks_density(mat, COST_N_H))
        for tag, R, C in rays:
            t_ker, _ = _time(lambda R=R, C=C: (closed_S1(mat, R, C, COST_N_H), closed_S2(mat, R, C, COST_N_H)))
            t_ex, _ = _time(lambda R=R, C=C: assemble_exact(mat, SectionMap(R, C), COST_N_H, COST_N_T)['S'], reps=1)
            rows.append({'profile': tag, 'material': mat.name, 'n_h': COST_N_H, 'n_t_exact': COST_N_T, 'repeats': COST_REPEATS, 'material_base_s': round(t_base, 6), 'ray_kernels_s': round(t_ker, 6), 'exact_solve_s': round(t_ex, 4), 'material_base_over_exact': t_base / t_ex, 'ray_kernels_over_exact': t_ker / t_ex, 'material_base_share': t_base / (t_base + t_ker)})
            print(f"  {tag:>22} {mat.name:<14} base {t_base:.4f}s kernels {t_ker:.4f}s exact {t_ex:.3f}s  base share {rows[-1]['material_base_share']:.3f}", flush=True)
    return rows

def write(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {path.relative_to(ROOT).as_posix()} ({len(rows)} rows)')

def main() -> int:
    C_ex = ex_coefficients()
    print('coefficients (c_0..c_3):', ', '.join((f'{v.real:+.8f}' for v in C_ex)))
    convention_gate(C_ex)
    geom = admissibility(SectionMap(EX_R, C_ex))
    print(f"       section: {geom['extent_y']:.4f} across by {geom['extent_x']:.4f} high, crown at {geom['crown_x']:+.4f}, invert at {geom['invert_x']:+.4f}")
    hs = SectionMap(P.R_HS, np.asarray(P.C_HS, dtype=complex))
    hs_geom = admissibility(hs)
    print(f"       horseshoe for comparison: sum j|c_j| = {hs_geom['coefficient_decay_sum']:.4f}")
    scale_gate(C_ex)
    print()
    rows = ladder_rows('HS', P.R_HS, np.asarray(P.C_HS, dtype=complex))
    rows += ladder_rows(PROFILE, EX_R, C_ex)
    replication_gate(rows)
    print()
    costs = cost_rows(C_ex)
    print()
    write(LADDER_CSV, rows)
    write(COST_CSV, costs)
    wall = max((float(r['rel2']) for r in rows if r['profile'] == PROFILE and float(r['rho_out']) == 1.0 and (float(r['t']) == 1.0)))
    wide = max((float(r['rel2']) for r in rows if r['profile'] == PROFILE and float(r['rho_out']) >= 1.5))
    share = [r['material_base_share'] for r in costs if r['profile'] == PROFILE]
    ker = [r['ray_kernels_over_exact'] for r in costs if r['profile'] == PROFILE]
    print(f'\n{PROFILE}: wall grade, full strength : {wall:.4e}')
    print(f'  worst over the two wider grades      : {wide:.4e}')
    print(f'  material base as a share of the pair : {min(share):.3f} to {max(share):.3f}')
    print(f'  ray kernels against one exact solve  : {min(ker):.4f} to {max(ker):.4f}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
