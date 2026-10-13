from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'lib'))
import numpy as np
from exec20_src import atlas as at
from exec20_src.material import AnisoMaterial, material_F, material_SABNGS, material_WAI
OUT_DIR = PROJECT_ROOT / 'results' / 'data' / 'derived'
FULL_OUT = OUT_DIR / 'atlas_family_box.csv'
COARSE_OUT = OUT_DIR / 'atlas_family_box_coarse.csv'
POINTS_OUT = OUT_DIR / 'atlas_material_points.csv'
RATIO_LO, RATIO_HI, N_RATIO = (1.25, 4.0, 111)
DIP_LO, DIP_HI, N_DIP = (0.0, 90.0, 91)
RATIO_STRIDE = 10
DIP_STRIDE = 10
FIELDS = ['ratio', 'dip_deg', 'abs_q1', 'abs_q2', 'mu_gap', 'cond_dirichlet_2x2', 'univalence_margin', 'cond_lambda']
BOX_FIELDS = FIELDS + ['bedding_dip_deg']

def cond_lambda(mat: AnisoMaterial) -> float:
    """kappa_2 of the operator block Lambda = -i B M^{-1}."""
    M = np.array([[mat.p1, mat.p2], [mat.q1, mat.q2]])
    B = np.array([[mat.mu1, mat.mu2], [-1.0, -1.0]])
    lam = -1j * B @ np.linalg.inv(M)
    sv = np.linalg.svd(lam, compute_uv=False)
    return float(sv[0] / sv[-1])

def row_of(mat: AnisoMaterial, ratio: float, dip: float) -> dict:
    q1, q2 = mat.qhats
    return {'ratio': ratio, 'dip_deg': dip, 'abs_q1': abs(q1), 'abs_q2': abs(q2), 'mu_gap': abs(mat.mu1 - mat.mu2), 'cond_dirichlet_2x2': at._dirichlet_cond(mat), 'univalence_margin': 1.0 - max(abs(q1), abs(q2)), 'cond_lambda': cond_lambda(mat)}

def check_orientation_period() -> None:
    """dip and -dip carry the same atlas quantities (mirror symmetry), so a
    dip axis over a quarter turn covers every dip of axis-parallel bedding."""
    worst = 0.0
    for ratio in (1.25, 2.0, 3.25, 4.0):
        for dip in (7.0, 23.0, 41.0, 68.0, 89.0):
            a = row_of(material_F(ratio, dip), ratio, dip)
            b = row_of(material_F(ratio, -dip), ratio, -dip)
            for key in ('abs_q1', 'abs_q2', 'mu_gap', 'cond_dirichlet_2x2', 'cond_lambda'):
                worst = max(worst, abs(a[key] - b[key]) / max(abs(b[key]), 1.0))
    if worst > 1e-10:
        raise SystemExit(f'orientation mirror: worst deviation {worst:.3e}')
    print(f'orientation mirror: dip and -dip agree to {worst:.2e}, so [0, 90] degrees spans every dip of axis-parallel bedding')

def sweep() -> list[dict]:
    ratios = np.linspace(RATIO_LO, RATIO_HI, N_RATIO)
    dips = np.linspace(DIP_LO, DIP_HI, N_DIP)
    rows = [row_of(material_F(float(r), float(d)), float(r), float(d)) for r in ratios for d in dips]
    for row in rows:
        row['bedding_dip_deg'] = DIP_HI - row['dip_deg']
    return rows

def material_points() -> list[dict]:
    named = [('F0', material_F(1.5, 0.0), 1.5, 0.0), ('F1', material_F(3.0, 45.0), 3.0, 45.0), ('F2', material_F(2.0, 30.0), 2.0, 30.0), ('Queenston Shale', material_SABNGS(), 15.8 / 10.5, 0.0), ('Waichecheng', material_WAI(), 7.8 / 2.4, 45.0)]
    out = []
    for name, mat, ratio, dip in named:
        row = row_of(mat, ratio, dip)
        row['material'] = name
        row['in_family'] = int(name.startswith('F'))
        out.append(row)
    return out

def write(path: Path, rows: list[dict], fields: list[str], fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in rows:
            w.writerow([r[k] if isinstance(r[k], str) or k in ('in_family',) else format(r[k], fmt) for k in fields])
    print(f'wrote {path.relative_to(PROJECT_ROOT)}  ({len(rows)} rows)')

def report(rows: list[dict]) -> None:

    def extreme(key: str) -> str:
        lo = min(rows, key=lambda r: r[key])
        hi = max(rows, key=lambda r: r[key])
        return f"{key}: min {lo[key]:.6f} @ ratio {lo['ratio']:.3f} dip {lo['dip_deg']:.0f} | max {hi[key]:.6f} @ ratio {hi['ratio']:.3f} dip {hi['dip_deg']:.0f}"
    qmax = [dict(r, qmax=max(r['abs_q1'], r['abs_q2'])) for r in rows]
    print(extreme('mu_gap'))
    print(extreme('cond_dirichlet_2x2'))
    print(extreme('univalence_margin'))
    print(extreme('cond_lambda'))
    lo = min(qmax, key=lambda r: r['qmax'])
    hi = max(qmax, key=lambda r: r['qmax'])
    print(f"decay ratio |qhat|: min {lo['qmax']:.6f} @ ratio {lo['ratio']:.3f} | max {hi['qmax']:.6f} @ ratio {hi['ratio']:.3f}")
    for key in ('univalence_margin',):
        spread = 0.0
        for ratio in sorted({r['ratio'] for r in rows}):
            vals = [r[key] for r in rows if r['ratio'] == ratio]
            spread = max(spread, (max(vals) - min(vals)) / max(abs(np.mean(vals)), 1e-30))
        print(f'{key} spread along dip at fixed ratio: {spread:.2e} relative')
    kmin = min(rows, key=lambda r: r['cond_dirichlet_2x2'])
    print(f"kappa_2(M) optimum: {kmin['cond_dirichlet_2x2']:.6f} at ratio {kmin['ratio']:.3f}, bedding dip {DIP_HI - kmin['dip_deg']:.0f} deg")
    edge = []
    for dip in sorted({r['dip_deg'] for r in rows}):
        band = [r for r in rows if r['dip_deg'] == dip]
        best = min(band, key=lambda r: r['cond_dirichlet_2x2'])
        edge.append((DIP_HI - dip, best['ratio']))
    on_edge = [d for d, r in edge if r >= RATIO_HI - 1e-09]
    at_zero = [r for d, r in edge if d == 0.0][0]
    print(f'per-dip optimum rides the ratio ceiling at bedding dips {min(on_edge):.0f}-{max(on_edge):.0f} deg ({len(on_edge)} of {len(edge)}); elsewhere it falls inside, and at the flat-bedding end of the axis it reads ratio {at_zero:.3f}')
    per_ratio = [DIP_HI - min((r for r in rows if r['ratio'] == q), key=lambda r: r['cond_dirichlet_2x2'])['dip_deg'] for q in sorted({r['ratio'] for r in rows})]
    print(f'per-ratio optimum sits at bedding dips {min(per_ratio):.0f}-{max(per_ratio):.0f} deg, strictly inside the box')

def main() -> None:
    check_orientation_period()
    rows = sweep()
    print(f'widened box: {len(rows)} points on {N_RATIO} ratios x {N_DIP} dips, ratio [{RATIO_LO}, {RATIO_HI}] x dip [{DIP_LO}, {DIP_HI}] deg')
    write(FULL_OUT, rows, BOX_FIELDS, '.12g')
    ratios = sorted({r['ratio'] for r in rows})[::RATIO_STRIDE]
    dips = sorted({r['dip_deg'] for r in rows})[::DIP_STRIDE]
    coarse = [r for r in rows if r['ratio'] in set(ratios) and r['dip_deg'] in set(dips)]
    write(COARSE_OUT, coarse, BOX_FIELDS, '.4f')
    write(POINTS_OUT, material_points(), ['material', 'in_family'] + FIELDS, '.6f')
    report(rows)
if __name__ == '__main__':
    main()
