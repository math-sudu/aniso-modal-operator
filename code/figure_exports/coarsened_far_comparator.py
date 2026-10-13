from __future__ import annotations
import csv
import gc
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
from exec20_src.material import material_F
from exec26_src import protocol as P
from exec26_src.fe_mapped import AnisoAnnulusFE, MappedMesh, mapped_released_load
from exterior_treatment_sweep import NEAR_LADDER, P0_H, P0_S, P0_V, PIN_CASE, REF_N_COLLOCATION, REF_N_ORDER, RHO, SMAP, certified_reference, far_mesh_solve, operator_solve, rel_to_ls, _stat
OUT = ROOT / 'results/data/derived/coarsened_far_comparator.csv'
SHIPPED_COST = ROOT / 'results/data/derived/exterior_cost_ladder.csv'
RHO_FAR = 16.0
GROWTH_LADDER = (1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5)
ERROR_SLACK = 0.02
COST_REPEATS = 15
CLOCK_ROW = ('operator', '12x64')

def coarsened_mesh(n_r_near: int, n_t: int, rho_far: float, growth: float):
    """Operator-route near mesh, then exterior rings at a fixed ratio.

    The near rings are ``rho^(i / n_r_near)`` for ``i = 0..n_r_near``, which is
    exactly what ``MappedMesh(SMAP, RHO, n_r_near, n_t)`` lays down, so the
    near block is shared with the condensed model rather than resembling it.
    The exterior takes the fewest rings at ``growth`` or finer that reach
    ``rho_far``, and the ratio is then set exactly by that count so the last
    ring lands on the remote radius.
    """
    near = RHO ** (np.arange(n_r_near + 1) / n_r_near)
    n_out = int(np.ceil(np.log(rho_far / RHO) / np.log(growth)))
    g = (rho_far / RHO) ** (1.0 / n_out)
    outer = RHO * g ** np.arange(1, n_out + 1)
    radii = np.concatenate([near, outer])
    radii[-1] = rho_far
    n_r = len(radii) - 1
    mesh = MappedMesh(SMAP, rho_far, n_r, n_t)
    mesh.radii = radii
    rr = np.repeat(radii, n_t)
    tt = np.tile(mesh.theta_nodes, n_r + 1)
    mesh.r_node, mesh.th_node = (rr, tt)
    z = SMAP.omega(rr * np.exp(1j * tt))
    mesh.xy = np.column_stack([np.real(z), np.imag(z)])
    return (mesh, g, n_out)

def coarsened_solve(mat, n_r_near: int, n_t: int, rho_far: float, growth: float, bracket: str='free'):
    """One comparator solve with its cost breakdown, timed as the ladder is."""
    t0 = time.perf_counter()
    mesh, g, n_out = coarsened_mesh(n_r_near, n_t, rho_far, growth)
    fe = AnisoAnnulusFE(mesh, mat)
    t_assemble = time.perf_counter() - t0
    f = mapped_released_load(mesh, P0_V, P0_H, P0_S)
    t0 = time.perf_counter()
    if bracket == 'free':
        u = fe.solve_neumann(f)
    else:
        u = fe.solve_dirichlet({'outer': np.zeros((n_t, 2))}, f)
    t_solve = time.perf_counter() - t0
    del fe
    gc.collect()
    return (mesh, u, (t_assemble, 0.0, t_solve), (g, n_out))

def near_block_gate(mesh, mesh_near, n_near: int) -> float:
    dev = np.max(np.abs(mesh.xy[:n_near] - mesh_near.xy[:n_near])) / np.max(np.abs(mesh_near.xy[:n_near]))
    print(f"[gate] comparator near block against the condensed model's: max deviation {dev:.3e}")
    if dev >= 1e-13:
        raise SystemExit('near-block fidelity gate failed; nothing written')
    return float(dev)

def shipped_row(treatment: str, configuration: str) -> dict:
    with SHIPPED_COST.open(encoding='utf-8', newline='') as fh:
        for r in csv.DictReader(fh):
            if r['treatment'] == treatment and r['configuration'] == configuration:
                return r
    raise SystemExit(f'{SHIPPED_COST.name} carries no {treatment} row at {configuration}; nothing written')

def clock_gate(mat, ref, idx) -> dict:
    """Re-time the condensed row and refuse a machine in a different state."""
    n_r, n_t = NEAR_LADDER[0]
    samples, err = ([], None)
    for _ in range(COST_REPEATS):
        mesh, u, cost = operator_solve(mat, n_r, n_t)
        e = rel_to_ls(mesh, u, ref, idx)
        err = e if err is None else err
        samples.append(sum(cost))
    med, lo, hi = _stat(samples)
    ship = shipped_row(*CLOCK_ROW)
    s_lo, s_hi = (float(ship['total_s_min']), float(ship['total_s_max']))
    s_err = float(ship['rel_error_vs_completed_basis'])
    inside = s_lo <= med <= s_hi
    print(f'[gate] condensed {n_r}x{n_t} re-timed: median {med:.4f}s [{lo:.4f}, {hi:.4f}] against the shipped range [{s_lo:.4f}, {s_hi:.4f}]; error {err:.6e} against {s_err:.6e}')
    if not inside:
        raise SystemExit('clock fidelity gate failed: this machine is not in the state the shipped cost table was measured on, so a second in this file cannot be read beside one in that table. Nothing written.')
    if abs(err - s_err) / s_err >= 1e-06:
        raise SystemExit("the condensed row's error column moved against the shipped table; nothing written")
    return {'median': med, 'min': lo, 'max': hi, 'error': err}

def emit(rows: list[dict], treatment: str, label: str, mesh, samples: list[tuple], err: float, extra: dict) -> None:
    stats = {k: _stat([s[i] for s in samples]) for i, k in enumerate(('assemble', 'operator', 'solve'))}
    stats['total'] = _stat([sum(s) for s in samples])
    row = {'treatment': treatment, 'configuration': label, 'total_dof': mesh.n_dof, 'interface_dof': 2 * mesh.n_t, 'n_rings': mesh.n_r, 'repeats': len(samples)}
    for stage in ('assemble', 'operator', 'solve', 'total'):
        med, lo, hi = stats[stage]
        row[f'{stage}_s'] = round(med, 4)
        row[f'{stage}_s_min'] = round(lo, 4)
        row[f'{stage}_s_max'] = round(hi, 4)
        row[f'{stage}_spread_rel'] = round((hi - lo) / med, 4) if med > 0 else ''
    row.update({'rel_error_vs_completed_basis': err, 'ref_n_order': REF_N_ORDER, 'ref_n_collocation': REF_N_COLLOCATION})
    row.update(extra)
    rows.append(row)
    print(f"[X] {treatment:>18s} {label:>16s}: dof={mesh.n_dof:6d} rings={mesh.n_r:3d} t={row['total_s']:7.4f}s [{row['total_s_min']:.4f}, {row['total_s_max']:.4f}] err={err:.4e}", flush=True)

def main() -> None:
    mat = material_F(*PIN_CASE, f'r{PIN_CASE[0]}d{PIN_CASE[1]}')
    ref, ref_dev, gate_solve = certified_reference(mat, f'ratio={PIN_CASE[0]} dip={PIN_CASE[1]}')
    n_r0, n_t0 = NEAR_LADDER[0]
    n_near0 = (n_r0 + 1) * n_t0
    idx = np.arange(n_near0)
    mesh_near, _, _ = operator_solve(mat, n_r0, n_t0)
    probe, _, _ = coarsened_mesh(n_r0, n_t0, RHO_FAR, GROWTH_LADDER[0])
    near_dev = near_block_gate(probe, mesh_near, n_near0)
    clock = clock_gate(mat, ref, idx)
    print()
    rows: list[dict] = []
    k = int(round(np.log2(RHO_FAR)))
    samples, err_ss, mesh_ss = ([], None, None)
    for _ in range(COST_REPEATS):
        mesh_ss, u, cost = far_mesh_solve(mat, n_r0, n_t0, k, 'free')
        e = rel_to_ls(mesh_ss, u, ref, idx)
        err_ss = e if err_ss is None else err_ss
        samples.append(cost)
    emit(rows, 'far_mesh_free', f'rho_far=2^{k}', mesh_ss, samples, err_ss, {'rho_far': RHO_FAR, 'near_mesh': f'{n_r0}x{n_t0}', 'growth_ratio': '', 'n_exterior_rings': mesh_ss.n_r - n_r0, 'grading': 'self-similar', 'shipped_row': False, 'near_block_dev': near_dev, 'clock_gate_median_s': clock['median'], 'ref_window_dev': f'{ref_dev:.3e}'})
    for growth in GROWTH_LADDER:
        samples, err, mesh_c, meta = ([], None, None, None)
        for _ in range(COST_REPEATS):
            mesh_c, u, cost, meta = coarsened_solve(mat, n_r0, n_t0, RHO_FAR, growth, 'free')
            e = rel_to_ls(mesh_c, u, ref, idx)
            err = e if err is None else err
            samples.append(cost)
        emit(rows, 'coarsened_free', f'growth={growth:.2f}', mesh_c, samples, err, {'rho_far': RHO_FAR, 'near_mesh': f'{n_r0}x{n_t0}', 'growth_ratio': round(meta[0], 6), 'n_exterior_rings': meta[1], 'grading': 'coarsened', 'shipped_row': False, 'near_block_dev': near_dev, 'clock_gate_median_s': clock['median'], 'ref_window_dev': f'{ref_dev:.3e}'})
    free = [r for r in rows if r['treatment'] == 'coarsened_free']
    ok = [r for r in free if r['rel_error_vs_completed_basis'] <= err_ss * (1.0 + ERROR_SLACK)]
    if not ok:
        raise SystemExit('no coarsening in the ladder stays inside the slack; the shipped comparator cannot be selected by the declared rule. Nothing written.')
    pick = max(ok, key=lambda r: r['growth_ratio'])
    pick['shipped_row'] = True
    print(f"\n[pick] {pick['configuration']} at growth {pick['growth_ratio']:.4f} over {pick['n_exterior_rings']} exterior rings: {pick['total_dof']} dof, {pick['total_s']:.4f}s, error {pick['rel_error_vs_completed_basis']:.4e} against the self-similar {err_ss:.4e} at the same radius (slack {ERROR_SLACK:.0%})")
    with OUT.open('w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'  wrote {OUT.relative_to(ROOT)} ({len(rows)} rows)')
if __name__ == '__main__':
    main()
