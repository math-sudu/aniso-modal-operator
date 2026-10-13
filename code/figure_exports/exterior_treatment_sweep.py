from __future__ import annotations
import argparse
import csv
import gc
import sys
import time
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as spla
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code' / 'closure'))
sys.path.insert(0, str(ROOT / 'code' / 'closure' / 'numeric'))
from closure_env import bridge_pinned
bridge_pinned()
from exec20_src.material import material_F
from exec26_src import protocol as P
from exec26_src.fe_mapped import AnisoAnnulusFE, MappedMesh, condensed_mapped_system, mapped_released_load, solve_condensed, strip_translation
from exec26_src.geometry import horseshoe_map
from exec26_src.ls_ref import LSReference
from modal_condensation import condense_with, nodal_from_modal
from tj_operator_grade_convergence import composed_modal
OUT = ROOT / 'results/data/derived'
P0_V, P0_H, P0_S = (-15000000.0, -13500000.0, 0.0)
RHO = 2.0
NEAR_LADDER = ((12, 64), (24, 128), (48, 256))
CASES = ((1.25, 0.0), (2.625, 45.0), (4.0, 90.0))
SMAP = horseshoe_map(1.0, P.R_HS, P.C_HS)
K_FAR = 14
K_COST = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14)
RHO_TIGHT = 1.25
K_FAR_TIGHT = 44
ROUTES_TIGHT = ('spectral', 'composed')
COST_REPEATS = 15
N_H_LADDER = (1, 2, 3, 4, 6, 8, 12, 16, 24, 31, 47, 63)
N_H_MESH = (24, 128)
REF_N_ORDER, REF_N_COLLOCATION = (48, 1200)
REF_WINDOW = (40, 56)
REF_REL_TOL = 0.0001
GATE_MESH = (48, 256)
PIN_CASE = (2.625, 45.0)

def strip_near(u: np.ndarray, n_near: int) -> np.ndarray:
    """Near-block displacements in the translation quotient."""
    v = u[:2 * n_near].copy()
    v[0::2] -= np.mean(v[0::2])
    v[1::2] -= np.mean(v[1::2])
    return v

def rigid_quotient(v: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Near-block displacements with all three rigid-body modes projected out.

    The finite traction-free reference admits arbitrary translations and
    rotation, selected by its outer nodal constraints. The coupled exterior
    resists rotation. Projecting both near fields onto the same complement
    defines the deformation-comparison metric; it does not change either
    solved field or the coupled problem's displacement representative.
    """
    n = len(xy)
    tx = np.zeros(2 * n)
    tx[0::2] = 1.0
    ty = np.zeros(2 * n)
    ty[1::2] = 1.0
    rot = np.zeros(2 * n)
    rot[0::2], rot[1::2] = (-xy[:, 1], xy[:, 0])
    Q, _ = np.linalg.qr(np.column_stack([tx, ty, rot]))
    return v - Q @ (Q.T @ v)

def solve_neumann_pinned(fe, f: np.ndarray) -> np.ndarray:
    """Pure-traction solve with three degrees of freedom pinned.

    Equivalent to the bordered Lagrange form of the inherited solver up to a
    rigid motion, which the comparison quotients out.  The bordered form adds
    three DENSE columns -- the rigid-body vectors -- and their fill exhausts
    the sparse factorization at the far model's size, where the same matrix
    factorizes without them in seconds.  Pinning both components at one outer
    node and the first component a quarter turn away removes exactly the three
    rigid modes and leaves the system sparse.
    """
    mesh = fe.mesh
    ring = mesh.ring_nodes('outer')
    a, b = (int(ring[0]), int(ring[mesh.n_t // 4]))
    fixed = np.array([2 * a, 2 * a + 1, 2 * b])
    free = np.setdiff1d(np.arange(mesh.n_dof), fixed)
    lu = spla.splu(fe.K[free][:, free].tocsc(), permc_spec='MMD_AT_PLUS_A')
    u = np.zeros(mesh.n_dof)
    u[free] = lu.solve(f[free])
    del lu
    return u

def solve_dirichlet_lean(fe, f: np.ndarray, n_t: int) -> np.ndarray:
    """Solve the clamped-outer-ring system with fill-reducing column ordering."""
    mesh = fe.mesh
    fixed = mesh.ring_dofs('outer')
    free = np.setdiff1d(np.arange(mesh.n_dof), fixed)
    lu = spla.splu(fe.K[free][:, free].tocsc(), permc_spec='MMD_AT_PLUS_A')
    u = np.zeros(mesh.n_dof)
    u[free] = lu.solve(f[free])
    del lu
    return u

def operator_solve(mat, n_r: int, n_t: int, s_scale: float=1.0):
    """Operator-closed solve at ``rho = 2`` with its cost breakdown."""
    t0 = time.perf_counter()
    mesh = MappedMesh(SMAP, RHO, n_r, n_t)
    fe = AnisoAnnulusFE(mesh, mat)
    t_assemble = time.perf_counter() - t0
    t0 = time.perf_counter()
    K_sys, _ = condensed_mapped_system(fe, mat, SMAP, RHO, n_modes=P.N_MODES_GENERAL)
    t_operator = time.perf_counter() - t0
    if s_scale != 1.0:
        K_sys = (fe.K - (fe.K - K_sys) * s_scale).tocsr()
    f = mapped_released_load(mesh, P0_V, P0_H, P0_S)
    t0 = time.perf_counter()
    u = solve_condensed(fe, K_sys, f)
    t_solve = time.perf_counter() - t0
    return (mesh, strip_translation(u), (t_assemble, t_operator, t_solve))

def far_mesh_solve(mat, n_r: int, n_t: int, k: int, bracket: str, lean: bool=False, rho: float=RHO):
    """Plain far-mesh truncation at ``rho_far = rho^k``.

    ``n_r`` layers per interface-radius factor keeps the near block identical
    to the operator route's mesh, which is what makes ``rho ** k`` the only
    admissible remote radius; ``bracket`` is the remote boundary condition,
    ``free`` for a traction-free outer ring and ``fixed`` for a clamped one.
    ``lean`` selects the memory-lean factorization path needed at the
    exhausted radius; the cost ladder leaves it off so both treatments are
    priced on the same solver.
    """
    t0 = time.perf_counter()
    mesh = MappedMesh(SMAP, float(rho ** k), n_r * k, n_t)
    fe = AnisoAnnulusFE(mesh, mat)
    t_assemble = time.perf_counter() - t0
    f = mapped_released_load(mesh, P0_V, P0_H, P0_S)
    t0 = time.perf_counter()
    if bracket == 'free':
        u = solve_neumann_pinned(fe, f) if lean else fe.solve_neumann(f)
    elif lean:
        u = solve_dirichlet_lean(fe, f, n_t)
    else:
        u = fe.solve_dirichlet({'outer': np.zeros((n_t, 2))}, f)
    t_solve = time.perf_counter() - t0
    del fe
    gc.collect()
    return (mesh, u, (t_assemble, 0.0, t_solve))

def assert_shared_near_block(mesh_far, mesh_near, n_near: int) -> float:
    """The far model must carry the near mesh, not merely resemble it."""
    dev = np.max(np.abs(mesh_far.xy[:n_near] - mesh_near.xy[:n_near])) / np.max(np.abs(mesh_near.xy[:n_near]))
    if dev >= 1e-13:
        raise AssertionError(f'near block not shared: max deviation {dev:.3e}')
    return float(dev)

def shared_index(n_r: int, n_t: int) -> np.ndarray:
    """The 832 nodes of the coarsest near mesh, as indices into a finer one.

    One error axis across meshes of different resolution: the node set is held
    fixed while the mesh under it refines.
    """
    n_r0, n_t0 = NEAR_LADDER[0]
    step_r, step_t = (n_r // n_r0, n_t // n_t0)
    return np.array([i * step_r * n_t + j * step_t for i in range(n_r0 + 1) for j in range(n_t0)])

def rel_to_ls(mesh, u, ref, idx=None) -> float:
    """Relative L2 difference from the completed-basis reference.

    ``idx`` restricts the comparison to a node subset, which is how the cost
    ladder keeps one error axis across meshes of different resolution.
    """
    nodes = np.arange(mesh.n_nodes) if idx is None else idx
    dof = np.column_stack([2 * nodes, 2 * nodes + 1]).ravel()
    ux, uy = ref.displacements(mesh.xy[nodes, 0], mesh.xy[nodes, 1])
    u_e = np.empty(2 * len(nodes))
    u_e[0::2], u_e[1::2] = (ux, uy)
    a = strip_near(u[dof], len(nodes))
    b = strip_near(u_e, len(nodes))
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))

def certified_reference(mat, tag: str):
    """Compute the order-48 reference and compare neighbouring resolutions.

Return the reference, displacement difference on shared nodes and the finest-mesh spectral solve. Check displacement sensitivity and wall-traction residuals to detect loss of accuracy at excessive order."""
    per_order = REF_N_COLLOCATION // REF_N_ORDER

    def build(n_o):
        return LSReference(mat, SMAP, P0_V, P0_H, P0_S, n_order=n_o, n_collocation=per_order * n_o)
    n_r, n_t = GATE_MESH
    mesh, u, _ = operator_solve(mat, n_r, n_t)
    idx = shared_index(n_r, n_t)
    ref = build(REF_N_ORDER)
    lo, hi = (build(REF_WINDOW[0]), build(REF_WINDOW[1]))
    better = lo if lo.wall_residual <= hi.wall_residual else hi
    at_pin = rel_to_ls(mesh, u, ref, idx)
    dev = abs(at_pin - rel_to_ls(mesh, u, better, idx)) / at_pin
    worst_nb = max(lo.wall_residual, hi.wall_residual)
    print(f'[gate] reference {tag}: order {REF_N_ORDER} on {REF_N_COLLOCATION} points, wall residual {ref.wall_residual:.3e} against a window maximum of {worst_nb:.3e}, readout moves {dev:.2e}')
    if dev > REF_REL_TOL or ref.wall_residual > worst_nb:
        raise SystemExit(f"reference pin gate failed for {tag}: the {n_r}x{n_t} shared-node level moves {dev:.3e} against the better neighbour of {REF_WINDOW} (tolerance {REF_REL_TOL:.0e}), and the pin's wall residual is {ref.wall_residual:.3e} against a window maximum of {worst_nb:.3e}. Nothing written.")
    return (ref, dev, (mesh, u))

def solver_gate() -> None:
    """Check the lean and bordered solves on the smallest far-domain mesh.

Compare solutions after removing rigid motions before writing results."""
    mat = material_F(1.25, 0.0, 'gate')
    n_r, n_t, k = (12, 64, 8)
    n_near = (n_r + 1) * n_t
    mesh = MappedMesh(SMAP, float(2 ** k), n_r * k, n_t)
    fe = AnisoAnnulusFE(mesh, mat)
    f = mapped_released_load(mesh, P0_V, P0_H, P0_S)
    xy = mesh.xy[:n_near]
    for bracket, inherited, leaned in (('free', fe.solve_neumann(f), solve_neumann_pinned(fe, f)), ('fixed', fe.solve_dirichlet({'outer': np.zeros((n_t, 2))}, f), solve_dirichlet_lean(fe, f, n_t))):
        a = rigid_quotient(inherited[:2 * n_near], xy)
        b = rigid_quotient(leaned[:2 * n_near], xy)
        dev = float(np.linalg.norm(a - b) / np.linalg.norm(a))
        print(f'[gate] {bracket:>5s} bracket, lean vs inherited solve: {dev:.3e}')
        if dev >= 1e-10:
            raise SystemExit('lean solver gate failed; nothing written')
    del fe
    gc.collect()

def stage_reference_ladder() -> list[dict]:
    rows = []
    for ratio, dip in CASES:
        mat = material_F(ratio, dip, f'r{ratio}d{dip}')
        for n_r, n_t in NEAR_LADDER:
            n_near = (n_r + 1) * n_t
            far, far_dof, xy_near = ({}, None, None)
            for bracket in ('free', 'fixed'):
                mesh_far, u_far, _ = far_mesh_solve(mat, n_r, n_t, K_FAR, bracket, lean=True)
                xy_near = mesh_far.xy[:n_near]
                far_dof = mesh_far.n_dof
                far[bracket] = (mesh_far, u_far[:2 * n_near])
            q_free = rigid_quotient(far['free'][1], xy_near)
            q_fixed = rigid_quotient(far['fixed'][1], xy_near)
            bracket_gap = float(np.linalg.norm(q_free - q_fixed) / np.linalg.norm(q_free))
            del q_free, q_fixed
            for s_scale in (1.0, 1.01, 0.0):
                mesh_op, u_op, _ = operator_solve(mat, n_r, n_t, s_scale)
                dev = assert_shared_near_block(far['free'][0], mesh_op, n_near)
                op_t = strip_near(u_op, n_near)
                op_tr = rigid_quotient(u_op[:2 * n_near], xy_near)
                row = {'ratio': ratio, 'dip_deg': dip, 'near_mesh': f'{n_r}x{n_t}', 's_scale': s_scale, 'rho_far': 2 ** K_FAR, 'operator_dof': mesh_op.n_dof, 'far_dof': far_dof, 'shared_block_dev': dev, 'bracket_gap_rel': bracket_gap}
                for bracket, (_, u_far_near) in far.items():
                    row[f'diff_{bracket}'] = float(np.linalg.norm(rigid_quotient(u_far_near, xy_near) - op_tr) / np.linalg.norm(op_tr))
                    row[f'diff_{bracket}_translation_only'] = float(np.linalg.norm(strip_near(u_far_near, n_near) - op_t) / np.linalg.norm(op_t))
                row['bracket_spread'] = abs(row['diff_free'] - row['diff_fixed']) / row['diff_free']
                rows.append(row)
                print(f"[A] ratio={ratio} dip={dip} {n_r}x{n_t} s_scale={s_scale}: free={row['diff_free']:.4e} fixed={row['diff_fixed']:.4e} gap={row['bracket_gap_rel']:.3e} spread={row['bracket_spread']:.2%}", flush=True)
            del far
            gc.collect()
    return rows

def operator_solve_tight(mat, n_r: int, n_t: int, route: str):
    """Operator-closed solve at ``rho = 1.25`` on either assembly route.

``spectral`` is the library assembly; ``composed`` closes the
same near field on the closed-form composition instead, condensed through
the shared module.  ``check_route_replication`` below is what says the two
paths differ in the operator and in nothing else."""
    mesh = MappedMesh(SMAP, RHO_TIGHT, n_r, n_t)
    fe = AnisoAnnulusFE(mesh, mat)
    if route == 'spectral':
        K_sys, _ = condensed_mapped_system(fe, mat, SMAP, RHO_TIGHT, n_modes=P.N_MODES_GENERAL)
    elif route == 'composed':
        n_h = n_t // 2 - 1
        S_mod = composed_modal(mat, RHO_TIGHT, n_h)
        K_sys = condense_with(fe, nodal_from_modal(S_mod, mesh.theta_nodes, n_h))
    else:
        raise ValueError(f'unknown route {route!r}')
    f = mapped_released_load(mesh, P0_V, P0_H, P0_S)
    return (mesh, strip_translation(solve_condensed(fe, K_sys, f)))

def check_route_replication() -> None:
    """Check local synthesis and condensation using the library's spectral matrix.

Agreement with the library's condensed system isolates operator changes from changes in finite-element condensation."""
    from exec26_src.sf_general import assemble_modal_dtn
    n_r, n_t = NEAR_LADDER[0]
    n_h = n_t // 2 - 1
    mat = material_F(2.625, 45.0, 'routegate')
    mesh = MappedMesh(SMAP, RHO_TIGHT, n_r, n_t)
    fe = AnisoAnnulusFE(mesh, mat)
    K_lib, _ = condensed_mapped_system(fe, mat, SMAP, RHO_TIGHT, n_modes=P.N_MODES_GENERAL)
    tier = SMAP.scaled(RHO_TIGHT)
    S_mod = assemble_modal_dtn(mat, tier, n_h, P.N_MODES_GENERAL)['S'][2:, :]
    K_loc = condense_with(fe, nodal_from_modal(S_mod, mesh.theta_nodes, n_h))
    rel = float(abs(K_loc - K_lib).max() / abs(K_lib).max())
    print(f"[gate] composed-route condensation against the library's, fed the library's own modal matrix: {rel:.3e}")
    if rel >= 1e-13:
        raise SystemExit('route replication gate failed; nothing written')

def stage_reference_tight() -> list[dict]:
    """The meshed-exterior comparison at the tight radius, both routes."""
    rows = []
    for ratio, dip in CASES:
        mat = material_F(ratio, dip, f't{ratio}d{dip}')
        for n_r, n_t in NEAR_LADDER:
            n_near = (n_r + 1) * n_t
            far, far_dof, xy_near = ({}, None, None)
            for bracket in ('free', 'fixed'):
                mesh_far, u_far, _ = far_mesh_solve(mat, n_r, n_t, K_FAR_TIGHT, bracket, lean=True, rho=RHO_TIGHT)
                xy_near = mesh_far.xy[:n_near]
                far_dof = mesh_far.n_dof
                far[bracket] = (mesh_far, u_far[:2 * n_near])
            q_free = rigid_quotient(far['free'][1], xy_near)
            q_fixed = rigid_quotient(far['fixed'][1], xy_near)
            bracket_gap = float(np.linalg.norm(q_free - q_fixed) / np.linalg.norm(q_free))
            del q_free, q_fixed
            for route in ROUTES_TIGHT:
                mesh_op, u_op = operator_solve_tight(mat, n_r, n_t, route)
                dev = assert_shared_near_block(far['free'][0], mesh_op, n_near)
                op_t = strip_near(u_op, n_near)
                op_tr = rigid_quotient(u_op[:2 * n_near], xy_near)
                row = {'ratio': ratio, 'dip_deg': dip, 'near_mesh': f'{n_r}x{n_t}', 'route': route, 'rho': RHO_TIGHT, 'rho_far': RHO_TIGHT ** K_FAR_TIGHT, 'operator_dof': mesh_op.n_dof, 'far_dof': far_dof, 'shared_block_dev': dev, 'bracket_gap_rel': bracket_gap}
                for bracket, (_, u_far_near) in far.items():
                    row[f'diff_{bracket}'] = float(np.linalg.norm(rigid_quotient(u_far_near, xy_near) - op_tr) / np.linalg.norm(op_tr))
                    row[f'diff_{bracket}_translation_only'] = float(np.linalg.norm(strip_near(u_far_near, n_near) - op_t) / np.linalg.norm(op_t))
                row['bracket_spread'] = abs(row['diff_free'] - row['diff_fixed']) / row['diff_free']
                rows.append(row)
                print(f"[D] ratio={ratio} dip={dip} {n_r}x{n_t} {route}: free={row['diff_free']:.4e} fixed={row['diff_fixed']:.4e} gap={row['bracket_gap_rel']:.3e} spread={row['bracket_spread']:.2%}", flush=True)
            del far
            gc.collect()
    return rows

def _stat(samples: list[float]) -> tuple[float, float, float]:
    """(median, min, max) of a stage's repeated wall-clock readings."""
    s = sorted(samples)
    n = len(s)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return (med, s[0], s[-1])

def stage_cost_ladder(mat, ref, ref_dev: float, repeats: int=COST_REPEATS) -> list[dict]:
    """One section, one material point, both treatments, one error axis.

    Every wall-clock cell is the median of ``repeats`` runs, and the spread
    columns carry the min and max of the same samples.  A single run cannot
    say whether a sub-0.1 s cell is a measurement or a scheduling artefact, so
    the dispersion is part of the reading rather than a refinement of it.  The
    error column is deterministic and is asserted identical across the repeats
    rather than averaged.
    """
    n_r0, n_t0 = NEAR_LADDER[0]
    n_near0 = (n_r0 + 1) * n_t0
    rows = []

    def emit(treatment, label, mesh, samples, err, extra):
        stats = {k: _stat([s[i] for s in samples]) for i, k in enumerate(('assemble', 'operator', 'solve'))}
        stats['total'] = _stat([sum(s) for s in samples])
        totals = stats['total']
        row = {'treatment': treatment, 'configuration': label, 'total_dof': mesh.n_dof, 'interface_dof': 2 * mesh.n_t, 'repeats': len(samples)}
        for stage in ('assemble', 'operator', 'solve', 'total'):
            med, lo, hi = stats[stage]
            row[f'{stage}_s'] = round(med, 4)
            row[f'{stage}_s_min'] = round(lo, 4)
            row[f'{stage}_s_max'] = round(hi, 4)
            row[f'{stage}_spread_rel'] = round((hi - lo) / med, 4) if med > 0 else ''
        row.update({'rel_error_vs_completed_basis': err, 'ref_n_order': REF_N_ORDER, 'ref_n_collocation': REF_N_COLLOCATION, 'ref_window_dev': f'{ref_dev:.3e}'})
        row.update(extra)
        rows.append(row)
        worst = max(((row[f'{s}_spread_rel'], s) for s in ('assemble', 'operator', 'solve') if row[f'{s}_spread_rel'] != ''))
        print(f"[B] {treatment:>10s} {label:>12s}: dof={mesh.n_dof:7d} t={row['total_s']:7.3f}s [{row['total_s_min']:.3f}, {row['total_s_max']:.3f}] n={len(samples)} worst stage spread {worst[0]:.1%} ({worst[1]}) err={err:.4e}", flush=True)

    def repeat(run, idx):
        """Run one configuration ``repeats`` times; the error must not move."""
        samples, mesh, err = ([], None, None)
        for _ in range(repeats):
            mesh, u, cost = run()
            e = rel_to_ls(mesh, u, ref, idx)
            if err is None:
                err = e
            elif e != err:
                raise RuntimeError(f'deterministic error column moved across repeats: {err!r} -> {e!r}')
            samples.append(cost)
        return (mesh, samples, err)
    for n_r, n_t in NEAR_LADDER:
        mesh, samples, err = repeat(lambda n_r=n_r, n_t=n_t: operator_solve(mat, n_r, n_t), shared_index(n_r, n_t))
        emit('operator', f'{n_r}x{n_t}', mesh, samples, err, {'rho_far': RHO, 'near_mesh': f'{n_r}x{n_t}'})
    for k in K_COST:
        for bracket in ('free', 'fixed'):
            mesh, samples, err = repeat(lambda k=k, b=bracket: far_mesh_solve(mat, n_r0, n_t0, k, b), np.arange(n_near0))
            emit(f'far_mesh_{bracket}', f'rho_far=2^{k}', mesh, samples, err, {'rho_far': 2 ** k, 'near_mesh': f'{n_r0}x{n_t0}'})
    return rows

def stage_harmonic_selection(certified: dict) -> list[dict]:
    """Retained harmonics against attained accuracy, by decay ratio."""
    n_r, n_t = N_H_MESH
    rows = []
    for ratio, dip in CASES:
        mat, ref, ref_dev = certified[ratio, dip]
        decay = float(np.max(np.abs(mat.qhats)))
        mesh = MappedMesh(SMAP, RHO, n_r, n_t)
        fe = AnisoAnnulusFE(mesh, mat)
        f = mapped_released_load(mesh, P0_V, P0_H, P0_S)
        for n_h in N_H_LADDER:
            t0 = time.perf_counter()
            K_sys, _ = condensed_mapped_system(fe, mat, SMAP, RHO, n_h=n_h, n_modes=P.N_MODES_GENERAL)
            t_operator = time.perf_counter() - t0
            u = strip_translation(solve_condensed(fe, K_sys, f))
            err = rel_to_ls(mesh, u, ref)
            rows.append({'ratio': ratio, 'dip_deg': dip, 'decay_ratio': decay, 'near_mesh': f'{n_r}x{n_t}', 'n_harmonics': n_h, 'operator_s': round(t_operator, 4), 'rel_error_vs_completed_basis': err, 'ref_n_order': REF_N_ORDER, 'ref_n_collocation': REF_N_COLLOCATION, 'ref_window_dev': f'{ref_dev:.3e}'})
            print(f'[C] ratio={ratio} n_h={n_h:3d}: err={err:.4e} ({t_operator:.2f}s)', flush=True)
    return rows

def write_csv(name: str, rows: list[dict]) -> None:
    path = OUT / name
    with path.open('w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'  wrote {path.relative_to(ROOT)} ({len(rows)} rows)')
STAGES = ('reference', 'reference_tight', 'cost', 'harmonic')

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description='Compare exterior treatments and near-field refinement.')
    ap.add_argument('--stages', nargs='*', default=list(STAGES), choices=list(STAGES), help='which products to rebuild; a product not selected keeps the file it already wrote')
    ap.add_argument('--repeats', type=int, default=COST_REPEATS, help='wall-clock repeats behind every cost-ladder cell; the shipped table is read from the default')
    args = ap.parse_args(argv)
    if args.repeats < 10:
        ap.error('--repeats below 10 cannot support the dispersion the cost table reports')
    OUT.mkdir(parents=True, exist_ok=True)
    need = set(CASES) if 'harmonic' in args.stages else {PIN_CASE}
    certified = {}
    for ratio, dip in CASES:
        if (ratio, dip) not in need:
            continue
        mat = material_F(ratio, dip, f'r{ratio}d{dip}')
        ref, dev, gate_solve = certified_reference(mat, f'ratio={ratio} dip={dip}')
        certified[ratio, dip] = (mat, ref, dev)
    if 'reference' in args.stages or 'reference_tight' in args.stages:
        solver_gate()
    if 'reference_tight' in args.stages:
        check_route_replication()
    print()
    if 'reference' in args.stages:
        write_csv('exterior_reference_ladder.csv', stage_reference_ladder())
        print()
    if 'reference_tight' in args.stages:
        write_csv('exterior_reference_tight.csv', stage_reference_tight())
        print()
    if 'cost' in args.stages:
        write_csv('exterior_cost_ladder.csv', stage_cost_ladder(*certified[PIN_CASE], repeats=args.repeats))
        print()
    if 'harmonic' in args.stages:
        write_csv('harmonic_selection.csv', stage_harmonic_selection(certified))
        print()
if __name__ == '__main__':
    main()
