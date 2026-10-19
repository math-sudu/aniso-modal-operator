from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from closure_env import bridge_pinned
cfg = bridge_pinned()
from anisotropic_elasticity.material import material_F
from mapped_elasticity import protocol as P
from mapped_elasticity.fe_mapped import AnisoAnnulusFE, MappedMesh, condensed_mapped_system, mapped_released_load, solve_condensed, strip_translation
from mapped_elasticity.geometry import horseshoe_map
from mapped_elasticity.ls_ref import LSReference
from mapped_elasticity.sf_twoterm import modal_blocks_density
from modal_condensation import condense_with, nodal_from_modal
from graded_kernels import closed_S1, closed_S2
FAILURES: list[str] = []
RAN: list[str] = []
ROOT = Path(__file__).resolve().parents[3]
P0_V, P0_H, P0_S = (-15000000.0, -13500000.0, 0.0)
RHO = 1.25
RHO_PUBLISHED = 2.0
LADDER = ((12, 64), (24, 128), (48, 256))
CASES = ((1.25, 0.0), (2.625, 45.0), (4.0, 90.0), (4.0, 0.0), (1.25, 90.0))
REF_N_ORDER, REF_N_COLLOCATION = (48, 1200)
REF_CHECK_N_ORDER, REF_CHECK_N_COLLOCATION = (64, 1600)
REF_SETTLE_TOL = 0.02
SMAP = horseshoe_map(1.0, P.R_HS, P.C_HS)

def _report(label: str, ok: bool, detail: str) -> None:
    RAN.append(label)
    print(f"{('PASS' if ok else 'FAIL')}  {label}: {detail}")
    if not ok:
        FAILURES.append(label)

def graded_map_constants(rho: float) -> tuple[float, np.ndarray]:
    """Return the scaled-map constants R*rho and c_j*rho^(-j-1)."""
    C = np.asarray(P.C_HS, dtype=complex)
    j = np.arange(len(C))
    return (P.R_HS * rho, C * rho ** (-(j + 1.0)))

def composed_modal(mat, rho: float, n_h: int, order: int=2) -> np.ndarray:
    """S0 + S1 + S2 at full mapping strength t = 1.

The order argument truncates the composition; order 0 keeps only S0."""
    R_g, C_g = graded_map_constants(rho)
    S = modal_blocks_density(mat, n_h)
    if order >= 1:
        S = S + closed_S1(mat, R_g, C_g, n_h)
    if order >= 2:
        S = S + closed_S2(mat, R_g, C_g, n_h)
    return S

def _solve(fe, mesh, K_sys) -> np.ndarray:
    f_load = mapped_released_load(mesh, P0_V, P0_H, P0_S)
    return strip_translation(solve_condensed(fe, K_sys, f_load))

def scored(mesh, u_c, ref) -> float:
    """Relative L2 difference between one operator solve and one reference."""
    ux, uy = ref.displacements(mesh.xy[:, 0], mesh.xy[:, 1])
    u_e = np.empty(mesh.n_dof)
    u_e[0::2], u_e[1::2] = (ux, uy)
    u_e = strip_translation(u_e)
    return float(np.linalg.norm(u_c - u_e) / np.linalg.norm(u_e))

def solve_spectral(mat, nr: int, nt: int, s_scale: float=1.0):
    """Solve with the general spectral exterior; return the mesh and displacement."""
    mesh = MappedMesh(SMAP, RHO, nr, nt)
    fe = AnisoAnnulusFE(mesh, mat)
    K_sys, _ = condensed_mapped_system(fe, mat, SMAP, RHO, n_modes=P.N_MODES_GENERAL)
    if s_scale != 1.0:
        A = (fe.K - K_sys) * s_scale
        K_sys = (fe.K - A).tocsr()
    return (mesh, _solve(fe, mesh, K_sys))

def solve_composed(mat, nr: int, nt: int, s_scale: float=1.0, order: int=2):
    """Solve with S_f from the closed-form composition S0 + S1 + S2."""
    mesh = MappedMesh(SMAP, RHO, nr, nt)
    fe = AnisoAnnulusFE(mesh, mat)
    n_h = nt // 2 - 1
    S_mod = composed_modal(mat, RHO, n_h, order=order)
    S_dens = nodal_from_modal(S_mod, mesh.theta_nodes, n_h) * s_scale
    return (mesh, _solve(fe, mesh, condense_with(fe, S_dens)))

def rel_spectral(mat, ref, nr: int, nt: int, s_scale: float=1.0) -> float:
    return scored(*solve_spectral(mat, nr, nt, s_scale), ref)

def rel_composed(mat, ref, nr: int, nt: int, s_scale: float=1.0, order: int=2) -> float:
    return scored(*solve_composed(mat, nr, nt, s_scale, order), ref)

def check_library_condensation() -> None:
    """Check local synthesis against the library assembly."""
    from mapped_elasticity.sf_general import assemble_modal_dtn
    nr, nt = (12, 64)
    n_h = nt // 2 - 1
    mat = material_F(2.625, 45.0, 'condensation_check')
    mesh = MappedMesh(SMAP, RHO, nr, nt)
    fe = AnisoAnnulusFE(mesh, mat)
    _, S_lib = condensed_mapped_system(fe, mat, SMAP, RHO, n_modes=P.N_MODES_GENERAL)
    tier = SMAP.scaled(RHO)
    S_mod = assemble_modal_dtn(mat, tier, n_h, P.N_MODES_GENERAL)['S'][2:, :]
    S_loc = nodal_from_modal(S_mod, mesh.theta_nodes, n_h)
    rel = float(np.linalg.norm(S_loc - S_lib) / np.linalg.norm(S_lib))
    R_g, C_g = graded_map_constants(RHO)
    dR = abs(tier.R - R_g) / abs(R_g)
    dC = float(np.max(np.abs(np.asarray(tier.c) - C_g)) / np.max(np.abs(C_g)))
    _report('local synthesis reproduces the library condensation', rel < 1e-13 and dR < 1e-13 and (dC < 1e-13), f'nodal density agrees to {rel:.2e}; graded constants agree to R {dR:.2e}, c {dC:.2e} (all must be < 1e-13)')

def check_collapsed_annulus() -> None:
    """rho = 1.0 is not a near field -- measured, not asserted."""
    with np.errstate(divide='ignore', invalid='ignore'):
        mesh = MappedMesh(SMAP, 1.0, 12, 64)
        n_radii = len(np.unique(mesh.r_node))
        fe = AnisoAnnulusFE(mesh, material_F(2.625, 45.0, 'collapsed_annulus'))
        bad = int(np.count_nonzero(~np.isfinite(fe.K.toarray())))
    mesh_ok = MappedMesh(SMAP, RHO, 12, 64)
    _report('the wall grade is a limit, not a configuration', n_radii == 1 and bad > 0 and (len(np.unique(mesh_ok.r_node)) == 13), f'rho = 1.0 collapses 13 node rings onto {n_radii} radius and leaves {bad} non-finite stiffness entries, while rho = {RHO} keeps {len(np.unique(mesh_ok.r_node))} distinct rings')

def strength(rho: float) -> float:
    tier = SMAP.scaled(rho) if rho != 1.0 else SMAP
    return float(sum((j * abs(c) for j, c in enumerate(tier.c))))

def build_tables():
    """Solve each mesh and compare its displacement with the reference.

Retain the finest-mesh solution for the reference-order comparison."""
    tabs = {}
    for ratio, dip in CASES:
        mat = material_F(ratio, dip, f'r{ratio}d{dip}')
        ref = LSReference(mat, SMAP, P0_V, P0_H, P0_S, n_order=REF_N_ORDER, n_collocation=REF_N_COLLOCATION)
        sol_s = [solve_spectral(mat, nr, nt) for nr, nt in LADDER]
        sol_c = [solve_composed(mat, nr, nt) for nr, nt in LADDER]
        spec = [scored(mesh, u_c, ref) for mesh, u_c in sol_s]
        comp = [scored(mesh, u_c, ref) for mesh, u_c in sol_c]
        tabs[ratio, dip] = (spec, comp, mat, ref, {'spectral': sol_s[-1], 'composed': sol_c[-1]})
    return tabs

def check_reference_truncation(tabs) -> dict:
    """Check reference truncation at the finest near-field mesh.

Re-evaluate each assembly route against a higher-order reference to distinguish operator error from reference truncation error."""
    ok = True
    shown = []
    out = {}
    for key, (spec, comp, mat, _, fine) in tabs.items():
        ref_hi = LSReference(mat, SMAP, P0_V, P0_H, P0_S, n_order=REF_CHECK_N_ORDER, n_collocation=REF_CHECK_N_COLLOCATION)
        for route, series in (('spectral', spec), ('composed', comp)):
            mesh, u_c = fine[route]
            hi = scored(mesh, u_c, ref_hi)
            moved = abs(hi - series[-1]) / series[-1]
            ok &= moved <= REF_SETTLE_TOL
            out[key, route] = hi
            shown.append(f'ratio={key[0]} {route}: {moved * 100:.3f}%')
    _report(f'reference resolved (finest rung moves under {REF_SETTLE_TOL * 100:.0f}% at order {REF_CHECK_N_ORDER})', ok, '; '.join(shown))
    return out

def check_rate(tabs, which: int, label: str) -> None:
    ok = True
    shown = []
    for key, vals in tabs.items():
        series = vals[which]
        r = series[-2] / series[-1]
        ok &= 3.8 <= r <= 4.2
        shown.append(f'ratio={key[0]}: {r:.2f}')
    _report(label, ok, '; '.join(shown) + ' (target [3.8, 4.2])')

def check_composition_error(tabs) -> float:
    """The two routes agree well inside the finest mesh's own level."""
    ok = True
    shown = []
    worst = 0.0
    for key, (spec, comp, _, _, _) in tabs.items():
        gap = abs(spec[-1] - comp[-1]) / spec[-1]
        worst = max(worst, gap)
        ok &= gap < 1.0
        shown.append(f'ratio={key[0]}: routes differ by {gap:.2e} of the finest level {spec[-1]:.3e}')
    _report('the composition is not the binding approximation', ok, '; '.join(shown) + ' -- each must sit under 1.0, i.e. the route difference stays inside the discretization it is measured in')
    return worst

def negative_controls(tabs, composition_error: float) -> dict:
    """Evaluate scaled-operator and base-only controls; return base-only errors."""
    (ratio, dip), (spec, comp, mat, ref, _) = next(iter(tabs.items()))
    bad = [rel_composed(mat, ref, nr, nt, s_scale=1.01) for nr, nt in LADDER[-2:]]
    r_bad = bad[0] / bad[1]
    _report('Negative control (composed S_f scaled by 1.01)', not 3.8 <= r_bad <= 4.2, f'1 percent operator error gives finest ratio {r_bad:.2f}, outside [3.8, 4.2] as required (clean composed ratio {comp[-2] / comp[-1]:.2f})')
    base = [rel_composed(mat, ref, nr, nt, order=0) for nr, nt in LADDER]
    base_ratios = [base[i - 1] / base[i] for i in range(1, len(base))]
    gap_base = abs(spec[-1] - base[-1]) / spec[-1]
    _report('Negative control (perturbation layer dropped, base S0 only)', gap_base > max(composition_error, 1e-12) * 10.0 and (not all((3.8 <= r <= 4.2 for r in base_ratios))), f'dropping S1 + S2 moves the route difference from {composition_error:.2e} to {gap_base:.2e} of the finest level, more than a decade, and leaves the sequence ' + ' -> '.join((f'{v:.4e}' for v in base)) + ' at ratios ' + ', '.join((f'{r:.2f}' for r in base_ratios)) + f' showing the effect of omitting the correction terms')
    return {(ratio, dip): base}

def export(tabs, reference_errors, base) -> Path:
    """Write refinement results, reference resolutions and base-only errors.

Export the complete refinement sequences so their convergence and relative error levels can be evaluated from the recorded data."""
    out = ROOT / 'results/data/derived/grade_convergence_ladder.csv'
    lines = ['ratio,dip_deg,near_mesh,route,ref_n_order,ref_n_collocation,rel_difference']
    pin = f'{REF_N_ORDER},{REF_N_COLLOCATION}'
    for (ratio, dip), (spec, comp, _, _, _) in tabs.items():
        for name, series in (('spectral', spec), ('composed', comp)):
            for (nr, nt), v in zip(LADDER, series):
                lines.append(f'{ratio},{dip},{nr}x{nt},{name},{pin},{v!r}')
    nr, nt = LADDER[-1]
    for ((ratio, dip), route), v in reference_errors.items():
        lines.append(f'{ratio},{dip},{nr}x{nt},{route},{REF_CHECK_N_ORDER},{REF_CHECK_N_COLLOCATION},{v!r}')
    for (ratio, dip), series in base.items():
        for (bnr, bnt), v in zip(LADDER, series):
            lines.append(f'{ratio},{dip},{bnr}x{bnt},invariant_base,{pin},{v!r}')
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    print(f'wrote {out.relative_to(ROOT).as_posix()} ({len(lines) - 1} rows)')
    return out

def main() -> int:
    print(f'section  : horseshoe (P.R_HS, P.C_HS), interface grade rho = {RHO} (published ladders run {RHO_PUBLISHED})')
    print(f'strength : sum_j j|c_j| = {strength(RHO):.4f} at this grade against {strength(RHO_PUBLISHED):.4f} at {RHO_PUBLISHED} and {strength(1.0):.4f} on the wall')
    print(f'routes   : A general spectral assembly (n_modes = {P.N_MODES_GENERAL}); B closed-form composition S0 + S1 + S2')
    print(f'reference: LSReference completed-basis collocation (full unbounded domain, no S_f), n_order = {REF_N_ORDER}, n_collocation = {REF_N_COLLOCATION}')
    print(f'ladder   : ' + ', '.join((f'{a}x{b}' for a, b in LADDER)))
    print()
    check_library_condensation()
    if FAILURES:
        print('\nOperator convergence checks failed: the replication gate failed, so no route-B number below would mean anything.')
        return 1
    check_collapsed_annulus()
    print()
    tabs = build_tables()
    hdr = f"{'case':18s}{'route':10s}" + ''.join((f"{f'{a}x{b}':>13s}" for a, b in LADDER))
    print(hdr)
    for (ratio, dip), (spec, comp, _, ref, _) in tabs.items():
        for name, series in (('spectral', spec), ('composed', comp)):
            print(f"{f'ratio={ratio} dip={dip}':18s}{name:10s}" + ''.join((f'{v:13.4e}' for v in series)) + (f'   wall_resid {ref.wall_residual:.2e}' if name == 'spectral' else ''))
    print()
    check_rate(tabs, 0, 'h^2 rate at the tight grade, spectral route')
    check_rate(tabs, 1, 'h^2 rate at the tight grade, composed route')
    composition_error = check_composition_error(tabs)
    reference_errors = check_reference_truncation(tabs)
    print()
    base = negative_controls(tabs, composition_error)
    print()
    export(tabs, reference_errors, base)
    print()
    if len(RAN) != 8:
        print(f'Operator convergence checks failed: {len(RAN)} checks ran, expected 8 -- the suite did not complete: {RAN}')
        return 1
    if FAILURES:
        print(f'Operator convergence checks failed: {FAILURES}')
        return 1
    print('Operator convergence checks passed: at the tight interface grade both assembly')
    print('  routes fall at h^2 against the completed-basis reference and')
    print('  agree inside the finest mesh, the reference is resolved where')
    print('  the difference is smallest, and the ladder resolves the')
    print('  perturbation layer it is there to test; controls fire.')
    return 0
if __name__ == '__main__':
    sys.exit(main())
