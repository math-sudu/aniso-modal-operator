import argparse
import csv
import importlib.util
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'results/data/derived'
spec = importlib.util.spec_from_file_location('plasticity', Path(__file__).with_name('localized_plasticity.py'))
plasticity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plasticity)

def first_yield(law, initial, increment):
    """First exit of an affine stress path from the convex DP elastic domain."""

    def excess(release):
        return float(law.yield_value(initial + release * increment).max())
    if excess(0.0) >= 0.0 or excess(1.0) <= 0.0:
        raise ValueError('The prescribed study requires first yield between releases 0 and 1')
    return brentq(excess, 0.0, 1.0, xtol=1e-12)

def check_onset():
    pressure = 10.0
    law = plasticity.DruckerPrager(np.eye(4), cohesion=2.0)
    initial = np.array([-pressure, -pressure, -0.5 * pressure, 0.0])
    increment = np.array([pressure, -pressure, 0.0, 0.0])
    expected = np.sqrt(((law.intercept + 2.5 * law.alpha * pressure) ** 2 - 0.25 * pressure ** 2) / (3 * pressure ** 2))
    np.testing.assert_allclose(first_yield(law, initial, increment), expected, atol=1e-12)
    print('Analytic DP first-yield check passed', flush=True)

def onset_case(nr, nt, history_name, operator, check_history):
    model = plasticity.CoupledSection(nr, nt, 2.0, 5.0)
    unlined = model.linear_solve(model.elastic, model.load)
    increment = model.strain(unlined) @ model.law.c.T
    release = first_yield(model.law, model.s0, increment)
    stress = model.s0 + release * increment
    values = model.law.yield_value(stress)
    index = np.unravel_index(np.argmax(values), values.shape)
    xy, sig = (model.xy[index], stress[index])
    lower = np.floor(release / 0.02) * 0.02
    upper = lower + 0.02
    assert lower < release < upper
    u0 = np.zeros(model.mesh.n_dof)
    ep0 = np.zeros((*model.weight.shape, 4))
    u, ep, _, _ = model.solve_step(u0, ep0, lower, None)
    np.testing.assert_allclose(u, lower * unlined, atol=1e-12, rtol=1e-09)
    np.testing.assert_array_equal(ep, ep0)
    u, ep, returned, _ = model.solve_step(u, ep, upper, None)
    equivalent = np.sqrt(2 / 3 * np.sum((ep @ plasticity.DEV) ** 2, axis=-1))
    area = float(model.weight[equivalent > 1e-10].sum())
    assert equivalent.max() > 0.0
    assert model.law.yield_value(returned).max() < 1e-08
    if check_history:
        with (DATA / history_name).open(encoding='utf-8') as stream:
            history = [r for r in csv.DictReader(stream) if r['stage'] != 'elastic_superposition' and r.get('operator', operator) == operator]
        first = next((i for i, r in enumerate(history) if float(r['plastic_area_m2']) > 0.0))
        before, after = history[first - 1:first + 1]
        np.testing.assert_allclose([lower, upper], [float(before['release']), float(after['release'])], atol=1e-14)
        np.testing.assert_allclose(area, float(after['plastic_area_m2']), atol=1e-12)
        np.testing.assert_allclose(plasticity.lined.closures(model.mesh, u), [float(after['closure_vertical_mm']), float(after['closure_horizontal_mm'])], rtol=1e-07, atol=1e-06)
        installation = max((r for r in history if r['stage'] == 'unsupported'), key=lambda r: float(r['release']))
        assert float(installation['release']) == 0.82
        assert float(installation['plastic_area_m2']) > 0.0
        assert float(installation['peak_thrust_MN_per_m']) == 0.0
        assert float(installation['peak_moment_kNm_per_m']) == 0.0
        assert all((float(r['outer_layer_plastic_strain']) == 0.0 for r in history))
        assert max((float(r['boundary_elastic_yield_MPa']) for r in history)) < 0.0
        print(f'{nr}x{nt}: saved first plastic increment reproduced; installation and exterior containment checked', flush=True)
    dv, dh = plasticity.lined.closures(model.mesh, release * unlined)
    print(f'{nr}x{nt}: first yield {release:.9f}, incremental bracket ({lower:g}, {upper:g})', flush=True)
    return dict(n_r=nr, n_t=nt, rho=2.0, cohesion_MPa=5.0, friction_deg=30.0, dilation_deg=0.0, operator='projected', release=release, sampled_elastic_release=lower, sampled_plastic_release=upper, rho_at_onset=float(model.rho_gp[index]), x_physical_m=float(-xy[1]), y_physical_m=float(xy[0]), sigma_raw_xx_MPa=float(sig[0]), sigma_raw_yy_MPa=float(sig[1]), sigma_zz_MPa=float(sig[2]), sigma_raw_xy_MPa=float(sig[3] / plasticity.SQRT2), yield_residual_MPa=float(values.max()), closure_vertical_mm=dv, closure_horizontal_mm=dh, elastic_yield_at_installation_MPa=float(model.law.yield_value(model.s0 + 0.82 * increment).max()))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DATA / 'dp_onset.csv')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--check-history', action='store_true', help='Compare the onset steps with the saved staged histories')
    args = parser.parse_args()
    check_onset()
    if args.check_only:
        return
    with threadpool_limits(limits=2):
        rows = [onset_case(24, 128, 'localized_plasticity.csv', 'spectral', args.check_history), onset_case(48, 256, 'support_pairing_plastic.csv', 'projected', args.check_history)]
    plasticity.write_csv(args.output, rows)
    print(f'Saved {args.output}', flush=True)
if __name__ == '__main__':
    main()
