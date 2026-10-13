from __future__ import annotations
import argparse
import importlib.util
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
capacity = load('lining_capacity')
contact = load('frictional_contact')
plasticity = load('localized_plasticity')
thickness = load('lining_thickness')
pole = load('pole_placement')
lined = capacity.lined

class Operators:

    def __init__(self, ratio, dip, nt, refined_spectral=False):
        self.identity = dict(ratio=ratio, material_dip_deg=dip, bedding_dip_deg=90 - dip, n_t=nt, n_h=nt // 2 - 1, rho=lined.RHO)
        mat = lined.material_F(ratio, dip)
        tier = lined.SMAP.scaled(lined.RHO)
        nh = nt // 2 - 1
        spectral = pole.assemble_modal_dtn(mat, tier, nh, 96, n_grid=512)
        self.modal = {'spectral': spectral['S'][2:]}
        if refined_spectral:
            modal, stats = pole.assemble(mat, tier, nh, {'eta': 0.97}, 384, 2048)
            self.modal['spectral_refined'] = modal
            self.refined_stats = stats
        for extra in (32, 64):
            out = pole.assemble_exact(mat, tier, nh, nh + extra, eta_override=0.97)
            self.modal[f'cauchy_{extra}'] = out['S']
        ref = self.modal['cauchy_64']
        self.rows = []
        for route, modal in self.modal.items():
            self.rows.append(dict(**self.identity, operator=route, spectral_modes=96 if route == 'spectral' else 384 if route == 'spectral_refined' else '', spectral_grid=512 if route == 'spectral' else 2048 if route == 'spectral_refined' else '', cauchy_resultant_order=nh + int(route.split('_')[1]) if route.startswith('cauchy') else '', cauchy_radius=0.97 if route.startswith('cauchy') else '', relative_operator_difference=float(np.linalg.norm(modal - ref) / np.linalg.norm(ref))))
            if route == 'spectral_refined':
                self.rows[-1].update(self.refined_stats)
        theta = 2 * np.pi * np.arange(nt) / nt
        self.nodal = {route: pole.nodal_from_modal(modal, theta, nh) for route, modal in self.modal.items()}
        print('Operators', self.identity, self.rows[0]['relative_operator_difference'], 'reference change', self.rows[1]['relative_operator_difference'], flush=True)

    def assembler(self, route):

        def assemble(fe, mat, smap, rho, **kwargs):
            if fe.mesh.n_t != self.identity['n_t'] or rho != self.identity['rho']:
                raise ValueError('The cached exterior does not match this FE interface')
            density = self.nodal[route]
            return (pole.condense_with(fe, density), density)
        return assemble

def lining_record(ring, ur, identity):
    row, profiles = capacity.recover(ring, ur, identity)
    n = np.array([p['thrust_MN_per_m'] for p in profiles])
    m = np.array([p['moment_kNm_per_m'] for p in profiles])
    angles = lined.station_theta(ring.n_t)
    im = int(np.argmax(np.abs(m)))
    ic = int(np.argmax([p['compression_MPa'] for p in profiles]))
    row.update(peak_thrust_MN_per_m=float(np.abs(n).max()), peak_moment_kNm_per_m=float(abs(m[im])), moment_at_peak_kNm_per_m=float(m[im]), moment_peak_theta_deg=float(angles[im]), compression_peak_theta_deg=float(angles[ic]))
    return (row, profiles)

def contact_case(bank, nr):
    rows, profiles, interfaces = ([], [], [])
    ratio, dip = (bank.identity['ratio'], bank.identity['material_dip_deg'])
    states = {}
    for route in bank.modal:
        model = contact.ContactSection(nr, bank.identity['n_t'], ratio, dip, exterior_system=bank.assembler(route))
        bonded_peak = None
        for mu in (None, 0.0, 0.53, 0.73, 1.15):
            result = model.solve(0.0 if mu is None else mu, bonded=mu is None)
            row, _, interface = model.records(result, 'bonded' if mu is None else 'Coulomb', 0.0 if mu is None else mu, 1)
            identity = dict(case='contact', operator=route, **row)
            row, profile = lining_record(model.ring, result['lining'], identity)
            if mu is None:
                bonded_peak = row['peak_moment_kNm_per_m']
            row['moment_change_from_bonded_pct'] = 100 * (row['peak_moment_kNm_per_m'] / bonded_peak - 1)
            states[route, mu] = result['state'].copy()
            rows.append(row)
            profiles.extend(profile)
            interfaces.extend((dict(operator=route, **p) for p in interface))
            print('Contact', route, ratio, dip, mu, row['peak_moment_kNm_per_m'], flush=True)
        lengths = model.length
    for row in rows:
        mu = None if row['law'] == 'bonded' else row['mu']
        mismatch = states[row['operator'], mu] != states['cauchy_64', mu]
        row['contact_state_disagreement_fraction'] = float(lengths[mismatch].sum() / lengths.sum())
    return (rows, profiles, interfaces)

def bonded_case(bank, nr, layers, patch=False):
    nt = bank.identity['n_t']
    ratio, dip = (bank.identity['ratio'], bank.identity['material_dip_deg'])
    mesh = lined.MappedMesh(lined.SMAP, lined.RHO, nr, nt)
    mat = lined.material_F(ratio, dip)
    fe = lined.AnisoAnnulusFE(mesh, mat)
    pressure = 0.9 if patch else 0.6
    force = lined.mapped_released_load(mesh, *lined.insitu(pressure))
    rows, profiles = ([], [])
    for route in bank.modal:
        kg, _ = bank.assembler(route)(fe, mat, lined.SMAP, lined.RHO)
        for change in (0.0, -0.1) if patch else (0.0,):
            t = thickness.thickness(mesh.theta_nodes, 247.5, change, 20.0)
            ring = lined.LiningRing(lined.SMAP, mesh.theta_nodes, t, layers)
            if ring.min_detj() <= 0:
                raise ValueError('Inverted lining element')
            _, _, ug, ur = lined._assemble_and_solve(mesh, kg, force, lined.E_LIN, pin_ring=None, ring=ring)
            identity = dict(case='foot_thinning' if patch else 'release_control', operator=route, **bank.identity, n_r=nr, layers=layers, pressure_coefficient=pressure, thickness_change_m=change, patch_center_deg=247.5 if patch else '', patch_halfwidth_deg=20.0 if patch else '')
            row, profile = lining_record(ring, ur, identity)
            dv, dh = lined.closures(mesh, ug)
            row.update(closure_vertical_mm=dv, closure_horizontal_mm=dh, lambda_lower=max(0.0, 1 - 20.0 / row['compression_MPa']))
            rows.append(row)
            profiles.extend(profile)
            print('Bonded', identity['case'], route, nt, change, row['compression_MPa'], row['lambda_lower'], flush=True)
    return (rows, profiles)

def plastic_case(bank, nr):
    rows, profiles, final = ([], [], {})
    for route in ('spectral', 'cauchy_64'):
        model = plasticity.CoupledSection(nr, bank.identity['n_t'], 2.0, 5.0, exterior_system=bank.assembler(route))
        u = np.zeros(model.mesh.n_dof)
        plastic = np.zeros((*model.weight.shape, 4))
        work, installed_at = (0.0, None)
        for lo, hi, stage in ((0.0, 0.82, 'unsupported'), (0.82, 1.0, 'supported')):
            if stage == 'supported':
                installed_at = u.copy()
            for release in np.linspace(lo, hi, int(np.ceil((hi - lo) / 0.02)) + 1)[1:]:
                old = plastic
                u, plastic, stress, iterations = model.solve_step(u, old, release, installed_at)
                increment = np.sum(stress * (plastic - old), axis=-1)
                if increment.min() < -1e-08:
                    raise RuntimeError('Negative plastic dissipation')
                work += float(np.sum(increment * model.weight))
                row = model.row(u, plastic, stress, release, stage, installed_at, work, iterations)
                row.update(case='plasticity', operator=route, installation_release=0.82, maximum_load_step=0.02, **bank.identity)
                if installed_at is not None:
                    wall = (u - installed_at)[model.wall]
                    ur = np.concatenate([model.ring_transfer @ wall, wall])
                    row, profile = lining_record(model.ring, ur, row)
                    profiles.extend(profile)
                rows.append(row)
                print('Plastic', route, stage, round(release, 4), row['closure_vertical_mm'], flush=True)
        eq = np.sqrt(2 / 3 * np.sum((plastic @ plasticity.DEV) ** 2, axis=-1))
        final[route] = dict(eq=eq, u=u, stress=stress)
    difference = final['spectral']['eq'] > 1e-10
    difference ^= final['cauchy_64']['eq'] > 1e-10
    disagreement = float(model.weight[difference].sum())
    for row in rows:
        if row['release'] == 1.0:
            row['plastic_state_disagreement_area_m2'] = disagreement
    return (rows, profiles)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=('contact', 'thickness', 'bonded', 'plastic', 'all'), default='all')
    parser.add_argument('--output-prefix', type=Path, default=Path(__file__).resolve().parents[2] / 'results/data/derived/support_operator')
    parser.add_argument('--refined-spectral', action='store_true', help='Include the 384/2048, radius-0.97 spectral block in the bonded control')
    args = parser.parse_args()
    prefix = args.output_prefix
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        for study in ('contact', 'thickness', 'bonded', 'plastic'):
            if args.study not in ('all', study):
                continue
            rows, profiles, operators, interfaces = ([], [], [], [])
            specs = {'contact': [(4.0, 45.0, 48, 256, 16), (1.25, 0.0, 48, 256, 16)], 'thickness': [(2.625, 45.0, 48, 256, 32)], 'bonded': [(1.25, 90.0, 24, 128, 16), (1.25, 90.0, 48, 256, 32)], 'plastic': [(4.0, 0.0, 48, 256, 16)]}[study]
            for ratio, dip, nr, nt, layers in specs:
                bank = Operators(ratio, dip, nt, refined_spectral=args.refined_spectral and study == 'bonded')
                operators.extend(bank.rows)
                if study == 'contact':
                    rr, pp, ii = contact_case(bank, nr)
                    interfaces.extend(ii)
                elif study == 'plastic':
                    rr, pp = plastic_case(bank, nr)
                else:
                    rr, pp = bonded_case(bank, nr, layers, patch=study == 'thickness')
                rows.extend(rr)
                profiles.extend(pp)
            for suffix, data in ((study, rows), (study + '_profiles', profiles), (study + '_operators', operators), (study + '_interface', interfaces)):
                if data:
                    fields = list(dict.fromkeys((k for row in data for k in row)))
                    data = [{key: row.get(key, '') for key in fields} for row in data]
                    capacity.write_csv(prefix.with_name(prefix.name + '_' + suffix + '.csv'), data)
            print('Saved', study, 'elapsed', time.perf_counter() - start, flush=True)
if __name__ == '__main__':
    main()
