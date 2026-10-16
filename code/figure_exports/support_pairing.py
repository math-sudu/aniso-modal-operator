import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from threadpoolctl import threadpool_limits
spec = importlib.util.spec_from_file_location('support', Path(__file__).with_name('support_operator.py'))
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)
lined = support.lined
from exterior_operator import assemble_modal, p1_projection, projected_work
from modal_condensation import condense_work, nodal_from_modal
from exec26_src.fe_mapped import theta_ring_mass, solve_condensed, strip_translation

class Pairings:

    def __init__(self, ratio, dip, nt):
        self.identity = dict(ratio=ratio, material_dip_deg=dip, bedding_dip_deg=90 - dip, n_t=nt, n_h=nt // 2 - 1, rho=lined.RHO)
        nh = nt // 2 - 1
        theta = 2 * np.pi * np.arange(nt) / nt
        modal = assemble_modal(lined.material_F(ratio, dip), lined.SMAP.scaled(lined.RHO), nh)
        self.modal = {'cauchy_64': modal, 'projected': modal}
        mass = np.kron(theta_ring_mass(nt), np.eye(2))
        self.work = {'cauchy_64': mass @ nodal_from_modal(modal, theta, nh), 'projected': projected_work(modal, theta)}
        self.rows, self.traces = ([], [])
        transform = p1_projection(theta, nh)
        weights = np.repeat(np.sinc(np.arange(1, nh + 1) / nt) ** 2, 4)
        basis = transform.T / weights[None, :] * np.sqrt(nt / 2)
        for name, work in self.work.items():
            quotient = basis.T @ work @ basis
            energy = np.linalg.eigvalsh(-(quotient + quotient.T) / 2)
            assert energy[0] > 0
            self.rows.append(dict(**self.identity, operator=name, work_reciprocity_defect=np.linalg.norm(work - work.T) / np.linalg.norm(work), minimum_exterior_eigenvalue=energy[0], work_difference=np.linalg.norm(work - self.work['cauchy_64']) / np.linalg.norm(self.work['cauchy_64'])))

    def assembler(self, route):

        def assemble(fe, mat, smap, rho, **kwargs):
            assert fe.mesh.n_t == self.identity['n_t'] and rho == self.identity['rho']
            work = self.work[route]
            return (condense_work(fe, work), work)
        return assemble

def check_projection():
    spec = importlib.util.spec_from_file_location('trace', Path(__file__).with_name('trace_order.py'))
    trace = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trace)
    nt, nh = (32, 15)
    theta = 2 * np.pi * np.arange(nt) / nt
    fe = SimpleNamespace(p=1, nt=nt, nth=nt, dt=2 * np.pi / nt)
    analytic = p1_projection(theta, nh)
    numerical = trace.CurvedFE.projection(fe, nh)
    np.testing.assert_allclose(analytic, numerical, atol=2e-15, rtol=2e-13)
    rigid = np.tile(np.eye(2), (nt, 1))
    nyquist = np.repeat((-1.0) ** np.arange(nt), 2)[:, None] * rigid
    np.testing.assert_allclose(analytic @ np.c_[rigid, nyquist], 0, atol=5e-15)
    print('Exact P1 projection, translation and Nyquist checks passed', flush=True)

def unlined_case(bank, nr, pressure):
    nt = bank.identity['n_t']
    mat = lined.material_F(bank.identity['ratio'], bank.identity['material_dip_deg'])
    mesh = lined.MappedMesh(lined.SMAP, lined.RHO, nr, nt)
    fe = lined.AnisoAnnulusFE(mesh, mat)
    force = lined.mapped_released_load(mesh, *lined.insitu(pressure))
    fields, rows = ({}, [])
    if nr == 12:
        from exec26_src.ls_ref import LSReference
        ref = LSReference(mat, lined.SMAP, *lined.insitu(pressure), n_order=48, n_collocation=1200)
        ur = strip_translation(np.column_stack(ref.displacements(mesh.xy[:, 0], mesh.xy[:, 1])).ravel())
    for route in bank.modal:
        system, _ = bank.assembler(route)(fe, mat, lined.SMAP, lined.RHO)
        u = strip_translation(solve_condensed(fe, system, force))
        fields[route] = u
        dv, dh = lined.closures(mesh, u)
        row = dict(**bank.identity, n_r=nr, pressure_coefficient=pressure, operator=route, closure_vertical_mm=dv, closure_horizontal_mm=dh, displacement_difference_from_cauchy=np.linalg.norm(u - fields['cauchy_64']) / np.linalg.norm(fields['cauchy_64']))
        if nr == 12:
            row['reference_error'] = np.linalg.norm(u - ur) / np.linalg.norm(ur)
        rows.append(row)
    print('Unlined', rows, flush=True)
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=('contact', 'thickness', 'bonded', 'plastic', 'unlined', 'all'), default='all')
    args = parser.parse_args()
    prefix = Path(__file__).resolve().parents[2] / 'results/data/derived/support_pairing'
    with threadpool_limits(limits=2):
        check_projection()
        if args.study in ('all', 'unlined'):
            rows = []
            for ratio, dip, nr, nt, pressure in ((2.625, 45.0, 12, 64, 0.9), (2.625, 0.0, 24, 128, 1.4), (2.625, 0.0, 48, 256, 1.4)):
                rows.extend(unlined_case(Pairings(ratio, dip, nt), nr, pressure))
            fields = list(dict.fromkeys((k for row in rows for k in row)))
            support.capacity.write_csv(prefix.with_name(prefix.name + '_unlined.csv'), [{k: row.get(k, '') for k in fields} for row in rows])
        for study, specs in {'bonded': [(1.25, 90.0, 24, 128, 16), (1.25, 90.0, 48, 256, 32)], 'contact': [(4.0, 45.0, 48, 256, 16), (1.25, 0.0, 48, 256, 16)], 'thickness': [(2.625, 45.0, 48, 256, 32)], 'plastic': [(4.0, 0.0, 48, 256, 16)]}.items():
            if args.study not in ('all', study):
                continue
            rows, profiles, operators, interfaces = ([], [], [], [])
            for ratio, dip, nr, nt, layers in specs:
                bank = Pairings(ratio, dip, nt)
                operators.extend(bank.rows)
                if study == 'contact':
                    rr, pp, ii = support.contact_case(bank, nr)
                    interfaces.extend(ii)
                elif study == 'plastic':
                    rr, pp = support.plastic_case(bank, nr, routes=('cauchy_64', 'projected'))
                else:
                    rr, pp = support.bonded_case(bank, nr, layers, patch=study == 'thickness')
                rows.extend(rr)
                profiles.extend(pp)
            for suffix, data in ((study, rows), (study + '_profiles', profiles), (study + '_operators', operators), (study + '_interface', interfaces)):
                if data:
                    fields = list(dict.fromkeys((k for row in data for k in row)))
                    support.capacity.write_csv(prefix.with_name(prefix.name + '_' + suffix + '.csv'), [{k: row.get(k, '') for k in fields} for row in data])
            print('Saved', study, flush=True)
if __name__ == '__main__':
    main()
