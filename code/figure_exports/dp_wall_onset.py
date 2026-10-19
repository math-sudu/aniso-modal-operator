import csv
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits
import near_field_clouds as clouds
ROOT = Path(__file__).resolve().parents[2]
import elastic_admissibility as ea
import localized_plasticity as lp

def main():
    with threadpool_limits(limits=2):
        mat = lp.lined.material_F(4.0, 0.0)
        shape = lp.lined.SMAP
        law = lp.DruckerPrager(lp.stiffness_3d(mat), cohesion=5.0, friction=30.0)
        initial = np.array([-15.0, -9.0, -10.95, 0.0])
        clouds.REF_N_MODES = 512
        clouds.REF_N_COLLOCATION = 6144
        clouds.REF_ETA_RING = 0.97
        ref = clouds.RingReference(mat, shape, *lp.lined.insitu(0.6))
        theta = 2.0 * np.pi * np.arange(2880) / 2880
        z = shape.omega(np.exp(1j * theta))
        sig = np.column_stack(ref.stresses_perturbation(z.real, z.imag)) / 1000000.0
        increment = np.column_stack([sig[:, :2], ea.axial_stress(sig, mat), np.sqrt(2.0) * sig[:, 2]])
        release = brentq(lambda a: float(law.yield_value(initial + a * increment).max()), 0.0, 1.0, xtol=1e-12)
        row = {'reference': 'elastic_complex_potential_wall', 'criterion': 'Drucker_Prager_c5_phi30', 'n_modes': 512, 'n_collocation': 6144, 'wall_stations': 2880, 'eta_ring': 0.97, 'release': float(release), 'wall_traction_residual': float(ref.wall_residual)}
    output = ROOT / 'results/data/derived/dp_wall_onset.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(row, flush=True)
if __name__ == '__main__':
    main()
