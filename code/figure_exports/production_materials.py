from __future__ import annotations
import csv
import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/closure'))
from exterior_operator import assemble_modal
from anisotropic_elasticity.material import material_F, material_ISO_path
from mapped_elasticity.geometry import circle_map, horseshoe_map
from mapped_elasticity import protocol

def cases():
    shape = horseshoe_map(1.0, protocol.R_HS, protocol.C_HS)
    existing = {(2.625, 45.0), (4.0, 0.0)}
    for band, angles in ((63, (0.0, 22.5, 45.0, 67.5, 90.0)), (127, (0.0, 45.0, 90.0))):
        for ratio in (1.25, 2.625, 4.0):
            for angle in angles:
                if (ratio, angle) not in existing:
                    yield ('horseshoe', ratio, angle, 2.0, band, material_F(ratio, angle), shape.scaled(2.0))
    yield ('horseshoe', 2.625, 45.0, 1.5, 63, material_F(2.625, 45.0), shape.scaled(1.5))
    yield ('horseshoe', 4.0, 0.0, 4.0, 63, material_F(4.0, 0.0), shape.scaled(4.0))
    for band in (63, 127):
        yield ('W4_circle', 1.0, 0.0, '', band, material_ISO_path(0.0, E=20500000.0, nu=0.32), circle_map(0.6))

def main():
    rows = []
    output = ROOT / 'results/data/derived/production_materials.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2):
        for geometry, ratio, angle, radius, band, material, interface in cases():
            modal = assemble_modal(material, interface, band)
            lower = assemble_modal(material, interface, band, resultant_order=band + 32)
            nodes = 2 * (band + 1)
            weights = np.repeat(np.sinc(np.arange(1, band + 1) / nodes) ** 2, 4)
            work = 2 * np.pi / nodes * weights[:, None] * modal * weights[None, :]
            energy = np.linalg.eigvalsh(-(work + work.T) / 2)
            if energy[0] <= 0:
                raise ValueError(f'Nonpositive exterior energy: {geometry}, {ratio}, {angle}, {band}')
            row = dict(geometry=geometry, ratio=ratio, material_angle_deg=angle, rho=radius, n_h=band, interface_nodes=nodes, retained_rank=4 * band, branch='analytic_twoterm' if interface.n_map <= 1 else 'cauchy', resultant_order='' if interface.n_map <= 1 else band + 64, relative_order_gap=np.linalg.norm(modal - lower) / np.linalg.norm(modal), work_reciprocity=np.linalg.norm(work - work.T) / np.linalg.norm(work), minimum_energy_Pa=energy[0], maximum_energy_Pa=energy[-1], projected_condition=np.linalg.cond(work))
            rows.append(row)
            print(row, flush=True)
    with output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
if __name__ == '__main__':
    main()
