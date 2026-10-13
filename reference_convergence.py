"""Recompute the order-48/64 wall-reference displacement comparison."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'lib'))
from exec20_src.material import material_F
from exec26_src import protocol as P
from exec26_src.geometry import horseshoe_map
from exec26_src.fe_mapped import MappedMesh, strip_translation
from exec26_src.ls_ref import LSReference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--central-only', action='store_true')
    parser.add_argument('--output', type=Path,
                        default=ROOT/'results/data/derived/reference_self_convergence.csv')
    args = parser.parse_args()
    shape = horseshoe_map(1., P.R_HS, P.C_HS)
    cases = [(2.625, 45.)] if args.central_only else [
        (1.25, 0.), (2.625, 45.), (4., 90.), (1.25, 90.), (4., 0.)]
    rows = []
    with threadpool_limits(limits=2):
        for ratio, dip in cases:
            mat = material_F(ratio, dip)
            refs = [LSReference(mat, shape, -15e6, -13.5e6, 0.,
                                n_order=n, n_collocation=25*n) for n in (48, 64)]
            for nr, nt in ((12, 64), (24, 128), (48, 256), (96, 512)):
                mesh = MappedMesh(shape, 2., nr, nt)
                fields = []
                for reference in refs:
                    ux, uy = reference.displacements(mesh.xy[:, 0], mesh.xy[:, 1])
                    fields.append(strip_translation(np.column_stack([ux, uy]).ravel()))
                gap = float(np.linalg.norm(fields[0]-fields[1])/np.linalg.norm(fields[1]))
                rows.append(dict(ratio=ratio, material_dip_deg=dip,
                    bedding_dip_deg=90-dip, n_r=nr, n_t=nt, order=48,
                    collocation=1200, comparison_order=64, comparison_collocation=1600,
                    relative_displacement_difference=gap))
                print(ratio, dip, nr, nt, gap, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
