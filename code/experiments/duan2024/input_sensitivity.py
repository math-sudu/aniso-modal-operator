"""Reproduce the four one-at-a-time W4 input comparisons in Table 5(b)."""
from __future__ import annotations

import argparse
import csv

import numpy as np

import plastic_forward

elastic = plastic_forward.elastic
CASES = {
    "thickness": (4., .70, .83),
    "installation": (6., .50, .83),
    "lower_k": (6., .70, .73),
    "upper_k": (6., .70, .93),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs="+", choices=tuple(CASES), default=list(CASES))
    args = parser.parse_args()
    output = elastic.HERE / "input_sensitivity.csv"
    records = {}
    if output.exists():
        with output.open(encoding="utf-8", newline="") as stream:
            records = {row["case"]: row for row in csv.DictReader(stream)}
    mesh_path = elastic.HERE / "supported_dtn_r1_R600_t6_a0.7_s0.05_k0.83.npz"
    with np.load(mesh_path) as saved:
        mesh = tuple(saved[key].copy() for key in ("xy", "cells", "outer", "wall", "gauges"))
    original_mesh = elastic.make_mesh
    elastic.make_mesh = lambda *args, **kwargs: mesh
    try:
        for name in args.case:
            thickness, installation, lateral_ratio = CASES[name]
            result = plastic_forward.run(
                "dtn", 1, 600., thickness, installation, .05,
                lateral_ratio=lateral_ratio, save=False, verbose=False)
            if result["max_outer_plastic_strain"] != 0.:
                raise RuntimeError(f"Plasticity reaches the elastic exterior in {name}")
            records[name] = dict(case=name, **result)
            elastic.write_rows(output, [records[k] for k in CASES if k in records])
            print(name, result, flush=True)
    finally:
        elastic.make_mesh = original_mesh


if __name__ == "__main__":
    main()
