"""Reproduce the W4 finite-box boundary comparison in Section 5.6."""
from __future__ import annotations

import argparse
import csv
import json

import plastic_forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement", type=int, nargs="+", choices=(1, 2),
                        default=[1, 2])
    args = parser.parse_args()
    directory = plastic_forward.elastic.HERE
    inputs = json.loads((directory / "inputs.json").read_text(encoding="utf-8"))
    baseline = inputs["table5"]["dtn"]
    output = directory / "boundary_sensitivity.csv"
    records = {}
    if output.exists():
        with output.open(encoding="utf-8", newline="") as stream:
            records = {int(row["refinement"]): row for row in csv.DictReader(stream)}
    for refinement in args.refinement:
        result = plastic_forward.run(
            "box_rigid_top", refinement,
            baseline["interface_radius_mm"], baseline["support_thickness_mm"],
            baseline["installation_release"], .05,
            lateral_ratio=baseline["lateral_ratio"], save=False, verbose=False)
        records[refinement] = result
        plastic_forward.elastic.write_rows(
            output, [records[key] for key in sorted(records)])
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
