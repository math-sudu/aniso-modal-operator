"""Generate the numerical inputs used by the manuscript figure scripts."""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

STUDIES = {
    "geometry": [("code/figure_exports/section_family_boundary.py",),
                 ("code/figure_exports/hs_section_geometry.py",)],
    "materials": [("code/figure_exports/atlas_family_box.py",),
                  ("code/closure/numeric/tv_constraint_conditioning_law.py",)],
    "kernels": [("code/figure_exports/closed_kernel_ladder.py",)],
    "fields": [("code/figure_exports/near_field_clouds.py",)],
    "exterior": [("code/figure_exports/exterior_treatment_sweep.py",
                  "--stages", "reference", "cost")],
    "lining": [("code/figure_exports/lined_section_demo.py",),
               ("code/figure_exports/lining_capacity.py",),
               ("code/figure_exports/support_screening.py",)],
    "contact": [("code/figure_exports/frictional_contact.py", "--mesh", "48", "256",
                 "--output", "results/data/derived/frictional_contact_mesh_refined.csv")],
}


def field_geometry():
    sys.path.insert(0, str(ROOT / "code/figure_exports"))
    from moxi_geometry import sample

    folder = ROOT / "results/data/field/moxi675"
    geometry = json.loads((folder / "geometry_reconstruction.json").read_text(encoding="utf-8"))
    levels = [gauge["left_endpoint_m"][1] for gauge in geometry["gauge_coordinates"]]
    arcs = geometry["analytic_definitions"]["initial_shotcrete_intrados"]["circular_arcs"]
    points, tags = sample(arcs, math.radians(0.5), levels)
    output = folder / "geometry_contours.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["contour", "point_index", "x_m", "y_m", "segment"])
        for index, ((x, y), tag) in enumerate(zip(points, tags)):
            writer.writerow(["initial_shotcrete_intrados", index, f"{x:.9f}", f"{y:.9f}", tag])
    print(f"Wrote {output.relative_to(ROOT)} ({len(points)} points)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["inputs", "figures", *STUDIES], default="inputs", nargs="?")
    parser.add_argument("--list", action="store_true", help="Print commands without running them")
    args = parser.parse_args()
    commands = ([command for study in STUDIES.values() for command in study]
                if args.study == "figures" else STUDIES.get(args.study, []))
    if args.list:
        print("Generate the Moxi contour from the supplied arc definitions.")
        for command in commands:
            print("python -B " + " ".join(command))
        return
    field_geometry()
    (ROOT / "results/data/derived").mkdir(parents=True, exist_ok=True)
    for command in commands:
        print("python -B " + " ".join(command), flush=True)
        subprocess.run([sys.executable, "-B", *command], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
