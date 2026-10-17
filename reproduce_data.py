"""Generate the numerical inputs used by the manuscript figure scripts."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

STUDIES = {
    "geometry": [("code/figure_exports/section_family_boundary.py",),
                 ("code/figure_exports/hs_section_geometry.py",)],
    "materials": [("code/figure_exports/atlas_family_box.py",),
                  ("code/closure/numeric/tv_constraint_conditioning_law.py",)],
    "kernels": [("code/figure_exports/closed_kernel_ladder.py",),
                ("code/figure_exports/closed_kernel_paths.py",)],
    "fields": [("code/figure_exports/near_field_clouds.py",)],
    "exterior": [("code/figure_exports/exterior_treatment_sweep.py",
                  "--stages", "reference", "cost"),
                 ("code/figure_exports/exterior_wall_profiles.py",)],
    "displacement": [("code/figure_exports/orthotropic_hole.py", "--from-data",
                      "--case", "fatima", "--refinement", "2", "--harmonics", "64"),
                     ("code/figure_exports/displacement_contours.py",)],
    "notches": [("code/figure_exports/notch_scf.py",),
                ("code/figure_exports/notch_scf.py", "--refinement", "2", "--harmonics", "64"),
                ("code/figure_exports/notch_scf.py", "--assess-only"),
                ("code/figure_exports/notch_independent.py",)],
    "lining": [("code/figure_exports/lined_section_demo.py",),
               ("code/figure_exports/lining_capacity.py",),
               ("code/figure_exports/support_screening.py",)],
    "contact": [("code/figure_exports/frictional_contact.py", "--mesh", "48", "256",
                 "--output", "results/data/derived/frictional_contact_mesh_refined.csv")],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["figures", *STUDIES])
    parser.add_argument("--list", action="store_true", help="Print commands without running them")
    args = parser.parse_args()
    commands = ([command for name, study in STUDIES.items() if name != "notches" for command in study]
                if args.study == "figures" else STUDIES.get(args.study, []))
    if args.list:
        for command in commands:
            print("python -B " + " ".join(command))
        return
    (ROOT / "results/data/derived").mkdir(parents=True, exist_ok=True)
    for command in commands:
        print("python -B " + " ".join(command), flush=True)
        subprocess.run([sys.executable, "-B", *command], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
