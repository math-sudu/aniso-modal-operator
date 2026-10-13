"""Check the distributed numerical modules on small existing analytical cases."""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
for folder in ("lib", "code/closure", "code/closure/numeric", "code/figure_exports"):
    sys.path.insert(0, str(ROOT / folder))

from exec20_src.material import material_F
from exec26_src.geometry import SectionMap
from exec26_src.sf_twoterm import modal_blocks_density
from p21_exact_evaluation import assemble_exact, heldout_certificate
from p22_graded_kernels import closed_S1, closed_S2
from support_screening import screening_rows, verify_saved_partition
from elastic_admissibility import checks as check_elastic_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solvers-only", action="store_true",
                        help="Check solver code without the archive's saved numerical tables")
    args = parser.parse_args()
    # Import the complete distributed computation chain without solving fields.
    bundled_modules = set()
    for path in sorted((ROOT / "lib").rglob("*.py")):
        relative = path.relative_to(ROOT / "lib").with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts.pop()
        if parts:
            bundled_modules.add(".".join(parts))
    for folder in ("code/closure", "code/closure/numeric", "code/figure_exports"):
        bundled_modules.update(path.stem for path in (ROOT / folder).glob("*.py"))
    for name in sorted(bundled_modules):
        importlib.import_module(name)
    mat = material_F(3.0, 45.0)
    reference = modal_blocks_density(mat, 3)
    for shape in (SectionMap(1.0, [0.0]), SectionMap(2.0, [0.0, 0.3])):
        result = assemble_exact(mat, shape, 3, 12)
        error = np.linalg.norm(result["S"] - reference) / np.linalg.norm(reference)
        residual = heldout_certificate(result)
        if error > 1e-10 or residual > 1e-10:
            raise AssertionError((error, residual))
        print(f"Exact two-term block: relative error {error:.3e}; residual {residual:.3e}")
    zero = np.zeros(4, dtype=complex)
    if np.linalg.norm(closed_S1(mat, 1.0, zero, 3)) > 1e-12 \
            or np.linalg.norm(closed_S2(mat, 1.0, zero, 3)) > 1e-12:
        raise AssertionError("Zero map perturbation must have zero correction kernels")
    external = []
    for name in sorted(bundled_modules):
        module = sys.modules[name]
        filename = getattr(module, "__file__", None)
        if filename is None or not Path(filename).resolve().is_relative_to(ROOT):
            external.append((name, filename))
    if external:
        raise AssertionError(f"Unexpected external project imports: {external}")
    check_elastic_path()
    if not args.solvers_only:
        screening = screening_rows()
        verify_saved_partition(screening)
        print(f"Support screening: {len(screening)} cases; saved release partition reproduced.")
    print("Standalone solver imports and zero-perturbation checks passed.")


if __name__ == "__main__":
    main()
