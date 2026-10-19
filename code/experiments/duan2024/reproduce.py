"""Run the supported-tunnel cases defined in inputs.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from plastic_forward import run


def main():
    inputs = json.loads(Path(__file__).with_name("inputs.json").read_text(encoding="utf-8"))
    cases = {**inputs["table5"], **inputs.get("diagnostics", {})}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary", choices=tuple(cases), default="dtn")
    parser.add_argument("--refinement", type=int, choices=(1, 2), default=2)
    parser.add_argument("--step", type=float)
    args = parser.parse_args()
    case = cases[args.boundary]
    step = args.step if args.step is not None else case["step"]
    result = run(args.boundary, args.refinement, case["interface_radius_mm"],
                 case["support_thickness_mm"], case["installation_release"],
                 step, case["lateral_ratio"], result_prefix="supported")
    print(result, flush=True)


if __name__ == "__main__":
    main()
