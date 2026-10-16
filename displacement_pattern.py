"""Reproduce displacement magnitude and pattern metrics from the saved fields."""

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results/data/derived"


def rms(values):
    return np.sqrt(np.mean(np.asarray(values) ** 2))


def main():
    fields = np.genfromtxt(DATA / "fatima2017_fields_refined.csv",
                          delimiter=",", names=True)
    rows = []
    for model in ("finite_orthotropic", "clamped_orthotropic",
                  "dtn_orthotropic", "finite_isotropic"):
        predicted = fields[model + "_mm"]
        for target, column in (("band_midpoint", "u_mid_mm"),
                               ("palette_value", "u_low_mm")):
            observed = fields[column]
            rows.append(dict(
                model=model, target=target, samples=len(observed),
                observed_rms_um=1000 * rms(observed),
                rms_difference_um=1000 * rms(predicted - observed),
                relative_rms_difference=rms(predicted - observed) / rms(observed),
                normalized_shape_difference=rms(
                    predicted / rms(predicted) - observed / rms(observed))))
    output = DATA / "fatima2017_anisotropy_metrics.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} displacement comparisons to {output}")


if __name__ == "__main__":
    main()
