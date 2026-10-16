from __future__ import annotations
import csv
import importlib
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'results/data/derived'
sys.path.insert(0, str(EVIDENCE))
model = importlib.import_module('orthotropic_hole')
OUTPUT = ROOT / 'results/data/derived/displacement_contours.csv'
FIELDS = ('dtn_orthotropic_mm', 'finite_orthotropic_mm', 'finite_isotropic_mm')

def main():
    radius, theta = np.meshgrid(np.linspace(1.1, 1.98, 90), np.linspace(0.0, np.pi / 2, 181), indexing='ij')
    x, y = (radius * np.cos(theta), radius * np.sin(theta))
    dense = 12.7 * np.c_[x.ravel(), y.ravel()]
    saved = np.genfromtxt(EVIDENCE / 'fatima2017_fields_refined.csv', names=True, delimiter=',')
    original = np.c_[saved['transverse_mm'], saved['axial_mm']]
    zero = np.c_[np.full(5, 2.5 * 12.7), np.linspace(-0.4, 0.4, 5)]
    samples = np.vstack((dense, original, zero))
    orth = model.material(104100.0, 28100.0, 3000.0, 0.16)
    iso = model.material(104100.0, 104100.0, 104100.0 / 2.32, 0.16)
    sigma = 4450 / (76.2 * 5.28)
    predictions = {}
    with threadpool_limits(limits=1):
        finite = model.mesh(12.7, 76.2, 279.4, 'traction', 2)
        infinite = model.mesh(12.7, 76.2, 279.4, 'infinite', 2)
        for key, data, material, kind in ((FIELDS[0], infinite, orth, 'infinite'), (FIELDS[1], finite, orth, 'traction'), (FIELDS[2], finite, iso, 'traction')):
            displacement, _, info = model.solve(data, material, kind, sigma, 64)
            axial = model.sample_displacement(data, displacement, samples)[:, 1]
            axial -= axial[-len(zero):].mean()
            matched = axial[len(dense):len(dense) + len(original)]
            maximum_difference = float(np.max(np.abs(matched - saved[key])))
            np.testing.assert_allclose(matched, saved[key], rtol=1e-09, atol=1e-12)
            predictions[key] = axial[:len(dense)]
            print(json.dumps(dict(case=key, display_samples=len(dense), original_samples=len(original), max_original_difference_mm=maximum_difference, minimum_mm=float(predictions[key].min()), maximum_mm=float(predictions[key].max()), **info)), flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(('radius_R', 'theta_rad', 'x_R', 'y_R', *FIELDS))
        writer.writerows(zip(radius.ravel(), theta.ravel(), x.ravel(), y.ravel(), *(predictions[key] for key in FIELDS)))
    print(f'Wrote {len(dense)} displacement contour locations to {OUTPUT}', flush=True)
if __name__ == '__main__':
    main()
