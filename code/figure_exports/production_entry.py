import csv
import importlib.util
import sys
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/closure'))
from exterior_operator import assemble_modal, twoterm_modal
from cauchy_evaluation import assemble_exact, require_simple_roots
from anisotropic_elasticity.material import material_F, material_ISO_path
from mapped_elasticity.geometry import SectionMap, horseshoe_map
from mapped_elasticity import protocol as P
from mapped_elasticity.sf_twoterm import modal_blocks_density

def check_roots():
    for mat in P.materials_main() + [material_F(r, d) for r in (1.25, 2.625, 4.0) for d in (0, 45, 90)]:
        require_simple_roots(mat)
        np.testing.assert_allclose(twoterm_modal(mat, 4), modal_blocks_density(mat, 4), rtol=1e-12, atol=0.0001)
    iso = material_ISO_path(0)
    block = twoterm_modal(iso, 1)
    expected = -2 * iso.Gvh * np.array([1, 1, 0.5, 0.5])
    np.testing.assert_allclose(np.linalg.eigvalsh(block), np.sort(expected), rtol=1e-13)
    for eps in (0, 1e-15):
        mat = material_ISO_path(eps)
        try:
            assemble_exact(mat, horseshoe_map(1, P.R_HS, P.C_HS), 2, 4)
        except ValueError as error:
            assert 'simple characteristic roots' in str(error)
        else:
            raise AssertionError('Unresolved roots reached the general-map solve')
        assert np.isfinite(assemble_modal(mat, SectionMap(1, [0, 0.2]), 4)).all()
    require_simple_roots(material_ISO_path(1e-08))
    print('Simple-root domain and two-term coalescent limit passed', flush=True)

def main():
    spec = importlib.util.spec_from_file_location('pole', Path(__file__).with_name('pole_placement.py'))
    pole = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pole)
    rows = []
    with threadpool_limits(limits=2):
        check_roots()
        mat = material_F(2.625, 45)
        tier = horseshoe_map(1, P.R_HS, P.C_HS).scaled(2)
        for band in (31, 63, 127):
            started = time.perf_counter()
            modal = assemble_modal(mat, tier, band)
            elapsed = time.perf_counter() - started
            refined = assemble_modal(mat, tier, band, resultant_order=band + 32)
            energy = np.linalg.eigvalsh(-(modal + modal.T) / 2)
            assert energy[0] > 0
            rows.append(dict(n_h=band, route='cauchy', setup_seconds=elapsed, relative_difference=np.linalg.norm(modal - refined) / np.linalg.norm(modal), reciprocity_defect=np.linalg.norm(modal - modal.T) / np.linalg.norm(modal), minimum_exterior_eigenvalue=energy[0]))
            started = time.perf_counter()
            spectral, _ = pole.assemble(mat, tier, band, {'eta': 0.97}, 384, 2048)
            elapsed = time.perf_counter() - started
            rows.append(dict(n_h=band, route='spectral_384_2048', setup_seconds=elapsed, relative_difference=np.linalg.norm(spectral - modal) / np.linalg.norm(modal), reciprocity_defect=np.linalg.norm(spectral - spectral.T) / np.linalg.norm(spectral), minimum_exterior_eigenvalue=np.linalg.eigvalsh(-(spectral + spectral.T) / 2)[0]))
            print(rows[-2:], flush=True)
    with (ROOT / 'results/data/derived/production_entry.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
if __name__ == '__main__':
    main()
