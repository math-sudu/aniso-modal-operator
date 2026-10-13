from __future__ import annotations
import numpy as np
from .material import AnisoMaterial, material_F, qhat, pq_coeffs
__all__ = ['family_atlas_rows', 'iso_ladder_rows']

def _dirichlet_cond(mat: AnisoMaterial) -> float:
    M = np.array([[mat.p1, mat.p2], [mat.q1, mat.q2]])
    sv = np.linalg.svd(M, compute_uv=False)
    return float(sv[0] / sv[-1])

def family_atlas_rows(n_ratio: int=41, n_dip: int=46):
    """Scan the Fixed-G' material family box ratio x dip = [1.25, 3] x
[0, 45 deg]."""
    rows = []
    for ratio in np.linspace(1.25, 3.0, n_ratio):
        for dip in np.linspace(0.0, 45.0, n_dip):
            m = material_F(float(ratio), float(dip))
            q1, q2 = m.qhats
            rows.append({'ratio': ratio, 'dip_deg': dip, 'abs_q1': abs(q1), 'abs_q2': abs(q2), 'mu_gap': abs(m.mu1 - m.mu2), 'cond_dirichlet_2x2': _dirichlet_cond(m), 'univalence_margin': 1.0 - max(abs(q1), abs(q2))})
    return rows

def iso_ladder_rows(path_factory, eps_ladder=(0.1, 0.03, 0.01, 0.003, 0.001, 0.0001, 1e-05, 1e-06)):
    """Confluence-limit ladder: mu gap, |qhat|, 2x2 conditioning growth
    along an isotropic-limit path (measured law: gap ~ sqrt(eps),
    cond ~ 1/gap)."""
    rows = []
    for eps in eps_ladder:
        m = path_factory(eps)
        q1, q2 = m.qhats
        rows.append({'eps': eps, 'mu_gap': abs(m.mu1 - m.mu2), 'abs_q1': abs(q1), 'abs_q2': abs(q2), 'cond_dirichlet_2x2': _dirichlet_cond(m)})
    return rows
