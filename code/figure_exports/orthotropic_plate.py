import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/closure'))
from closure_env import bridge_pinned
bridge_pinned()
sys.path.insert(0, str(ROOT / 'code/closure/numeric'))
from exec20_src.material import mu_roots, pq_coeffs
from exec26_src.geometry import SectionMap
from exec26_src.sf_twoterm import solve_neumann_twoterm
from p21_exact_evaluation import assemble_exact

def traction(force, xy, nodes, stress):
    for a, b in zip(nodes, np.roll(nodes, -1)):
        dx, dy = xy[b] - xy[a]
        load = stress @ np.array([dy, -dx]) / 2
        force[2 * a:2 * a + 2] += load
        force[2 * b:2 * b + 2] += load

def cst(xy, cells):
    q = xy[cells]
    dx, dy = (q[:, [1, 2, 0], 1] - q[:, [2, 0, 1], 1], q[:, [2, 0, 1], 0] - q[:, [1, 2, 0], 0])
    determinant = dx[:, 0] * (q[:, 0, 0] - q[:, 2, 0]) + dy[:, 0] * (q[:, 0, 1] - q[:, 2, 1])
    if np.any(determinant <= 0):
        raise ValueError('Nonpositive element area')
    b = np.zeros((len(cells), 3, 6))
    b[:, 0, 0::2], b[:, 1, 1::2] = (dx / determinant[:, None], dy / determinant[:, None])
    b[:, 2, 0::2], b[:, 2, 1::2] = (dy / determinant[:, None], dx / determinant[:, None])
    return (b, determinant / 2, (2 * cells[:, :, None] + [0, 1]).reshape(-1, 6))

def exterior(mat, count, harmonics, section):
    theta = np.arange(count) * 2 * np.pi / count
    transform = np.zeros((4 * harmonics, 2 * count))
    for n in range(1, harmonics + 1):
        weight = 2 / count * np.sinc(n / count) ** 2
        co, si = (weight * np.cos(n * theta), weight * np.sin(n * theta))
        transform[4 * n - 4, 0::2], transform[4 * n - 3, 0::2] = (co, si)
        transform[4 * n - 2, 1::2], transform[4 * n - 1, 1::2] = (co, si)
    modal = assemble_exact(mat, section, harmonics, 2 * harmonics)['S']
    return -np.pi * transform.T @ modal @ transform
