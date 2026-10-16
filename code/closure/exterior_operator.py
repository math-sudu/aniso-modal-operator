from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from closure_env import bridge_pinned
bridge_pinned()
sys.path.insert(0, str(Path(__file__).resolve().parent / 'numeric'))
from p21_exact_evaluation import assemble_exact
from modal_condensation import condense_work

def twoterm_modal(mat, n_h):
    """Material block using exact divided differences, including double roots."""
    mu1, mu2 = mat.mus
    p1, q1 = mat.pq[0]
    b = mat.beta
    dp = b['11'] * (mu1 + mu2) - b['16']
    dq = b['12'] - b['22'] / (mu1 * mu2)
    displacement = np.array([[p1, dp], [q1, dq]], dtype=complex)
    resultant = np.array([[mu1, 1], [-1, 0]], dtype=complex)
    impedance = -1j * np.linalg.solve(displacement.T, resultant.T).T
    data = 0.5 * np.array([[1, 1j, 0, 0], [0, 0, 1, 1j]])
    density = impedance @ data
    block = 2 * np.array([density[0].real, density[0].imag, density[1].real, density[1].imag])
    return np.kron(np.diag(np.arange(1, n_h + 1)), block)

def assemble_modal(mat, interface, n_h, resultant_order=None):
    """Return the retained real modal density block on an interface map.

    The default resultant order N+64 and radius floor 0.97 resolve the
    manuscript's N=31, 63 and 127 bands. Resultant order remains independent
    of the displacement band for refinement on a new material or interface.
    """
    if interface.n_map <= 1:
        return twoterm_modal(mat, n_h)
    order = n_h + 64 if resultant_order is None else resultant_order
    return assemble_exact(mat, interface, n_h, order, radius_floor=0.97)['S']

def p1_projection(theta_nodes, n_h):
    """Exact Fourier coefficients of uniform periodic P1 hat functions.

    Both Cartesian components use T=(2/nt) diag(sinc(n/nt)^2) E.T.
    Translation and the even-grid Nyquist alternations project to zero.
    """
    theta = np.asarray(theta_nodes)
    nt = len(theta)
    phase = np.arange(1, n_h + 1)[:, None] * theta
    transform = np.zeros((4 * n_h, 2 * nt))
    transform[0::4, 0::2] = np.cos(phase)
    transform[1::4, 0::2] = np.sin(phase)
    transform[2::4, 1::2] = np.cos(phase)
    transform[3::4, 1::2] = np.sin(phase)
    weights = np.repeat(np.sinc(np.arange(1, n_h + 1) / nt) ** 2, 4)
    return 2 / nt * weights[:, None] * transform

def projected_work(modal, theta_nodes):
    """Exterior work with the same projected trial and test trace."""
    transform = p1_projection(theta_nodes, len(modal) // 4)
    return np.pi * transform.T @ modal @ transform

def condensed_mapped_system(fe, mat, smap, rho_out, n_h=None, resultant_order=None):
    """Condense the Cauchy exterior using two-sided P1 Fourier projection.

    The second return value is the boundary-work matrix on the outer ring.
    """
    mesh = fe.mesh
    n_h = mesh.n_t // 2 - 1 if n_h is None else n_h
    if not 1 <= n_h < mesh.n_t / 2:
        raise ValueError('The retained band must exclude translation and Nyquist modes')
    modal = assemble_modal(mat, smap.scaled(rho_out), n_h, resultant_order)
    work = projected_work(modal, mesh.theta_nodes)
    return (condense_work(fe, work), work)
