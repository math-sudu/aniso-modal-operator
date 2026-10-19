import os
import sys
from dataclasses import dataclass
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_SEED = os.path.normpath(os.path.join(_HERE, os.pardir, os.pardir, 'seed_skeleton_km'))
if _SEED not in sys.path:
    sys.path.insert(0, _SEED)
from refsol import kirsch

@dataclass(frozen=True)
class Params:
    """Problem parameter set (compression-positive in-situ inputs)."""
    a: float = 2.0
    sigma_v: float = 5.0
    K0: float = 0.5
    E: float = 1800.0
    nu: float = 0.3
    label: str = 'A'

    @property
    def G(self):
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def kappa(self):
        return 3.0 - 4.0 * self.nu

    @property
    def m(self):
        return 0.5 * (self.sigma_v + self.K0 * self.sigma_v)

    @property
    def d(self):
        return 0.5 * (self.sigma_v - self.K0 * self.sigma_v)

    def exact_far_coeffs(self, n_coeff):
        """Exact decaying potential coefficients A_k, B_k (z^-k, k=1..n)."""
        if n_coeff < 3:
            raise ValueError('need n_coeff >= 3 to hold the exact solution')
        A = np.zeros(n_coeff, dtype=complex)
        B = np.zeros(n_coeff, dtype=complex)
        A[0] = self.d * self.a ** 2
        B[0] = self.m * self.a ** 2
        B[2] = self.d * self.a ** 4
        return (A, B)

def pert_stress_polar(r, theta, prm):
    """Tension-positive perturbation stresses (s_rr, s_tt, s_rt)."""
    s_rr, s_tt, s_rt = kirsch.stresses(r, theta, prm.a, prm.sigma_v, prm.K0, part='excavation')
    return (-s_rr, -s_tt, -s_rt)

def disp_polar(r, theta, prm):
    """Tension-positive perturbation displacements (u_r, u_t)."""
    u_r, u_t = kirsch.displacements(r, theta, prm.a, prm.sigma_v, prm.K0, prm.E, prm.nu)
    return (-u_r, -u_t)

def inner_wall_traction_polar(theta, prm):
    """Applied traction on the near zone at r=a (outward normal -e_r)."""
    s_rr, _, s_rt = pert_stress_polar(prm.a, theta, prm)
    return (-s_rr, -s_rt)

def vec_polar_to_cart(v_r, v_t, theta):
    c, s = (np.cos(theta), np.sin(theta))
    return (v_r * c - v_t * s, v_r * s + v_t * c)

def vec_cart_to_polar(v_x, v_y, theta):
    c, s = (np.cos(theta), np.sin(theta))
    return (v_x * c + v_y * s, -v_x * s + v_y * c)

def stress_cart_to_polar(s_xx, s_yy, s_xy, theta):
    c, s = (np.cos(theta), np.sin(theta))
    s_rr = s_xx * c * c + s_yy * s * s + 2.0 * s_xy * c * s
    s_tt = s_xx * s * s + s_yy * c * c - 2.0 * s_xy * c * s
    s_rt = (s_yy - s_xx) * c * s + s_xy * (c * c - s * s)
    return (s_rr, s_tt, s_rt)
