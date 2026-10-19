from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
__all__ = ['compliance_voigt_principal', 'reduced_beta_rotated', 'mu_roots', 'pq_coeffs', 'qhat', 'dmat_plane_strain_aniso', 'AnisoMaterial', 'material_F', 'material_SABNGS', 'material_WAI', 'material_ISO_path']

def compliance_voigt_principal(Eh: float, Ev: float, nu_vh: float, nu_hh: float, Gvh: float) -> np.ndarray:
    """6x6 Voigt compliance in principal axes.

    Axes: x' in the bedding (isotropic) plane, y' normal to bedding,
    z' the in-plane isotropic direction (out-of-plane of the section).
    nu_vh = effect of the y' (normal) stress on in-plane strain,
    nu_hh = in-plane Poisson ratio, Gvh = cross shear modulus.
    """
    a = np.zeros((6, 6))
    a[0, 0] = 1.0 / Eh
    a[1, 1] = 1.0 / Ev
    a[2, 2] = 1.0 / Eh
    a[0, 1] = a[1, 0] = -nu_vh / Ev
    a[1, 2] = a[2, 1] = -nu_vh / Ev
    a[0, 2] = a[2, 0] = -nu_hh / Eh
    a[3, 3] = 1.0 / Gvh
    a[4, 4] = 2.0 * (1.0 + nu_hh) / Eh
    a[5, 5] = 1.0 / Gvh
    return a

def _voigt_to_tensor(a: np.ndarray) -> np.ndarray:
    idx = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (1, 2): 3, (2, 1): 3, (0, 2): 4, (2, 0): 4, (0, 1): 5, (1, 0): 5}
    S = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    I, J = (idx[i, j], idx[k, l])
                    fac = 1.0
                    if I >= 3:
                        fac *= 0.5
                    if J >= 3:
                        fac *= 0.5
                    S[i, j, k, l] = fac * a[I, J]
    return S

def _rotate_z(S: np.ndarray, phi: float) -> np.ndarray:
    """Rotate the compliance tensor about z by phi (= dip angle: material
    x' axis vs global x axis, check_carrier convention)."""
    c, s = (np.cos(phi), np.sin(phi))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.einsum('pi,qj,rk,sl,ijkl->pqrs', R, R, R, R, S)

def _tensor_to_voigt(S: np.ndarray) -> np.ndarray:
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    a = np.zeros((6, 6))
    for I, (i, j) in enumerate(pairs):
        for J, (k, l) in enumerate(pairs):
            fac = 1.0
            if I >= 3:
                fac *= 2.0
            if J >= 3:
                fac *= 2.0
            a[I, J] = fac * S[i, j, k, l]
    return a

def reduced_beta_rotated(Eh: float, Ev: float, nu_vh: float, nu_hh: float, Gvh: float, dip: float) -> dict:
    """Plane-strain reduced compliances beta_ij in GLOBAL axes after
    rotating the material by ``dip`` (radians) about z:
    beta_ij = a_ij - a_i3 a_j3 / a_33 for i, j in {1, 2, 6}."""
    a0 = compliance_voigt_principal(Eh, Ev, nu_vh, nu_hh, Gvh)
    a = _tensor_to_voigt(_rotate_z(_voigt_to_tensor(a0), dip))
    out = {}
    for key, (I, J) in {'11': (0, 0), '12': (0, 1), '16': (0, 5), '22': (1, 1), '26': (1, 5), '66': (5, 5)}.items():
        out[key] = a[I, J] - a[I, 2] * a[J, 2] / a[2, 2]
    return out

def mu_roots(b: dict) -> tuple[complex, complex]:
    """Upper-half-plane roots of
    b11 mu^4 - 2 b16 mu^3 + (2 b12 + b66) mu^2 - 2 b26 mu + b22 = 0,
    sorted by ascending imaginary part (check_carrier labeling)."""
    coeffs = [b['11'], -2.0 * b['16'], 2.0 * b['12'] + b['66'], -2.0 * b['26'], b['22']]
    r = np.roots(coeffs)
    up = sorted([z for z in r if z.imag > 0], key=lambda z: z.imag)
    if len(up) != 2:
        raise ValueError(f'expected 2 upper-half roots, got {up}')
    return (complex(up[0]), complex(up[1]))

def pq_coeffs(b: dict, mu: complex) -> tuple[complex, complex]:
    """Displacement coefficients: u = 2Re[p_k phi_k], v = 2Re[q_k phi_k]."""
    p = b['11'] * mu ** 2 + b['12'] - b['16'] * mu
    q = b['12'] * mu + b['22'] / mu - b['26']
    return (p, q)

def qhat(mu: complex) -> complex:
    """Interface mapping ratio (1 + i mu)/(1 - i mu); 0 at isotropy."""
    return (1.0 + 1j * mu) / (1.0 - 1j * mu)

def dmat_plane_strain_aniso(b: dict) -> np.ndarray:
    """Plane-strain stiffness D (3x3, engineering shear) = inverse of the
    reduced compliance block [[b11,b12,b16],[b12,b22,b26],[b16,b26,b66]]."""
    C = np.array([[b['11'], b['12'], b['16']], [b['12'], b['22'], b['26']], [b['16'], b['26'], b['66']]])
    return np.linalg.inv(C)

@dataclass
class AnisoMaterial:
    name: str
    Eh: float
    Ev: float
    nu_vh: float
    nu_hh: float
    Gvh: float
    dip: float = 0.0
    beta: dict = field(init=False)
    mu1: complex = field(init=False)
    mu2: complex = field(init=False)
    p1: complex = field(init=False)
    q1: complex = field(init=False)
    p2: complex = field(init=False)
    q2: complex = field(init=False)

    def __post_init__(self):
        self.beta = reduced_beta_rotated(self.Eh, self.Ev, self.nu_vh, self.nu_hh, self.Gvh, self.dip)
        self.mu1, self.mu2 = mu_roots(self.beta)
        self.p1, self.q1 = pq_coeffs(self.beta, self.mu1)
        self.p2, self.q2 = pq_coeffs(self.beta, self.mu2)

    @property
    def mus(self):
        return (self.mu1, self.mu2)

    @property
    def qhats(self):
        return (qhat(self.mu1), qhat(self.mu2))

    @property
    def pq(self):
        return ((self.p1, self.q1), (self.p2, self.q2))

    def D(self) -> np.ndarray:
        return dmat_plane_strain_aniso(self.beta)
_EV_BASE = 10500000000.0

def material_F(ratio: float, dip_deg: float, name: str | None=None) -> AnisoMaterial:
    """Fixed-G' material family: E = ratio*EV_BASE, E' = EV_BASE,
nu = 0.25, nu' = 0.2, G' = EV_BASE/2.5."""
    return AnisoMaterial(name or f'F(r={ratio},d={dip_deg})', Eh=ratio * _EV_BASE, Ev=_EV_BASE, nu_vh=0.2, nu_hh=0.25, Gvh=_EV_BASE / 2.5, dip=np.deg2rad(dip_deg))

def material_SABNGS() -> AnisoMaterial:
    """Hefny-Lo 1999 Queenston Shale.
nu_hv is taken reciprocity-consistent (eq. (3)): nu_vh*Eh/Ev ~ 0.451;
the paper's printed 0.44 is noted as a rounding."""
    return AnisoMaterial('SABNGS', Eh=15800000000.0, Ev=10500000000.0, nu_vh=0.3, nu_hh=0.3, Gvh=3950000000.0, dip=0.0)

def material_WAI(dip_deg: float=45.0) -> AnisoMaterial:
    """Bobet 2011 ABAQUS case = Tonon-Amadei Waichecheng series
with E_x = 7800, E_y = 2400,
nu_xy = 0.07, nu_xz = 0.22, G_xy = 830 MPa.  Map: Eh = E_x,
Ev = E_y, nu_hh = nu_xz, nu_vh = nu_yx = nu_xy*E_y/E_x ~ 0.02154
(Bobet Eq. 3 reciprocity; the paper's separately printed
nu_yz = 0.02 is the same quantity rounded)."""
    return AnisoMaterial(f'WAI(d={dip_deg})', Eh=7800000000.0, Ev=2400000000.0, nu_vh=0.07 * 2.4 / 7.8, nu_hh=0.22, Gvh=830000000.0, dip=np.deg2rad(dip_deg))

def material_ISO_path(eps: float, E: float=_EV_BASE, nu: float=0.25) -> AnisoMaterial:
    """Isotropic-limit path with G multiplied by 1 + eps.

The path preserves x-y exchange symmetry and tests convergence of the modal block to isotropy."""
    G_iso = E / (2.0 * (1.0 + nu))
    return AnisoMaterial(f'ISO_G({eps:g})', Eh=E, Ev=E, nu_vh=nu, nu_hh=nu, Gvh=G_iso * (1.0 + eps), dip=0.0)

def material_ISO_path_E(eps: float, E: float=_EV_BASE, nu: float=0.25) -> AnisoMaterial:
    """Isotropic-limit path with E_h multiplied by 1 + eps.

The twofold anisotropic component and its polar harmonic coupling vanish as eps tends to zero."""
    G_iso = E / (2.0 * (1.0 + nu))
    return AnisoMaterial(f'ISO_E({eps:g})', Eh=E * (1.0 + eps), Ev=E, nu_vh=nu, nu_hh=nu, Gvh=G_iso, dip=0.0)
