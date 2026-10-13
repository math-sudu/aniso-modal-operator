from __future__ import annotations
import numpy as np
__all__ = ['jaeger_strength', 'uj_point_check', 'uj_strength_sweep', 'apparent_modulus_vs_dip']

def jaeger_strength(sigma3: float, c_j: float, phi_j: float, beta_n: float) -> float:
    """Closed-form slip strength sigma1 - sigma3 (np.inf outside the
    slip window).  Angles in radians; compression-positive."""
    t = np.tan(phi_j)
    denom = (1.0 - t / np.tan(beta_n)) * np.sin(2.0 * beta_n)
    if denom <= 0.0:
        return np.inf
    return 2.0 * (c_j + sigma3 * t) / denom

def _plane_stress_components(sig1, sig3, beta_n):
    sn = sig1 * np.cos(beta_n) ** 2 + sig3 * np.sin(beta_n) ** 2
    tau = (sig1 - sig3) * np.sin(beta_n) * np.cos(beta_n)
    return (sn, abs(tau))

def uj_point_check(sig1: float, sig3: float, beta_n: float, c_m: float, phi_m: float, c_j: float, phi_j: float):
    """One UJ material-point check at a compression-positive principal
    state: matrix MC first, then weak plane on the (possibly corrected)
    stress; weak-plane correction returns |tau| to the envelope at fixed
    sigma_n (probe-tier correction).

    Returns dict with matrix_yield, joint_slip, and the corrected
    (sigma_n, tau) pair on the plane.
    """
    f_matrix = sig1 - sig3 - (2.0 * c_m * np.cos(phi_m) + (sig1 + sig3) * np.sin(phi_m))
    matrix_yield = f_matrix > 0.0
    sn, tau = _plane_stress_components(sig1, sig3, beta_n)
    f_joint = tau - (c_j + sn * np.tan(phi_j))
    joint_slip = f_joint > 0.0
    tau_corr = min(tau, c_j + sn * np.tan(phi_j))
    return {'matrix_yield': bool(matrix_yield), 'joint_slip': bool(joint_slip), 'sigma_n': sn, 'tau': tau, 'tau_corrected': tau_corr, 'f_joint': f_joint}

def uj_strength_sweep(sigma3: float, c_j: float, phi_j: float, betas_deg: np.ndarray, c_m: float=1000000000.0, phi_m: float=np.deg2rad(89.0), ds: float=1000.0):
    """Driven sigma1 ramp per beta: measured slip-onset strength from the
    UJ point check vs the closed form.  Returns rows of
    (beta_deg, onset_measured, onset_closed_form)."""
    rows = []
    for bdeg in betas_deg:
        b = np.deg2rad(bdeg)
        ref = jaeger_strength(sigma3, c_j, phi_j, b)
        if not np.isfinite(ref):
            rows.append((bdeg, np.inf, np.inf))
            continue
        lo, hi = (0.0, 10.0 * ref + 1000000.0)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            r = uj_point_check(sigma3 + mid, sigma3, b, c_m, phi_m, c_j, phi_j)
            if r['joint_slip']:
                hi = mid
            else:
                lo = mid
        rows.append((bdeg, hi, ref))
    return rows

def apparent_modulus_vs_dip(mat_factory, dips_deg: np.ndarray):
    """Division-of-labor readout: in-plane apparent uniaxial modulus
    E_app(dip) = 1/beta11(dip) (plane-strain reduced, x-direction) for
    the structure-level anisotropic D; the standard UJ elastic response
    (single E/nu) is dip-flat by construction."""
    rows = []
    for d in dips_deg:
        m = mat_factory(d)
        rows.append((d, 1.0 / m.beta['11']))
    return rows
