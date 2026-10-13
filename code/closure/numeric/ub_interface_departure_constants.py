from __future__ import annotations
import mpmath as mp
import numpy as np
from scipy.special import zeta

def scalar_symbols(x: float) -> tuple[float, float, float]:
    """Return assembled, absolute-alias and signed-alias coefficients."""
    assembled = 2 * x * (2 + np.cos(2 * x)) / 3
    absolute = 2 * np.sin(x) ** 4 * (zeta(3, x / np.pi) + zeta(3, 1 - x / np.pi)) / np.pi ** 3
    return (float(assembled), float(absolute), float(np.sin(2 * x)))

def matrix_symbols(n_t: int, m: int, impedance: np.ndarray):
    """Nodal stiffness eigenblocks, including translations and Nyquist."""
    if m == 0:
        zero = np.zeros((2, 2), dtype=complex)
        return (zero.copy(), zero)
    x = np.pi * abs(m) / n_t
    s, g, signed = scalar_symbols(x)
    z = impedance if m > 0 else impedance.conj()
    exact = g * z.real + 1j * signed * z.imag
    assembled = s * z
    if abs(m) == n_t // 2:
        assembled = np.zeros((2, 2), dtype=complex)
        exact = g * z.real.astype(complex)
    return (assembled, exact)

def direct_alias_symbol(n_t: int, m: int, impedance: np.ndarray, pairs: int=100000):
    """Sum actual Fourier multipliers; bound the omitted tails by integrals."""
    h = 2 * np.pi / n_t
    k = m + n_t * np.arange(-pairs, pairs + 1, dtype=float)
    weights = h * np.abs(k) * np.sinc(k / n_t) ** 4
    direct = weights[k > 0].sum() * impedance
    direct += weights[k < 0].sum() * impedance.conj()
    x = np.pi * abs(m) / n_t
    tail = np.sin(x) ** 4 / np.pi * ((pairs * np.pi - x) ** (-2) + (pairs * np.pi + x) ** (-2)) * np.linalg.norm(impedance, 2)
    return (direct, tail)

def real_nodal_assembly(n_t: int, impedance: np.ndarray) -> np.ndarray:
    """P1 mass times modal density synthesis in interleaved Cartesian order."""
    theta = 2 * np.pi * np.arange(n_t) / n_t
    density = np.zeros((2 * n_t, 2 * n_t))
    material = np.empty((4, 4))
    material[0::2, 0::2] = impedance.real
    material[0::2, 1::2] = -impedance.imag
    material[1::2, 0::2] = impedance.imag
    material[1::2, 1::2] = impedance.real
    for m in range(1, n_t // 2):
        basis = np.zeros((2 * n_t, 4))
        basis[0::2, 0] = basis[1::2, 2] = np.cos(m * theta)
        basis[0::2, 1] = basis[1::2, 3] = np.sin(m * theta)
        density += 2 / n_t * basis @ (m * material) @ basis.T
    h = 2 * np.pi / n_t
    eye = np.eye(n_t)
    mass = h * (4 * eye + np.roll(eye, 1, axis=1) + np.roll(eye, -1, axis=1)) / 6
    return np.kron(mass, np.eye(2)) @ density

def fourier_pairing(v: np.ndarray, w: np.ndarray, impedance: np.ndarray, exact: bool) -> float:
    """Pair real nodal vectors using normalized e^(-i*m*theta) coefficients."""
    n_t = len(v)
    vhat, what = (np.fft.ifft(v, axis=0), np.fft.ifft(w, axis=0))
    modes = np.rint(np.fft.fftfreq(n_t, d=1 / n_t)).astype(int)
    value = 0j
    for j, m in enumerate(modes):
        symbol = matrix_symbols(n_t, int(m), impedance)[int(exact)]
        value += np.vdot(vhat[j], symbol @ what[j])
    assert abs(value.imag) < 1e-11 * max(1, abs(value.real))
    return float(n_t * value.real)

def verify_aliases(materials: dict[str, np.ndarray]) -> None:
    worst = 0.0
    for z in materials.values():
        assert np.linalg.eigvalsh(z)[0] > 0
        for n_t in (32, 64):
            for m in (1, n_t // 4, n_t // 2 - 1, n_t // 2, -3):
                _, exact = matrix_symbols(n_t, m, z)
                direct, tail = direct_alias_symbol(n_t, m, z)
                error = np.linalg.norm(direct - exact, 2)
                rounding = 5e-13 * np.linalg.norm(exact, 2)
                assert error <= tail + rounding, (n_t, m, error, tail)
                worst = max(worst, error / np.linalg.norm(exact, 2))
    print(f'PASS alias matrices, including Nyquist: max relative gap {worst:.3e}')

def verify_nodal_pairing(materials: dict[str, np.ndarray]) -> None:
    rng = np.random.default_rng(220)
    worst = 0.0
    for z in materials.values():
        n_t = 32
        stiffness = real_nodal_assembly(n_t, z)
        eigenvalues = np.linalg.eigvalsh(stiffness)
        tolerance = 1e-11 * eigenvalues[-1]
        assert eigenvalues[0] > -tolerance
        assert np.count_nonzero(abs(eigenvalues) < tolerance) == 4
        for _ in range(3):
            v, w = rng.normal(size=(2, n_t, 2))
            nodal = v.ravel() @ stiffness @ w.ravel()
            modal = fourier_pairing(v, w, z, exact=False)
            scale = np.linalg.norm(v) * np.linalg.norm(w) * np.linalg.norm(z, 2)
            error = abs(nodal - modal) / scale
            assert error < 1e-12
            worst = max(worst, error)
        alternating = np.zeros((n_t, 2))
        alternating[:, 0] = (-1.0) ** np.arange(n_t)
        exact = fourier_pairing(alternating, alternating, z, exact=True)
        assembled = alternating.ravel() @ stiffness @ alternating.ravel()
        assert exact > 0 and abs(assembled) < 1e-12 * exact
    print(f'PASS nodal mass/DFT pairing and four null modes: gap {worst:.3e}')

def verify_scaled_bounds(materials: dict[str, np.ndarray]) -> None:
    worst = np.zeros(6)
    for z in materials.values():
        norm_z = np.linalg.norm(z, 2)
        for n_t in (32, 64, 128, 256, 512, 1024):
            h = 2 * np.pi / n_t
            for m in range(1, n_t // 2 + 1):
                assembled, exact = matrix_symbols(n_t, m, z)
                defect = np.linalg.norm((exact - assembled) / h, 2) / norm_z
                scaled = [defect / (h ** r * m ** (r + 1)) for r in range(4)]
                _, g, _ = scalar_symbols(np.pi * m / n_t)
                scaled += [defect / (h * np.sqrt(g / h) * m ** 1.5), defect / (h ** 2 * m ** 3)]
                worst = np.maximum(worst, scaled)
    assert np.all(np.isfinite(worst)) and np.all(worst < 1), worst
    print('PASS full-band scaled defects (r=0,1,2,3; mixed; smooth): ' + ', '.join((f'{value:.6f}' for value in worst)))

def verify_low_frequency() -> None:
    with mp.workdps(60):
        x = mp.mpf('0.0001')
        s = 2 * x * (2 + mp.cos(2 * x)) / 3
        g = 2 * mp.sin(x) ** 4 * (mp.zeta(3, x / mp.pi) + mp.zeta(3, 1 - x / mp.pi)) / mp.pi ** 3
        absolute_limit = -4 * mp.zeta(3) / mp.pi ** 3
        signed_limit = mp.mpf(8) / 45
        assert abs((s - g) / x ** 4 - absolute_limit) < mp.mpf('0.00001')
        assert abs((s - mp.sin(2 * x)) / x ** 5 - signed_limit) < mp.mpf('1e-8')
        print(f'PASS low-frequency cancellations: (s-g)/x^4={float((s - g) / x ** 4):.9f}, (s-sin(2x))/x^5={float((s - mp.sin(2 * x)) / x ** 5):.9f}')

def main() -> int:
    materials = {'isotropic': np.array([[1.5, -0.5j], [0.5j, 1.5]]), 'coupled': np.array([[3.0, 0.6 - 0.4j], [0.6 + 0.4j, 1.4]]), 'strong_skew': np.array([[1.2, 0.15 - 0.7j], [0.15 + 0.7j, 1.1]])}
    verify_aliases(materials)
    verify_nodal_pairing(materials)
    verify_scaled_bounds(materials)
    verify_low_frequency()
    z = materials['isotropic']
    assembled, exact = matrix_symbols(64, 26, z)
    polarization = np.array([1, -1j]) / np.sqrt(2)
    relative = np.vdot(polarization, (assembled - exact) @ polarization).real
    relative /= np.vdot(polarization, exact @ polarization).real
    _, g, _ = scalar_symbols(26 * np.pi / 64)
    wrong = g * z
    gap = np.linalg.norm(wrong - exact) / np.linalg.norm(exact)
    assert gap > 0.1
    print(f'PASS isotropic conjugate-alias witness: scalar-factor gap {gap:.6f}, polarized relative pairing error {relative:.6f}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
