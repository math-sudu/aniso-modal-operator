from __future__ import annotations
import csv
import math
from pathlib import Path
R_HS = 5.034
C_HS = (-0.068772, -0.0604, 0.0656, -0.0031, -0.0212, 0.006, 0.0042, -0.0051, 0.0025, 0.0034)
ELL_C1 = 0.3
HS_T_LADDER = (0.25, 0.5, 1.0)
RHO_GRADES = (1.0, 1.5, 2.0)
N_THETA = 120
OUT = Path(__file__).resolve().parents[2] / 'results' / 'data' / 'derived' / 'section_family_boundary.csv'

def scaled(coeffs: tuple[float, ...], rho: float) -> tuple[float, ...]:
    """The Gamma_rho tier as a family map (geometry.py::SectionMap.scaled)."""
    return tuple((c * rho ** (-(j + 1.0)) for j, c in enumerate(coeffs)))

def omega(theta: float, radius: float, coeffs: tuple[float, ...]) -> complex:
    zeta = complex(math.cos(theta), math.sin(theta))
    total = zeta
    for j, c in enumerate(coeffs):
        total += c if j == 0 else c * zeta ** (-j)
    return radius * total

def domega(theta: float, radius: float, coeffs: tuple[float, ...]) -> complex:
    zeta = complex(math.cos(theta), math.sin(theta))
    total = complex(1.0, 0.0)
    for j, c in enumerate(coeffs):
        if j >= 1:
            total -= j * c * zeta ** (-(j + 1))
    return radius * total

def members() -> list[tuple[str, float, tuple[float, ...]]]:
    out: list[tuple[str, float, tuple[float, ...]]] = [('CIRC', R_HS, (0.0,)), ('ELL', R_HS, (0.0, ELL_C1))]
    for t in HS_T_LADDER:
        out.append((f'HS{int(round(t * 100)):03d}', R_HS, tuple((t * c for c in C_HS))))
    return out

def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['member', 'mapping_strength_t', 'rho', 'theta_deg', 'x_m', 'y_m', 'abs_domega', 'univalence_gauge'])
        for name, radius, coeffs in members():
            t = 0.0
            if name.startswith('HS'):
                t = int(name[2:]) / 100.0
            for rho in RHO_GRADES:
                c_rho = scaled(coeffs, rho)
                gauge = sum((j * abs(c) for j, c in enumerate(c_rho)))
                for k in range(N_THETA):
                    theta = 2.0 * math.pi * k / N_THETA
                    z = omega(theta, radius * rho, c_rho)
                    dz = domega(theta, radius * rho, c_rho)
                    writer.writerow([name, f'{t:.2f}', f'{rho:.1f}', f'{math.degrees(theta):.1f}', f'{z.real:.4f}', f'{z.imag:.4f}', f'{abs(dz):.4f}', f'{gauge:.4f}'])
    print(f'wrote {OUT}')
if __name__ == '__main__':
    main()
