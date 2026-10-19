from __future__ import annotations
import numpy as np
from ._paths import bridge
bridge()
from anisotropic_elasticity.material import AnisoMaterial, material_F, material_SABNGS, material_WAI, material_ISO_path, material_ISO_path_E
R_HS = 5.034
C_HS = np.array([-0.068772, -0.0604, 0.0656, -0.0031, -0.0212, 0.006, 0.0042, -0.0051, 0.0025, 0.0034], dtype=complex)
ELL_C1 = 0.3
HS_T_LADDER = (0.25, 0.5, 1.0)
RHO_OUT = (1.5, 2.0)
MESHES = ((12, 64), (24, 128))
N_H_MAIN = 16
N_H_DIAG = 24
N_MODES_GENERAL = 96
ISO_EPS_LADDER = (0.1, 0.03, 0.01, 0.003, 0.001, 0.0001, 1e-05, 1e-06)

def materials_main() -> list[AnisoMaterial]:
    """Return the five material parameter sets used in the calculations."""
    return [material_F(1.5, 0.0, 'F0'), material_F(3.0, 45.0, 'F1'), material_F(2.0, 30.0, 'F2'), material_SABNGS(), material_WAI()]

def load_case(mat: AnisoMaterial) -> tuple[float, float, float]:
    """Released-load far-field state (tension-positive sxx, syy, sxy):
generic biaxial K0 = 0.5 frame for F materials (sigma_v = 1 scale);
SAB / WAI keep their literature loads."""
    name = mat.name
    if name == 'SABNGS':
        return (-21000000.0, -5200000.0, 0.0)
    if name.startswith('WAI'):
        return (-500000.0, -1000000.0, 0.0)
    return (-0.5, -1.0, 0.0)
