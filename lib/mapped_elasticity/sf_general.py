from __future__ import annotations
import numpy as np
from anisotropic_elasticity.material import AnisoMaterial
from .geometry import SectionMap
__all__ = ['ring_poles', 'GeneralExterior', 'assemble_modal_dtn', 'structure_spectra', 'rho_crit_of_map']
_COMPS = ('Uc', 'Us', 'Vc', 'Vs')

def rho_crit_of_map(smap: SectionMap) -> float:
    """Largest modulus of the interior critical zeros of omega'
    (omega'(zeta) = 0 <=> zeta^{n+1} - sum_j j c_j zeta^{n-j} = 0)."""
    n = smap.n_map
    if n < 1:
        return 0.0
    coeffs = np.zeros(n + 2, dtype=complex)
    coeffs[0] = 1.0
    for j in range(1, n + 1):
        coeffs[j + 1] = -j * smap.c[j]
    roots = np.roots(coeffs)
    return float(np.max(np.abs(roots))) if roots.size else 0.0
ETA_FLOOR = 0.85
'Minimum pole-ring radius: the 1/eta^m mode normalisation amplifies the\nfloating-point noise of the m-th trace harmonic by eta^{-m} -- at the\ncircle default eta = 0.5 the m = 47 column loses ~14 digits and the\nhigh-harmonic Dirichlet fit collapses (measured fitres 0.21).  Poles\nnear the wall (eta -> 1) keep the high-mode information at the cost of\na longer quadrature sum J (linear).'

def _poles_inside_wall(poles: np.ndarray, wall: np.ndarray) -> bool:
    """Even-odd crossing test of every pole against the closed wall
polygon (vectorised)."""
    x, y = (np.real(wall), np.imag(wall))
    x2, y2 = (np.roll(x, -1), np.roll(y, -1))
    px = np.real(poles)[:, None]
    py = np.imag(poles)[:, None]
    dy = np.where(y2 == y, 1e-300, y2 - y)[None, :]
    t = (py - y[None, :]) / dy
    crosses = ((y[None, :] <= py) != (y2[None, :] <= py)) & (px < x[None, :] + t * (x2 - x)[None, :])
    inside = np.sum(crosses, axis=1) % 2 == 1
    return bool(inside.all())

def _interior_crit_zeros(smap: SectionMap, zero_min: float) -> np.ndarray:
    """Interior critical zeros of omega' with modulus > zero_min for lightning clusters."""
    n = smap.n_map
    if n < 1:
        return np.zeros(0, dtype=complex)
    coeffs = np.zeros(n + 2, dtype=complex)
    coeffs[0] = 1.0
    for j in range(1, n + 1):
        coeffs[j + 1] = -j * smap.c[j]
    roots = np.roots(coeffs)
    roots = roots[np.abs(roots) > zero_min]
    return roots[np.abs(roots) < 1.0]

def ring_poles(smap: SectionMap, mat: AnisoMaterial, n_modes: int, eta: float | None=None, quad_multiple: int=4, mfs_offset: float=0.0, cluster_len: int=12, cluster_ratio: float=0.5, cluster_zero_min: float=0.35, lightning: bool=True):
    """Construct quadrature pole rings, DFT weights and lightning clusters.

Ring modes recombine J quadrature poles on the eta ring by DFT. For each interior critical value of omega, add a geometric chain of simple poles approaching it from the map-centre direction.

Returns (poles[2] (J,), light[2] (n_light,), w_dft (J, n_modes), s[2], eta, J)."""
    rc = rho_crit_of_map(smap)
    if eta is None:
        eta = max(0.5 * (1.0 + rc), ETA_FLOOR)
    if not 0.0 < eta < 1.0:
        raise ValueError('eta must be in (0, 1)')
    j_alias = int(np.ceil(np.log(1e-15) / np.log(eta)))
    J = max(quad_multiple * n_modes, j_alias)
    thj = mfs_offset + 2.0 * np.pi / J * np.arange(J)
    ring_iso = smap.omega(eta * np.exp(1j * thj))
    zc = _interior_crit_zeros(smap, cluster_zero_min) if lightning else np.zeros(0, dtype=complex)
    wc_iso = smap.omega(zc) if zc.size else np.zeros(0, dtype=complex)
    z0_iso = smap.R * smap.c[0]
    th_wall = 2.0 * np.pi * np.arange(1024) / 1024
    wall_iso = smap.boundary(th_wall)
    poles = []
    light = []
    scales = []
    for mu in mat.mus:
        bm, bp = SectionMap._betas(mu)
        pk = bm * ring_iso + bp * np.conj(ring_iso)
        wk = bm * wall_iso + bp * np.conj(wall_iso)
        wc_k = bm * wc_iso + bp * np.conj(wc_iso)
        z0_k = bm * z0_iso + bp * np.conj(z0_iso)
        cl = []
        for w in wc_k:
            d0 = 0.5 * abs(w - z0_k)
            if d0 == 0.0:
                continue
            dirn = (z0_k - w) / abs(z0_k - w)
            for jj in range(1, cluster_len + 1):
                cl.append(w + dirn * d0 * cluster_ratio ** jj)
        lk = np.asarray(cl, dtype=complex)
        allp = np.concatenate([pk, lk])
        if not _poles_inside_wall(allp, wk):
            raise ValueError(f'basis poles leave the mapped hole at eta={eta:.3f} (map geometry outside the pole-layout validity domain)')
        poles.append(pk)
        light.append(lk)
        scales.append(0.1 * smap.R * abs(bm))
    m = np.arange(n_modes)
    w_dft = np.exp(-1j * np.outer(thj, m)) / (J * eta ** m)[None, :]
    return (poles, light, w_dft, scales, eta, J)

class GeneralExterior:
    """Ring-mode decaying two-branch representation outside the wall of
    ``smap`` (rho = 1 curve; feed smap.scaled(rho) for interface
    tiers)."""

    def __init__(self, mat: AnisoMaterial, smap: SectionMap, n_modes: int, eta: float | None=None, quad_multiple: int=4, mfs_offset: float=0.0, lightning: bool=True):
        self.mat = mat
        self.smap = smap
        self.n_modes = int(n_modes)
        self.poles, self.light, self.w_dft, self.s, self.eta, self.J = ring_poles(smap, mat, self.n_modes, eta, quad_multiple, mfs_offset, lightning=lightning)
        self.n_light = int(self.light[0].size)
        self.n_cols = self.n_modes + self.n_light

    def phi_columns(self, z, k: int, derivative: bool=False):
        """Basis columns [ring modes | lightning poles] (and optionally
        d/dz_k) at physical points z: shape z.shape + (n_cols,)."""
        z = np.asarray(z, dtype=complex)
        mu = self.mat.mus[k]
        bm, bp = SectionMap._betas(mu)
        zk = bm * z + bp * np.conj(z)
        d = zk[..., None] - self.poles[k][None, :]
        val_r = self.s[k] / d @ self.w_dft
        if self.n_light:
            dl = zk[..., None] - self.light[k][None, :]
            val_l = self.s[k] / dl
            val = np.concatenate([val_r, val_l], axis=-1)
        else:
            val = val_r
        if not derivative:
            return val
        der_r = -self.s[k] / (d * d) @ self.w_dft
        if self.n_light:
            der_l = -self.s[k] / (dl * dl)
            der = np.concatenate([der_r, der_l], axis=-1)
        else:
            der = der_r
        return (val, der)

    def wall_traces(self, theta):
        """Displacement / resultant traces of every basis column on the
        wall z(theta) = smap.boundary(theta).

        Returns dict with U (2 n_g, 2 n_modes) interleaved-free block
        rows [u_x; u_y], and X, Y (n_g, 2 n_modes): the real traces of
        each REAL-coefficient column pair is handled by the caller (the
        columns here are complex basis functions; real/imag splitting
        happens in the design matrix)."""
        z = self.smap.boundary(theta)
        n_g = len(np.atleast_1d(theta))
        nm = self.n_cols
        Ux = np.zeros((n_g, 2 * nm), dtype=complex)
        Uy = np.zeros((n_g, 2 * nm), dtype=complex)
        X = np.zeros((n_g, 2 * nm), dtype=complex)
        Y = np.zeros((n_g, 2 * nm), dtype=complex)
        for k in range(2):
            p, q = self.mat.pq[k]
            mu = self.mat.mus[k]
            cols = self.phi_columns(z, k)
            sl = slice(k * nm, (k + 1) * nm)
            Ux[:, sl] = p * cols
            Uy[:, sl] = q * cols
            X[:, sl] = mu * cols
            Y[:, sl] = -cols
        return {'Ux': Ux, 'Uy': Uy, 'X': X, 'Y': Y}

def _real_design(mat_c: np.ndarray) -> np.ndarray:
    """Complex trace columns c -> real columns [2Re(c), -2Im(c)]:
    a real coefficient pair (a, b) on column psi means the field
    2Re[(a + i b) psi]."""
    return np.concatenate([2.0 * np.real(mat_c), -2.0 * np.imag(mat_c)], axis=1)

def _spectral_dtheta(f: np.ndarray) -> np.ndarray:
    """d/dtheta by FFT (rows = theta samples, uniform grid)."""
    N = f.shape[0]
    k = np.fft.fftfreq(N, d=1.0 / N)
    F = np.fft.fft(f, axis=0)
    return np.real(np.fft.ifft(1j * k[:, None] * F, axis=0))

def assemble_modal_dtn(mat: AnisoMaterial, smap: SectionMap, n_h: int, n_modes: int, n_grid: int=512, eta: float | None=None):
    """Spectral modal DtN on the general tier (density pairing).

    For each input harmonic basis entry (n, comp), n = 1..n_h: solve
    the Dirichlet trace fit on the ring-mode basis (one SVD, many
    RHS), synthesize the traction-density profile by the resultant
    identities + spectral d/dtheta, and project onto content harmonics
    n' = 0..n_h_content (= n_h + bandpad).

    Returns dict:
      S      (4 n_h + 2, 4 n_h)  modal matrix incl. the two n' = 0 rows
                                  (net-force rows, expected ~0);
      fitres (4 n_h,)             relative Dirichlet fit residual per
                                  input entry (basis completeness
                                  spectrum);
      cond   float                design-matrix condition number;
      rank   int                  design-matrix rank.
    """
    th = 2.0 * np.pi * np.arange(n_grid) / n_grid
    ext = GeneralExterior(mat, smap, n_modes, eta=eta)
    tr = ext.wall_traces(th)
    A_dec = np.concatenate([_real_design(tr['Ux']), _real_design(tr['Uy'])], axis=0)
    ones = np.ones(n_grid)
    zer = np.zeros(n_grid)
    rig = np.column_stack([np.concatenate([ones, zer]), np.concatenate([zer, ones])])
    rig = rig / np.linalg.norm(rig, axis=0, keepdims=True) * float(np.median(np.linalg.norm(A_dec, axis=0)))
    n_dec = A_dec.shape[1]
    A = np.concatenate([A_dec, rig], axis=1)
    U_svd, sv, Vt = np.linalg.svd(A, full_matrices=False)
    tol = sv[0] * max(A.shape) * np.finfo(float).eps
    rank = int(np.sum(sv > tol))
    inv_s = np.where(sv > tol, 1.0 / sv, 0.0)
    Xr = _real_design(tr['X'])
    Yr = _real_design(tr['Y'])
    rows_n = [(0, 'Uc'), (0, 'Vc')] + [(n, c) for n in range(1, n_h + 1) for c in _COMPS]
    S = np.zeros((len(rows_n), 4 * n_h))
    fitres = np.zeros(4 * n_h)
    c_all = {n: np.cos(n * th) for n in range(n_h + 1)}
    s_all = {n: np.sin(n * th) for n in range(n_h + 1)}
    for jin, (n, comp) in enumerate([(n, c) for n in range(1, n_h + 1) for c in _COMPS]):
        ux = c_all[n] if comp == 'Uc' else s_all[n] if comp == 'Us' else np.zeros(n_grid)
        uy = c_all[n] if comp == 'Vc' else s_all[n] if comp == 'Vs' else np.zeros(n_grid)
        b = np.concatenate([ux, uy])
        coef = Vt.T @ (inv_s * (U_svd.T @ b))
        fit = A @ coef
        fitres[jin] = np.linalg.norm(fit - b) / np.linalg.norm(b)
        X = Xr @ coef[:n_dec]
        Yv = Yr @ coef[:n_dec]
        tx = _spectral_dtheta(X[:, None])[:, 0]
        ty = _spectral_dtheta(Yv[:, None])[:, 0]
        for i, (npr, cpr) in enumerate(rows_n):
            w = (1.0 if npr == 0 else 2.0) / n_grid
            if cpr == 'Uc':
                S[i, jin] = w * (tx @ c_all[npr])
            elif cpr == 'Us':
                S[i, jin] = w * (tx @ s_all[npr])
            elif cpr == 'Vc':
                S[i, jin] = w * (ty @ c_all[npr])
            else:
                S[i, jin] = w * (ty @ s_all[npr])
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    return {'S': S, 'fitres': fitres, 'cond': cond, 'rank': rank, 'eta': ext.eta, 'J': ext.J}

def structure_spectra(S_full: np.ndarray, n_h: int):
    """Structure readouts of a modal matrix WITH net-force rows:
    strips the two n' = 0 rows, then computes

    * offblock_ratio: Frobenius mass outside the per-harmonic 4x4
      diagonal blocks over the in-block mass;
    * decay: max |entry| over components at each harmonic offset
      dn = 0..n_h-1 (aggregated over the matrix);
    * netforce: max |n' = 0 row entry|.
    """
    net = float(np.max(np.abs(S_full[:2, :])))
    S = S_full[2:, :]
    mask = np.zeros_like(S, dtype=bool)
    for n in range(1, n_h + 1):
        i0 = 4 * (n - 1)
        mask[i0:i0 + 4, i0:i0 + 4] = True
    inb = float(np.linalg.norm(S[mask]))
    offb = float(np.linalg.norm(S[~mask]))
    offblock = offb / inb if inb > 0 else np.inf
    decay = np.zeros(n_h)
    for dn in range(n_h):
        vals = []
        for n in range(1, n_h + 1):
            for npr in (n - dn, n + dn):
                if 1 <= npr <= n_h:
                    blk = S[4 * (npr - 1):4 * npr, 4 * (n - 1):4 * n]
                    vals.append(np.max(np.abs(blk)))
        decay[dn] = max(vals) if vals else 0.0
    return {'offblock_ratio': offblock, 'decay': decay, 'netforce': net}
