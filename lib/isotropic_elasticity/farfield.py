import numpy as np
_KINDS = ('traction', 'displacement', 'robin_minus', 'robin_plus')

def eval_fields(A, B, r, theta, G, kappa):
    """Evaluate polar tension-positive fields of decaying KM potentials.

    Parameters
    ----------
    A, B : complex arrays (N,) -- coefficients of z^-k, k = 1..N.
    r, theta : array_like, broadcast together.

    Returns
    -------
    dict with s_rr, s_tt, s_rt, u_r, u_t (broadcast shape).
    """
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)
    z = r * np.exp(1j * theta)
    phi = np.zeros_like(z)
    dphi = np.zeros_like(z)
    ddphi = np.zeros_like(z)
    psi = np.zeros_like(z)
    dpsi = np.zeros_like(z)
    zinv = 1.0 / z
    zk = np.ones_like(z)
    for k in range(1, len(A) + 1):
        zk = zk * zinv
        Ak, Bk = (A[k - 1], B[k - 1])
        if Ak != 0.0:
            phi = phi + Ak * zk
            dphi = dphi - k * Ak * zk * zinv
            ddphi = ddphi + k * (k + 1) * Ak * zk * zinv * zinv
        if Bk != 0.0:
            psi = psi + Bk * zk
            dpsi = dpsi - k * Bk * zk * zinv
    inv = 4.0 * np.real(dphi)
    dev = 2.0 * np.exp(2j * theta) * (np.conj(z) * ddphi + dpsi)
    s_rr = 0.5 * (inv - np.real(dev))
    s_tt = 0.5 * (inv + np.real(dev))
    s_rt = 0.5 * np.imag(dev)
    U = (kappa * phi - z * np.conj(dphi) - np.conj(psi)) / (2.0 * G)
    W = np.exp(-1j * theta) * U
    return {'s_rr': s_rr, 's_tt': s_tt, 's_rt': s_rt, 'u_r': np.real(W), 'u_t': np.imag(W)}

def params_to_coeffs(p):
    """Real parameter vector (4N,) -> complex coefficient arrays (A, B)."""
    p = np.asarray(p, dtype=float)
    A = p[0::4] + 1j * p[1::4]
    B = p[2::4] + 1j * p[3::4]
    return (A, B)

def coeffs_to_params(A, B):
    p = np.empty(4 * len(A))
    p[0::4], p[1::4] = (A.real, A.imag)
    p[2::4], p[3::4] = (B.real, B.imag)
    return p

def _rows_for(kind, f, beta):
    """Stack interface data rows [component1; component2] for a fit kind."""
    if kind == 'traction':
        return np.concatenate([f['s_rr'], f['s_rt']])
    if kind == 'displacement':
        return np.concatenate([f['u_r'], f['u_t']])
    if kind == 'robin_minus':
        return np.concatenate([f['s_rr'] - beta * f['u_r'], f['s_rt'] - beta * f['u_t']])
    if kind == 'robin_plus':
        return np.concatenate([f['s_rr'] + beta * f['u_r'], f['s_rt'] + beta * f['u_t']])
    raise ValueError(f'unknown kind {kind!r}')

def design_matrix(kind, r, theta, G, kappa, n_coeff, beta=0.0):
    """Real design matrix (2M, 4N): interface rows vs real coefficient params.

    Column order per k: (Re A_k, Im A_k, Re B_k, Im B_k) -- matching
    ``params_to_coeffs``.
    """
    theta = np.asarray(theta, dtype=float)
    cols = []
    for k in range(n_coeff):
        for which in ('A', 'B'):
            for unit in (1.0, 1j):
                A = np.zeros(n_coeff, dtype=complex)
                B = np.zeros(n_coeff, dtype=complex)
                if which == 'A':
                    A[k] = unit
                else:
                    B[k] = unit
                f = eval_fields(A, B, r, theta, G, kappa)
                cols.append(_rows_for(kind, f, beta))
    return np.column_stack(cols)

class FarBase:
    """Shared machinery: cached scaled design matrices + field evaluation.

    State ``self.p`` is the real parameter vector (4N,) of the current
    far-zone solution.  Subclasses implement ``_solve(Ds, data, warm)``
    returning the SCALED parameter vector q (p = q / s).
    """
    name = 'base'

    def __init__(self, R, theta_nodes, prm, n_coeff=8):
        self.R = float(R)
        self.theta = np.asarray(theta_nodes, dtype=float)
        self.G = prm.G
        self.kappa = prm.kappa
        self.n_coeff = int(n_coeff)
        self.p = np.zeros(4 * self.n_coeff)
        self.misfit = np.nan
        self._designs = {}

    def _design(self, kind, beta=0.0):
        key = (kind, float(beta))
        if key not in self._designs:
            D = design_matrix(kind, self.R, self.theta, self.G, self.kappa, self.n_coeff, beta)
            s = np.linalg.norm(D, axis=0)
            s[s == 0.0] = 1.0
            self._designs[key] = (D / s, s)
        return self._designs[key]

    def fit(self, kind, data, beta=0.0):
        """Determine coefficients from interface data (stacked 2M vector)."""
        Ds, s = self._design(kind, beta)
        q = self._solve(Ds, np.asarray(data, dtype=float), (kind, float(beta)))
        self.p = q / s
        resid = Ds @ q - data
        self.misfit = np.linalg.norm(resid) / max(np.linalg.norm(data), 1e-300)
        return self

    def _solve(self, Ds, data, key):
        raise NotImplementedError

    def coeffs(self):
        return params_to_coeffs(self.p)

    def fields(self, r, theta):
        A, B = self.coeffs()
        return eval_fields(A, B, r, theta, self.G, self.kappa)

    def eval_kind(self, kind, beta=0.0):
        """Evaluate the fitted representation's own rows at interface nodes."""
        Ds, s = self._design(kind, beta)
        return Ds @ (s * self.p)

class FarLaurentLS(FarBase):
    """Carrier 1: analytic truncated Laurent, coefficients by least squares."""
    name = 'laurent_ls'

    def _solve(self, Ds, data, key):
        q, *_ = np.linalg.lstsq(Ds, data, rcond=None)
        return q
