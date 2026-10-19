from __future__ import annotations
import numpy as np
IDENTITY = np.array([1.0, 1.0, 1.0, 0.0])
II = np.outer(IDENTITY, IDENTITY)
DEV = np.eye(4) - II / 3.0
YIELD_TOL = 1e-09

def isotropic_moduli(cmat):
    """Validate the elastic tensor used by the isotropic return maps."""
    c = np.asarray(cmat, dtype=float).copy()
    if c.shape != (4, 4) or not np.isfinite(c).all():
        raise ValueError('Expected a finite 4x4 Mandel elastic stiffness')
    shear = c[3, 3] / 2.0
    bulk = float(IDENTITY @ c @ IDENTITY / 9.0)
    if shear <= 0.0 or bulk <= 0.0 or (not np.allclose(c, 2.0 * shear * DEV + bulk * II, rtol=1e-12, atol=1e-12 * np.max(np.abs(c)))):
        raise ValueError('Principal return requires isotropic positive elastic moduli')
    return (c, shear, bulk)

class DruckerPragerReturnError(RuntimeError):
    """Failed local points, indexed in the flattened input to ``update``."""

    def __init__(self, reason, points):
        self.reason = reason
        self.points = points
        details = '; '.join((f"index={p['flat_index']}, trial I1={p['trial_I1_MPa']:.12g} MPa, k/alpha={p['apex_I1_MPa']:.12g} MPa, trial q={p['trial_q_MPa']:.12g} MPa, smooth q_return={p['smooth_return_q_MPa']:.12g} MPa" for p in points[:8]))
        if len(points) > 8:
            details += f'; {len(points) - 8} further points in exception.points'
        super().__init__(f'Drucker-Prager {reason}: {details}')

class DruckerPrager:

    def __init__(self, cmat, cohesion=10.0, friction=30.0, dilation=0.0):
        self.c, self.shear, self.bulk = isotropic_moduli(cmat)
        if not np.isfinite([cohesion, friction, dilation]).all() or not (cohesion >= 0.0 and 0.0 <= friction < 90.0 and (0.0 <= dilation < 90.0)):
            raise ValueError('Require cohesion >= 0 and friction/dilation in [0, 90) degrees')
        if cohesion == 0.0 and friction == 0.0:
            raise ValueError('The degenerate q=0 yield set is not implemented')
        phi, psi = np.deg2rad([friction, dilation])
        self.alpha = 2.0 * np.sin(phi) / (3.0 - np.sin(phi))
        self.beta = 2.0 * np.sin(psi) / (3.0 - np.sin(psi))
        self.intercept = 6.0 * cohesion * np.cos(phi) / (3.0 - np.sin(phi))
        self.apex_I1 = self.intercept / self.alpha if self.alpha > 0.0 else np.inf
        self.last_diagnostics = {}

    def yield_value(self, stress):
        stress = np.asarray(stress, dtype=float)
        s = stress @ DEV
        return np.sqrt(1.5 * np.sum(s * s, axis=-1)) + self.alpha * (stress @ IDENTITY) - self.intercept

    def shear_loading_surface(self, trial):
        """DP activation value and its gradient with respect to total strain."""
        trial = np.asarray(trial, dtype=float)
        dev = trial @ DEV
        q = np.sqrt(1.5 * np.sum(dev * dev, axis=-1))
        normal = np.divide(1.5 * dev, q[..., None], out=np.zeros_like(dev), where=q[..., None] > 0.0)
        return (self.yield_value(trial), (normal + self.alpha * IDENTITY) @ self.c)

    def loading_switches(self, trial):
        """Material branch coordinates and strain gradients for continuation."""
        value, gradient = self.shear_loading_surface(trial)
        return (value[..., None], gradient[..., None, :], np.zeros((*value.shape, 1), dtype=int))

    def update(self, trial):
        trial = np.asarray(trial, dtype=float)
        if trial.ndim == 0 or trial.shape[-1] != 4 or (not np.isfinite(trial).all()):
            raise ValueError('Expected finite trial stresses with final dimension 4')
        shape = trial.shape
        t = trial.reshape(-1, 4)
        stress = t.copy()
        dep = np.zeros_like(t)
        tangent = np.broadcast_to(self.c, (len(t), 4, 4)).copy()
        dev = t @ DEV
        q = np.sqrt(1.5 * np.sum(dev * dev, axis=1))
        i1 = t @ IDENTITY
        f = q + self.alpha * i1 - self.intercept
        active = np.flatnonzero(f > YIELD_TOL)
        self.last_diagnostics = dict(return_residual_MPa=0.0, active_yield_residual_MPa=0.0, yield_violation_MPa=float(np.maximum(f, 0.0).max(initial=0.0)), minimum_plastic_multiplier=0.0, active_return_points=len(active), return_iterations=0, smooth_return_points=0, apex_return_points=0, apex_I1_MPa=float(self.apex_I1))
        if len(active):
            g, bulk = (self.shear, self.bulk)
            den = 3.0 * g + 9.0 * bulk * self.alpha * self.beta
            dl = f[active] / den
            qr = (3.0 * g * (self.intercept - self.alpha * i1[active]) + 9.0 * bulk * self.alpha * self.beta * q[active]) / den
            at_apex = qr <= 0.0
            if self.beta == 0.0 and np.any(at_apex):
                failed = np.flatnonzero(at_apex)
                points = [dict(flat_index=int(active[j]), trial_I1_MPa=float(i1[active[j]]), trial_q_MPa=float(q[active[j]]), trial_yield_MPa=float(f[active[j]]), smooth_return_q_MPa=float(qr[j]), apex_I1_MPa=float(self.apex_I1), reason='no_admissible_zero_dilation_return' if qr[j] < 0.0 else 'zero_dilation_apex_not_differentiable') for j in failed]
                self.last_diagnostics.update(failed_return_points=points)
                reason = 'zero-dilation return is infeasible (I1 cannot decrease)' if np.any(qr[failed] < 0.0) else 'zero-dilation apex has no full consistent tangent'
                raise DruckerPragerReturnError(reason, points)
            smooth = np.flatnonzero(~at_apex)
            if len(smooth):
                ids = active[smooth]
                n = 1.5 * dev[ids] / q[ids, None]
                lam = dl[smooth]
                factor = qr[smooth] / q[ids]
                i1_return = i1[ids] - 9.0 * bulk * self.beta * lam
                stress[ids] = factor[:, None] * dev[ids] + (i1_return / 3.0)[:, None] * IDENTITY
                dep[ids] = lam[:, None] * (n + self.beta * IDENTITY)
                cg = 2.0 * g * n + 3.0 * bulk * self.beta * IDENTITY
                cf = 2.0 * g * n + 3.0 * bulk * self.alpha * IDENTITY
                tangent[ids] = (2.0 * g * factor)[:, None, None] * DEV + bulk * II + (4.0 * g * g * lam / q[ids])[:, None, None] * n[:, :, None] * n[:, None, :] - cg[:, :, None] * cf[:, None, :] / den
            apex = np.flatnonzero(at_apex)
            if len(apex):
                ids = active[apex]
                dl[apex] = (i1[ids] - self.apex_I1) / (9.0 * bulk * self.beta)
                stress[ids] = self.apex_I1 / 3.0 * IDENTITY
                dep[ids] = dev[ids] / (2.0 * g) + ((i1[ids] - self.apex_I1) / (9.0 * bulk))[:, None] * IDENTITY
                tangent[ids] = 0.0
            closure = stress[active] - t[active] + dep[active] @ self.c.T
            returned_f = self.yield_value(stress)
            self.last_diagnostics.update(return_residual_MPa=float(max(np.abs(closure).max(), np.abs(returned_f[active]).max())), active_yield_residual_MPa=float(np.abs(returned_f[active]).max()), yield_violation_MPa=float(np.maximum(returned_f, 0.0).max(initial=0.0)), minimum_plastic_multiplier=float(dl.min()), smooth_return_points=len(smooth), apex_return_points=len(apex), minimum_smooth_return_q_MPa=float(qr.min()), maximum_active_trial_I1_MPa=float(i1[active].max()))
        return (stress.reshape(shape), dep.reshape(shape), tangent.reshape((*shape[:-1], 4, 4)))

def checks():
    """Compare smooth Drucker-Prager returns and differentiate the strain map."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / 'figure_exports/localized_plasticity.py'
    spec = importlib.util.spec_from_file_location('plasticity_reference', path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    young, nu = (560.15, 0.3)
    bulk, shear = (young / (3.0 * (1.0 - 2.0 * nu)), young / (2.0 * (1.0 + nu)))
    c = 2.0 * shear * DEV + bulk * II
    trial = np.array([[-2.0, -2.0, -2.0, 0.0], [-10.0, -1.0, -2.0, 2.0], [-1.0, -0.1, -0.3, 1.2], [0.25, 0.15, 0.05, 0.5]])
    comparison_error = tangent_error = closure_error = 0.0
    for dilation in (0.0, 7.0, 25.6):
        law = DruckerPrager(c, 0.2, 25.6, dilation)
        old = reference.DruckerPrager(c, 0.2, 25.6, dilation)
        returned = law.update(trial)
        legacy = old.update(trial)
        for actual, expected in zip(returned, legacy):
            np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
            comparison_error = max(comparison_error, float(np.abs(actual - expected).max()))
        stress, dep, tangent = returned
        closure_error = max(closure_error, float(np.abs(stress + dep @ c.T - trial).max()))
        np.testing.assert_allclose(stress + dep @ c.T, trial, rtol=1e-12, atol=1e-12)
        numerical = np.empty_like(tangent)
        h = 1e-07
        for j in range(4):
            plus = law.update(trial + h * c[:, j])[0]
            minus = law.update(trial - h * c[:, j])[0]
            numerical[:, :, j] = (plus - minus) / (2.0 * h)
        error = np.linalg.norm(numerical - tangent) / np.linalg.norm(tangent)
        tangent_error = max(tangent_error, float(error))
        np.testing.assert_allclose(numerical, tangent, rtol=2e-07, atol=2e-06)
        unloaded = law.update(stress - 0.01 * IDENTITY)
        np.testing.assert_allclose(unloaded[1], 0.0, atol=0.0)
    law = DruckerPrager(c, 0.2, 25.6, 0.0)
    bad = np.array([[law.apex_I1 / 3.0 + 0.1] * 3 + [0.0]])
    try:
        law.update(bad)
    except DruckerPragerReturnError as error:
        failure = error.points[0]
        assert failure['reason'] == 'no_admissible_zero_dilation_return'
        assert failure['trial_I1_MPa'] > failure['apex_I1_MPa']
    else:
        raise AssertionError('A zero-dilation tensile trial must fail explicitly')
    law = DruckerPrager(c, 0.2, 25.6, 7.0)
    apex_trial = np.array([[2.0, 1.8, 1.9, 0.02]])
    stress, dep, tangent = law.update(apex_trial)
    assert law.last_diagnostics['apex_return_points'] == 1
    np.testing.assert_allclose(stress + dep @ c.T, apex_trial, atol=1e-12)
    np.testing.assert_allclose(law.yield_value(stress), 0.0, atol=1e-12)
    lam = (apex_trial @ IDENTITY - law.apex_I1) / (9.0 * bulk * law.beta)
    subgradient = dep / lam[:, None] - law.beta * IDENTITY
    assert np.linalg.norm(subgradient) <= np.sqrt(1.5)
    for j in range(4):
        plus = law.update(apex_trial + 1e-07 * c[:, j])[0]
        minus = law.update(apex_trial - 1e-07 * c[:, j])[0]
        np.testing.assert_allclose((plus - minus) / 2e-07, tangent[:, :, j], atol=0.0)
    return dict(smooth_reference_max_absolute_difference=comparison_error, smooth_tangent_relative_difference=tangent_error, strain_stress_closure_error_MPa=closure_error, zero_dilation_failure=failure, positive_dilation_apex='subgradient, closure and tangent checks passed')
MANDEL_BASIS = np.zeros((4, 3, 3))
MANDEL_BASIS[0, 0, 0] = MANDEL_BASIS[1, 1, 1] = MANDEL_BASIS[2, 2, 2] = 1.0
MANDEL_BASIS[3, 0, 1] = MANDEL_BASIS[3, 1, 0] = 1.0 / np.sqrt(2.0)
DEV3 = np.eye(3) - np.ones((3, 3)) / 3.0

def spectral_return(trial, stress, vectors, derivative, c3, shear):
    """Principal projectors and full Mandel strain tangent, with batch support."""
    projection = np.einsum('...ai,jab,...bi->...ij', vectors, MANDEL_BASIS, vectors)
    tangent = np.einsum('...ia,...ij,...jb->...ab', projection, derivative @ c3, projection)
    repeated_tol = 64.0 * np.finfo(float).eps * np.maximum(1.0, np.max(np.abs(trial), axis=-1))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        gap = trial[..., i] - trial[..., j]
        ratio = 0.5 * (derivative[..., i, i] - derivative[..., i, j] - derivative[..., j, i] + derivative[..., j, j])
        ratio = np.divide(stress[..., i] - stress[..., j], gap, out=np.array(ratio), where=np.abs(gap) > repeated_tol)
        cross = (vectors[..., :, i, None] * vectors[..., None, :, j] + vectors[..., :, j, None] * vectors[..., None, :, i]) / np.sqrt(2.0)
        shear_projection = np.einsum('aij,...ij->...a', MANDEL_BASIS, cross)
        tangent += (2.0 * shear * ratio)[..., None, None] * np.einsum('...a,...b->...ab', shear_projection, shear_projection)
    return (projection, tangent)

class RankineReturnError(RuntimeError):
    """A trial for which none of the enumerated multisurface sets was valid."""

    def __init__(self, points):
        self.reason = 'no admissible Drucker-Prager/Rankine active set'
        self.points = points
        super().__init__(f'{self.reason}: {points}')

class DruckerPragerRankine(DruckerPrager):
    """Zero-dilation DP with associated Rankine tensile plastic flow.

    ``ft=0`` is an explicit no-tension material assumption. It is not inferred
    from the DP cohesion. The cap must lie below the hydrostatic DP apex; this
    excludes a coincident nonsmooth apex from the present active-set model.
    The original DruckerPrager class and its uncapped failures are unchanged.

    Principal stresses are ordered from largest to smallest. A cap multiplier
    multiplies its principal projector; DP retains g=q and its original strain
    increment. Surface switches are semismooth; the returned tangent belongs
    to the selected active set. No stress component is clipped after return.
    """

    def __init__(self, cmat, cohesion=10.0, friction=30.0, dilation=0.0, ft=0.0):
        super().__init__(cmat, cohesion, friction, dilation)
        if self.beta != 0.0:
            raise ValueError('DruckerPragerRankine implements zero DP dilation only')
        if not np.isfinite(ft) or ft < 0.0:
            raise ValueError('Rankine tensile strength ft must be finite and nonnegative')
        if self.alpha > 0.0 and 3.0 * self.alpha * ft >= self.intercept:
            raise ValueError('Rankine cap must lie strictly below the hydrostatic DP apex')
        self.ft = float(ft)
        self.c3 = 2.0 * self.shear * np.eye(3) + (self.bulk - 2.0 * self.shear / 3.0) * np.ones((3, 3))
        self.multiplier_tol = YIELD_TOL / np.linalg.norm(self.c3, 2)

    def rankine_value(self, stress):
        stress = np.asarray(stress, dtype=float)
        largest_xy = (stress[..., 0] + stress[..., 1]) / 2.0 + np.sqrt(((stress[..., 0] - stress[..., 1]) / 2.0) ** 2 + stress[..., 3] ** 2 / 2.0)
        return np.maximum(largest_xy, stress[..., 2]) - self.ft

    def _q_normal(self, stress):
        dev = stress @ DEV3
        q = np.sqrt(1.5 * np.dot(dev, dev))
        return (q, 1.5 * dev / q)

    def _valid_set(self, trial, stress, lp, lt, m, dp):
        q = np.sqrt(1.5 * np.sum((stress @ DEV3) ** 2))
        f = q + self.alpha * stress.sum() - self.intercept
        tol = YIELD_TOL + 64.0 * np.finfo(float).eps * np.max(np.abs(trial))
        if np.any(lt < -self.multiplier_tol) or lp < -self.multiplier_tol:
            return False
        if np.max(stress - self.ft) > tol or f > tol or (dp and abs(f) > tol):
            return False
        if np.max(np.abs(stress[:m] - self.ft)) > tol:
            return False
        dep = np.zeros(3)
        dep[:m] = lt
        if dp:
            if q == 0.0:
                return False
            dep += lp * (1.5 * (stress @ DEV3) / q)
        return bool(np.max(np.abs(stress - trial + self.c3 @ dep)) <= tol)

    def _mixed_derivative(self, stress, lp, m):
        """Implicit derivative d sigma_principal / d trial_principal."""
        q, normal = self._q_normal(stress)
        hessian = (1.5 * DEV3 - np.outer(normal, normal)) / q
        jac = np.zeros((4 + m, 4 + m))
        jac[:3, :3] = np.eye(3) + lp * self.c3 @ hessian
        jac[:3, 3] = self.c3 @ normal
        jac[:3, 4:] = self.c3[:, :m]
        jac[3, :3] = normal + self.alpha
        jac[4:, :3] = np.eye(3)[:m]
        rhs = np.zeros((4 + m, 3))
        rhs[:3] = np.eye(3)
        return np.linalg.solve(jac, rhs)[:3]

    def _rankine_return(self, trial):
        """Associated cap projection in principal space, before testing DP."""
        tol = YIELD_TOL + 64.0 * np.finfo(float).eps * np.max(np.abs(trial))
        for m in (1, 2, 3):
            lt = np.linalg.solve(self.c3[:m, :m], trial[:m] - self.ft)
            stress = trial - self.c3[:, :m] @ lt
            if np.min(lt) >= -self.multiplier_tol and np.max(stress - self.ft) <= tol:
                derivative = np.eye(3) - self.c3[:, :m] @ np.linalg.solve(self.c3[:m, :m], np.eye(3)[:m])
                return (stress, lt, derivative, m)
        raise RuntimeError('No admissible associated Rankine projection')

    def shear_loading_surface(self, trial):
        """DP activation after the currently required Rankine projection.

        A tensile trial can already require cap flow when DP activates. Its
        switching surface is f_DP(sigma_cap(trial)), not f_DP(trial).
        The chain rule gives C_cap.T @ n_DP as its total-strain gradient.
        This only defines a continuation coordinate; it changes no return.
        """
        trial = np.asarray(trial, dtype=float)
        shape = trial.shape
        t = trial.reshape(-1, 4)
        value, gradient = super().shear_loading_surface(t)
        ids = np.flatnonzero(self.rankine_value(t) > 0.0)
        values, vectors = np.linalg.eigh(np.einsum('na,aij->nij', t[ids], MANDEL_BASIS))
        for index, principal, frame in zip(ids, values[:, ::-1], vectors[:, :, ::-1]):
            stress, _, derivative, _ = self._rankine_return(principal)
            projection, tangent = spectral_return(principal, stress, frame, derivative, self.c3, self.shear)
            capped = projection.T @ stress
            dev = capped @ DEV
            q = np.sqrt(1.5 * dev @ dev)
            normal = 1.5 * dev / q if q > 0.0 else np.zeros(4)
            value[index] = self.yield_value(capped)
            gradient[index] = tangent.T @ (normal + self.alpha * IDENTITY)
        return (value.reshape(shape[:-1]), gradient.reshape(shape))

    def _cap_return(self, trial):
        """Enumerate cap-only m=1..3 and mixed DP/cap m=1..2 sets."""
        stress, lt, derivative, m = self._rankine_return(trial)
        if self._valid_set(trial, stress, 0.0, lt, m, False):
            return (stress, 0.0, lt, derivative, m, False, 0)
        caa, lame = (self.c3[0, 0], self.c3[1, 0])
        a = self.ft - (trial[1] + trial[2]) / 2.0 + lame / caa * (trial[0] - self.ft)
        d = (trial[1] - trial[2]) / 2.0
        h = self.bulk / caa
        cap_margin = self.intercept - 3.0 * self.alpha * self.ft

        def mixed_yield(z):
            gap = a * z / (h + (1.0 - h) * z)
            return np.sqrt(gap * gap + 3.0 * (d * z) ** 2) - 2.0 * self.alpha * gap - cap_margin
        if a > 0.0 and mixed_yield(1.0) > 0.0:
            from scipy.optimize import brentq
            z, root_info = brentq(mixed_yield, 0.0, 1.0, xtol=1e-14, rtol=4.0 * np.finfo(float).eps, full_output=True)
            gap = a * z / (h + (1.0 - h) * z)
            stress = np.array([self.ft, self.ft - gap + d * z, self.ft - gap - d * z])
            q, normal = self._q_normal(stress)
            lp = (1.0 - z) * np.sqrt((a / (h + (1.0 - h) * z)) ** 2 + 3.0 * d * d) / (3.0 * self.shear)
            lt = np.array([(trial[0] - self.ft - 2.0 * self.shear * lp * normal[0]) / caa])
            if self._valid_set(trial, stress, lp, lt, 1, True):
                return (stress, lp, lt, self._mixed_derivative(stress, lp, 1), 1, True, root_info.iterations)
        stress = np.array([self.ft, self.ft, self.ft - cap_margin / (1.0 - self.alpha)])
        dep = np.linalg.solve(self.c3, trial - stress)
        lp = -dep[2]
        lt = dep[:2] - 0.5 * lp
        if self._valid_set(trial, stress, lp, lt, 2, True):
            return (stress, lp, lt, self._mixed_derivative(stress, lp, 2), 2, True, 0)
        return None

    def update(self, trial):
        trial = np.asarray(trial, dtype=float)
        if trial.ndim == 0 or trial.shape[-1] != 4 or (not np.isfinite(trial).all()):
            raise ValueError('Expected finite trial stresses with final dimension 4')
        shape = trial.shape
        t = trial.reshape(-1, 4)
        stress = t.copy()
        dep = np.zeros_like(t)
        tangent = np.broadcast_to(self.c, (len(t), 4, 4)).copy()
        f = self.yield_value(t)
        cap = self.rankine_value(t)
        active = (f > YIELD_TOL) | (cap > YIELD_TOL)
        q = np.sqrt(1.5 * np.sum((t @ DEV) ** 2, axis=1))
        i1 = t @ IDENTITY
        qr = self.intercept - self.alpha * i1
        candidates = np.flatnonzero((f > YIELD_TOL) & (qr > 0.0))
        free_dp = candidates[i1[candidates] / 3.0 + qr[candidates] / q[candidates] * (cap[candidates] + self.ft - i1[candidates] / 3.0) <= self.ft + YIELD_TOL]
        multipliers = []
        dp_active = np.zeros(len(t), dtype=bool)
        cap_active_residual = complementarity = 0.0
        cap_only = mixed = active_planes = root_iterations = 0
        if len(free_dp):
            stress[free_dp], dep[free_dp], tangent[free_dp] = super().update(t[free_dp])
            multipliers.extend((f[free_dp] / (3.0 * self.shear)).tolist())
            dp_active[free_dp] = True
        remaining = active.copy()
        remaining[free_dp] = False
        ids = np.flatnonzero(remaining)
        tensors = np.einsum('na,aij->nij', t[ids], MANDEL_BASIS)
        values, vectors = np.linalg.eigh(tensors)
        values, vectors = (values[:, ::-1], vectors[:, :, ::-1])
        for index, principal_trial, frame in zip(ids, values, vectors):
            returned = self._cap_return(principal_trial)
            if returned is None:
                points = [dict(flat_index=int(index), trial_principal_MPa=principal_trial.tolist(), trial_mandel_MPa=t[index].tolist(), trial_I1_MPa=float(i1[index]), trial_q_MPa=float(q[index]), tensile_strength_MPa=self.ft)]
                self.last_diagnostics = dict(failed_return_points=points)
                raise RankineReturnError(points)
            principal_stress, lp, lt, derivative, m, is_mixed, iterations = returned
            root_iterations = max(root_iterations, iterations)
            projection, tangent[index] = spectral_return(principal_trial, principal_stress, frame, derivative, self.c3, self.shear)
            principal_dep = np.zeros(3)
            principal_dep[:m] = lt
            if is_mixed:
                principal_dep += lp * self._q_normal(principal_stress)[1]
                dp_active[index] = True
                multipliers.append(float(lp))
                mixed += 1
            else:
                cap_only += 1
            multipliers.extend(lt.tolist())
            active_planes += m
            stress[index], dep[index] = (projection.T @ principal_stress, projection.T @ principal_dep)
            cap_active_residual = max(cap_active_residual, float(np.max(np.abs(principal_stress[:m] - self.ft))))
            complementarity = max(complementarity, float(np.max(np.abs(lt * (principal_stress[:m] - self.ft)))))
        returned_f = self.yield_value(stress)
        returned_cap = self.rankine_value(stress)
        dp_residual = float(np.abs(returned_f[dp_active]).max(initial=0.0))
        closure = float(np.abs(stress - t + dep @ self.c.T).max(initial=0.0))
        self.last_diagnostics = dict(return_residual_MPa=max(closure, dp_residual, cap_active_residual), active_yield_residual_MPa=max(dp_residual, cap_active_residual), yield_violation_MPa=float(np.maximum(returned_f, 0.0).max(initial=0.0)), rankine_violation_MPa=float(np.maximum(returned_cap, 0.0).max(initial=0.0)), minimum_plastic_multiplier=float(min(multipliers)) if multipliers else 0.0, active_return_points=int(active.sum()), return_iterations=root_iterations, smooth_return_points=len(free_dp), apex_return_points=0, rankine_only_return_points=cap_only, mixed_return_points=mixed, active_rankine_planes=active_planes, tensile_strength_MPa=self.ft, rankine_complementarity_MPa=complementarity)
        return (stress.reshape(shape), dep.reshape(shape), tangent.reshape((*shape[:-1], 4, 4)))

def rankine_checks():
    """Check active sets, strain history and the full rotated Mandel tangent."""
    young, nu = (560.15, 0.3)
    bulk, shear = (young / (3.0 * (1.0 - 2.0 * nu)), young / (2.0 * (1.0 + nu)))
    c = 2.0 * shear * DEV + bulk * II
    law = DruckerPragerRankine(c, 0.2, 25.6, 0.0, ft=0.0)
    cases = {'elastic': [-2.0, -2.0, -2.0], 'compression_shear': [-1.0, -2.0, -10.0], 'rankine_one': [0.5, -0.1, -0.2], 'rankine_two': [0.5, 0.4, -0.2], 'rankine_three': [0.5, 0.5, 0.5], 'mixed_one': [10.0, -1.0, -2.0], 'mixed_two': [10.0, 9.0, -2.0], 'repeated_rankine_one': [0.5, -0.2, -0.2], 'repeated_rankine_two': [0.5, 0.5, -0.2], 'repeated_mixed_one': [10.0, -2.0, -2.0], 'repeated_mixed_two': [10.0, 10.0, -2.0]}
    rotation = np.array([[np.cos(0.37), -np.sin(0.37), 0.0], [np.sin(0.37), np.cos(0.37), 0.0], [0.0, 0.0, 1.0]])
    results = {}
    max_fd = 0.0
    for name, principal in cases.items():
        tensor = rotation @ np.diag(principal) @ rotation.T
        trial = np.einsum('aij,ij->a', MANDEL_BASIS, tensor)
        stress, dep, tangent = law.update(trial)
        diagnostics = law.last_diagnostics.copy()
        np.testing.assert_allclose(stress + dep @ c.T, trial, atol=1e-11)
        assert law.yield_value(stress) <= YIELD_TOL
        assert law.rankine_value(stress) <= YIELD_TOL
        assert diagnostics['minimum_plastic_multiplier'] >= -law.multiplier_tol
        numerical = np.empty((4, 4))
        h = 2e-08
        for j in range(4):
            plus = law.update(trial + h * c[:, j])[0]
            minus = law.update(trial - h * c[:, j])[0]
            numerical[:, j] = (plus - minus) / (2.0 * h)
        error = float(np.linalg.norm(numerical - tangent) / max(1.0, np.linalg.norm(tangent)))
        max_fd = max(max_fd, error)
        np.testing.assert_allclose(numerical, tangent, rtol=2e-06, atol=2e-05)
        unloaded = law.update(stress - 0.01 * IDENTITY)
        np.testing.assert_allclose(unloaded[1], 0.0, atol=0.0)
        results[name] = dict(dp_only=diagnostics['smooth_return_points'], cap_only=diagnostics['rankine_only_return_points'], mixed=diagnostics['mixed_return_points'], cap_planes=diagnostics['active_rankine_planes'], tangent_relative_difference=error)
    switch_trial = np.array([-1.0706638351237299, 0.02732291763248007, -0.5776570741210296, 0.4213823031750979])
    value, gradient = law.shear_loading_surface(switch_trial)
    assert law.yield_value(switch_trial) > 0.1 and abs(value) <= 2 * YIELD_TOL
    numerical = np.empty(4)
    for j in range(4):
        plus = law.shear_loading_surface(switch_trial + 1e-08 * c[:, j])[0]
        minus = law.shear_loading_surface(switch_trial - 1e-08 * c[:, j])[0]
        numerical[j] = (plus - minus) / 2e-08
    switch_gradient_error = float(np.linalg.norm(numerical - gradient) / np.linalg.norm(gradient))
    np.testing.assert_allclose(numerical, gradient, rtol=2e-07, atol=2e-06)
    direction = gradient / np.linalg.norm(gradient)
    law.update(switch_trial - 1e-08 * (c @ direction))
    assert law.last_diagnostics['rankine_only_return_points'] == 1
    law.update(switch_trial + 1e-08 * (c @ direction))
    assert law.last_diagnostics['mixed_return_points'] == 1
    failed_i1, failed_q = (1.2522966557730637, 0.011501030659675334)
    rebuilt = np.array([failed_i1 / 3.0 + 2.0 * failed_q / 3.0, failed_i1 / 3.0 - failed_q / 3.0, failed_i1 / 3.0 - failed_q / 3.0, 0.0])
    stress, dep, tangent = law.update(rebuilt)
    assert law.last_diagnostics['active_rankine_planes'] == 3
    np.testing.assert_allclose(stress, 0.0, atol=1e-14)
    np.testing.assert_allclose(stress + dep @ c.T, rebuilt, atol=1e-14)
    assert law.yield_value(stress) <= YIELD_TOL and law.rankine_value(stress) <= YIELD_TOL
    reconstructed_stress = stress.tolist()
    reconstructed_diagnostics = law.last_diagnostics.copy()
    initial = -IDENTITY.copy()
    plastic = np.zeros(4)
    previous_strain = np.zeros(4)
    corners = np.array([[-0.02, 0.01, 0.0, 0.02], [0.015, -0.015, 0.0, 0.025], [0.035, -0.03, 0.0, 0.03], [0.05, -0.05, 0.0, 0.005], [0.035, -0.045, 0.0, 0.005]])
    path_counts = dict(dp=0, cap=0, mixed=0, elastic=0)
    path_recovery_error = 0.0
    compliance = np.linalg.inv(c)
    for end in corners:
        for strain in np.linspace(previous_strain, end, 11)[1:]:
            trial = initial + (strain - plastic) @ c.T
            stress, increment, _ = law.update(trial)
            diagnostics = law.last_diagnostics
            path_counts['dp'] += diagnostics['smooth_return_points']
            path_counts['cap'] += diagnostics['rankine_only_return_points']
            path_counts['mixed'] += diagnostics['mixed_return_points']
            path_counts['elastic'] += int(diagnostics['active_return_points'] == 0)
            plastic = plastic + increment
            recovered = (stress - initial) @ compliance.T + plastic
            path_recovery_error = max(path_recovery_error, float(np.max(np.abs(recovered - strain))))
            np.testing.assert_allclose(recovered, strain, atol=2e-12)
            assert abs(recovered[2]) < 2e-12
            assert law.yield_value(stress) <= YIELD_TOL and law.rankine_value(stress) <= YIELD_TOL
        previous_strain = end
    assert path_counts['dp'] > 0 and path_counts['mixed'] > 0 and (path_counts['elastic'] > 0)
    assert abs(plastic[2]) > 1e-05
    rng = np.random.default_rng(675)
    trial = rng.normal(size=(2000, 4)) * 5.0
    stress, increment, _ = law.update(trial)
    batch_diagnostics = law.last_diagnostics.copy()
    np.testing.assert_allclose(stress + increment @ c.T, trial, atol=2e-11)
    assert np.max(law.yield_value(stress)) <= YIELD_TOL
    assert np.max(law.rankine_value(stress)) <= YIELD_TOL
    assert batch_diagnostics['minimum_plastic_multiplier'] >= -law.multiplier_tol
    return dict(cases=results, maximum_tangent_relative_difference=max_fd, capped_dp_switch_gradient_relative_difference=switch_gradient_error, reconstructed_failure_return_MPa=reconstructed_stress, reconstructed_failure_diagnostics=reconstructed_diagnostics, plane_strain_path_counts=path_counts, plane_strain_path_recovery_error=path_recovery_error, batch_diagnostics=batch_diagnostics)
if __name__ == '__main__':
    import json
    print(json.dumps(dict(drucker_prager=checks(), dp_rankine=rankine_checks()), indent=2))
