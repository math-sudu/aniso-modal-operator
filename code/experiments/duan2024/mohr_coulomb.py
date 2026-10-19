from itertools import combinations
import numpy as np
from isotropic_plasticity import MANDEL_BASIS, YIELD_TOL, isotropic_moduli, spectral_return

class MohrCoulombReturnError(RuntimeError):

    def __init__(self, points):
        self.reason = 'no admissible Mohr-Coulomb/Rankine active set'
        self.points = points
        super().__init__(f'{self.reason}: {points}')

class MohrCoulombRankine:

    def __init__(self, cmat, cohesion, friction, dilation=0.0, ft=0.0):
        self.c, self.shear, self.bulk = isotropic_moduli(cmat)
        if not np.isfinite([cohesion, friction, dilation, ft]).all() or not (cohesion > 0.0 and 0.0 <= friction < 90.0 and (dilation == 0.0) and (ft >= 0.0)):
            raise ValueError('Require c > 0, phi in [0, 90), zero dilation and ft >= 0')
        self.sin_phi = np.sin(np.deg2rad(friction))
        self.intercept = 2.0 * cohesion * np.cos(np.deg2rad(friction))
        if 2.0 * self.sin_phi * ft >= self.intercept:
            raise ValueError('Rankine cap must lie strictly below the MC apex')
        self.ft = float(ft)
        self.c3 = 2.0 * self.shear * np.eye(3) + (self.bulk - 2.0 * self.shear / 3.0) * np.ones((3, 3))
        self.normals = np.zeros((6, 3))
        self.flows = np.zeros((6, 3))
        for face, (i, j) in enumerate(((0, 2), (0, 1), (1, 2))):
            self.normals[face, i], self.normals[face, j] = (1.0 + self.sin_phi, -1.0 + self.sin_phi)
            self.flows[face, i], self.flows[face, j] = (1.0, -1.0)
        self.normals[3:] = self.flows[3:] = np.eye(3)
        self.limits = np.r_[np.full(3, self.intercept), np.full(3, self.ft)]
        self.branches = [dict(surfaces=np.array([], dtype=int), derivative=np.eye(3))]
        for count in (1, 2, 3):
            for indices in combinations(range(6), count):
                ids = np.array(indices)
                normal, flow = (self.normals[ids], self.flows[ids])
                system = normal @ self.c3 @ flow.T
                if np.linalg.matrix_rank(system) < count:
                    continue
                multiplier_map = np.linalg.solve(system, normal)
                multiplier_offset = np.linalg.solve(system, self.limits[ids])
                self.branches.append(dict(surfaces=ids, multiplier_map=multiplier_map, multiplier_offset=multiplier_offset, derivative=np.eye(3) - self.c3 @ flow.T @ multiplier_map))
        self.last_diagnostics = {}

    def _extrema(self, stress):
        stress = np.asarray(stress, dtype=float)
        middle = (stress[..., 0] + stress[..., 1]) / 2.0
        radius = np.sqrt(((stress[..., 0] - stress[..., 1]) / 2.0) ** 2 + stress[..., 3] ** 2 / 2.0)
        return (np.maximum(middle + radius, stress[..., 2]), np.minimum(middle - radius, stress[..., 2]))

    def yield_value(self, stress):
        largest, smallest = self._extrema(stress)
        return largest - smallest + self.sin_phi * (largest + smallest) - self.intercept

    def rankine_value(self, stress):
        return self._extrema(stress)[0] - self.ft

    def _trial_principals(self, trial):
        trial = np.asarray(trial, dtype=float)
        if trial.ndim == 0 or trial.shape[-1] != 4 or (not np.isfinite(trial).all()):
            raise ValueError('Expected finite trial stresses with final dimension 4')
        flat = trial.reshape(-1, 4)
        values, vectors = np.linalg.eigh(np.einsum('na,aij->nij', flat, MANDEL_BASIS))
        return (trial.shape, flat, values[:, ::-1], vectors[:, :, ::-1])

    def _principal_return(self, trial):
        stress, plastic = (trial.copy(), np.zeros_like(trial))
        derivative = np.broadcast_to(np.eye(3), (len(trial), 3, 3)).copy()
        branch = np.zeros(len(trial), dtype=int)
        multipliers = np.zeros((len(trial), 3))
        tol = YIELD_TOL + 64.0 * np.finfo(float).eps * np.max(np.abs(trial), axis=1)
        pending = np.max(trial @ self.normals.T - self.limits, axis=1) > tol
        for number, data in enumerate(self.branches[1:], 1):
            ids = np.flatnonzero(pending)
            if not len(ids):
                break
            surfaces = data['surfaces']
            lam = trial[ids] @ data['multiplier_map'].T - data['multiplier_offset']
            dep = lam @ self.flows[surfaces]
            returned = trial[ids] - dep @ self.c3.T
            valid = (np.min(lam, axis=1) >= -tol[ids] / (2.0 * self.shear)) & (np.max(returned @ self.normals.T - self.limits, axis=1) <= tol[ids]) & (np.max(np.diff(returned, axis=1), axis=1) <= tol[ids])
            selected = ids[valid]
            stress[selected], plastic[selected] = (returned[valid], dep[valid])
            derivative[selected] = data['derivative']
            branch[selected] = number
            multipliers[selected, :len(surfaces)] = lam[valid]
            pending[selected] = False
        if pending.any():
            points = [dict(flat_index=int(i), trial_principal_MPa=trial[i].tolist()) for i in np.flatnonzero(pending)]
            self.last_diagnostics = dict(failed_return_points=points)
            raise MohrCoulombReturnError(points)
        return (stress, plastic, derivative, branch, multipliers)

    def update(self, trial):
        shape, flat, principal, vectors = self._trial_principals(trial)
        returned, plastic, derivative, branch, multipliers = self._principal_return(principal)
        projection, tangent = spectral_return(principal, returned, vectors, derivative, self.c3, self.shear)
        stress = np.einsum('nia,ni->na', projection, returned)
        dep = np.einsum('nia,ni->na', projection, plastic)
        elastic = branch == 0
        stress[elastic], dep[elastic], tangent[elastic] = (flat[elastic], 0.0, self.c)
        counts = dict(shear_face_return_points=0, shear_edge_return_points=0, rankine_only_return_points=0, mixed_return_points=0)
        active_residual = complementarity = 0.0
        minimum_multiplier = np.inf
        for number in np.unique(branch[~elastic]):
            selected = branch == number
            surfaces = self.branches[number]['surfaces']
            shear_count, cap_count = (np.sum(surfaces < 3), np.sum(surfaces >= 3))
            kind = 'mixed_return_points' if shear_count and cap_count else 'rankine_only_return_points' if cap_count else 'shear_edge_return_points' if shear_count == 2 else 'shear_face_return_points'
            counts[kind] += int(selected.sum())
            f = returned[selected] @ self.normals[surfaces].T - self.limits[surfaces]
            lam = multipliers[selected, :len(surfaces)]
            active_residual = max(active_residual, float(np.max(np.abs(f))))
            complementarity = max(complementarity, float(np.max(np.abs(lam * f))))
            minimum_multiplier = min(minimum_multiplier, float(lam.min()))
        closure = float(np.max(np.abs(stress - flat + dep @ self.c.T), initial=0.0))
        self.last_diagnostics = dict(**counts, active_return_points=int((~elastic).sum()), return_iterations=0, return_residual_MPa=max(closure, active_residual), active_yield_residual_MPa=active_residual, yield_violation_MPa=float(np.maximum(self.yield_value(stress), 0.0).max(initial=0.0)), rankine_violation_MPa=float(np.maximum(self.rankine_value(stress), 0.0).max(initial=0.0)), minimum_plastic_multiplier=minimum_multiplier if np.isfinite(minimum_multiplier) else 0.0, complementarity_MPa=complementarity)
        return (stress.reshape(shape), dep.reshape(shape), tangent.reshape((*shape[:-1], 4, 4)))

    def loading_switches(self, trial):
        """Exact branch-boundary coordinates, strain gradients and identities.

        Inactive yield values reach zero at activation; active multipliers
        reach zero on unloading. Their derivatives include the selected return
        branch, so an edge or cap intersection is handled as well as first
        shear yield. Multipliers are scaled by 2G to give MPa coordinates.
        Coordinates for identities within a degenerate corner may change even
        without a stress-map change; the caller also requires a tangent jump.
        """
        shape, _, principal, vectors = self._trial_principals(trial)
        returned, _, _, branch, multipliers = self._principal_return(principal)
        projection = np.einsum('nai,jab,nbi->nij', vectors, MANDEL_BASIS, vectors)
        values = np.full((len(principal), 9), -np.inf)
        principal_gradients = np.zeros((len(principal), 9, 3))
        for number in np.unique(branch):
            selected = np.flatnonzero(branch == number)
            data = self.branches[number]
            surfaces = data['surfaces']
            values[selected, :6] = returned[selected] @ self.normals.T - self.limits
            face_gradients = self.normals @ data['derivative']
            roundoff = 64.0 * np.finfo(float).eps * max(1.0, np.linalg.norm(data['derivative']))
            stationary = np.linalg.norm(face_gradients, axis=1) <= roundoff * np.linalg.norm(self.normals, axis=1)
            face_gradients[stationary] = 0.0
            principal_gradients[selected, :6] = face_gradients
            if len(surfaces):
                values[np.ix_(selected, surfaces)] = -np.inf
                principal_gradients[np.ix_(selected, surfaces)] = 0.0
                values[selected, 6:6 + len(surfaces)] = -2.0 * self.shear * multipliers[selected, :len(surfaces)]
                principal_gradients[selected, 6:6 + len(surfaces)] = -2.0 * self.shear * data['multiplier_map']
        gradients = np.einsum('nki,nia,ab->nkb', principal_gradients, projection, self.c)
        identifiers = branch[:, None] * 9 + np.arange(9)
        return (values.reshape((*shape[:-1], 9)), gradients.reshape((*shape[:-1], 9, 4)), identifiers.reshape((*shape[:-1], 9)))
