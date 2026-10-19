"""Dry-clay MC/Rankine near field with the existing elastic DtN exterior."""
from __future__ import annotations

import argparse
import json
import os
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import brentq

import forward as elastic
from mohr_coulomb import MohrCoulombRankine


def run(boundary, refinement, radius, thickness, install, step, lateral_ratio=None,
        save=True, verbose=True, result_prefix=None):
    inputs=json.loads((elastic.HERE/"inputs.json").read_text(encoding="utf-8"))
    outline,side_y=elastic.load_geometry(inputs)
    data=elastic.make_mesh(outline,side_y,inputs,boundary,refinement,radius)
    xy,cells,outer,wall,gauges=data
    cmat=elastic.elastic_matrix(inputs["ground"]["young_MPa"],inputs["ground"]["poisson"])
    law=MohrCoulombRankine(cmat,inputs["ground"]["cohesion_MPa"],inputs["ground"]["friction_angle_deg"],ft=0.)
    support_active=bool(thickness and install==0)
    model=elastic.assemble(data,inputs,boundary,radius,thickness if support_active else 0.,lateral_ratio)
    stress0=np.r_[np.diag(model["stress"]), inputs["ground"]["poisson"]*np.trace(model["stress"]),0.]
    old_stress=np.tile(stress0,(len(cells),1))
    old_u=np.zeros((len(xy),2)); plastic=np.zeros_like(old_stress)
    q=np.zeros(model["system"].shape[0]); support_reference=np.zeros(model["fixed_k"].shape[0])
    b,area,edof=model["b"],model["area"],model["dofs"]
    rowidx,colidx=np.repeat(edof,6,axis=1).ravel(),np.tile(edof,(1,6)).ravel()
    history=[]; load=0.; increment=step
    tag=f"{boundary}_r{refinement}_R{radius:g}_t{thickness:g}_a{install:g}_s{step:g}"
    prefix="plastic"
    if lateral_ratio is not None:
        tag+=f"_k{lateral_ratio:.9g}"
    if result_prefix is not None:
        prefix=result_prefix

    def evaluate(unknown,target,need_tangent=True):
        transform,cons=model["transform"],model["constraints"]
        size=transform.shape[1]
        u=transform@unknown[:size]
        strain=np.einsum("eai,ei->ea",b,(u[:2*len(xy)].reshape(-1,2)-old_u).ravel()[edof])
        trial=old_stress+strain@cmat.T
        stress,dep,tangent=law.update(trial)
        element_force=np.einsum("eai,ea,e->ei",b,stress-stress0,area)
        force=np.bincount(edof.ravel(),weights=element_force.ravel(),minlength=len(u))
        force+=model["fixed_k"]@u-model["beam_k"]@support_reference-target*model["force"]
        reduced=transform.T@force
        if cons.shape[1]:
            reduced+=cons@unknown[size:]
            residual=np.r_[reduced,cons.T@unknown[:size]]
        else:
            residual=reduced
        if not need_tangent:
            return residual,None,stress,dep,u
        ke=np.einsum("eai,eab,ebj,e->eij",b,tangent,b,area)
        kt=sp.coo_matrix((ke.ravel(),(rowidx,colidx)),shape=model["fixed_k"].shape).tocsc()+model["fixed_k"]
        kr=transform.T@kt@transform
        system=sp.bmat([[kr,cons],[cons.T,None]],format="csc") if cons.shape[1] else kr.tocsc()
        return residual,system,stress,dep,u

    while load<1-1e-12:
        if thickness and not support_active and load>=install-1e-12:
            old_multipliers=q[model["transform"].shape[1]:]
            model=elastic.assemble(data,inputs,boundary,radius,thickness,lateral_ratio)
            support_reference=np.r_[old_u.ravel(),np.zeros(len(wall))]
            t=model["transform"]
            nodal=(t.T@support_reference)/np.asarray(t.sum(axis=0)).ravel()
            q=np.r_[nodal,old_multipliers]
            support_active=True
        target=min(1.,load+increment)
        if thickness and not support_active:
            target=min(target,install)
        candidate=q.copy()
        scale=np.linalg.norm(model["transform"].T@model["force"])*max(target,.01)
        converged=False
        for iteration in range(50):
            result=evaluate(candidate,target)
            residual,system=result[:2]
            norm=np.linalg.norm(residual)
            if norm/scale<1e-7:
                converged=True
                break
            correction=sla.spsolve(system,-residual)
            if not np.isfinite(correction).all():
                break
            # Nonassociated yield switches need not decrease the residual
            # norm along a Newton direction. Use the existing solver's
            # directional-residual-work search, with full equilibrium checked
            # above before accepting any load increment.
            def work(alpha):
                return float(correction@evaluate(candidate+alpha*correction,target,need_tangent=False)[0])
            alpha=1.
            if float(correction@residual)*work(1.)<0.:
                alpha=brentq(work,0.,1.,xtol=1e-10)
            trial=candidate+alpha*correction
            if np.linalg.norm(trial-candidate)<=64*np.finfo(float).eps*max(np.linalg.norm(candidate),1.):
                break
            candidate=trial
        if not converged:
            increment*=.5
            if increment<1e-5:
                raise RuntimeError(f"Unresolved equilibrium at load={load}, target={target}, residual={norm/scale}")
            continue
        q=candidate; load=target
        old_stress=result[2].copy(); plastic+=result[3]
        old_u=result[4][:2*len(xy)].reshape(-1,2).copy()
        magnitude=np.linalg.norm(plastic,axis=1)
        outer_cells=np.any(np.isin(cells,outer),axis=1)
        centers=xy[cells].mean(1)
        active=magnitude>1e-8
        values=[-old_u[gauges[0],1],-old_u[gauges[1],0]]
        ring_stress=elastic.ring_fiber_stresses(data,inputs,thickness,
            result[4]-support_reference) if support_active else np.zeros(1)
        history.append(dict(boundary=boundary,refinement=refinement,interface_radius_mm=radius,
            plaster_thickness_mm=thickness,installation_release=install,nominal_step=step,
            load_fraction=load,crown_settlement_mm=float(values[0]),side_inward_mm=float(values[1]),
            plastic_elements=int(active.sum()),plastic_radius_mm=float(np.linalg.norm(centers[active],axis=1).max(initial=0.)),
            outer_plastic_strain=float(magnitude[outer_cells].max(initial=0.)),
            maximum_plastic_strain=float(magnitude.max(initial=0.)),equilibrium_relative=float(norm/scale),
            iterations=iteration,actual_step=increment,
            lateral_ratio=float(-stress0[0]/inputs["load"]["vertical_pressure_MPa"]),
            ring_peak_compression_MPa=float(-ring_stress.min()),
            ring_max_tension_MPa=float(ring_stress.max())))
        if save:
            elastic.write_rows(elastic.HERE/f"{prefix}_{tag}.csv",history)
        if verbose and (len(history)%5==0 or load==1):
            print(history[-1],flush=True)
        if iteration<5:
            increment=min(step,increment*1.5)
    if save:
        np.savez_compressed(elastic.HERE/f"{prefix}_{tag}.npz",xy=xy,cells=cells,u=old_u,stress=old_stress,plastic=plastic,outer=outer,wall=wall,gauges=gauges)
    summary = dict(history[-1])
    summary.update(nodes=len(xy), elements=len(cells), interface_nodes=len(outer),
        wall_nodes=len(wall), accepted_steps=len(history),
        max_equilibrium_relative=max(row["equilibrium_relative"] for row in history),
        max_outer_plastic_strain=max(row["outer_plastic_strain"] for row in history),
        max_plastic_radius_mm=max(row["plastic_radius_mm"] for row in history))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary",choices=("dtn","box_rigid_top"),default="dtn")
    parser.add_argument("--refinement",type=int,default=1)
    parser.add_argument("--radius",type=float,default=600.)
    parser.add_argument("--thickness",type=float,default=0.)
    parser.add_argument("--install",type=float,default=0.)
    parser.add_argument("--step",type=float,default=.05)
    parser.add_argument("--lateral-ratio",type=float)
    args=parser.parse_args()
    run(args.boundary,args.refinement,args.radius,args.thickness,args.install,args.step,args.lateral_ratio)


if __name__=="__main__":
    main()
