"""Duan 2024 dry-excavation geometry and elastic response.

Units: mm, MPa, force per mm axial thickness. Material/load inputs come from
inputs.json; reproduce.py supplies the Table 5 calculation entry point.
The production exterior
operator is used in its isotropic limit around the noncircular excavation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import fitz
import gmsh
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "code/closure"))
from exterior_operator import assemble_modal, projected_work
from anisotropic_elasticity.material import material_ISO_path
from mapped_elasticity.geometry import circle_map

PDF = ROOT / "references/papers/duan2024lining/source.pdf"


def elastic_matrix(young, poisson):
    identity = np.array([1., 1., 1., 0.])
    deviator = np.eye(4)-np.outer(identity, identity)/3
    shear = young/(2*(1+poisson))
    bulk = young/(3*(1-2*poisson))
    return 2*shear*deviator+bulk*np.outer(identity, identity)


def triangles(xy, cells):
    """CST Mandel strain matrix, ordered xx, yy, zz, sqrt(2)*xy."""
    q = xy[cells]
    determinant = ((q[:,1,0]-q[:,0,0])*(q[:,2,1]-q[:,0,1])
                   -(q[:,2,0]-q[:,0,0])*(q[:,1,1]-q[:,0,1]))
    if np.any(determinant <= 0):
        raise ValueError("The mesh contains non-positive triangle areas")
    dx = (q[:,[1,2,0],1]-q[:,[2,0,1],1])/determinant[:,None]
    dy = (q[:,[2,0,1],0]-q[:,[1,2,0],0])/determinant[:,None]
    b = np.zeros((len(cells),4,6))
    b[:,0,0::2], b[:,1,1::2] = dx, dy
    b[:,3,0::2], b[:,3,1::2] = dy/np.sqrt(2), dx/np.sqrt(2)
    dofs = (2*cells[:,:,None]+[0,1]).reshape(-1,6)
    return b, determinant/2, dofs


def frame_local(length, ea, ei):
    k = np.zeros((6,6))
    k[np.ix_([0,3],[0,3])] = ea/length*np.array([[1,-1],[-1,1]])
    k[np.ix_([1,2,4,5],[1,2,4,5])] = ei/length**3*np.array([
        [12,6*length,-12,6*length],
        [6*length,4*length**2,-6*length,2*length**2],
        [-12,-6*length,12,-6*length],
        [6*length,2*length**2,-6*length,4*length**2]])
    return k


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_data(inputs):
    """Read plotted vector coordinates, retaining their page/path provenance."""
    document = fitz.open(PDF)
    drawings = document[5].get_drawings()
    rows = []
    for channel, axes_id, curves in (
        ("crown", 58, (59, 72, 84, 97)),
        ("side", 118, (119, 132, 144, 157)),
    ):
        axes = drawings[axes_id]["items"]
        # Use every labelled tick: the source plot spacing is slightly uneven.
        xticks = np.array([axes[j][1].x for j in range(1, 7)])
        tick_items = axes[8:]
        yticks = np.array(sorted(item[1].y for item in tick_items))
        yrange = 2.0 if channel == "crown" else 1.0
        for label, path_id in zip(("W1", "W2", "W3", "W4"), curves):
            items = drawings[path_id]["items"]
            points = [items[0][1]] + [item[2] for item in items]
            for point in points:
                rows.append(dict(
                    case=label, channel=channel,
                    footage_cm=float(np.interp(point.x, xticks, np.arange(6)*10)),
                    signed_displacement_mm=float(np.interp(point.y, yticks, -np.linspace(0, yrange, len(yticks)))),
                    pdf_page=6, pdf_path=path_id, pdf_x=point.x, pdf_y=point.y,
                    data_role="digitized_published_vector_curve",
                ))
    write_rows(HERE / "measurements.csv", rows)

    # Figure 2(a)'s outer contour is split among five PDF subpaths. The other
    # subpaths draw the lining intrados, dimensions and lettering.
    items = document[4].get_drawings()[25]["items"]
    selected = [items[j] for lo, hi in ((26,53),(65,77),(77,90),(103,115),(127,154))
                for j in range(lo, hi)]
    pixels = np.unique(np.round([tuple(p) for item in selected for p in item[1:]], 6), axis=0)
    lo, hi = pixels.min(0), pixels.max(0)
    xy = (pixels-(lo+hi)/2)/ (hi-lo) * [inputs["geometry"]["opening_width_mm"],
                                       -inputs["geometry"]["opening_height_mm"]]
    angle = np.arctan2(xy[:,1], xy[:,0])
    order = np.argsort(angle)
    xy, pixels = xy[order], pixels[order]
    side_y = -((108.6065-(lo[1]+hi[1])/2)/(hi[1]-lo[1]))*inputs["geometry"]["opening_height_mm"]
    write_rows(HERE / "geometry.csv", [dict(x_mm=float(x), y_mm=float(y), pdf_page=5,
        pdf_path=25, pdf_x=float(px), pdf_y=float(py), data_role="dimension_scaled_published_contour")
        for (x,y),(px,py) in zip(xy,pixels)])
    return xy, float(side_y), rows


def sample_closed(xy, count):
    q = np.vstack([xy, xy[0]])
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    t = np.arange(count)*s[-1]/count
    return np.column_stack([np.interp(t,s,q[:,j]) for j in (0,1)])


def load_geometry(inputs):
    """Load the extracted contour without reading or changing observations."""
    rows = np.genfromtxt(HERE / "geometry.csv", delimiter=",", names=True)
    return np.column_stack([rows["x_mm"], rows["y_mm"]]), inputs["geometry"]["side_gauge_y_mm"]


def make_mesh(outline, side_y, inputs, boundary, refinement, radius, circle=False):
    wall = sample_closed(outline, 192*refinement)
    height = inputs["geometry"]["opening_height_mm"]
    width = inputs["geometry"]["opening_width_mm"]
    if circle:
        theta = np.arange(len(wall))*2*np.pi/len(wall)
        wall = width/2*np.column_stack([np.cos(theta),np.sin(theta)])
        height, side_y = width, 0.
    offset = inputs["geometry"]["gauge_offset_mm"]
    probes = np.array([[0.,height/2+offset], [width/2+offset,side_y]])
    if boundary == "dtn":
        theta = np.arange(128*refinement)*2*np.pi/(128*refinement)
        outer = radius*np.column_stack([np.cos(theta),np.sin(theta)])
    else:
        top = height/2+inputs["geometry"]["crown_cover_mm"]
        bottom = top-inputs["geometry"]["box_height_mm"]
        half = inputs["geometry"]["box_width_mm"]/2
        corners = np.array([[-half,bottom],[half,bottom],[half,top],[-half,top]])
        t = np.arange(32*refinement)/(32*refinement)
        outer = np.concatenate([a+(b-a)*t[:,None] for a,b in zip(corners,np.roll(corners,-1,axis=0))])
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("duan_dry_excavation")
        groups, loops = [], []
        for points, size in ((outer,30/refinement),(wall,3/refinement)):
            tags = [gmsh.model.geo.addPoint(float(x),float(y),0,size) for x,y in points]
            lines = [gmsh.model.geo.addLine(a,b) for a,b in zip(tags,tags[1:]+tags[:1])]
            groups.append(tags)
            loops.append(gmsh.model.geo.addCurveLoop(lines))
        surface = gmsh.model.geo.addPlaneSurface(loops)
        gauges = [gmsh.model.geo.addPoint(float(x),float(y),0,4/refinement) for x,y in probes]
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(0,gauges,2,surface)
        for _,tag in gmsh.model.getEntities(1):
            gmsh.model.mesh.setTransfiniteCurve(tag,2)
        gmsh.option.setNumber("Mesh.Algorithm",6)
        gmsh.model.mesh.generate(2)
        tags,xyz,_ = gmsh.model.mesh.getNodes()
        xy = np.asarray(xyz).reshape(-1,3)[:,:2]
        index = {int(tag):i for i,tag in enumerate(tags)}
        types,_,elements = gmsh.model.mesh.getElements(2)
        nodes, = [nodes for typ,nodes in zip(types,elements) if typ==2]
        cells = np.array([index[int(tag)] for tag in nodes]).reshape(-1,3)
        mapped = [np.array([index[int(gmsh.model.mesh.getNodes(0,tag)[0][0])] for tag in group])
                  for group in groups+[gauges]]
        return xy,cells,*mapped
    finally:
        gmsh.finalize()


def assemble(data, inputs, boundary, radius, thickness=0., lateral_ratio=None):
    xy,cells,outer,wall,gauges = data
    young,nu = inputs["ground"]["young_MPa"], inputs["ground"]["poisson"]
    pressure = inputs["load"]["vertical_pressure_MPa"]
    ratio = nu/(1-nu) if lateral_ratio is None else lateral_ratio
    stress = -pressure*np.diag([ratio,1.])
    b,area,dofs = triangles(xy,cells)
    ke = np.einsum("eai,ab,ebj,e->eij",b,elastic_matrix(young,nu),b,area)
    count = 2*len(xy)+(len(wall) if thickness else 0)
    k = sp.coo_matrix((ke.ravel(),(np.repeat(dofs,6,axis=1).ravel(),np.tile(dofs,(1,6)).ravel())),shape=(count,count)).tocsc()
    ground_k = k.copy()
    beam_k = sp.csc_matrix(k.shape)
    f = np.zeros(count)
    for a,c in zip(wall,np.roll(wall,-1)):
        dx,dy = xy[c]-xy[a]
        traction = stress @ np.array([dy,-dx])/2
        f[2*a:2*a+2] += traction
        f[2*c:2*c+2] += traction
    if thickness:
        rows,cols,vals = [],[],[]
        es = inputs["support"]["young_MPa"]/(1-inputs["support"]["poisson"]**2)
        for j,(a,c) in enumerate(zip(wall,np.roll(wall,-1))):
            vector = xy[c]-xy[a]
            length = np.linalg.norm(vector)
            co,si = vector/length
            rotate = np.array([[co,si,0],[-si,co,0],[0,0,1.]])
            transform = np.zeros((6,6)); transform[:3,:3]=rotate; transform[3:,3:]=rotate
            beam = transform.T@frame_local(length,es*thickness,es*thickness**3/12)@transform
            dof = np.array([2*a,2*a+1,2*len(xy)+j,2*c,2*c+1,2*len(xy)+(j+1)%len(wall)])
            rows.extend(np.repeat(dof,6)); cols.extend(np.tile(dof,6)); vals.extend(beam.ravel())
        beam_k = sp.coo_matrix((vals,(rows,cols)),shape=k.shape).tocsc()
        k += beam_k
    classes = np.arange(count)
    if boundary == "dtn":
        material = material_ISO_path(0.,E=young,nu=nu)
        modal = assemble_modal(material,circle_map(radius),len(outer)//2-1)
        work = projected_work(modal,np.arange(len(outer))*2*np.pi/len(outer))
        odof = (2*outer[:,None]+[0,1]).ravel()
        k -= sp.coo_matrix((work.ravel(),(np.repeat(odof,len(odof)),np.tile(odof,len(odof)))),shape=k.shape).tocsc()
        constraints = np.zeros((count,2))
        constraints[2*outer,0]=1/len(outer); constraints[2*outer+1,1]=1/len(outer)
        fixed = np.array([],dtype=int)
    else:
        left,right = xy[outer,0].min(),xy[outer,0].max()
        bottom,top = xy[outer,1].min(),xy[outer,1].max()
        sides = outer[np.isclose(xy[outer,0],left)|np.isclose(xy[outer,0],right)]
        floor = outer[np.isclose(xy[outer,1],bottom)]
        roof = outer[np.isclose(xy[outer,1],top)]
        fixed = np.r_[2*sides,2*floor+1]
        if boundary == "box_rigid_top":
            classes[2*roof+1]=2*roof[0]+1
        constraints = np.zeros((count,0))
    _,inverse = np.unique(classes,return_inverse=True)
    transform = sp.coo_matrix((np.ones(count),(np.arange(count),inverse)),shape=(count,inverse.max()+1)).tocsc()
    free = np.setdiff1d(np.arange(transform.shape[1]),inverse[fixed])
    transform = transform[:,free]
    kr,fr = transform.T@k@transform,transform.T@f
    constraint = sp.csc_matrix(transform.T@constraints)
    system = sp.bmat([[kr,constraint],[constraint.T,None]],format="csc") if constraint.shape[1] else kr.tocsc()
    rhs = np.r_[fr,np.zeros(constraint.shape[1])]
    return dict(data=data, system=system, rhs=rhs, transform=transform,
                constraints=constraint, fixed_k=k-ground_k, beam_k=beam_k,
                b=b, area=area, dofs=dofs, force=f, stress=stress)


def solve(data, inputs, boundary, radius, thickness=0., lateral_ratio=None):
    model = assemble(data, inputs, boundary, radius, thickness, lateral_ratio)
    solved = sla.spsolve(model["system"],model["rhs"])
    residual = np.linalg.norm(model["system"]@solved-model["rhs"])/np.linalg.norm(model["rhs"])
    if not np.isfinite(residual):
        raise RuntimeError("Nonfinite equilibrium solve")
    xy,_,_,_,gauges = data
    transform = model["transform"]
    displacement = (transform@solved[:transform.shape[1]])[:2*len(xy)].reshape(-1,2)
    values = np.array([-displacement[gauges[0],1],-displacement[gauges[1],0]])
    return values,displacement,float(residual)


def support_demand(data, inputs, boundary, thickness, lateral_ratio=None):
    """Recover elastic beam-fiber demand against the printed gypsum strength."""
    model=assemble(data,inputs,boundary,600.,thickness,lateral_ratio)
    q=sla.spsolve(model["system"],model["rhs"])
    full=model["transform"]@q[:model["transform"].shape[1]]
    fibers=ring_fiber_stresses(data,inputs,thickness,full)
    compression=-min(fibers)
    strength=inputs["support"]["compressive_strength_MPa"]
    return dict(boundary=boundary,thickness_mm=thickness,
        peak_compression_MPa=compression,largest_tension_positive_stress_MPa=max(fibers),
        printed_compressive_strength_MPa=strength,
        elastic_minimum_release_from_compression=max(0.,1-strength/compression))


def ring_fiber_stresses(data, inputs, thickness, full):
    """Recover axial-plus-bending stress from ring displacement increments."""
    xy,_,_,wall,_=data
    es=inputs["support"]["young_MPa"]/(1-inputs["support"]["poisson"]**2)
    fibers=[]
    for j,(a,c) in enumerate(zip(wall,np.roll(wall,-1))):
        vector=xy[c]-xy[a]; length=np.linalg.norm(vector); co,si=vector/length
        rotate=np.array([[co,si,0],[-si,co,0],[0,0,1.]])
        transform=np.zeros((6,6)); transform[:3,:3]=rotate; transform[3:,3:]=rotate
        dof=np.array([2*a,2*a+1,2*len(xy)+j,2*c,2*c+1,2*len(xy)+(j+1)%len(wall)])
        force=frame_local(length,es*thickness,es*thickness**3/12)@transform@full[dof]
        fibers.extend([-force[0]/thickness+6*m/thickness**2*s
                       for m in (force[2],force[5]) for s in (-1,1)])
    return np.array(fibers)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--support-check",action="store_true")
    args=parser.parse_args()
    inputs=json.loads((HERE/"inputs.json").read_text(encoding="utf-8"))
    outline,side_y,measurements=source_data(inputs)
    if args.support_check:
        checks=[]
        for boundary in ("dtn","box_rigid_top"):
            for refinement in (1,2):
                data=make_mesh(outline,side_y,inputs,boundary,refinement,600.)
                for thickness in (4.,6.):
                    row=support_demand(data,inputs,boundary,thickness)
                    row["refinement"]=refinement
                    checks.append(row)
                    print(row,flush=True)
        write_rows(HERE/"support_demand.csv",checks)
        return
    for channel in ("crown","side"):
        curve=[r for r in measurements if r["case"]=="W4" and r["channel"]==channel]
        print(channel, "W4 initial/final/increment mm",curve[0]["signed_displacement_mm"],curve[-1]["signed_displacement_mm"],curve[-1]["signed_displacement_mm"]-curve[0]["signed_displacement_mm"],flush=True)
    cases=[("dtn",1,600.,0.),("box_rigid_top",1,600.,0.)]
    if not args.quick:
        cases += [(boundary,2,600.,0.) for boundary in ("dtn","box_rigid_top","box_traction_top")]
        cases += [("dtn",2,400.,0.),("dtn",3,600.,0.),("box_rigid_top",3,600.,0.)]
        cases += [(boundary,2,600.,t) for boundary in ("dtn","box_rigid_top") for t in (4.,6.)]
    rows=[]; cache={}; unsupported={}
    for boundary,refinement,radius,thickness in cases:
        key=("dtn" if boundary=="dtn" else "box",refinement,radius)
        if key not in cache:
            cache[key]=make_mesh(outline,side_y,inputs,boundary,refinement,radius)
        data=cache[key]
        value,u,residual=solve(data,inputs,boundary,radius,thickness)
        if thickness==0:
            unsupported[(boundary,refinement,radius)]=value
        for release in ([0.,.5,1.] if thickness else [1.]):
            result=value if not thickness else release*unsupported[(boundary,refinement,radius)]+(1-release)*value
            row=dict(boundary=boundary,refinement=refinement,interface_radius_mm=radius,
                     plaster_thickness_mm=thickness,release_before_installation=release,
                     crown_settlement_mm=float(result[0]),side_inward_mm=float(result[1]),
                     nodes=len(data[0]),elements=len(data[1]),equilibrium_relative=residual)
            rows.append(row)
            print(row,flush=True)
        if boundary=="dtn" and refinement==3 and thickness==0:
            np.savez_compressed(HERE/"field.npz",xy=data[0],cells=data[1],u=u,outer=data[2],wall=data[3],gauges=data[4])
        write_rows(HERE/"predictions.csv",rows)
    if not args.quick:
        circle=make_mesh(outline,side_y,inputs,"dtn",2,600.,circle=True)
        values,_,residual=solve(circle,inputs,"dtn",600.,lateral_ratio=1.)
        a=inputs["geometry"]["opening_width_mm"]/2
        expected=(1+inputs["ground"]["poisson"])*inputs["load"]["vertical_pressure_MPa"]*a*a/(inputs["ground"]["young_MPa"]*(a+50))
        check=[dict(channel=name,computed_mm=float(value),analytic_mm=expected,relative_difference=float(value/expected-1),equilibrium_relative=residual)
               for name,value in zip(("crown","side"),values)]
        write_rows(HERE/"circle_check.csv",check)
        print("Circle check",check,flush=True)


if __name__=="__main__":
    main()
