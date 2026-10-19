# Supported W4 tunnel comparison

This directory reproduces manuscript Section 5.6/Table 5 of
*A complex-potential DtN–FEM formulation for nonlinear
tunnel–ground interaction* (revision dated 2026-10-20).
Units are mm, MPa and force per mm of tunnel-axis length.

The experimental source is Duan, Feng, Song and Yao (2024),
[*Model test study on the mechanical response of the deep buried tunnel lining*](https://doi.org/10.1038/s41598-024-63438-5),
Scientific Reports 14, 12863. The original article is included under its
[CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/) at
`references/papers/duan2024lining/source.pdf` in the archive.

## Calculation

From the extracted archive root, run:

```shell
python -B code/experiments/duan2024/reproduce.py
```

The command runs the forward calculation from the fixed inputs.
`--refinement 1 --step 0.05` and `--refinement 1 --step 0.025` reproduce the
coarse mesh/loading-step checks. The default is refinement 2 and step 0.025.

For the four one-at-a-time input comparisons in Table 5(b), run:

```shell
python -B code/experiments/duan2024/reproduce.py --refinement 1 --step 0.05
python -B code/experiments/duan2024/input_sensitivity.py
```

This uses the coarse baseline mesh produced by the first command and a maximum release step of 0.05.
It changes thickness to 4 mm, installation release to 0.50, or k to 0.73 or
0.93, holding the other inputs at their baseline values. The optional
`--case thickness installation lower_k upper_k` selects individual rows.
`input_sensitivity.csv` stores the final displacements, peak ring compression,
equilibrium residual and outer-adjacent plastic strain for each case.

For the finite-box boundary comparison in Section 5.6, run:

```shell
python -B code/experiments/duan2024/boundary_sensitivity.py
```

This retains the Table 5 baseline material, loading and support inputs and
uses maximum release increment 0.05 at both box mesh levels. Optional
`--refinement 1` or `--refinement 2` selects one level. The box has zero
horizontal displacement increments at its sides and zero vertical increments
at its base. The top nodes share one vertical displacement under unchanged
resultant load. Movements use that box reference; the DtN comparison uses
zero translation at infinity. `boundary_sensitivity.csv` stores both final
responses, mesh sizes and maximum accepted equilibrium residuals.

The calculation chain is:

1. `reproduce.py` reads the fixed case in `inputs.json:table5.dtn`.
2. `plastic_forward.run` loads the extracted opening from `geometry.csv`.
   It initializes homogeneous stress, zero plastic strain and zero disturbance
   displacement, then retains the complete material history during excavation.
3. `forward.make_mesh` creates a triangular near field between the noncircular
   wall and a circular artificial interface of radius 600 mm. The refined
   mesh has 24,631 nodes and 48,622 triangles; its wall/interface have 384/256
   vertices. The coarse mesh has 6,380 nodes, 12,440 triangles and 192/128
   wall/interface vertices.
4. `forward.assemble` calls `material_ISO_path(0, E=20.5, nu=0.32)` and
   `circle_map(600)`. In `code/closure/exterior_operator.py`, this map selects
   `twoterm_modal`, the analytic divided-difference block at coincident roots.
   The retained bands are N=127/63 on the refined/coarse meshes.
5. `projected_work` applies the same P1 Fourier projection on both sides.
   The near-field tangent contains minus this exterior work matrix; two
   mean-interface constraints remove translations.
   Each component has weight 1/n on the n equally spaced interface nodes,
   with zero weights on all other nodes. The isotropic, zero-resultant circular
   exterior has zero-mean decaying displacement harmonics, so this selects
   zero translation at infinity. The saved refined field has interface-mean
   displacement components below 1e-16 mm. The comparison assumes a fixed
   experimental reference.
6. `MohrCoulombRankine` returns the full four-component plane-strain stress
   with zero-dilation shear flow and an associated zero-tension cap. The
   Newton solve accepts a step only below relative equilibrium residual 1e-7.
   The 6 mm ring activates unstressed at release 0.70, with the attained
   wall translations as its reference and ground history retained.
7. The two ground-node movements are extracted as `-u_y` at (0,170) mm and
   `-u_x` at (180,-20.4808983) mm. The resulting history and final field are
   saved as `supported_*.csv` and `supported_*.npz`.

The ring uses Euler–Bernoulli beam segments, bonded translations and independent
rotations. Its stiffness per unit axial length is Ebar*t and Ebar*t^3/12,
where Ebar=269/(1-0.21^2) MPa. The near-field elastic matrix retains zero total
axial strain, with initial stress (xx,yy,zz,xy)=(-0.2075,-0.25,-0.1464,0) MPa.

## Inputs and observations

| Quantity | Source | Model representation |
|---|---|---|
| Opening | Source Figs. 1–2: 260×240 mm; dimension-scaled contour extraction | `geometry.csv`, with original PDF page/path coordinates; centred bounding box, x right/y up |
| Ground | Table 2 Clay 1: E=20.5 MPa, nu=0.32, c=0.031 MPa, phi=46.93 degrees | Homogeneous clay under the prescribed initial stress; no additional self-weight |
| Load | Procedure: 200 kN top load | Full top area 1.6×0.5 m gives pv=0.25 MPa; ph/pv=0.83 and the axial stress are prescribed model inputs |
| Support | Table 3 M4: E=269 MPa, nu=0.21, compressive strength=2.4 MPa; Table 4: 6 mm (source prose: 4 mm) | Bonded gypsum ring using Table 4; copper arches, anchors and connections are outside this component model |
| Installation | W4 has initial support and no secondary lining (Table 5) | Prescribed release lambda=0.70; unstressed ring activation |
| Gauges | Fig. 2(a): 50 mm outside crown/sidewall | Crown ground settlement and interpreted one-sided inward ground movement, with a fixed datum |
| Observation stage | Fig. 3 W4, excavation footage 0–50 cm | Digitized endpoint-minus-first-reading increments: crown 1.708987−0.106363=1.602624 mm; side 0.882686−0.000150=0.882536 mm |
| Exterior | Source box 1600×1600×500 mm, crown cover 790 mm | Unbounded isotropic exterior outside the 600 mm artificial circle |

The prescribed k=0.83 and lambda=0.70 are literature-informed model inputs.
Vardakos (2003), *Distinct Element Modeling of the
Shimizu Tunnel No. 3 in Japan*, Section 6.5.1, p.121
([source](https://hdl.handle.net/10919/36153)), provides a precedent for k=0.83.
Negro and Eisenstein (1997), *Panel discussion: Delayed lining activation and
ground stress relaxation in shallow tunnels*, p.2396
([source](https://www.issmge.org/uploads/publications/1/31/1997_04_0051.pdf)),
reports a 20–70% activation-release range. The calculation adopts these
literature-informed values independently of the displacement observations.

`measurements.csv` retains the plotted curves with PDF page/path coordinates.
`forward.source_data` performs the original vector-curve/contour extraction;
the staged forward calculation reads only the stored geometry and model inputs.
Stored digits retain extraction precision. The final full-sectional-release
solution is compared with the digitized stage increment.

## Results and their interpretation

The refined result is 1.456779 mm crown and 0.823841 mm side movement, or
1.457/0.824 mm in Table 5. Its 40 accepted increments retain zero plastic
strain adjacent to the 600 mm interface; the maximum plastic radius is
140.295 mm and final peak ring compression is 0.934481 MPa. The default command and the two coarse-mesh options above reproduce the three mesh/step histories.

Table 5(b) compares its coarse baseline, 1.454687/0.823275 mm, with the four
input changes. Crown and side increments span 1.316977–1.543477 mm and
0.619184–1.028531 mm. Earlier installation reduces both movements; increasing
k decreases crown settlement and increases side movement. Every increment
satisfies the 1e-7 equilibrium tolerance, and the outer-adjacent elements
remain elastic. The thickness and k rows reuse the completed forward runs;
the installation row adds the 0.50 activation case on the same mesh.

The finite-box coarse result is 1.986655/0.779694 mm. Relative to the coarse
DtN result at the same maximum increment, crown movement increases by
0.531967 mm (36.6%) and side movement decreases by 0.043581 mm (5.3%).
The box mesh refinement changes the movements by 0.003567/0.000493 mm.
The outer boundary and displacement reference therefore influence the crown
comparison more than its baseline observation difference. This is a change
of boundary model; selection of the experimental model requires the actual
wall restraint and gauge-reference definition. Both models predict movements
on the observed millimetre scale.

The W4 comparison checks the scale of ground–support displacement conditional
on these inputs, a fixed measurement datum and an unbounded plane-strain
exterior. General noncircular anisotropic exterior accuracy is assessed by
the separate numerical comparisons.
