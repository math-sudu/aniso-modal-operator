# Code and input data

For **A complex-potential DtN–FEM model for noncircular tunnel deformation
and lining response**, by Pengcheng Zhu, Tielin Chen, Dingli Zhang and
Yulong Wang. Correspondence: tlchen1@bjtu.edu.cn.

The repository contains the complex-potential exterior operator, its
finite-element coupling, support and contact calculations, figure scripts and
measured input data. Numerical tables and figures are generated locally.
Original field photographs provide the engineering context in Figures 1 and 9.

## Install and check

Use Python 3.11 or later and run these commands from the repository root:

```shell
python -m pip install -r requirements.txt
python -B check_release.py --solvers-only
python -B reproduce_data.py --help
python -B reference_convergence.py --central-only
```

The solver check imports the distributed modules, verifies exact circle and
ellipse blocks, checks zero shape corrections and tests the elastic stress
path. The last command recomputes the order-48/64 displacement-reference
comparison for the central horseshoe case and writes
`results/data/derived/reference_self_convergence.csv`.

The source release runs independently of any other project checkout. The
solver library derives from revision `23e7962d241005360f1f70d72db1387788b69a2a`;
paper-specific operator corrections and calculations are included here.
Comparisons that merely reread old result tables are omitted from the source
release. Independent analytical and reference-resolution checks remain.

## Reproduce figures and tables

For all ten figures, first generate their numerical data, then render:

```shell
python -B reproduce_data.py figures --list
python -B reproduce_data.py figures
python -B reproduce_figures.py
```

The first command lists the calculation sequence. The second runs the
geometry, material, kernel, excavation, exterior, displacement, lining and
refined-contact studies. These full mesh and case sequences take substantially
longer than plotting.
Each study can also run separately, for example
`python -B reproduce_data.py contact`. A failed calculation stops the sequence.
For `exterior`, run `fields` first: the wall-profile calculation reads the
24x128 error fields and extends them with the coarse and fine wall solutions.
Results go to `results/data/derived/`; figure PDF/SVG files go to `figures/`.
Times New Roman and STIX fonts are required for the manuscript typography.
`reproduce_figures.py --only <display_job_id>` selects one of Figures 1–10;
the figure IDs and sizes are in `paper/figures/content_figure_spec.json`.

The additional table studies use the following commands. Their output tables
are also written under `results/data/derived/`.

| Result | Command after `python -B` |
|---|---|
| Table 3: pole placement | `code/figure_exports/pole_placement.py` |
| Retained bands N=31/63/127 | `code/figure_exports/pole_placement.py --case bands --output results/data/derived/band_resolution.csv` |
| Table 4: interface order | `code/figure_exports/trace_order.py` |
| Table 5: noncircular-hole stress concentration | `code/figure_exports/notch_scf.py --refinement 2 --harmonics 64` |
| Table 6: staged plasticity, near domain | `code/figure_exports/localized_plasticity.py --mesh 48 256 --output results/data/derived/localized_plasticity_mesh_refined.csv` |
| Table 6: staged plasticity, expanded domain | `code/figure_exports/localized_plasticity.py --mesh 48 128 --rho 4 --output results/data/derived/localized_plasticity_domain_expanded.csv` |
| Section 5.3: fixed-FE exterior replacements | `code/figure_exports/support_operator.py --study contact`, then `--study thickness`, `--study bonded --refined-spectral` and `--study plastic` |
| Section 7.2; response Table R2-6: lining modulus | `code/figure_exports/lining_modulus.py` |
| Section 7.2; response Table R2-19: local thickness | `code/figure_exports/lining_thickness.py` |
| Ground-yield constraint | `code/figure_exports/elastic_admissibility.py` |
| Pressure uncertainty | `code/figure_exports/pressure_uncertainty.py` |
| Capacity variability | `code/figure_exports/capacity_variability.py` |

Run the lining study before pressure uncertainty; run pressure uncertainty
before capacity variability. The lining study generates the continuum stress
demands and the stress–closure screen in dependency order. Running
`check_release.py` without `--solvers-only` additionally checks these generated
support tables against their release partition.

Table 7 summarizes parameters and responses from seven cited studies.
The refined Figure 10 contact data use a 48-by-256 ground mesh. For Table 4's
band check, use `trace_order.py --harmonics 191 --fine-only --output
results/data/derived/trace_order_modal_refined.csv` under
`code/figure_exports/`. Mesh, case and material options are documented by each
script's `--help` where an argument parser is provided.

## Experimental inputs and comparisons

The source observations are supplied under `results/data/derived/` and are
kept separate from the generated predictions.

For Figure 6, `fatima2017_observations.csv` preserves the 2108 candidate
image locations and decoding mask from Fatima and Rowlands (2017),
DOI [10.17265/2159-5275/2017.04.005](https://doi.org/10.17265/2159-5275/2017.04.005).
The comparison uses the 1278 readable locations, reflected to x transverse
and y axial, with the source's 0.1R hole-edge exclusion and a 1.05 mm square
subset average. The measured image and printed palette are retained in
`figures/displacement-measurement/assets/fatima2017_measured_quadrant.npz`.
For display, printed contour strokes take the nearest decoded fill color;
the decoded bands, white gaps and numerical observations are preserved.

`python -B reproduce_data.py displacement` generates the refined fields and
the three continuous model contours used in Figure 6. To reproduce the
coarse/fine assessment and normalized spatial-pattern metrics, run:

```shell
python -B code/figure_exports/orthotropic_hole.py --from-data --case fatima
python -B code/figure_exports/orthotropic_hole.py --from-data --case fatima --refinement 2 --harmonics 64
python -B code/figure_exports/orthotropic_hole.py --assess-only
python -B displacement_pattern.py
```

For Table 5, `khechai2018_table3.csv` preserves the source stress-concentration
measurements from Khechai et al. (2018),
DOI [10.1016/j.compstruct.2017.11.060](https://doi.org/10.1016/j.compstruct.2017.11.060).
The paper compares all seven noncircular cases using the independently
reported material properties and the 0.5 mm corner radius. P2 triangles
evaluate axial stress at hole-edge midpoints. The finite plate fixes its
lower end and loads its upper end, as in the source's Section 5. The
infinite model uses the shared Cauchy exterior with a projected P1 trace.
Experimental differences, source-FEM differences and mesh refinement are
reported separately. `--protocol` reproduces the boundary, element-order
and stress-recovery controls on two representative holes, including the
source's 20 segments per rounded corner.

```shell
python -B code/figure_exports/notch_scf.py
python -B code/figure_exports/notch_scf.py --refinement 2 --harmonics 64
python -B code/figure_exports/notch_scf.py --assess-only
python -B code/figure_exports/notch_scf.py --protocol
python -B code/figure_exports/notch_scf.py --check-elements
```

`python -B reproduce_data.py notches` regenerates both resolution tables
and their assessment.

## Model and data interpretation

- `lib/exec20_src/` contains TI material, circular-harmonic and finite-element
  routines. `lib/exec26_src/` contains conformal maps, exterior assembly,
  mapped finite elements and independent complex-potential references.
- `code/closure/numeric/p21_exact_evaluation.py` builds the general-map
  Cauchy-constraint operator. The shared production entry in
  `code/closure/exterior_operator.py` uses N+64 resultant harmonics,
  enforces the simple-root domain for general maps, and supplies a stable
  coincident-root block for circles and ellipses. It couples periodic P1
  traces through two-sided Fourier projection, `pi*T.T*S_modal*T`;
  `p22_graded_kernels.py` evaluates the explicit
  first- and second-order mapping corrections. The pole-based realization is
  in `lib/exec26_src/sf_general.py`.
- Numerical horseshoe parameters are prescribed scenarios. Geometry follows
  Zeng et al. (2024), DOI
  [10.1016/j.ijrmms.2024.105880](https://doi.org/10.1016/j.ijrmms.2024.105880),
  with radius 5.034 m and the coefficients in `lib/exec26_src/protocol.py`.
  The physical frame is `x_phys = -y_map`, `y_phys = x_map`.
- The material family has `E_v = 10.50 GPa`, `nu_vh = 0.20`,
  `nu_hh = 0.25` and `G_vh = 4.20 GPa`. In the upright horseshoe,
  `bedding_dip_deg = 90 - material_dip_deg`. All reported loading cases are
  dry-ground, tension-positive stress calculations.
- CSV suffixes `_m`, `_mm`, `_um`, `_MPa`, `_GPa` and `_s` give metres,
  millimetres, micrometres, megapascals, gigapascals and seconds. Lining thrust
  and moment are per metre of tunnel axis. Timing columns depend on the computer.
- The lining screen uses prescribed compressive/tensile strengths of
  20/1.5 MPa and 24 mm chord allowances. Its interval must also satisfy the
  ground strength condition. It is an unfactored scenario screen.

Numerical-reference errors concern the prescribed horseshoe benchmark.

Please cite the accompanying article when using the implementations or
inputs. Bibliographic details of the scientific sources are in
`references.bib`.
