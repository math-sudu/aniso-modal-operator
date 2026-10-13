# Code and input data

For **A complex-potential DtN–FEM model for noncircular tunnel deformation
and lining response**, by Pengcheng Zhu, Tielin Chen, Dingli Zhang and
Yulong Wang. Correspondence: tlchen1@bjtu.edu.cn.

The repository contains the complex-potential exterior operator, its
finite-element coupling, support and contact calculations, figure scripts,
original field photographs and Moxi monitoring inputs. Numerical tables and
figures are generated locally. The accompanying research archive contains
the manuscript's saved calculation results.

## Install and check

Use Python 3.11 or later and run these commands from the repository root:

```shell
python -m pip install -r requirements.txt
python -B check_release.py --solvers-only
python -B reproduce_data.py inputs
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

Figure 10 uses the supplied observations and the six-arc section definition:

```shell
python -B reproduce_data.py inputs
python -B code/figure_exports/moxi_monitoring.py
```

For all ten figures, first generate their numerical data, then render:

```shell
python -B reproduce_data.py figures --list
python -B reproduce_data.py figures
python -B reproduce_figures.py
```

The first command lists the calculation sequence. The second runs the
material, kernel, excavation, exterior, lining and refined-contact studies;
these full mesh and case sequences take substantially longer than plotting.
Each study can also run separately, for example
`python -B reproduce_data.py contact`. A failed calculation stops the sequence.
Results go to `results/data/derived/`; figure PDF/SVG files go to `figures/`.
Times New Roman and STIX fonts are required for the manuscript typography.
`reproduce_figures.py --only <display_job_id>` selects one of Figures 1–9;
the figure IDs and sizes are in `paper/figures/content_figure_spec.json`.

The additional table studies use the following commands. Their output tables
are also written under `results/data/derived/`.

| Result | Command after `python -B` |
|---|---|
| Table 3: pole placement | `code/figure_exports/pole_placement.py` |
| Retained bands N=31/63/127 | `code/figure_exports/pole_placement.py --case bands --output results/data/derived/band_resolution.csv` |
| Table 4: interface order | `code/figure_exports/trace_order.py` |
| Table 5: lining modulus | `code/figure_exports/lining_modulus.py` |
| Table 6: local thickness | `code/figure_exports/lining_thickness.py` |
| Table 7: staged plasticity | `code/figure_exports/localized_plasticity.py --mesh 48 256 --output results/data/derived/localized_plasticity_mesh_refined.csv` |
| Table 8: exterior replacement | `code/figure_exports/support_operator.py --study contact`, then `--study thickness`, `--study bonded --refined-spectral` and `--study plastic` |
| Ground-yield constraint | `code/figure_exports/elastic_admissibility.py` |
| Pressure uncertainty | `code/figure_exports/pressure_uncertainty.py` |
| Capacity variability | `code/figure_exports/capacity_variability.py` |

Run the lining study before pressure uncertainty; run pressure uncertainty
before capacity variability. The lining study generates the continuum stress
demands and the stress–closure screen in dependency order. Running
`check_release.py` without `--solvers-only` additionally checks these generated
support tables against their release partition.

The refined Figure 9 contact data use a 48-by-256 ground mesh. For Table 4's
band check, use `trace_order.py --harmonics 191 --fine-only --output
results/data/derived/trace_order_modal_refined.csv` under
`code/figure_exports/`. Mesh, case and material options are documented by each
script's `--help` where an argument parser is provided. The original Abaqus
comparison requires the commercial solver and is represented by its result
table in the accompanying research archive.

## Model and data interpretation

- `lib/exec20_src/` contains TI material, circular-harmonic and finite-element
  routines. `lib/exec26_src/` contains conformal maps, exterior assembly,
  mapped finite elements and independent complex-potential references.
- `code/closure/numeric/p21_exact_evaluation.py` builds the general-map
  Cauchy-constraint operator; `p22_graded_kernels.py` evaluates the explicit
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
- CSV suffixes `_m`, `_mm`, `_MPa`, `_GPa` and `_s` give metres,
  millimetres, megapascals, gigapascals and seconds. Lining thrust and moment
  are per metre of tunnel axis. Timing columns depend on the computer.
- The lining screen uses prescribed compressive/tensile strengths of
  20/1.5 MPa and 24 mm chord allowances. Its interval must also satisfy the
  ground strength condition. It is an unfactored scenario screen.

`results/data/field/moxi675/` contains 121 original monitoring entries,
construction events and the 12.86-by-10.45 m inner-section specification.
Survey chords lie 2.5, 6.0 and 9.5 m below the crown. Original day/value text,
repeated readings, units, value flags and source-table cell locators remain
in the CSV. All four channels end on reported day 39. The plot uses cumulative
values on the reported days; flagged unreadable or malformed entries break
the line. Reported events are lower-bench excavation at day 4, primary-ring
closure at day 8 and invert-concrete placement at day 25.

These are measured histories. Quantitative prediction of the monitored
section remains unvalidated: the records do not independently determine
support-installation release or the evolution of ground and support
properties. Numerical-reference errors concern the prescribed horseshoe
benchmark. Original field photographs provide engineering context.

Please cite the accompanying article when using the implementations or
inputs. Bibliographic details of the scientific sources are in
`references.bib`.
