# Code and input data

For **A complex-potential DtN–FEM formulation for nonlinear tunnel–ground
interaction**, by Pengcheng Zhu, Tielin Chen, Dingli Zhang and
Yulong Wang. Correspondence: tlchen1@bjtu.edu.cn.

The repository contains the complex-potential exterior operator, its
finite-element coupling, support and contact calculations, figure scripts and
material and loading inputs. Numerical tables and figures are generated locally.
Original field photographs provide the engineering context in Figures 1 and 8.

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

The source release runs independently of any other project checkout and
includes the operator, near-field and support calculations used in the article.
Comparisons that merely reread old result tables are omitted from the source
release. Independent analytical and reference-resolution checks remain.

## Reproduce figures and tables

For all nine figures, first generate their numerical data, then render:

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
`reproduce_figures.py --only <figure_id>` selects one of Figures 1–9;
the figure IDs and sizes are in `paper/figures/content_figure_spec.json`.

The additional table studies use the following commands. Outputs go to
`results/data/derived/`, except for the W4 files written beside its scripts.

| Result | Command after `python -B` |
|---|---|
| Table 3: projected pole-placement comparison | `code/figure_exports/pole_placement.py --case projected --output results/data/derived/projected_work.csv` |
| Automatic pole selection | `code/figure_exports/pole_placement.py --case automatic --output results/data/derived/automatic_poles.csv` |
| Retained bands N=31/63/127 | `code/figure_exports/pole_placement.py --case bands --output results/data/derived/band_resolution.csv` |
| Table 4: interface order | `code/figure_exports/trace_order.py` |
| Table 5: supported W4 model-test comparison | `code/experiments/duan2024/reproduce.py` |
| Table 6: staged PINN ground, 1,632 neurons | `code/closure/near_field_pinn.py --output results/data/derived/near_field_pinn/medium` |
| Table 6: 3,312-neuron refinement | `code/closure/near_field_pinn.py --radial 18 --angular 144 --output results/data/derived/near_field_pinn/fine` |
| Table 6: half loading step | `code/closure/near_field_pinn.py --step 0.01 --output results/data/derived/near_field_pinn/step_refined` |
| Section 6.4: independent history/field verification | `code/closure/check_near_field_pinn.py results/data/derived/near_field_pinn/fine` |
| Section 6.4: elastic DP wall first yield, 0.42814 | `code/figure_exports/dp_wall_onset.py` |
| Section 5.3: fixed-FE exterior replacements | `code/figure_exports/support_operator.py --study contact`, then `--study thickness`, `--study bonded --refined-spectral` and `--study plastic` |
| Section 6.2; response Table R2-6: lining modulus | `code/figure_exports/lining_modulus.py` |
| Section 6.2; response Table R2-19: local thickness | `code/figure_exports/lining_thickness.py` |
| Ground-yield constraint | `code/figure_exports/elastic_admissibility.py` |
| Pressure uncertainty | `code/figure_exports/pressure_uncertainty.py` |
| Capacity variability | `code/figure_exports/capacity_variability.py` |

Run the lining study before pressure uncertainty; run pressure uncertainty
before capacity variability. The lining study generates the continuum stress
demands and the stress–closure screen in dependency order. Running
`check_release.py` without `--solvers-only` additionally checks these generated
support tables against their release partition.

Table 7 summarizes parameters and responses from seven cited studies.
The refined Figure 9 contact data use a 48-by-256 ground mesh. For Table 4's
band check, use `trace_order.py --harmonics 191 --fine-only --output
results/data/derived/trace_order_modal_refined.csv` under
`code/figure_exports/`. Mesh, case and material options are documented by each
script's `--help` where an argument parser is provided.

The W4 geometry, measured displacement curves and model inputs are in
`code/experiments/duan2024/`. Its [README](code/experiments/duan2024/README.md)
documents source locations, mesh/step checks and the input and boundary
comparisons. The PINN commands above generate their histories, weights and
independent checks locally.

## Model and data interpretation

- `lib/anisotropic_elasticity/` contains TI material, circular-harmonic and finite-element
  routines. `lib/mapped_elasticity/` contains conformal maps, exterior assembly,
  mapped finite elements and independent complex-potential references.
- `code/closure/numeric/cauchy_evaluation.py` builds the general-map
  Cauchy-constraint operator. The shared production entry in
  `code/closure/exterior_operator.py` uses N+64 resultant harmonics,
  enforces the simple-root domain for general maps, and supplies a stable
  coincident-root block for circles and ellipses. It couples periodic P1
  traces through two-sided Fourier projection, `pi*T.T*S_modal*T`;
  `graded_kernels.py` evaluates the explicit
  first- and second-order mapping corrections. The pole-based realization is
  in `lib/mapped_elasticity/sf_general.py`.
- Numerical horseshoe parameters are prescribed scenarios. Geometry follows
  Zeng et al. (2024), DOI
  [10.1016/j.ijrmms.2024.105880](https://doi.org/10.1016/j.ijrmms.2024.105880),
  with radius 5.034 m and the coefficients in `lib/mapped_elasticity/protocol.py`.
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
