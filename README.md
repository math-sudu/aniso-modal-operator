# Research data and code

[Download Research_Data_and_Code.zip](https://github.com/math-sudu/aniso-modal-operator/raw/refs/heads/main/Research_Data_and_Code.zip)

The archive contains 106 files, including 41 CSV datasets and 61 Python scripts.

SHA-256: `72229c02dcff7abf0bd7c36bb33272962919119f2d92ecc6c07115a21902b0f5`

Supporting material for **Far-field truncation effects and support response in
noncircular tunnels using a coupled DtN and FEM model based on complex potential
theory**, by
Pengcheng Zhu, Tielin Chen, Dingli Zhang, Jiaqiang Peng, Kai Yang and Yulong Wang.
Correspondence: tlchen1@bjtu.edu.cn.

This archive contains numerical data used in the manuscript, the scripts that
draw its eleven figures, and the material, exterior-operator, finite-element
and lining implementations used by the calculations. The data cover exterior
boundary position and constraint effects, noncircular DtN construction,
coupled mesh convergence, and lining contact and stress release. Extract the archive into
an empty directory. Python 3.11 or later is required.

```shell
python -m pip install -r requirements.txt
python -B check_release.py
python -B reproduce_figures.py
```

The figure command reads the supplied CSVs and writes PDF artwork under
`figures/`; Figures 1–6 also have SVG exports and panel-data CSVs. It does not
solve new tunnel cases. Plot text and geometry are retained from the manuscript
generators. The plotting scripts use Matplotlib and the dependencies listed
in `requirements.txt`.

## Code and calculations

- `lib/exec20_src/`: transversely isotropic compliance, characteristic roots,
  circular harmonics and anisotropic finite elements.
- `lib/exec26_src/`: conformal maps, exact circle/ellipse traction-density
  blocks, spectral assembly, mapped finite elements and complex-potential
  reference fields.
- `lib/proto_src/`, `lib/refsol/` and the two `lib/lekhnitskii_*.py` modules:
  supporting finite-element and analytical-reference routines.
- `code/closure/numeric/p21_exact_evaluation.py`: the square general-map
  Cauchy-constraint assembly, including constant potential values at infinity.
- `code/closure/numeric/p22_graded_kernels.py`: explicit first- and second-order
  mapping corrections. The square assembly and spectral assembly are separate
  constructions; a caller must select the intended one.
- `code/closure/numeric/ub_interface_departure_constants.py`: numerical checks
  of the full-vector periodic P1 consistency formulas.
- `code/figure_exports/`: retained-data plotting and calculation producers.
  In particular, `atlas_family_box.py`, `closed_kernel_ladder.py`,
  `exterior_treatment_sweep.py`, `near_field_clouds.py` and
  `lined_section_demo.py` generate their respective material, kernel,
  refinement, field and lining datasets.

Run a calculation producer by its path from the extracted archive. Parameters
are defined in that producer and in `lib/exec26_src/protocol.py`. These commands
can be substantially more expensive than drawing the figures and replace the
corresponding CSVs. Keep the original downloaded archive if generating new
cases. Only `check_release.py` and the figure command are needed to check the
distribution and reproduce the supplied graphics.

The library snapshot is taken from source revision
`23e7962d241005360f1f70d72db1387788b69a2a`; the paper-specific square-assembly
correction and explicit kernels are supplied in `code/closure/numeric/`.
The export adjusts import paths to the included library, edits documentation
for public use, and uses ordinary Matplotlib file export. Numerical operations,
parameters and result-field names are retained.
The archive needs no other project checkout. The reference-data subdirectory
contains the original material-atlas and circular-benchmark numerical records.

The supplied Abaqus comparison is a retained result table. Its job-path and
console-log columns are omitted; all scientific columns and rows are retained.
Re-executing the commercial-solver comparison requires Abaqus and is outside
the standalone Python checks. The archive's checks do not constitute a new
end-to-end execution of all the original experiments. Timing columns record
the authors' measured machine times, not portable performance guarantees.

## Units, coordinates and reading the data

Files are UTF-8, comma-separated, with a header row. Empty cells mean that the
quantity is not applicable to that row. Field names ending in `_m`, `_mm`,
`_um`, `_MPa`, `_GPa` and `_s` indicate metres, millimetres, micrometres,
megapascals, gigapascals and seconds. Lining thrust and moment are per metre
of tunnel axis, in kN/m and kN m/m. Relative errors, condition numbers, modulus
ratios and map coefficients are dimensionless.

The material family has `ratio = E_h/E_v`, `E_v = 10.50 GPa`,
`nu_vh = 0.20`, `nu_hh = 0.25` and `G_vh = 4.20 GPa`.
For the upright horseshoe, `material_dip_deg` and the older `dip_deg` denote
rotation from the raw map x-axis, which becomes the physical vertical.
`bedding_dip_deg = 90 - material_dip_deg` is the angle from the physical
horizontal. Published circular benchmarks use their stated principal frames.
The historical material keys are F0 = (1.5, 90 degrees bedding dip),
F1 = (3.0, 45 degrees), F2 = (2.0, 60 degrees), SABNGS = Queenston Shale,
and WAI(d=45.0) = Waichecheng. The manuscript gives their constants.

`rho`/`rho_out` is the conformal interface radius; `t` is mapping strength.
`near_mesh` lists radial-by-circumferential elements. `n_h` counts retained
displacement harmonics. In square-assembly result files, `n_t` denotes the
constraint truncation: each branch uses `n_t + 1` sampling points before
projection modulo constants. In finite-element routines, `n_t` denotes the
number of circumferential elements; these are different discretizations.

The geometry is the numerical horseshoe example in Zeng et al. (2024),
DOI [10.1016/j.ijrmms.2024.105880](https://doi.org/10.1016/j.ijrmms.2024.105880),
with radius 5.034 m and the ten coefficients stated in the manuscript.
The physical frame is `x_phys = -y_map`, `y_phys = x_map`.
Map angle `theta` starts at the crown; physical polar angle is
`arg(omega(exp(i*theta)))`. They are not interchangeable. The field plotting
code converts them explicitly.

The complex-potential convergence tables use their recorded reference order
and collocation count. The meshed-exterior comparison provides both full
rigid-motion removal (`diff_free`) and translation-only values
(`diff_free_translation_only`). The manuscript specifies which quantity it
reports. `rel0`, `rel1` and `rel2` compare the base, first-order and
second-order mapping approximations with the assembled operator. Lining
`lambda_released` is the fraction released before support engagement; the
20 MPa fiber-stress threshold is an illustrative design value.

Please cite the accompanying article when using these data or implementations.
Bibliographic details of the scientific sources are in `references.bib`.

Verified export environment: Python 3.13.7; numpy==2.4.4, scipy==1.16.2, matplotlib==3.10.6, mpmath==1.3.0.

## Data files and columns

### Geometry; Figures 1 and 3

- `section_family_boundary.csv` (1800 rows): `member`, `mapping_strength_t`, `rho`, `theta_deg`, `x_m`, `y_m`, `abs_domega`, `univalence_gauge`.

- `hs_section_geometry.csv` (18 rows): `quantity`, `value`, `note`.

- `hs_section_profile.csv` (360 rows): `theta_deg`, `x_map_m`, `y_map_m`, `x_phys_m`, `y_phys_m`, `height_above_invert_m`, `signed_curvature_per_m`, `local_radius_m`.

### Material representation and circular benchmarks; Table 1 and Section 5.2

- `atlas_material_points.csv` (5 rows): `material`, `in_family`, `ratio`, `dip_deg`, `abs_q1`, `abs_q2`, `mu_gap`, `cond_dirichlet_2x2`, `univalence_margin`, `cond_lambda`.

- `anchor_mode_content.csv` (384 rows): `family`, `case`, `ratio`, `dip_deg`, `coeff_err_rel`, `m1_magnitude`, `mode_ge2_magnitude`, `mode_ge2_over_m1`, `bracket_modulus_gap`, `bracket_literal_gap`, `operator_vs_bracket`, `root_real_fraction`.

- `abaqus_crosscode_ladder.csv` (6 rows): `family`, `mesh`, `n_nodes`, `n_dof`, `sta_ok`, `abq_vs_ours_nodal_u`, `ours_vs_ref_u`, `abq_vs_ref_u`, `solver`, `pinned_value`, `pin_deviation`, `reference`, `ref_n_order`, `ref_n_collocation`, `ref_window_dev`, `deck_bc_dev`, `wall_s`.

### Material conditioning and root evaluation; Figures 2 and 6, Table 2

- `atlas_family_box.csv` (10101 rows): `ratio`, `dip_deg`, `abs_q1`, `abs_q2`, `mu_gap`, `cond_dirichlet_2x2`, `univalence_margin`, `cond_lambda`, `bedding_dip_deg`.

- `degenerate_switch.csv` (74 rows): `record`, `ratio`, `label`, `g`, `mu_gap`, `kappa_M`, `err_2root`, `err_conf`.

### Exact assembly and mapping kernels; Figures 4 and 5

- `constraint_conditioning_law.csv` (25 rows): `material`, `n_t`, `eta_min`, `eta_max`, `ratio_predicted`, `cond`, `heldout_residual`, `operator_increment`.

- `exact_ellipse_invariance.csv` (45 rows): `R`, `c1`, `material`, `n_t`, `invariance_rel`, `offblock_ratio`, `cond`.

- `exact_general_structure.csv` (5 rows): `section`, `rho_out`, `material`, `n_t`, `offblock_ratio`, `decay_dn0`, `decay_dn1`, `decay_dn2`, `decay_dn3`, `decay_dn4`, `decay_dn5`, `decay_dn6`, `decay_dn7`, `cond`.

- `closed_kernel_ladder.csv` (45 rows): `rho_out`, `material`, `t`, `rel0`, `rel1`, `rel2`, `S1_over_S0`, `S2_over_S0`, `lowest_covering_order`.

- `closed_kernel_convergence.csv` (20 rows): `material`, `rho_out`, `t`, `n_t`, `rel0`, `rel1`, `rel2`, `cond`.

- `second_profile_ladder.csv` (90 rows): `profile`, `rho_out`, `material`, `t`, `rel0`, `rel1`, `rel2`, `rel2_check_n_t`, `saturation_drift`, `S1_over_S0`, `S2_over_S0`, `lowest_covering_order`.

- `kernel_reuse_cost.csv` (17 rows): `ray`, `material`, `n_h`, `t_member`, `n_t_exact`, `precompute_s`, `compose_s`, `exact_s`, `precompute_over_exact`, `compose_over_exact`, `composition_dev`, `recompose_dev`, `role`.

- `second_profile_cost.csv` (10 rows): `profile`, `material`, `n_h`, `n_t_exact`, `repeats`, `material_base_s`, `ray_kernels_s`, `exact_solve_s`, `material_base_over_exact`, `ray_kernels_over_exact`, `material_base_share`.

### Mesh refinement and computational cost; Figure 8, Table 3 and Appendix A

- `operator_perturbation_control.csv` (29 rows): `ratio`, `dip_deg`, `near_mesh`, `s_scale`, `ref_n_order`, `ref_n_collocation`, `rel_difference`.

- `grade_convergence_ladder.csv` (43 rows): `ratio`, `dip_deg`, `near_mesh`, `route`, `ref_n_order`, `ref_n_collocation`, `rel_difference`.

- `exterior_reference_ladder.csv` (27 rows): `ratio`, `dip_deg`, `near_mesh`, `s_scale`, `rho_far`, `operator_dof`, `far_dof`, `shared_block_dev`, `bracket_gap_rel`, `diff_free`, `diff_free_translation_only`, `diff_fixed`, `diff_fixed_translation_only`, `bracket_spread`.

- `exterior_reference_tight.csv` (18 rows): `ratio`, `dip_deg`, `near_mesh`, `route`, `rho`, `rho_far`, `operator_dof`, `far_dof`, `shared_block_dev`, `bracket_gap_rel`, `diff_free`, `diff_free_translation_only`, `diff_fixed`, `diff_fixed_translation_only`, `bracket_spread`.

- `exterior_cost_ladder.csv` (25 rows): `treatment`, `configuration`, `total_dof`, `interface_dof`, `repeats`, `assemble_s`, `assemble_s_min`, `assemble_s_max`, `assemble_spread_rel`, `operator_s`, `operator_s_min`, `operator_s_max`, `operator_spread_rel`, `solve_s`, `solve_s_min`, `solve_s_max`, `solve_spread_rel`, `total_s`, `total_s_min`, `total_s_max`, `total_spread_rel`, `rel_error_vs_completed_basis`, `ref_n_order`, `ref_n_collocation`, `ref_window_dev`, `rho_far`, `near_mesh`.

- `coarsened_far_comparator.csv` (8 rows): `treatment`, `configuration`, `total_dof`, `interface_dof`, `n_rings`, `repeats`, `assemble_s`, `assemble_s_min`, `assemble_s_max`, `assemble_spread_rel`, `operator_s`, `operator_s_min`, `operator_s_max`, `operator_spread_rel`, `solve_s`, `solve_s_min`, `solve_s_max`, `solve_spread_rel`, `total_s`, `total_s_min`, `total_s_max`, `total_spread_rel`, `rel_error_vs_completed_basis`, `ref_n_order`, `ref_n_collocation`, `rho_far`, `near_mesh`, `growth_ratio`, `n_exterior_rings`, `grading`, `shipped_row`, `near_block_dev`, `clock_gate_median_s`, `ref_window_dev`.

- `coarsened_far_comparator_clamped.csv` (7 rows): `treatment`, `configuration`, `total_dof`, `interface_dof`, `n_rings`, `rel_error_vs_completed_basis`, `ref_n_order`, `ref_n_collocation`, `rho_far`, `near_mesh`, `growth_ratio`, `n_exterior_rings`, `grading`, `shipped_counterpart`, `anchor_far_mesh_fixed`, `near_block_dev`, `ref_window_dev`.

- `harmonic_selection.csv` (36 rows): `ratio`, `dip_deg`, `decay_ratio`, `near_mesh`, `n_harmonics`, `operator_s`, `rel_error_vs_completed_basis`, `ref_n_order`, `ref_n_collocation`, `ref_window_dev`.

### Excavation fields; Figures 7, 9 and 10

- `near_field_case_summary.csv` (10 rows): `case`, `E_ratio`, `material_dip_deg`, `bedding_dip_deg`, `rho_out`, `ref_n_order`, `ref_n_collocation`, `ref_peak_window_dev_MPa`, `ls_wall_residual_Pa`, `peak_hoop_MPa`, `peak_hoop_theta_deg`, `max_wall_convergence_mm`, `coupled_vs_reference_rel`, `max_abs_diff_um`.

- `near_field_stress_hs100.csv` (80019 rows): `case`, `E_ratio`, `bedding_dip_deg`, `x_m`, `y_m`, `sigma_hh_MPa`, `sigma_vv_MPa`, `sigma_hv_MPa`, `sigma_1_MPa`, `sigma_3_MPa`, `max_shear_MPa`.

- `near_field_displacement_hs100.csv` (28800 rows): `case`, `E_ratio`, `bedding_dip_deg`, `x_m`, `y_m`, `on_wall`, `u_h_mm`, `u_v_mm`, `u_mag_mm`.

- `near_field_interface_error_hs100.csv` (32000 rows): `case`, `E_ratio`, `bedding_dip_deg`, `rho_out`, `x_m`, `y_m`, `node_grade`, `abs_diff_um`.

- `wall_hoop_hs100.csv` (3240 rows): `case`, `E_ratio`, `bedding_dip_deg`, `map_theta_deg`, `x_m`, `y_m`, `hoop_sigma_MPa`.

### Lining, contact and stress release; Figure 11 and Appendix B

- `lined_demo_actions.csv` (3456 rows): `ratio`, `material_dip_deg`, `near_mesh`, `route`, `station_theta_deg`, `thrust_kN_per_m`, `moment_kNm_per_m`.

- `lined_demo_summary.csv` (18 rows): `ratio`, `material_dip_deg`, `bedding_dip_deg`, `near_mesh`, `route`, `s_scale`, `closure_vertical_mm`, `closure_horizontal_mm`, `thrust_max_kN_per_m`, `thrust_max_theta_deg`, `moment_absmax_kNm_per_m`, `moment_absmax_theta_deg`, `fiber_stress_max_MPa`, `interface_net_force_rel`, `solve_s`.

- `lined_exterior_check.csv` (54 rows): `ratio`, `material_dip_deg`, `bedding_dip_deg`, `near_mesh`, `route`, `s_scale`, `closure_vertical_mm`, `closure_horizontal_mm`, `thrust_max_kN_per_m`, `thrust_max_theta_deg`, `moment_absmax_kNm_per_m`, `moment_absmax_theta_deg`, `fiber_stress_max_MPa`, `interface_net_force_rel`, `solve_s`.

- `lined_lame_gate_ladder.csv` (4 rows): `stations`, `layers`, `element_aspect_ratio`, `thrust_rel_err_integral`, `thrust_rel_err_pRo`, `moment_rel_err`, `thrust_station_spread`, `moment_err_ratio_to_previous`.

- `lined_ring_thickness_ladder.csv` (27 rows): `ratio`, `material_dip_deg`, `bedding_dip_deg`, `near_mesh`, `ring_layers`, `thrust_max_kN_per_m`, `moment_absmax_kNm_per_m`, `closure_vertical_mm`, `closure_horizontal_mm`, `thrust_shift_pct`, `moment_shift_pct`.

- `lined_lateral_sweep.csv` (36 rows): `ratio`, `material_dip_deg`, `bedding_dip_deg`, `near_mesh`, `route`, `s_scale`, `closure_vertical_mm`, `closure_horizontal_mm`, `thrust_max_kN_per_m`, `thrust_max_theta_deg`, `moment_absmax_kNm_per_m`, `moment_absmax_theta_deg`, `fiber_stress_max_MPa`, `interface_net_force_rel`, `solve_s`, `lateral_pressure_coefficient`.

- `lined_contact_sensitivity.csv` (9 rows): `ratio`, `material_dip_deg`, `bedding_dip_deg`, `near_mesh`, `thrust_bonded_kN_per_m`, `thrust_slip_kN_per_m`, `moment_bonded_kNm_per_m`, `moment_slip_kNm_per_m`, `fiber_bonded_MPa`, `fiber_slip_MPa`, `closure_vertical_bonded_mm`, `closure_vertical_slip_mm`, `closure_horizontal_bonded_mm`, `closure_horizontal_slip_mm`, `thrust_shift_pct`, `moment_shift_pct`, `fiber_shift_pct`, `closure_vertical_shift_pct`, `max_tangential_slip_mm`.

- `lined_release_ladder.csv` (72 rows): `ratio`, `material_dip_deg`, `bedding_dip_deg`, `lambda_released`, `closure_vertical_mm`, `closure_horizontal_mm`, `thrust_max_kN_per_m`, `moment_absmax_kNm_per_m`, `fiber_stress_max_MPa`.

- `lined_release_partition.csv` (27 rows): `lateral_pressure_coefficient`, `ratio`, `material_dip_deg`, `bedding_dip_deg`, `lambda_released`, `thrust_max_kN_per_m`, `moment_absmax_kNm_per_m`, `closure_vertical_mm`, `closure_horizontal_mm`, `fiber_stress_max_MPa`.

- `lined_release_threshold.csv` (1 rows): `fiber_stress_limit_MPa`, `binding_lateral_pressure_coefficient`, `binding_ratio`, `binding_bedding_dip_deg`, `fiber_stress_at_zero_release_MPa`, `lambda_required`, `lambda_reported`, `fiber_stress_at_reported_MPa`.
