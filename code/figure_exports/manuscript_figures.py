from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, ConnectionPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FixedLocator
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'results/data/derived'
SPEC = ROOT / 'paper/figures/content_figure_spec.json'
sys.path.insert(0, str(ROOT / 'code/closure/numeric'))
from p21_exact_evaluation import cauchy_radius
import origin_style as fs
MM = 1 / 25.4
NAVY, TEAL, RUST, PURPLE, GREY = ('#244663', '#147D92', '#BD5A37', '#746393', '#626870')
MATERIALS = ('F0', 'F1', 'F2', 'SABNGS', 'WAI(d=45.0)')
COLORS = dict(zip(MATERIALS, (NAVY, TEAL, RUST, PURPLE, GREY)))
MARKERS = dict(zip(MATERIALS, ('o', 's', '^', 'D', 'v')))
TEXT = {'F0': '$(1.5,90^\\circ)$', 'F1': '$(3.0,45^\\circ)$', 'F2': '$(2.0,60^\\circ)$', 'SABNGS': 'Queenston Shale', 'WAI(d=45.0)': 'Waichecheng', 'domain': 'Computational domain', 'boundary': 'Interface coupling', 'exterior': 'Homogeneous exterior', 'near': 'Finite-element near field', 'opening': 'Excavation', 'bedding': 'Bedding', 'interface': 'Interface', 'near_short': 'Near field', 'ext_short': 'Exterior', 'stiffness': 'Exterior stiffness', 'profiles': 'Two-term profiles', 'density': 'Traction density', 'traction': 'Traction values', 'circle': 'Circle', 'ellipse': 'Ellipse', 'theta': 'Parameter $\\theta$ (deg)', 'parameter': 'Parameter plane', 'branch1': 'Branch 1', 'branch2': 'Branch 2', 'fold': 'Fold shell', 'test_points': 'Test points', 'wall': 'Boundary', 'heldout': 'Independent test residual', 'increment': 'Operator increment', 'truncation': 'Constraint harmonics $n_c$', 'relative': 'Relative error', 'orders': 'Accuracy along the shape ray', 'placement': 'Interface placement', 'strength': 'Mapping strength $t$', 'rho': 'Interface radius parameter $\\rho$', 'base': '$S_0$', 'first': '$S_0+tS_1$', 'second': '$S_0+tS_1+t^2S_2$', 'target': '$10^{-3}$ target', 'box': 'Conditioning in the material box', 'switch': 'Approach to a double root', 'ratio': 'Modulus ratio $E_h/E_v$', 'condition': 'Condition number', 'gap': 'Root separation $|\\mu_1-\\mu_2|$', 'two_root': 'Two-root evaluation', 'confluent': 'Coalesced-root formula', 'dip_range': 'dip $0$--$90^\\circ$', 'second_at_one': '$t=1$, second order', 'residual_axis': 'Analyticity residual', 'increment_axis': 'Relative operator increment'}

def style() -> None:
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'mathtext.fontset': 'stixsans', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'lines.linewidth': 1.4, 'lines.markersize': 4, 'legend.frameon': False, 'savefig.facecolor': 'white', 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'svg.hashsalt': 'aniso-modal-operator-content-figures', 'axes.unicode_minus': True})

def read_csv(name: str) -> list[dict]:
    with (DATA / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))

def numbers(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows])

def panel(ax, letter: str, title: str) -> None:
    ax.set_title(title, loc='left', fontsize=9, pad=10)
    ax.text(-0.16, 1.055, letter, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')

def grid(ax) -> None:
    ax.grid(axis='y', color='#E2E5E8', linewidth=0.45)
    ax.set_axisbelow(True)
    ax.minorticks_off()

def legend_materials(fig, *, y=0.035) -> None:
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], markerfacecolor='white', label=TEXT[m]) for m in MATERIALS]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.51, y), ncol=3, columnspacing=1.25, handlelength=1.8, labelspacing=0.8)

def material(ratio=3.0, phi=45.0):
    """Equations beta_reduction, beta_rotation and characteristic_roots; GPa."""
    ev, eh, nvh, nhh, shear = (10.5, ratio * 10.5, 0.2, 0.25, 4.2)
    beta0 = np.array([[(1 - nhh ** 2) / eh, -nvh * (1 + nhh) / ev, 0], [-nvh * (1 + nhh) / ev, 1 / ev - nvh ** 2 * eh / ev ** 2, 0], [0, 0, 1 / shear]])
    c, s = (np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi)))
    qrot = np.array([[c * c, s * s, 2 * s * c], [s * s, c * c, -2 * s * c], [-s * c, s * c, c * c - s * s]])
    beta = qrot.T @ beta0 @ qrot
    b11, b12, b16 = beta[0]
    b22, b26, b66 = (beta[1, 1], beta[1, 2], beta[2, 2])
    roots = np.roots([b11, -2 * b16, 2 * b12 + b66, -2 * b26, b22])
    mus = np.array(sorted(roots[roots.imag > 0], key=lambda z: z.imag))
    p = b11 * mus ** 2 + b12 - b16 * mus
    q = b12 * mus + b22 / mus - b26
    mat_m = np.vstack((p, q))
    mat_b = np.vstack((mus, -np.ones(2)))
    lam = -1j * np.linalg.solve(mat_m.T, mat_b.T).T
    assert np.linalg.eigvalsh(beta).min() > 0
    assert np.linalg.norm(lam - lam.conj().T) < 1e-12
    assert np.linalg.eigvalsh(-lam).min() > 0
    return (mus, mat_m, mat_b, lam)

def omega(zeta, coeff, radius):
    zeta = np.asarray(zeta, dtype=complex)
    return radius * (zeta + sum((c * zeta ** (-j) for j, c in enumerate(coeff))))

def domega(zeta, coeff, radius):
    zeta = np.asarray(zeta, dtype=complex)
    return radius * (1 - sum((j * c * zeta ** (-j - 1) for j, c in enumerate(coeff))))

def hs(spec, rho=1.0, theta=None):
    if theta is None:
        theta = np.linspace(0, 2 * np.pi, 721)
    geo = spec['geometry']
    return omega(rho * np.exp(1j * theta), geo['coefficients'], geo['radius_m'])

def physical(z):
    return np.c_[-z.imag, z.real]

def verify_geometry(spec):
    rows = [r for r in read_csv('section_family_boundary.csv') if r['member'] == 'HS100']
    error = 0.0
    geo = spec['geometry']
    for row in rows:
        rho = float(row['rho'])
        angle = np.deg2rad(float(row['theta_deg']))
        value = hs(spec, rho, angle)
        recorded = complex(float(row['x_m']), float(row['y_m']))
        error = max(error, float(abs(value - recorded)))
    assert len(rows) == 360 and error < 7.2e-05, (len(rows), error)
    theta = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    jacobian_min = float(np.min(abs(domega(np.exp(1j * theta), geo['coefficients'], geo['radius_m']))))
    assert jacobian_min > 3.32
    return {'retained_coordinate_rows': len(rows), 'max_coordinate_error_m': error, 'coordinate_precision_m': 0.0001, 'minimum_wall_jacobian_m': jacobian_min}

def draw_domain(spec):
    from engineering_figures import draw_domain as draw_engineering_context
    return draw_engineering_context(spec)

def density_data():
    mus, mat_m, mat_b, lam = material()
    theta = np.linspace(0, 2 * np.pi, 721)
    sigma = np.exp(1j * theta)
    datum = np.array([0.5, 0.0])
    coeff = np.linalg.solve(mat_m, datum)
    density = 2 * np.real((3 * lam @ datum)[:, None] * np.exp(-3j * theta))
    rows, worst = ([], 0.0)
    for c1 in (0.0, 0.3):
        z = sigma + c1 / sigma
        dz = 1j * (sigma - c1 / sigma)
        gamma = (1 - 1j * mus + (1 + 1j * mus) * c1) / 2
        delta = ((1 - 1j * mus) * c1 + (1 + 1j * mus)) / 2
        prime = -3 * coeff[:, None] * sigma ** (-4) / (gamma[:, None] - delta[:, None] * sigma ** (-2))
        sxx = 2 * np.real(np.sum(mus[:, None] ** 2 * prime, axis=0))
        syy = 2 * np.real(np.sum(prime, axis=0))
        sxy = -2 * np.real(np.sum(mus[:, None] * prime, axis=0))
        direct = np.array([sxx * dz.imag - sxy * dz.real, sxy * dz.imag - syy * dz.real])
        worst = max(worst, float(np.max(abs(direct - density)) / np.max(abs(density))))
        for j, angle in enumerate(theta):
            rows.append({'c1': c1, 'theta_deg': np.rad2deg(angle), 'x_over_R': z[j].real, 'y_over_R': z[j].imag, 'jacobian_over_R': abs(dz[j]), 'density_x_over_Ev_u0': direct[0, j] / 10.5, 'density_y_over_Ev_u0': direct[1, j] / 10.5, 'R_traction_x_over_Ev_u0': direct[0, j] / abs(dz[j]) / 10.5, 'R_traction_y_over_Ev_u0': direct[1, j] / abs(dz[j]) / 10.5})
    assert worst < 1e-12, worst
    expected = read_csv('atlas_family_box.csv')
    point = next((r for r in expected if float(r['ratio']) == 3 and float(r['dip_deg']) == 45))
    assert abs(np.linalg.cond(lam) - float(point['cond_lambda'])) < 1e-09
    return (rows, {'density_vs_stress_traction_relative_error': worst, 'material': {'Eh_over_Ev': 3, 'phi_deg': 45, 'Ev_GPa': 10.5}, 'displacement': 'u_x=u0*cos(3*theta), u_y=0', 'c1': [0, 0.3]})

def draw_density(spec):
    NAVY, TEAL, RUST, GREY = (fs.BLACK, fs.BLUE, fs.RED, '#666666')
    rows, checks = density_data()
    fig = fs.figure(390 * 25.4 / 72.27, 114)
    profiles = fig.add_axes([0.075, 0.65, 0.4, 0.24])
    jac = fig.add_axes([0.615, 0.65, 0.355, 0.24])
    density = fig.add_axes([0.1, 0.18, 0.365, 0.29])
    traction = fig.add_axes([0.615, 0.18, 0.355, 0.29])
    for c1, offset, name in ((0.0, -1.7, 'circle'), (0.3, 1.55, 'ellipse')):
        r = [x for x in rows if x['c1'] == c1]
        x, y = (numbers(r, 'x_over_R'), numbers(r, 'y_over_R'))
        profiles.fill(x + offset, y, color='#F1F4F5', zorder=0)
        profiles.plot(x + offset, y, color=NAVY, lw=1.15)
        indices = np.arange(0, len(r) - 1, 60)
        profiles.plot(x[indices] + offset, y[indices], 'o', color=NAVY, ms=2.2)
        arrow_indices = np.arange(0, 720, 80)
        dx = 0.4 * np.cos(3 * np.deg2rad(numbers(r, 'theta_deg')[arrow_indices]))
        profiles.quiver(x[arrow_indices] + offset, y[arrow_indices], dx, np.zeros_like(dx), angles='xy', scale_units='xy', scale=1, color=TEAL, width=0.007, headwidth=3.6, headlength=4.5, zorder=4)
        profiles.text(offset, -1.48, TEXT[name], ha='center', fontsize=8)
        profiles.text(offset, 1.48, f'$c_1={c1:g}$', ha='center', fontsize=8)
    profiles.set(aspect='equal', xlim=(-3.1, 3.05), ylim=(-1.8, 1.85))
    profiles.axis('off')
    fig.text(0.275, 0.555, '$u_x=u_0\\cos3\\theta,\\quad u_y=0$', ha='center', fontsize=9)
    for c1, ls in ((0.0, '-'), (0.3, (0, (4, 2)))):
        r = [x for x in rows if x['c1'] == c1]
        theta = numbers(r, 'theta_deg')
        jac.plot(theta, numbers(r, 'jacobian_over_R'), color=GREY if c1 == 0 else TEAL, ls=ls, lw=1.35)
        for component, col in (('x', NAVY), ('y', RUST)):
            for ax, key in ((density, f'density_{component}_over_Ev_u0'), (traction, f'R_traction_{component}_over_Ev_u0')):
                ax.plot(theta, numbers(r, key), color=col, ls=ls, lw=2.3 if c1 == 0 else 1.15, alpha=0.28 if c1 == 0 else 1)
    ellipse = [x for x in rows if x['c1'] == 0.3]
    jac.fill_between(numbers(ellipse, 'theta_deg'), 1, numbers(ellipse, 'jacobian_over_R'), color=TEAL, alpha=0.09)
    jac.set(ylim=(0.6, 1.4), yticks=[0.7, 1.0, 1.3])
    jac.set_ylabel("$|\\omega'|/R$", labelpad=3)
    jac.text(0.53, 0.83, TEXT['ellipse'], color=TEAL, transform=jac.transAxes, fontsize=8)
    jac.text(0.77, 0.54, TEXT['circle'], color=GREY, transform=jac.transAxes, fontsize=8)
    for ax in (jac, density, traction):
        ax.set(xlim=(0, 360), xticks=[0, 90, 180, 270, 360])
        ax.set_xlabel(TEXT['theta'])
        fs.style_axes(ax)
    for ax in (density, traction):
        ax.set(ylim=(-4.5, 4.5), yticks=[-4, -2, 0, 2, 4])
        ax.axhline(0, color=GREY, lw=0.55, zorder=0)
    density.set_ylabel('$(dF/d\\theta)/(E_vu_0)$', labelpad=3)
    traction.set_ylabel('$R\\,t/(E_vu_0)$', labelpad=3)
    for xpos, ypos, letter, title in ((0.025, 0.94, 'a', 'One displacement harmonic'), (0.545, 0.94, 'b', 'Boundary Jacobian'), (0.025, 0.515, 'c', 'Identical traction densities'), (0.545, 0.515, 'd', 'Geometry-dependent traction')):
        fig.text(xpos, ypos, letter, fontweight='bold', fontsize=10)
        fig.text(xpos + 0.04, ypos, title, fontsize=9)
    fig.legend(handles=[Line2D([], [], color=NAVY, label='$x$ component'), Line2D([], [], color=RUST, label='$y$ component'), Line2D([], [], color=GREY, lw=2.3, alpha=0.4, label=TEXT['circle']), Line2D([], [], color=GREY, ls=(0, (4, 2)), label=TEXT['ellipse'])], loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.012), handlelength=2.0, columnspacing=1.3)
    checks['jacobian_range_over_R'] = [float(min(numbers(ellipse, 'jacobian_over_R'))), float(max(numbers(ellipse, 'jacobian_over_R')))]
    return (fig, rows, checks)

def branch_series(mu, spec):
    coeff = spec['geometry']['coefficients']
    base = {1: 1.0, **{-j: c for j, c in enumerate(coeff)}}
    gamma, delta = ((1 - 1j * mu) / 2, (1 + 1j * mu) / 2)
    keys = set(base) | {-j for j in base}
    return {j: gamma * base.get(j, 0) + delta * np.conj(base.get(-j, 0)) for j in keys}

def eval_series(series, zeta):
    return sum((a * np.asarray(zeta, dtype=complex) ** j for j, a in series.items()))

def branch_geometry(spec):
    mus, *_ = material()
    rows, facts = ([], [])
    theta = np.linspace(0, 2 * np.pi, 721)
    for k, mu in enumerate(mus, 1):
        ser = branch_series(mu, spec)
        lo, hi = (min(ser), max(ser))
        derivative = np.array([j * ser.get(j, 0) for j in range(lo, hi + 1)], complex)
        roots = np.roots(derivative[::-1])
        critical = roots[abs(roots) < 1]
        rcrit = float(max(abs(critical), default=0.0))
        eta = cauchy_radius(ser)
        wall = eval_series(ser, np.exp(1j * theta))
        points = eval_series(ser, eta * np.exp(2j * np.pi * (np.arange(24) + 0.31) / 24))
        curve = MplPath(np.c_[wall.real, wall.imag])
        assert all(curve.contains_points(np.c_[points.real, points.imag]))
        root_counts = []
        for w0 in points:
            polynomial = np.array([ser.get(j, 0) for j in range(lo, hi + 1)], complex)
            polynomial[-lo] -= w0
            preimages = np.roots(polynomial[::-1])
            root_counts.append(int(np.count_nonzero(abs(preimages) < 1)))
        assert set(root_counts) == {1 - lo}
        facts.append({'k': k, 'root': [float(mu.real), float(mu.imag)], 'fold_radius': rcrit, 'eta': eta, 'interior_preimage_count': root_counts[0]})
        for kind, values in (('boundary', wall), ('test', points), ('critical', critical)):
            for i, z in enumerate(values):
                rows.append({'branch': k, 'kind': kind, 'point': i, 'real': z.real, 'imag': z.imag, 'eta': eta, 'fold_radius': rcrit})
    recorded = read_csv('constraint_conditioning_law.csv')
    anchor = next((r for r in recorded if r['material'] == 'F1'))
    assert np.allclose(sorted((f['eta'] for f in facts)), [float(anchor['eta_min']), float(anchor['eta_max'])], atol=1e-11, rtol=0)
    return (rows, {'branches': facts, 'displayed_points_per_branch': 24, 'point_offset': 0.31, 'point_role': 'illustration of the sampling rule'})

def draw_sampling(spec):
    rows, checks = branch_geometry(spec)
    fig = plt.figure(figsize=(145 * MM, 126 * MM))
    angles = 2 * np.pi * (np.arange(24) + 0.31) / 24
    theta = np.linspace(0, 2 * np.pi, 721)
    for k, col, bottom in ((1, TEAL, 0.6), (2, RUST, 0.18)):
        left = fig.add_axes([0.1, bottom, 0.33, 0.3])
        right = fig.add_axes([0.62, bottom, 0.33, 0.3])
        fact = checks['branches'][k - 1]
        rcrit, eta = (fact['fold_radius'], fact['eta'])
        series = branch_series(complex(*fact['root']), spec)
        left.add_patch(Circle((0, 0), 1, fc='#F0F5F5', ec='#26313A', lw=1.2))
        left.add_patch(Circle((0, 0), rcrit, fc='#ECEDEF', ec=GREY, lw=0.7, ls=':'))
        critical = [r for r in rows if r['branch'] == k and r['kind'] == 'critical']
        left.plot(numbers(critical, 'real'), numbers(critical, 'imag'), 'x', color=GREY, ms=3.5, mew=0.7)
        left.plot(eta * np.cos(theta), eta * np.sin(theta), color=col, lw=1.0)
        left.plot(eta * np.cos(angles), eta * np.sin(angles), 'o', color=col, ms=2.5)
        left.plot(eta * np.cos(angles[3]), eta * np.sin(angles[3]), 's', color=PURPLE, mfc='white', ms=5, mew=1.15, zorder=6)
        ray_angle = -0.24 * np.pi
        left.plot([0, rcrit * np.cos(ray_angle)], [0, rcrit * np.sin(ray_angle)], color=GREY, lw=0.7, ls=':')
        left.text(0.14, -0.37, f'$r_{k}$', color=GREY, fontsize=9)
        for radius in np.linspace(eta, 1.0, 4):
            mapped = eval_series(series, radius * np.exp(1j * theta))
            right.plot(mapped.real, mapped.imag, color=col, lw=0.45, alpha=0.5)
        for angle in angles[::2]:
            radii = np.linspace(eta, 1.0, 50)
            mapped = eval_series(series, radii * np.exp(1j * angle))
            right.plot(mapped.real, mapped.imag, color=col, lw=0.5, alpha=0.65)
        boundary = [r for r in rows if r['branch'] == k and r['kind'] == 'boundary']
        pts = [r for r in rows if r['branch'] == k and r['kind'] == 'test']
        bx, by = (numbers(boundary, 'real'), numbers(boundary, 'imag'))
        right.fill(bx, by, color='#F4F6F6', zorder=0)
        right.plot(bx, by, color='#26313A', lw=1.2)
        right.plot(numbers(pts, 'real'), numbers(pts, 'imag'), 'o', color=col, ms=2.5)
        right.plot(float(pts[3]['real']), float(pts[3]['imag']), 's', color=PURPLE, mfc='white', ms=5, mew=1.15, zorder=6)
        right.text(0.05, 0.94, f'$\\Gamma_{k}$', transform=right.transAxes, fontsize=10, va='top')
        for ax in (left, right):
            ax.set(aspect='equal', xlim=(-1.35, 1.35), ylim=(-1.35, 1.35), xticks=[-1, 0, 1], yticks=[-1, 0, 1])
        left.set(xlabel='$\\mathrm{Re}\\,\\sigma$', ylabel='$\\mathrm{Im}\\,\\sigma$')
        right.set(xlabel=f'$\\mathrm{{Re}}\\,z_{k}/R$', ylabel=f'$\\mathrm{{Im}}\\,z_{k}/R$')
        ypos = bottom + 0.325
        for xpos, letter, title in ((0.035, chr(95 + 2 * k), f'Parameter ring, branch {k}'), (0.55, chr(96 + 2 * k), f'Characteristic image, branch {k}')):
            fig.text(xpos, ypos, letter, fontweight='bold', fontsize=10)
            fig.text(xpos + 0.04, ypos, title, fontsize=8.5)
        fig.add_artist(ConnectionPatch((1.1, 0.68), (-0.19, 0.68), 'axes fraction', 'axes fraction', axesA=left, axesB=right, arrowstyle='-|>', color=col, lw=1.1, mutation_scale=9))
        fig.text(0.508, bottom + 0.227, f'$z_{k}(\\sigma)$', ha='center', color=col, fontsize=10)
        fig.text(0.51, bottom + 0.09, f'$r_{k}={rcrit:.3f}$' + '\n' + f'$\\eta_{k}={eta:.3f}$', ha='center', color=col, fontsize=8, linespacing=1.6)
    fig.legend(handles=[Line2D([], [], color='#26313A', label=TEXT['wall']), Line2D([], [], color=GREY, ls=':', label=TEXT['fold']), Line2D([], [], color=GREY, marker='x', ls='none', label='Derivative zeros'), Line2D([], [], color=PURPLE, marker='s', mfc='white', ls='none', label='Corresponding point')], loc='lower center', bbox_to_anchor=(0.51, 0.012), ncol=4, columnspacing=0.8, handlelength=1.35, fontsize=7.5)
    checks['mapped_grid'] = 'four radii from each branch test ring to its boundary; twelve radial connections'
    return (fig, rows, checks)

def draw_constraints(spec):
    rows = read_csv('constraint_conditioning_law.csv')
    fig, axes = plt.subplots(1, 2, figsize=(145 * MM, 83 * MM))
    fig.subplots_adjust(left=0.115, right=0.98, bottom=0.32, top=0.86, wspace=0.39)
    for key, ax, title in zip(('heldout_residual', 'operator_increment'), axes, ('heldout', 'increment')):
        for mat in MATERIALS:
            group = [r for r in rows if r['material'] == mat and r[key] != '']
            ax.semilogy(numbers(group, 'n_t'), numbers(group, key), color=COLORS[mat], marker=MARKERS[mat], mfc='white', mew=0.9)
        ax.set(xlabel=TEXT['truncation'], xticks=[8, 16, 24, 32, 64], xlim=(5, 67))
        ax.set_ylabel(TEXT['residual_axis'] if key == 'heldout_residual' else TEXT['increment_axis'])
        grid(ax)
        panel(ax, 'a' if ax is axes[0] else 'b', TEXT[title])
    axes[0].set(ylim=(8e-15, 0.001), yticks=[1e-13, 1e-10, 1e-07, 0.0001])
    axes[1].set(ylim=(4e-16, 0.0001), yticks=[1e-14, 1e-11, 1e-08, 1e-05])
    for ax, key, label, ytext in ((axes[0], 'heldout_residual', '$1.2\\times10^{-13}$', 1e-09), (axes[1], 'operator_increment', '$1.5\\times10^{-13}$', 1e-09)):
        maximum = max((float(r[key]) for r in rows if int(r['n_t']) == 64))
        ax.annotate(label, xy=(64, maximum), xytext=(40, ytext), fontsize=8, arrowprops={'arrowstyle': '-', 'lw': 0.7, 'color': GREY})
    legend_materials(fig)
    return (fig, rows, {'source_rows': len(rows), 'residual_points': len(rows), 'increment_points': sum((r['operator_increment'] != '' for r in rows)), 'residual_definition': 'Cauchy value minus recovered infinity value, scaled by prescribed circular potential amplitude', 'missing_increment_rule': 'n_c=8 has no preceding refinement; no zero is inserted'})

def draw_kernels(spec):
    rows = read_csv('closed_kernel_ladder.csv')
    fig, axes = plt.subplots(1, 2, figsize=(145 * MM, 88 * MM))
    fig.subplots_adjust(left=0.115, right=0.98, bottom=0.34, top=0.86, wspace=0.39)
    ax = axes[0]
    ts = sorted({float(r['t']) for r in rows})
    for key, color, marker, name in (('rel0', GREY, 'o', 'base'), ('rel1', RUST, 's', 'first'), ('rel2', TEAL, 'D', 'second')):
        low, high = ([], [])
        for t in ts:
            values = [float(r[key]) for r in rows if float(r['rho_out']) == 1 and float(r['t']) == t]
            low.append(min(values))
            high.append(max(values))
        ax.fill_between(ts, low, high, color=color, alpha=0.11, linewidth=0)
        ax.loglog(ts, high, color=color, marker=marker, mfc='white', label=TEXT[name])
    ax.set(xlim=(0.22, 1.12), ylim=(7e-07, 0.03), xticks=ts, xticklabels=['0.25', '0.5', '1.0'], xlabel=TEXT['strength'], ylabel=TEXT['relative'])
    ax.set_yticks([1e-06, 1e-05, 0.0001, 0.001, 0.01])
    ax.text(0.045, 0.045, '$\\rho=1$', transform=ax.transAxes, fontsize=9)
    grid(ax)
    panel(ax, 'a', TEXT['orders'])
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=1, bbox_to_anchor=(0.29, 0.015), handlelength=2, labelspacing=0.5)
    ax = axes[1]
    for mat in MATERIALS:
        group = [r for r in rows if r['material'] == mat and float(r['t']) == 1]
        ax.semilogy(numbers(group, 'rho_out'), numbers(group, 'rel2'), color=COLORS[mat], marker=MARKERS[mat], mfc='white', mew=0.9)
    ax.axhline(0.001, color=GREY, lw=0.8, ls='--')
    ax.text(1.5, 0.00135, TEXT['target'], ha='center', fontsize=8)
    ax.set(xlim=(0.92, 2.07), ylim=(2e-09, 0.007), xticks=[1, 1.5, 2], xlabel=TEXT['rho'], ylabel=TEXT['relative'])
    ax.set_yticks([1e-08, 1e-06, 0.0001, 0.01])
    ax.text(0.06, 0.07, TEXT['second_at_one'], transform=ax.transAxes, fontsize=8)
    grid(ax)
    panel(ax, 'b', TEXT['placement'])
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], mfc='white', label=TEXT[m]) for m in MATERIALS]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.76, 0.015), ncol=2, columnspacing=0.9, handlelength=1.5, labelspacing=0.5, fontsize=7.8)
    return (fig, rows, {'source_rows': 45, 'panel_a': '15 wall-ray rows; all five materials contribute to each min-max envelope', 'panel_b': '15 full-strength rows; individual materials at all three interface radii', 'other_20_rows': 'intermediate strengths at wider interfaces retained in source CSV; full-strength endpoints answer the placement question', 'envelopes': 'deterministic extrema, not confidence intervals'})

def draw_roots(spec):
    atlas = read_csv('atlas_family_box.csv')
    switch = read_csv('degenerate_switch.csv')
    ratios = sorted({float(r['ratio']) for r in atlas})
    lower, upper, lambdas, spread = ([], [], [], [])
    aggregate = []
    for ratio in ratios:
        group = [r for r in atlas if float(r['ratio']) == ratio]
        km = numbers(group, 'cond_dirichlet_2x2')
        kl = numbers(group, 'cond_lambda')
        lower.append(km.min())
        upper.append(km.max())
        lambdas.append(kl.mean())
        spread.append(np.ptp(kl))
        aggregate.append({'ratio': ratio, 'cond_M_min': km.min(), 'cond_M_max': km.max(), 'cond_Lambda_mean': kl.mean(), 'cond_Lambda_spread': np.ptp(kl)})
    assert len(atlas) == 10101 and len(ratios) == 111
    assert max(spread) < 1e-09
    fig, axes = plt.subplots(1, 2, figsize=(145 * MM, 79 * MM))
    fig.subplots_adjust(left=0.1, right=0.985, bottom=0.24, top=0.8, wspace=0.37)
    ax = axes[0]
    ax.fill_between(ratios, lower, upper, color=GREY, alpha=0.16)
    ax.plot(ratios, lower, color=GREY, lw=0.85)
    ax.plot(ratios, upper, color=GREY, lw=0.85)
    ax.plot(ratios, lambdas, color=TEAL, lw=1.8)
    ax.text(2.5, 2.67, '$\\kappa_2(M)$' + '\n' + TEXT['dip_range'], fontsize=9)
    ax.text(2.8, 1.67, '$\\kappa_2(\\Lambda)$', fontsize=10, color=TEAL)
    ax.set(xlim=(1.2, 4.05), ylim=(1.5, 3.8), xticks=[1.25, 2, 3, 4], yticks=[1.5, 2, 2.5, 3, 3.5], xlabel=TEXT['ratio'], ylabel=TEXT['condition'])
    grid(ax)
    panel(ax, 'a', TEXT['box'])
    ax = axes[1]
    for ratio, marker in ((1.25, 'o'), (4.0, 's')):
        group = sorted((r for r in switch if r['record'] == 'walk' and float(r['ratio']) == ratio), key=lambda r: float(r['mu_gap']))
        x = numbers(group, 'mu_gap')
        assert np.all(x > 0)
        for key, color in (('err_2root', RUST), ('err_conf', TEAL)):
            y = numbers(group, key)
            assert np.all(y > 0)
            ax.loglog(x, y, color=color, marker=marker, mfc='white', ms=2.4, lw=0.85, mew=0.7)
    ax.text(2e-07, 0.05, TEXT['confluent'], color=TEAL, fontsize=8, ha='left')
    ax.text(8e-05, 6e-08, TEXT['two_root'], color=RUST, fontsize=8, ha='left')
    ax.set(xlim=(4e-09, 3), ylim=(5e-17, 2), xticks=[1e-08, 0.0001, 1], yticks=[1e-16, 1e-12, 1e-08, 0.0001, 1], xlabel=TEXT['gap'], ylabel=TEXT['relative'])
    grid(ax)
    panel(ax, 'b', TEXT['switch'])
    fig.legend(handles=[Line2D([], [], color=GREY, marker='o', mfc='white', ls='none', label='$E_h/E_v=1.25$'), Line2D([], [], color=GREY, marker='s', mfc='white', ls='none', label='$E_h/E_v=4$')], loc='lower center', bbox_to_anchor=(0.75, 0.025), ncol=2, columnspacing=0.9)
    return (fig, aggregate, {'atlas_rows': len(atlas), 'dip_values_per_ratio': 91, 'Lambda_max_dip_spread': max(spread), 'switch_source_rows': len(switch), 'plotted_walk_rows': 62, 'switch_exclusion': '12 endpoint, box, switch-estimate and coefficient records are diagnostics, not walk samples; zero-gap diagnostics cannot appear on a log axis'})
DRAWERS = {'domain-coupling': draw_domain, 'density-invariance': draw_density, 'analyticity-sampling': draw_sampling, 'constraint-convergence': draw_constraints, 'kernel-validity': draw_kernels, 'root-stability': draw_roots}

def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
