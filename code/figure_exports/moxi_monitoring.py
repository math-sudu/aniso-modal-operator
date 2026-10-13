from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
import origin_style as fs
DATA = ROOT / 'results/data/field/moxi675'
OUTPUT = ROOT / 'figures/moxi-monitoring/moxi_monitoring.pdf'
ACCEPTED_STATUS = {'reported_numeric', 'explicit_decimal_with_internal_whitespace'}
CHANNELS = (('deformation_01', 'Line 1', fs.BLACK, 'o', '-'), ('deformation_02', 'Line 2', fs.RED, 's', '--'), ('deformation_03', 'Line 3', fs.BLUE, '^', '-.'), ('deformation_04', 'Crown', '#222222', 'D', '-'))

def load_data():
    with (DATA / 'geometry_contours.csv').open(encoding='utf-8-sig', newline='') as stream:
        contour_rows = [row for row in csv.DictReader(stream) if row['contour'] == 'initial_shotcrete_intrados']
    contour_rows.sort(key=lambda row: int(row['point_index']))
    intrados = np.array([[float(row['x_m']), float(row['y_m'])] for row in contour_rows])
    geometry = json.loads((DATA / 'geometry_reconstruction.json').read_text(encoding='utf-8'))
    with (DATA / 'monitoring_deformation.csv').open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    series = {}
    summaries = []
    for channel, label, *_ in CHANNELS:
        rows = sorted((row for row in records if row['channel_id'] == channel), key=lambda row: int(row['sample_index']))
        retained = [row['value_status'] in ACCEPTED_STATUS and bool(row['literal_numeric_value']) for row in rows]
        accepted = [row for row, keep in zip(rows, retained) if keep]
        first_value = float(accepted[0]['literal_numeric_value'])
        last_value = float(accepted[-1]['literal_numeric_value'])
        day = np.array([float(row['source_day_text']) for row in rows])
        cumulative = np.array([float(row['literal_numeric_value']) if keep else np.nan for row, keep in zip(rows, retained)])
        series[channel] = (day, cumulative)
        summaries.append({'channel': channel, 'label': label, 'first': {'reported_day': int(day[retained][0]), 'table_value_mm': first_value}, 'last': {'reported_day': int(day[retained][-1]), 'table_value_mm': last_value}, 'total_change_mm': round(last_value - first_value, 6), 'used_count': len(accepted), 'omitted_count': len(rows) - len(accepted)})
    return (intrados, geometry, series, summaries)

def color_for(color, grayscale):
    if not grayscale:
        return color
    red, green, blue = matplotlib.colors.to_rgb(color)
    gray = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return (gray, gray, gray)

def section_panel(ax, intrados, geometry, grayscale):
    crown_y = geometry['geometric_interpretation']['initial_intrados_crown_y_m']
    ax.plot(intrados[:, 0], crown_y - intrados[:, 1], color='#222222', lw=1.15)
    text_positions = [(0, 2.02), (0, 5.52), (0, 9.02)]
    for gauge, spec, text_xy in zip(geometry['gauge_coordinates'], CHANNELS, text_positions):
        _, label, color, marker, linestyle = spec
        color = color_for(color, grayscale)
        left, right = (gauge['left_endpoint_m'], gauge['right_endpoint_m'])
        depth = gauge['depth_below_crown_m']
        ax.plot([left[0], right[0]], [depth] * 2, color=color, marker=marker, markersize=3.5, markerfacecolor='white', markeredgewidth=0.8, lw=0.8, ls=linestyle)
        ax.text(*text_xy, f'{label[-1]}: {depth:.1f} m', color=color, ha='center', va='center', fontsize=8)
    ax.plot(0, 0, 'D', color='#222222', markersize=3.5, markerfacecolor='white', mew=0.8)
    ax.annotate('Crown', xy=(0, 0), xytext=(1.0, -0.45), ha='left', va='center', fontsize=8, arrowprops={'arrowstyle': '-', 'color': '#555555', 'lw': 0.6})
    ax.set(xlim=(-7.2, 7.2), ylim=(11.5, -1.0), aspect='equal', xticks=[-6, 0, 6], yticks=[0, 2, 4, 6, 8, 10], xlabel='Horizontal position (m)', ylabel='Depth below crown (m)')

def event_markers(ax):
    for day, label_y in ((4, 0.99), (8, 0.92), (25, 0.99)):
        ax.axvline(day, color='#a5a5a5', lw=0.65, ls=(0, (3, 3)), zorder=0)
        ax.text(day, label_y, f'D{day}', transform=ax.get_xaxis_transform(), ha='center', va='top', color='#555555', fontsize=8, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.2})

def draw_figure(intrados, geometry, series, grayscale=False):
    width_mm, height_mm = (137.069, 108.0)
    fig = fs.figure(width_mm, height_mm)
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 8, 'axes.linewidth': 0.65, 'xtick.major.width': 0.65, 'ytick.major.width': 0.65, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})
    axes = [fig.add_axes([left / width_mm, bottom / height_mm, width / width_mm, height / height_mm]) for left, bottom, width, height in ((10, 34, 46, 46), (70, 65, 64.5, 34), (70, 16, 64.5, 34))]
    for ax in axes:
        fs.style_axes(ax)
        ax.tick_params(pad=2)
    section_panel(axes[0], intrados, geometry, grayscale)
    for channel, label, color, marker, linestyle in CHANNELS:
        ax = axes[2] if channel == 'deformation_04' else axes[1]
        day, cumulative = series[channel]
        ax.plot(day, cumulative, label=label, color=color_for(color, grayscale), marker=marker, markersize=3.0, markerfacecolor='white', markeredgewidth=0.65, lw=0.9, ls=linestyle, zorder=2)
    axes[1].set(ylim=(-12, 375), yticks=[0, 100, 200, 300], ylabel='Cumulative convergence (mm)')
    axes[2].set(ylim=(-2, 86), yticks=[0, 20, 40, 60, 80], ylabel='Cumulative settlement (mm)')
    for ax in axes[1:]:
        ax.set(xlim=(-1, 40), xticks=[0, 10, 20, 30, 40], xlabel='Reported monitoring day')
        event_markers(ax)
    axes[1].legend(loc='lower right', frameon=False, handlelength=2.1, handletextpad=0.45, borderpad=0.1, labelspacing=0.35)
    for left, top, title in ((10, 84, '(a) Section and gauges'), (70, 103, '(b) Horizontal convergence'), (70, 54, '(c) Crown settlement')):
        fig.text(left / width_mm, top / height_mm, title, fontsize=8.5, ha='left', va='center')
    fig.text(0.5, 2.2 / height_mm, 'D4: lower bench   |   D8: support ring closure   |   D25: invert concrete', fontsize=8, ha='center', va='center', color='#555555')
    return fig

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', type=Path, help='Optional temporary color PNG path.')
    parser.add_argument('--grayscale-preview', type=Path, help='Optional temporary grayscale PNG path.')
    args = parser.parse_args()
    intrados, geometry, series, summaries = load_data()
    fig = draw_figure(intrados, geometry, series)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fs.save(fig, OUTPUT, size_mm=(137.069, 108.0), preview_path=args.preview)
    plt.close(fig)
    if args.grayscale_preview:
        fig = draw_figure(intrados, geometry, series, grayscale=True)
        args.grayscale_preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.grayscale_preview, dpi=300)
        plt.close(fig)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
