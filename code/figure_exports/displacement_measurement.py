from __future__ import annotations
from pathlib import Path
import sys
from matplotlib import colors, patheffects
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle, PathPatch, Rectangle, Wedge
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
ROOT = Path(__file__).resolve().parents[2]
import origin_style as fs
from engineering_figures import axes_mm
INK, BLUE, GREY = (fs.BLACK, fs.BLUE, '#70787D')
RADIUS = 12.7
FIELDS = ('u_mid_mm', 'dtn_orthotropic_mm', 'finite_orthotropic_mm', 'finite_isotropic_mm')
NATIVE = ROOT / 'figures/displacement-measurement/assets/fatima2017_measured_quadrant.npz'

def extract_native():
    """Retain the measured half of source Figure 4(a) and its printed palette."""
    import fitz
    with fitz.open(ROOT / 'references/papers/fatima2017/paper.pdf') as document:
        pix = fitz.Pixmap(document, document[8].get_images(full=True)[0][0])
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    assert rgb.shape == (254, 512, 3)
    NATIVE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NATIVE, rgb=rgb[20:211, 51:242], palette=np.median(rgb[21:211, 450:466], axis=1), crop_xyxy=[51, 20, 242, 211], origin_xy=[241, 210], radius_px=95.25)

def measured_raster(cmap):
    """Show source color regions with printed contour linework omitted."""
    with np.load(NATIVE) as native:
        rgb, palette = (native['rgb'], native['palette'])
    distance, index = cKDTree(palette).query(rgb.reshape(-1, 3))
    bands = np.clip(np.rint((1 - index / 189) * 10).astype(int), 0, 9).reshape(rgb.shape[:2])
    decoded = distance.reshape(rgb.shape[:2]) < 12
    white = np.all(rgb >= 235, axis=2)
    strokes = ~decoded & ~white
    _, nearest = distance_transform_edt(~decoded, return_indices=True)
    displayed = bands.copy()
    displayed[strokes] = bands[tuple(nearest[:, strokes])]
    result = cmap(displayed)
    result[white, :3] = rgb[white] / 255
    np.testing.assert_array_equal(displayed[decoded], bands[decoded])
    return (np.flip(result.transpose(1, 0, 2), axis=(0, 1)), {'decoded_fill_pixels_preserved': int(decoded.sum()), 'printed_line_pixels_restored': int(strokes.sum()), 'white_pixels_preserved': int(white.sum()), 'display_restoration': 'Nearest decoded fill over printed strokes only; no field smoothing'})

def draw_displacement(width, height, labels):
    frame = pd.read_csv(ROOT / 'results/data/derived/fatima2017_fields_refined.csv')
    observations = pd.read_csv(ROOT / 'results/data/derived/fatima2017_observations.csv')
    valid = observations[observations.valid == 1]
    assert len(frame) == len(valid) == 1278
    for name in ('pixel_x', 'pixel_y', 'transverse_mm', 'axial_mm', 'u_mid_mm'):
        np.testing.assert_allclose(frame[name], valid[name], rtol=0, atol=1e-13)
    x, y = (frame.transverse_mm.to_numpy() / RADIUS, frame.axial_mm.to_numpy() / RADIUS)
    np.testing.assert_allclose(x, (210 - frame.pixel_y) / 95.25, atol=1e-14)
    np.testing.assert_allclose(y, (241 - frame.pixel_x) / 95.25, atol=1e-14)
    radius = np.hypot(x, y)
    assert np.all((radius >= 1.1) & (radius <= 1.98))
    levels = np.linspace(0, 12.7, 11)
    spectrum = colors.LinearSegmentedColormap.from_list('spectrum', ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000'])
    cmap = colors.ListedColormap(spectrum(np.linspace(0, 1, 10)))
    cmap.set_under(cmap(0))
    norm = colors.BoundaryNorm(levels, cmap.N)
    native, native_display = measured_raster(cmap)
    dense = pd.read_csv(ROOT / 'results/data/derived/displacement_contours.csv')
    nr, nt = (dense.radius_R.nunique(), dense.theta_rad.nunique())
    gx = dense.x_R.to_numpy().reshape(nr, nt)
    gy = dense.y_R.to_numpy().reshape(nr, nt)
    fig = fs.figure(width, height)
    canvas = fig.add_axes([0, 0, 1, 1], zorder=5)
    canvas.set(xlim=(0, width), ylim=(0, height), aspect='equal')
    canvas.set_axis_off()
    serial = 0

    def tag(artist, role):
        nonlocal serial
        serial += 1
        artist.set_gid(f'{role}-{serial:03d}')
        return artist

    def text(xp, yp, value, *, size=8, color=INK, **kwargs):
        return tag(canvas.text(xp, yp, labels.get(value, value), fontsize=size, color=color, va=kwargs.pop('va', 'center'), zorder=30, **kwargs), 'label')

    def line(points, *, role, color=INK, **kwargs):
        artist, = canvas.plot(*np.asarray(points).T, color=color, ls=kwargs.pop('ls', '-'), **kwargs)
        return tag(artist, role)

    def arrow(start, end, *, role, color=INK, style='-|>', **kwargs):
        return tag(canvas.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle=style, color=color, lw=0.7, mutation_scale=7, **kwargs)), role)

    def leader(points, role):
        obj = line(points, role=role, color=GREY, lw=0.6, zorder=12)
        obj.set_path_effects([patheffects.Stroke(linewidth=1.5, foreground='white'), patheffects.Normal()])
    centre, scale = (np.array([20.0, 109.0]), 0.2)
    plate_width, plate_length = (76.2 * scale, 279.4 * scale)
    lower = centre - np.array([plate_width, plate_length]) / 2
    tag(canvas.add_patch(Rectangle(lower, plate_width, plate_length, fc='#E9E4DC', ec=INK, lw=0.85)), 'plate')
    tag(canvas.add_patch(Circle(centre, RADIUS * scale, fc='white', ec=INK, lw=0.75, zorder=9)), 'physical-hole')
    tag(canvas.add_patch(Wedge(centre, 1.98 * RADIUS * scale, 0, 90, width=0.88 * RADIUS * scale, fc='#C5DCEE', ec=BLUE, lw=0.65, zorder=10)), 'sampled-quarter-annulus')
    roi = 2 * RADIUS * scale
    tag(canvas.add_patch(Rectangle(centre, roi, roi, fill=False, ec=BLUE, lw=0.8, zorder=11)), 'quadrant-locator')
    text(2, 148, 'Loaded graphite/epoxy plate', size=8.5)
    for sign in [-1, 1]:
        end = centre + [0, sign * plate_length / 2]
        arrow(end, end + [0, sign * (6.5 if sign > 0 else 4)], role='tensile-resultant', color=BLUE)
    text(24, 141, '4.45 kN', size=8, color=BLUE)
    for yp in [lower[1], lower[1] + plate_length]:
        line([(6, yp), (lower[0] - 0.5, yp)], role='length-extension', color=GREY, lw=0.5)
    arrow((7, lower[1]), (7, lower[1] + plate_length), role='plate-length', style='<->', color=GREY)
    text(3.5, centre[1], '279.4 mm', size=7.5, rotation=90, ha='center')
    arrow((lower[0], 76), (lower[0] + plate_width, 76), role='plate-width', style='<->', color=GREY)
    text(20, 72.7, '76.2 mm', size=7.5, ha='center')
    text(2, 67.8, 'Thickness: 5.28 mm', size=7.5)
    leader([centre + [roi, roi], [68, 136], [72, 136]], 'quadrant-upper-enlargement')
    leader([centre + [roi, 0], [68, 86], [72, 86]], 'quadrant-lower-enlargement')
    text(43, 132.5, 'Sampled', size=7.5, ha='center', color=BLUE)
    text(43, 128.5, 'quadrant', size=7.5, ha='center', color=BLUE)
    text(36, 112, '$R=12.7$ mm', size=7.5)
    leader([centre + [1.6, -1.8], [34, 112]], 'hole-radius-label')
    theta = np.linspace(0, np.pi / 2, 240)
    arc = np.c_[np.cos(theta), np.sin(theta)]
    region = np.vstack((1.98 * arc, 1.1 * arc[::-1], 1.98 * arc[:1]))
    clip_path = MplPath(region)
    artists = []

    def field(key, box, title, *, measured=False):
        xp, yp, side = box
        ax = axes_mm(fig, xp, yp, side, side)
        fs.style_axes(ax, minor=False)
        ax.tick_params(labelsize=7, pad=2, length=2.5, width=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)
        tag(ax.add_patch(Wedge((0, 0), 1.1, 0, 90, width=0.1, fc='#E3E5E7', ec='none')), 'excluded-hole-band')
        values = 1000 * frame[key].to_numpy()
        assert np.all((values >= levels[0]) & (values <= levels[-1]))
        if measured:
            artist = ax.imshow(native, origin='lower', interpolation='none', extent=(-0.5 / 95.25, 190.5 / 95.25, -0.5 / 95.25, 190.5 / 95.25), gid='native-measured-color-contours')
        else:
            grid = 1000 * dense[key].to_numpy().reshape(nr, nt)
            assert grid.max() <= levels[-1]
            artist = ax.contourf(gx, gy, grid, levels=levels, cmap=cmap, norm=norm, antialiased=False, extend='min')
            artist.set_edgecolor('face')
            tag(artist, f'computed-contours-{key}')
        artist.set_clip_path(PathPatch(clip_path, transform=ax.transData))
        ax.plot(*arc.T, color=INK, lw=0.8, ls='-', gid=f'hole-{key}')
        ax.plot(*(1.1 * arc).T, color=GREY, lw=0.65, ls=(0, (3, 2)), gid=f'exclusion-{key}')
        ax.plot(*(1.98 * arc).T, color=BLUE if measured else GREY, lw=0.55, ls=(0, (2, 2)), gid=f'extraction-edge-{key}')
        ax.set(aspect='equal', xlim=(0, 2), ylim=(0, 2), xticks=[0, 1, 2], yticks=[0, 1, 2])
        ax.set_xlabel('$x/R$', fontsize=8, labelpad=1.5)
        ax.set_ylabel('$y/R$', fontsize=8, labelpad=1.5)
        text(xp + side / 2, yp + side + 4.8, title, size=8, ha='center')
        artists.append(artist)
        return ax
    field('u_mid_mm', (72, 86, 50), 'Measured DIC', measured=True)
    text(97, 146, 'Published colour bands', size=7.5, ha='center', color=GREY)
    text(77, 91, 'Hole', size=8)
    text(84, 99, 'Excluded', size=7.3, ha='center')
    text(84, 95.5, '$0.1R$', size=7.3, ha='center')
    leader([(87, 101), (91.3, 105.3)], 'exclusion-band-label')
    text(72, 76, '$x$: transverse; $y$: axial', size=7.5)
    text(2, 60.7, 'Predicted displacement in the same region', size=8.5)
    for key, xp, title in zip(FIELDS[1:], [10, 55, 100], ['Infinite DtN–FEM', 'Finite orthotropic', 'Finite isotropic control']):
        field(key, (xp, 16.5, 32), title)
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=axes_mm(fig, 33, 3.4, 71, 2.6), orientation='horizontal', ticks=[0, 2.54, 5.08, 7.62, 10.16, 12.7])
    bar.ax.set_label('<colorbar>')
    bar.ax.tick_params(labelsize=7, pad=1, length=2)
    bar.ax.set_xticklabels(['0', '2.54', '5.08', '7.62', '10.16', '12.7'])
    for artist in artists[1:]:
        artist.colorbar = bar
    text(width / 2, 8.3, 'Axial displacement (μm)', size=8, ha='center')
    return (fig, {'source': 'fatima2017_fields_refined.csv', 'sample_count': len(frame), 'candidate_count': len(observations), 'masked_candidates': len(observations) - len(frame), 'sampling_mm': 0.4, 'extraction_radius_R': [1.1, 1.98], 'plotting': 'Recolored native measured contours and directly evaluated continuous model fields', 'native_source': 'Fatima and Rowlands (2017), Figure 4(a), measured left half', 'native_pixels': [191, 191], 'measured_display': native_display, 'computed_grid': [nr, nt], 'computed_ranges_um': {key: [float(dense[key].min() * 1000), float(dense[key].max() * 1000)] for key in FIELDS[1:]}, 'under_zero_color': 'Lowest blue; saved numerical values are retained', 'field_clipping': 'Annular quadrant, 1.1 <= r/R <= 1.98', 'palette': 'Ten blue-cyan-green-yellow-red spectrum bands', 'plate_dimensions_mm': [76.2, 279.4, 5.28], 'hole_radius_mm': RADIUS, 'plate_scale_mm_per_mm': scale, 'load_kN': 4.45, 'field_box_mm': {'measured': [72, 86, 50], 'predictions': [[xp, 16.5, 32] for xp in [10, 55, 100]]}, 'scale_um': [0, 12.7], 'color_band_edges_um': levels.tolist(), 'normalization': 'none; measured values are published band midpoints', 'coordinate_mapping': 'x/R=(210-pixel_y)/95.25; y/R=(241-pixel_x)/95.25', 'field_ranges_um': {key: [float(frame[key].min() * 1000), float(frame[key].max() * 1000)] for key in FIELDS}})
