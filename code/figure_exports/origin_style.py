from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import matplotlib as mpl
from matplotlib import font_manager
from cycler import cycler
STYLE_PATH = Path(__file__).with_name('origin-style.json')
STYLE = json.loads(STYLE_PATH.read_text(encoding='utf-8'))
BLACK, RED, BLUE = STYLE['colors']

def apply_style() -> None:
    """Install the sole default; a missing Times font is an actionable error."""
    try:
        font_manager.findfont(STYLE['font_family'], fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError('Times New Roman is required; install the font before rendering.') from exc
    mpl.rcParams.update(STYLE['rc'])
    mpl.rcParams['axes.prop_cycle'] = cycler(color=STYLE['colors']) + cycler(linestyle=STYLE['linestyles'][:3])

def figure(width_mm: float, height_mm: float, **kwargs):
    """Create a white canvas in millimetres; add content-specific axes yourself."""
    from matplotlib import pyplot as plt
    if not all((math.isfinite(v) and v > 0 for v in (width_mm, height_mm))):
        raise ValueError('Figure dimensions must be positive.')
    apply_style()
    return plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), **kwargs)

def style_axes(ax, *, minor: bool=True, categorical: str=''):
    """Box numeric axes; preserve log locators and omit categorical minor ticks.

    Photographs and schematics normally use ax.set_axis_off() instead. Fields
    keep physical coordinates, aspect, scientific colormap and a labelled bar.
    """
    from matplotlib.ticker import AutoMinorLocator, NullLocator
    for spine in ax.spines.values():
        spine.set(visible=True, linewidth=STYLE['rc']['axes.linewidth'], edgecolor=BLACK)
    ax.set_facecolor('white')
    ax.grid(False, which='both')
    ax.tick_params(which='major', direction='in', top=True, right=True, length=STYLE['rc']['xtick.major.size'], width=STYLE['rc']['xtick.major.width'], colors=BLACK, pad=3)
    ax.tick_params(which='minor', direction='in', top=True, right=True, length=STYLE['rc']['xtick.minor.size'], width=STYLE['rc']['xtick.minor.width'], colors=BLACK)
    for name, axis in (('x', ax.xaxis), ('y', ax.yaxis)):
        if not minor or name in categorical:
            axis.set_minor_locator(NullLocator())
        elif axis.get_scale() == 'linear':
            axis.set_minor_locator(AutoMinorLocator(2))
    return ax

def series(index: int, *, role: str, replicate: int=0) -> dict:
    """Return explicit experiment/model encoding, independent of plot order.

    This only chooses artist properties. It never filters, samples or changes
    data. Use markevery solely to space symbols on an otherwise complete line.
    """
    if role not in {'measured', 'model'}:
        raise ValueError('role must be measured or model')
    color = STYLE['colors'][index % len(STYLE['colors'])]
    marker = STYLE['markers'][index % len(STYLE['markers'])]
    return {'color': color, 'linestyle': 'none' if role == 'measured' else STYLE['linestyles'][(index + replicate) % 4], 'marker': marker if role == 'measured' else 'none', 'markerfacecolor': color if replicate % 2 == 0 else 'white', 'markeredgecolor': color, 'markeredgewidth': 0.7, 'markersize': STYLE['rc']['lines.markersize'], 'linewidth': STYLE['rc']['lines.linewidth']}

def save(fig, pdf_path, *, size_mm, spec=None, journal=None, preview_path=None):
    pdf = Path(pdf_path)
    actual = tuple((float(v) * 25.4 for v in fig.get_size_inches()))
    if any((abs(a - b) > 0.2 for a, b in zip(actual, size_mm))):
        raise ValueError('Figure dimensions differ from the specified dimensions')
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches=None, metadata={'CreationDate': None})
    fig.savefig(pdf.with_suffix('.svg'), bbox_inches=None, metadata={'Date': None})
    if preview_path is not None:
        preview = Path(preview_path)
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=180)
    return {'ok': True, 'outputs': {ext: {'sha256': hashlib.sha256(pdf.with_suffix('.' + ext).read_bytes()).hexdigest()} for ext in ('pdf', 'svg')}}