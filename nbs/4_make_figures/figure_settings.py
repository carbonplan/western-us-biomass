# plot_style.py
import matplotlib as mpl

style_settings = {
    # Font
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Helvetica",
        "Arial",
        "DejaVu Sans",
    ],  # Arial as fallback if Helvetica not installed
    "font.size": 24,
    "axes.titlesize": 26,
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 20,
    # Axes
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.labelpad": 10,
    # Lines
    "lines.linewidth": 8,
    # Layout
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.constrained_layout.use": False,
    # Legend
    "legend.framealpha": 0,
    "legend.frameon": False,
}


def apply_style():
    mpl.rcParams.update(style_settings)
