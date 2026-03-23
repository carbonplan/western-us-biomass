import matplotlib as mpl

panel_x_offset = 0.02
panel_y_offset = 0.98

# Figure 1 colors
color_MTBS = "#A24936"
color_MTBS_light = "#FBBBA8"
color_FIA_fire = "#E55713"

# Figure 3 colors
color_unburned = "#64b9c4"
color_net = "#282B28"
color_burned = "#f07071"

# Figure 5 colors
color_CMIP = "#A24936"
color_Liu = "#83BCA9"
color_this_study = "#282B28"
color_Li = "#3E5641"
color_EPA = "#D36135"

style_settings = {
    # Font
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Helvetica",
        "Arial",
        "DejaVu Sans",
    ],
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
    "xtick.major.pad": 5.5,
    "ytick.major.pad": 5.5,
    "xtick.top": True,
    "ytick.right": True,
    "axes.labelpad": 8,
    "axes.titlepad": 10,
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
