import geopandas as gpd
import matplotlib.pyplot as plt

from conus_biomass.dir_info import dir_shp


def get_western_state_shapefile(shp):
    western_state_names = [
        "California",
        "Washington",
        "Oregon",
        "Idaho",
        "Montana",
        "Arizona",
        "Colorado",
        "New Mexico",
        "Utah",
        "Wyoming",
        "Nevada",
    ]
    shp_western = shp[shp["NAME"].isin(western_state_names)]

    return shp_western


SHP = gpd.read_file(dir_shp + "cb_2018_us_state_20m.shp")
SHP_WESTERN = get_western_state_shapefile(SHP)


def plot_map(
    dataset,
    shp=SHP_WESTERN,
    latlon: bool = True,
    title: str = "Years Since Disturbance (2024)",
    cbar_label: str = "Years After Fire in 2024",
    cmap: str = "copper_r",
    clims: list[int] = [0, 30],
    savefig: str = None,
    cbar_location: str = "right",
    ax=None,
):
    plt.rcParams["font.size"] = 16
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 7))

    if latlon:
        shp_plot = shp.to_crs("EPSG:4326")
        dataset_latlon = dataset.rio.reproject("EPSG:4326")

        im = dataset_latlon.plot(
            ax=ax,
            vmin=clims[0],
            vmax=clims[1],
            cmap=cmap,
            cbar_kwargs={
                "shrink": 0.7,
                "pad": 0.02,
                "orientation": "horizontal" if cbar_location in ["top", "bottom"] else "vertical",
                "location": cbar_location,
            },
        )

        plt.xlim([-126, -101])
        plt.ylim([31, 50])
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
    else:
        shp_plot = shp.to_crs(dataset.rio.crs)
        im = dataset.plot(
            ax=ax,
            vmin=clims[0],
            vmax=clims[1],
            cmap=cmap,
            cbar_kwargs={
                "location": cbar_location,
                "orientation": "horizontal" if cbar_location in ["top", "bottom"] else "vertical",
            },
        )

    plt.tight_layout()
    ax.set_axis_off()
    shp_plot.boundary.plot(ax=ax, color="gray", linewidth=1)
    plt.title(title)
    im.colorbar.set_label(cbar_label, fontsize=20)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig, bbox_inches="tight", dpi=500)


def plot_hexbin_latlon(
    x,
    y,
    C,
    shp=SHP,
    clims: list[int] = [-30, 30],
    cmap=plt.cm.BrBG,
    gridsize: int = 70,
    cbar_label: str = "Biomass Delta",
    title: str = "",
    savefig: str = None,
    ax=None,
    show: bool = False,
    mincnt=5,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    shp_plot = shp.to_crs("EPSG:4326")
    # Plot the state outlines
    shp_plot.boundary.plot(ax=ax, color="gray", linewidth=1)

    # Overlay the hexbin
    hb = ax.hexbin(
        x=x,
        y=y,
        C=C,
        gridsize=gridsize,
        vmin=clims[0],
        vmax=clims[1],
        cmap=cmap,
        edgecolors="None",
        mincnt=mincnt,
    )

    plt.colorbar(hb, ax=ax, label=cbar_label)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    if savefig is not None:
        plt.savefig(savefig, bbox_inches="tight", dpi=500)
    if show:
        plt.show()
