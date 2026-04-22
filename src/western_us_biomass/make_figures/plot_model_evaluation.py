import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import xarray as xr

from western_us_biomass.dir_info import dir_QAQC
from western_us_biomass.make_figures.maps import plot_hexbin_latlon


def plot_histograms(
    y_test: xr.DataArray,
    y_pred: np.ndarray,
    fname=dir_QAQC + "initial_biomass/histograms_absolute_biomass.png",
):
    """Plot histograms of actual vs predicted biomass (three panel figure)

    Args:
        y_test (xr.DataArray): Actual biomass values for test subset.
        y_pred (np.ndarray): Predicted biomass values for test subset.
        fname (str, optional): File path to save the figure.
    """
    plt.figure(figsize=(15, 5))
    plt.rcParams["font.size"] = 16
    plt.subplot(1, 3, 1)
    plt.hist(y_test, alpha=0.5, bins=np.arange(-2.5, 400, 5), label="actual biomass")
    plt.hist(y_pred, alpha=0.5, bins=np.arange(-2.5, 400, 5), label="predicted biomass")
    plt.ylabel("Number of plots", fontsize=16)
    plt.xlabel("Live Aboveground Biomass Density \n (Mg/ha) in 2005", fontsize=16)
    plt.title("All plots", fontsize=16)
    plt.axvline(x=0, linestyle="--", color="k")
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.subplot(1, 3, 3)
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname, dpi=500)


def make_plot_level_comp_figure(
    y_test: xr.DataArray,
    X_test: pd.DataFrame,
    y: xr.DataArray,
    X: pd.DataFrame,
    model: sklearn.ensemble._forest.RandomForestRegressor,
    minval: int = -10,
    maxval: int = 300,
    fname: str = dir_QAQC + "initial_biomass/absolute_biomass_random_forest_model_performance.png",
):
    """Create a figure comparing predicted and actual biomass (scatter plots).
    Need to add R^2, β_0 and an RMSE.

    Args:
        y_test (array-like): Actual biomass values for test subset.
        X_test (DataFrame): Feature values for test subset.
        y (array-like): Actual biomass values.
        X (DataFrame): Feature values.
        model (object): Trained model for predictions.
        minval (int, optional): Minimum value for plot axes. Defaults to -10.
        maxval (int, optional): Maximum value for plot axes. Defaults to 300.
        fname (str, optional): File path to save the figure. Defaults to dir_QAQC+"initial_biomass/absolute_biomass_random_forest_model_performance.png".
    """
    y_pred = model.predict(X_test)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(y_test, y_pred, ".", alpha=0.1)
    plt.grid()
    plt.axhline(y=0, linestyle="--", color="k")
    plt.axvline(x=0, linestyle="--", color="k")
    plt.ylabel("Predicted Carbon Stock (MgC/ha)")
    plt.xlabel("Actual Carbon Stock (MgC/ha)")
    plt.xlim([minval, maxval])
    plt.ylim([minval, maxval])
    plt.plot([minval, maxval], [minval, maxval], "-", color="gray")
    plt.title("Testing data")

    plt.subplot(2, 2, 2)
    plt.plot(y, model.predict(X), ".", alpha=0.1)
    plt.axhline(y=0, linestyle="--", color="k")
    plt.axvline(x=0, linestyle="--", color="k")
    plt.plot([minval, maxval], [minval, maxval], "-", color="gray")
    plt.grid()
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.title("All data")

    plt.subplot(2, 2, 3)
    plt.plot(y_test, y_pred - y_test, ".", alpha=0.1)
    plt.grid()
    plt.axhline(y=0, linestyle="--", color="k")
    plt.xlabel("Actual")
    plt.ylabel("Residuals")
    plt.title("Testing data")

    plt.subplot(2, 2, 4)
    residuals = model.predict(X) - y
    plt.plot(y, residuals, ".", alpha=0.1)
    plt.grid()
    plt.axhline(y=0, linestyle="--", color="k")
    plt.xlabel("Actual")
    plt.ylabel("Residuals")
    plt.title("All data")
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)


def plot_feature_importance(
    shap_values: shap._explanation.Explanation,
    X_test: pd.DataFrame,
    dir_figures: str = dir_QAQC + "initial_biomass/",
):
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    if dir_figures is not None:
        plt.savefig(dir_figures + "feature_importance_scatter.png", dpi=500)

    plt.figure()
    shap.plots.bar(shap_values, max_display=12, show=False)
    plt.tight_layout()
    if dir_figures is not None:
        plt.savefig(dir_figures + "feature_importance_bar.png", dpi=500)
    plt.show()


def plot_partial_dependencies(
    shap_values: shap._explanation.Explanation,
    X: pd.DataFrame,
    dir_figures: str = dir_QAQC + "initial_biomass/",
):
    fig, axes = plt.subplots(nrows=8, ncols=7, figsize=(50, 25))
    for i, var in enumerate(X.columns):
        shap.plots.scatter(
            shap_values[:, i],
            ax=axes[int(i / 7), int(i % 7)],
            show=False,
            x_jitter=0.5,
            color=shap_values,
        )
    plt.tight_layout()
    if dir_figures is not None:
        plt.savefig(dir_figures + "partial_dependency_plots.png")
    plt.show()


def plot_hexbins_actual_vs_pred(
    fia_data: xr.DataArray,
    y: xr.DataArray,
    y_pred: np.ndarray,
    clims: list[int] = [0, 170],
    gridsize: int = 100,
    cmap=plt.cm.viridis,
    cbar_label: str = "Live Forest Biomass (Mg/ha)",
):
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(20, 8))
    plt.rcParams["font.size"] = 16

    plot_hexbin_latlon(
        x=fia_data["lon"][fia_data["plotid"].isin(y.plotid.values)],
        y=fia_data["lat"][fia_data["plotid"].isin(y.plotid.values)],
        C=y,
        ax=axes[0],
        title="Actual",
        savefig=None,
        cbar_label=cbar_label,
        clims=clims,
        cmap=cmap,
        gridsize=gridsize,
    )

    plot_hexbin_latlon(
        x=fia_data["lon"][fia_data["plotid"].isin(y.plotid.values)],
        y=fia_data["lat"][fia_data["plotid"].isin(y.plotid.values)],
        C=y_pred,
        ax=axes[1],
        title="Predicted",
        savefig=None,
        cbar_label=cbar_label,
        clims=clims,
        cmap=cmap,
        gridsize=gridsize,
    )
