import numpy as np
import pandas as pd
import xarray as xr

LAT_BINS = np.arange(24, 50, 0.5)
LON_BINS = np.arange(-125, -40, 0.5)


def calculate_ds_binned(
    ds, vars_year, vars_static, do_counts=False, lat_bins=LAT_BINS, lon_bins=LON_BINS
):
    df = ds.to_dataframe().reset_index()

    lat_bin = np.digitize(ds.lat, lat_bins, right=False) - 1
    lon_bin = np.digitize(ds.lon, lon_bins, right=False) - 1

    if do_counts:
        # Bin using the DataFrame's lat/lon columns
        df["lat_bin"] = np.digitize(df["lat"].values, lat_bins) - 1
        df["lon_bin"] = np.digitize(df["lon"].values, lon_bins) - 1

        grouped = df.groupby(["lat_bin", "lon_bin"])["plotid"].count().reset_index(name="n_plots")

        ds_binned = grouped.set_index(["lat_bin", "lon_bin"]).to_xarray()

        # Map bin indices to edges or midpoints
        ds_binned = ds_binned.assign_coords(
            {
                "lat_bin": ("lat_bin", lat_bins[ds_binned["lat_bin"].values.astype(int)]),
                "lon_bin": ("lon_bin", lon_bins[ds_binned["lon_bin"].values.astype(int)]),
            }
        )

    else:
        bin_df = pd.DataFrame(
            {
                "plotid": ds.plotid.values,
                "lat_bin": lat_bin,
                "lon_bin": lon_bin,
            }
        )

        df = df.merge(bin_df, on="plotid", how="left")

        # Separate variables by whether they have year dimension

        # 1. Year-dependent variables
        df_year = df[["lat_bin", "lon_bin", "year"] + vars_year]
        grouped_year = (
            df_year.groupby(["lat_bin", "lon_bin", "year"]).mean(numeric_only=True).reset_index()
        )
        ds_binned_year = grouped_year.set_index(["lat_bin", "lon_bin", "year"]).to_xarray()

        # 2. Year-independent variables
        df_static = df[["lat_bin", "lon_bin"] + vars_static]
        grouped_static = (
            df_static.groupby(["lat_bin", "lon_bin"]).mean(numeric_only=True).reset_index()
        )
        ds_binned_static = grouped_static.set_index(["lat_bin", "lon_bin"]).to_xarray()

        # 3. Merge datasets
        ds_binned = xr.merge([ds_binned_static, ds_binned_year])

        ds_binned = ds_binned.assign_coords(
            {
                "lat_bin": ("lat_bin", lat_bins[ds_binned["lat_bin"].values.astype(int)]),
                "lon_bin": ("lon_bin", lon_bins[ds_binned["lon_bin"].values.astype(int)]),
            }
        )

    # Map bin indices to midpoints
    # lat_mid = (lat_bins[:-1] + lat_bins[1:]) / 2
    # lon_mid = (lon_bins[:-1] + lon_bins[1:]) / 2

    # lat_idx = np.clip(ds_binned["lat_bin"].values.astype(int), 0, len(lat_mid) - 1)
    # lon_idx = np.clip(ds_binned["lon_bin"].values.astype(int), 0, len(lon_mid) - 1)

    # ds_binned = ds_binned.assign_coords(
    #    {
    #        "lat_bin": ("lat_bin", lat_mid[lat_idx]),
    #        "lon_bin": ("lon_bin", lon_mid[lon_idx]),
    #    }

    return ds_binned


def get_stacked_binned_data(
    plot_level_data, vars_year, vars_static, do_counts=False, lat_bins=LAT_BINS, lon_bins=LON_BINS
):  #

    var_list = vars_year + vars_static

    ds = plot_level_data[var_list]

    # Bin data
    ds_binned = calculate_ds_binned(
        ds, vars_year, vars_static, do_counts=do_counts, lat_bins=lat_bins, lon_bins=lon_bins
    )

    # Prepare 2D coordinates for pcolormesh
    lon_2d, lat_2d = np.meshgrid(ds_binned["lon_bin"].values, ds_binned["lat_bin"].values)

    # Stack bins into a single dimension
    ds_binned_stacked = ds_binned.stack(plotid=["lat_bin", "lon_bin"])

    # Calculate number of plots with measurements
    ds_notnan = plot_level_data[["biomass_delta", "lat", "lon"]]
    ds_notnan["biomass_delta"] = (~plot_level_data["biomass_delta"].isnull()).astype(int)
    ds_binned_counts = calculate_ds_binned(ds_notnan, vars_year, vars_static, do_counts=True)
    ds_binned_counts_stacked = ds_binned_counts.stack(plotid=["lat_bin", "lon_bin"])

    # Filter out gridcells with too few measurements
    ds_binned_stacked = ds_binned_stacked.where(ds_binned_counts_stacked["n_plots"] > 0, drop=True)

    ds_binned_stacked = ds_binned_stacked.drop_vars(["lat_bin", "lon_bin"])

    return ds_binned_stacked, lon_2d, lat_2d

    # ds_notnan = plot_level_data[['delta_live_canopy_cvr_pct_per_year','lat','lon']]
    # ds_notnan["delta_live_canopy_cvr_pct_per_year"]=(~ds_notnan['delta_live_canopy_cvr_pct_per_year'].isnull()).astype(int)

    # ds_binned_counts_canopy = calculate_ds_binned(ds_notnan, do_counts=True)

    # ds = fia_data_test_undisturbed[var_list]
    # ds_binned_test = calculate_ds_binned(ds)

    # lon_2d_test, lat_2d_test = np.meshgrid(ds_binned_test['lon_bin'].values, ds_binned_test['lat_bin'].values)

    # ds_binned_test_stacked = ds_binned_test.stack(plotid=['lat_bin','lon_bin'])

    # ds_binned_test_stacked = ds_binned_test_stacked.drop_vars(['lat_bin','lon_bin'])

    #
