import argparse
import logging
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
import xarray as xr

from western_us_biomass import dir_info
from western_us_biomass.train_models import (
    train_model_delta_burned,
    train_model_delta_unburned,
    train_model_init_biomass,
)

logging.basicConfig(level=logging.INFO)

PREDICTORS = {
    "init": train_model_init_biomass.FPATH_PREDICTORS,
    "unburned": train_model_delta_unburned.FPATH_PREDICTORS,
    "burned": train_model_delta_burned.FPATH_PREDICTORS,
}


def initialize_models(model_suffix=""):
    """Initialize models with an optional model_suffix (e.g., '_0001', '_0002')"""

    models = {
        "init": joblib.load(train_model_init_biomass.FPATH_MODEL + model_suffix + ".pkl"),
        "unburned": joblib.load(train_model_delta_unburned.FPATH_MODEL + model_suffix + ".pkl"),
        "burned": joblib.load(train_model_delta_burned.FPATH_MODEL + model_suffix + ".pkl"),
    }

    for k in models:
        models[k].n_jobs = 1

    return models


def select_model_and_predictors(disturbance: str, backwards: bool = False, models=None):
    if models is None:
        raise ValueError("models dict must be provided")
    key = disturbance + ("_backwards" if backwards else "")
    return models[key], PREDICTORS[key]


def save_gridded_dataset(ds, fname, fextension=".nc"):
    ds = ds.chunk({dim: -1 for dim in ds.dims})
    if fextension == ".zarr":
        ds.to_zarr(fname + ".zarr", mode="w")
    elif fextension == ".nc":
        ds.to_netcdf(fname + ".nc")


def get_var_2d(var: str, year: int = None, inputs_2d=None):
    var_lookup = {
        "STDAGE_start": "STDAGE",
        "ecosection": "Ecosection",
        "ecoprovince": "Ecoprovince",
        "slope": "SLOPE_preliminary",
        "aspect": "ASPECT_preliminary",
        "elevation": "ELEV_preliminary",
    }

    if var in var_lookup:
        var_str = var_lookup[var]
    else:
        var_str = var

    if var_str not in inputs_2d.data_vars:
        logging.warning(
            "Variable '%s' not found in inputs_2d; returning NaN array instead.", var_str
        )


    da = inputs_2d[var_str]
    if year is not None and "year" in da.dims:
        da = da.sel(year=year)

    return da


def prepare_input_data(
    fpath_predictor_list: str = PREDICTORS["unburned"],
    fia_plot_data=None,
    inputs_2d=None,
    year: int = None,
    **kwargs,
) -> list:
    """Prepare input data for the model."""

    predictors = pd.read_csv(
        fpath_predictor_list,
        header=None,
    )

    time_varying_vars = {
        "biomass_start": "biomass_start",
        "biomass_end": "biomass_start",
        "years_after_fire": "years_after_fire",
        "years_after_fire_start": "years_after_fire",
        "LIVE_CANOPY_CVR_PCT": "canopy_cover",
        "LIVE_CANOPY_CVR_PCT_start": "canopy_cover",
        "delta_live_canopy_cvr_pct_per_year": "delta_live_canopy_cvr",
        "delta_live_canopy_cvr_pct": "delta_live_canopy_cvr",
    }

    var_lookup = {"STDAGE_start": "STDAGE", "ecosection": "Ecosection",
                  "tmean_mean_10yr_start": "tmean_mean_10yr",
"ppt_mean_10yr_start": "ppt_mean_10yr",
"tmax_maxseason_10yr_start": "tmax_maxseason_10yr",
"tmin_minseason_10yr_start": "tmin_minseason_10yr",
"vpdmax_maxseason_10yr_start": "vpdmax_maxseason_10yr"
                 }

    for i, predictor in enumerate(predictors[0]):
        if predictor in time_varying_vars:
            kwarg_name = time_varying_vars[predictor]
            predictor_series = kwargs.get(kwarg_name, None)
        elif predictor in var_lookup:
            predictor_series = get_var_2d(var=var_lookup[predictor], year=year, inputs_2d=inputs_2d)
        else:
            predictor_series = get_var_2d(var=predictor, year=year,inputs_2d=inputs_2d)

        if i == 0:
            df_inputs = predictor_series.to_dataset(name=predictor)
        else:
            df_inputs[predictor] = predictor_series

    df_inputs["analysis_mask"] = get_var_2d(var="analysis_mask", inputs_2d=inputs_2d)

    df_inputs_consistent = df_inputs

    df_inputs_consistent = df_inputs.stack(flat=["x", "y"]).unify_chunks()
    df_inputs_consistent = df_inputs_consistent.drop_vars("analysis_mask")
    df_inputs_consistent = df_inputs_consistent.to_dask_dataframe()
    all_data_spatial = df_inputs_consistent[["flat", "band", "x", "y"]]  # "spatial_ref",

    if "year" in df_inputs_consistent.columns:
        df_inputs_consistent = df_inputs_consistent.drop(
            columns=["x", "y", "flat", "band", "year"],  # "spatial_ref",
        )
    else:
        df_inputs_consistent = df_inputs_consistent.drop(
            columns=["x", "y", "flat", "band"],  # "spatial_ref"
        )

    return df_inputs_consistent, all_data_spatial["x"], all_data_spatial["y"]


def predict_biomass(
    df_inputs: pd.DataFrame,
    model: sklearn.ensemble._forest.RandomForestRegressor,
    x_dask_array,
    y_dask_array,
    use_dask=True,
    inputs_2d=None,
) -> np.ndarray:
    """Predict biomass using the trained model."""
    if use_dask:

        def predict_partition(partition):
            nan_mask = partition[partition.columns[0]].isna()
            preds = model.predict(partition)
            preds = np.where(nan_mask, np.nan, preds)

            return pd.DataFrame({"value": preds}, index=partition.index)

        meta = pd.DataFrame({"value": pd.Series(dtype="float64")})

        predicted = df_inputs.map_partitions(predict_partition, meta=meta)
        predicted_biomass_flat = predicted.compute()
        predicted_biomass_flat = predicted_biomass_flat.squeeze()

    else:
        predicted_biomass_flat = model.predict(df_inputs)

    x_flat = x_dask_array.compute()
    y_flat = y_dask_array.compute()

    df = pd.DataFrame({"x": x_flat, "y": y_flat, "value": predicted_biomass_flat})

    pivoted = df.pivot(index="y", columns="x", values="value")

    predicted_biomass_da = xr.DataArray(
        pivoted,
        dims=["y", "x"],
        coords={"y": pivoted.index, "x": pivoted.columns},
        name="predicted_biomass",
    )

    return predicted_biomass_da


def calculate_delta_biomass(
    predicted_biomass_start,
    delta_live_canopy_cvr,
    delta_live_canopy_cvr_twoyear,
    year: int,
    years_since_fire=None,
    fpath_predictor_list_unburned=PREDICTORS["unburned"],
    fpath_predictor_list_burned=PREDICTORS["burned"],
    models=None,
    fia_plot_data=None,
    save_components=False,
    inputs_2d=None,
    tile_ind="",
    model_suffix="",
    **kwargs,
):

    if models is None:
        raise ValueError("models dict must be provided")

    model_unburned = models["unburned"]
    model_burned = models["burned"]

    """Calculate the change in biomass over time."""
    if years_since_fire is None:
        years_since_fire = get_var_2d(var="years_after_fire", year=year, inputs_2d=inputs_2d)

    fire_frac = get_var_2d(var="FIRE_FRACTION", year=year, inputs_2d=inputs_2d)
    undisturbed_frac = 1 - fire_frac

    [df_inputs, original_shape_x, original_shape_y] = prepare_input_data(
        fpath_predictor_list=fpath_predictor_list_unburned,
        biomass_start=predicted_biomass_start,
        years_after_fire=years_since_fire,
        fia_plot_data=fia_plot_data,
        inputs_2d=inputs_2d,
        year=year,
        delta_live_canopy_cvr=delta_live_canopy_cvr,
        **kwargs,
    )

    logging.info("Done preparing unburned inputs")
    predicted_biomass_delta_unburned = predict_biomass(
        df_inputs=df_inputs,
        model=model_unburned,
        x_dask_array=original_shape_x,
        y_dask_array=original_shape_y,
        inputs_2d=inputs_2d,
    )
    logging.info("Predicted unburned delta")

    [df_inputs, original_shape_x, original_shape_y] = prepare_input_data(
        fpath_predictor_list=fpath_predictor_list_burned,
        biomass_start=predicted_biomass_start,
        years_after_fire=years_since_fire,
        fia_plot_data=fia_plot_data,
        inputs_2d=inputs_2d,
        year=year,
        delta_live_canopy_cvr=delta_live_canopy_cvr_twoyear,
        **kwargs,
    )

    logging.info("Done preparing burned inputs")

    predicted_biomass_delta_burned = predict_biomass(
        df_inputs=df_inputs,
        model=model_burned,
        x_dask_array=original_shape_x,
        y_dask_array=original_shape_y,
        inputs_2d=inputs_2d,
    )
    logging.info("Predicted burned delta")

    if save_components:
        save_gridded_dataset(
            ds=(predicted_biomass_delta_unburned * undisturbed_frac).to_dataset(
                name="predicted_biomass_delta_unburned"
            ),
            fname=dir_info.dir_model_output
            + "unburned_predicted_biomass_unfiltered_"
            + str(year)
            + model_suffix
            + tile_ind,
        )

        save_gridded_dataset(
            ds=(predicted_biomass_delta_burned * fire_frac).to_dataset(
                name="predicted_biomass_delta_burned"
            ),
            fname=dir_info.dir_model_output
            + "burned_predicted_biomass_unfiltered_"
            + str(year)
            + model_suffix
            + tile_ind,
        )

        save_gridded_dataset(
            ds=years_since_fire.to_dataset(name="years_since_fire"),
            fname=dir_info.dir_model_output
            + "years_since_fire"
            + str(year)
            + model_suffix
            + tile_ind,
        )

    predicted_biomass_delta_burned = fire_frac * predicted_biomass_delta_burned
    predicted_biomass_delta_undisturbed = undisturbed_frac * predicted_biomass_delta_unburned

    predicted_biomass_delta = predicted_biomass_delta_burned + predicted_biomass_delta_undisturbed

    return predicted_biomass_delta


def increment_time_step(
    biomass_t_minus_1,
    year: int,
    delta_live_canopy_cvr,
    delta_live_canopy_cvr_twoyear,
    backwards: bool = False,  # True if running backwards, False if running forwards
    models=None,
    inputs_2d=None,
    tile_ind="",
    model_suffix="",
    **kwargs,
):
    """Increment the biomass for the next time step when running forward (e.g. 2005 to 2006)

    Returns:
        np.ndarray: Biomass at the next time step.
    """

    model_unburned, fpath_predictor_list_unburned = select_model_and_predictors(
        disturbance="unburned", backwards=backwards, models=models
    )
    model_burned, fpath_predictor_list_burned = select_model_and_predictors(
        disturbance="burned", backwards=backwards, models=models
    )

    predicted_biomass_delta = calculate_delta_biomass(
        predicted_biomass_start=biomass_t_minus_1,
        delta_live_canopy_cvr=delta_live_canopy_cvr,
        delta_live_canopy_cvr_twoyear=delta_live_canopy_cvr_twoyear,
        year=year,
        models=models,
        fpath_predictor_list_unburned=fpath_predictor_list_unburned,
        fpath_predictor_list_burned=fpath_predictor_list_burned,
        inputs_2d=inputs_2d,
        tile_ind=tile_ind,
        model_suffix=model_suffix,
        **kwargs,
    )

    if backwards:
        biomass_t = biomass_t_minus_1 - predicted_biomass_delta
    else:
        biomass_t = biomass_t_minus_1 + predicted_biomass_delta

    biomass_t = biomass_t.where(biomass_t > 0, 0)

    return biomass_t


def initialize_biomass(
    dir_in: str = dir_info.dir_model_input,
    dir_out: str = dir_info.dir_model_output,
    year: int = 2005,
    models=None,
    inputs_2d=None,
    tile_ind="",
    model_suffix="",
):
    """ """
    years_since_fire_initial = get_var_2d(var="years_after_fire", year=year, inputs_2d=inputs_2d)

    [df_inputs, original_shape_x, original_shape_y] = prepare_input_data(
        fpath_predictor_list=PREDICTORS["init"],
        years_after_fire=years_since_fire_initial,
        canopy_cover=get_var_2d(var="LIVE_CANOPY_CVR_PCT", year=year, inputs_2d=inputs_2d),
        inputs_2d=inputs_2d,
        year=year,
    )

    predicted_biomass_start = predict_biomass(
        df_inputs=df_inputs,
        model=models["init"],
        x_dask_array=original_shape_x,
        y_dask_array=original_shape_y,
        inputs_2d=inputs_2d,
    )

    save_gridded_dataset(
        ds=predicted_biomass_start.to_dataset(name="predicted_biomass"),
        fname=dir_out + "predicted_biomass_unfiltered_" + "init" + model_suffix + tile_ind,
    )

    return predicted_biomass_start


def calculate_biomass_changes_over_time(
    dir_out: str = dir_info.dir_model_output,
    start_year: int = None,
    year_range=np.arange(2005, 2025),  # np.arange(1996, 1989, -1)
    backwards=False,
    models=None,
    inputs_2d=None,
    tile_ind="",
    model_suffix="",
):
    """ """
    start_time = time.time()
    if start_year is None:
        predicted_biomass_start = xr.open_dataset(
            dir_out + "predicted_biomass_unfiltered_init" + model_suffix + tile_ind + ".nc"
        )["predicted_biomass"]
    else:
        predicted_biomass_start = xr.open_dataset(
            dir_out
            + "predicted_biomass_unfiltered_"
            + str(start_year)
            + model_suffix
            + tile_ind
            + ".nc"
        )["predicted_biomass"]

    # ecosection = get_var_2d(var="ecosection", inputs_2d=inputs_2d)
    # ecoprovince = get_var_2d(var="ecoprovince", inputs_2d=inputs_2d)
    slope = get_var_2d(var="slope", inputs_2d=inputs_2d)
    aspect = get_var_2d(var="aspect", inputs_2d=inputs_2d)
    elevation = get_var_2d(var="elevation", inputs_2d=inputs_2d)
    pct_own_public = get_var_2d(var="pct_own_public", inputs_2d=inputs_2d)

    biomass_t = predicted_biomass_start
    for i, year in enumerate(year_range):
        save_gridded_dataset(
            ds=biomass_t.to_dataset(name="predicted_biomass"),
            fname=dir_out + "predicted_biomass_unfiltered_" + str(year) + model_suffix + tile_ind,
        )
        logging.info(year)

        canopy_cover = get_var_2d(var="LIVE_CANOPY_CVR_PCT", year=year, inputs_2d=inputs_2d)

        canopy_cover_next = get_var_2d(
            var="LIVE_CANOPY_CVR_PCT", year=year + 1, inputs_2d=inputs_2d
        )
        #

        canopy_cover_prev = get_var_2d(
            var="LIVE_CANOPY_CVR_PCT", year=year - 1, inputs_2d=inputs_2d
        )
        delta_live_canopy_cvr_pct_per_year = canopy_cover - canopy_cover_prev
        # delta_live_canopy_cvr_pct_per_year_v2 = canopy_cover_next - canopy_cover
        delta_live_canopy_cvr_pct_per_year_twoyear = canopy_cover_next - canopy_cover_prev

        biomass_t = increment_time_step(
            biomass_t_minus_1=biomass_t,
            backwards=backwards,
            year=year,
            models=models,
            inputs_2d=inputs_2d,
            canopy_cover=canopy_cover,
            delta_live_canopy_cvr=delta_live_canopy_cvr_pct_per_year,
            delta_live_canopy_cvr_twoyear=delta_live_canopy_cvr_pct_per_year_twoyear,
            # ecosection=ecosection,
            # ecoprovince=ecoprovince,
            slope=slope,
            aspect=aspect,
            elevation=elevation,
            pct_own_public=pct_own_public,
            tile_ind=tile_ind,
            model_suffix=model_suffix,
        )

        end_time = time.time()
        logging.info(f"Elapsed time for running so far: {end_time - start_time:.2f} seconds")


def main(
    xtile=None,
    ytile=None,
    dir_in: str = dir_info.dir_model_input,
    dir_out: str = dir_info.dir_model_output,
    resolution: int = 1000,
    model_suffix: str = "",
):
    def process_tile(
        tile_ind, inputs_2d, year_range=np.arange(2005, 2023), model_suffix=model_suffix
    ):
        initialize_biomass(
            dir_in=dir_in,
            dir_out=dir_out,
            models=models,
            inputs_2d=inputs_2d,
            tile_ind=tile_ind,
            model_suffix=model_suffix,
            year=2005,
        )
        calculate_biomass_changes_over_time(
            dir_out=dir_out,
            models=models,
            inputs_2d=inputs_2d,
            year_range=year_range,
            tile_ind=tile_ind,
            model_suffix=model_suffix,
        )

    models = initialize_models(model_suffix=model_suffix)

    fpath_2d = dir_info.dir_model_input + "all_variables.nc"

    if resolution == 1000:
        tile_size = 200  # this is the appropriate tile size for a 1000m resolution run
    elif resolution == 100:
        tile_size = 2000  # this is the appropriate tile size for a 100m resolution run

    inputs_2d_all = xr.open_dataset(fpath_2d, chunks={"x": tile_size, "y": tile_size})

    x_slice = slice(xtile * tile_size, (xtile + 1) * tile_size)
    y_slice = slice(ytile * tile_size, (ytile + 1) * tile_size)
    inputs_2d = inputs_2d_all.isel(x=x_slice, y=y_slice)

    tile_ind = f"_{xtile}.{ytile}_"
    logging.info(f"Processing tile {tile_ind}")

    if inputs_2d["analysis_mask"].max() > 0:
        process_tile(tile_ind=tile_ind, model_suffix=model_suffix, inputs_2d=inputs_2d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xtile", type=int, required=True)
    parser.add_argument("--ytile", type=int, required=True)
    parser.add_argument(
        "--model-suffix", type=str, default="", help="Suffix for model files (e.g., '0001', '0002')"
    )
    args = parser.parse_args()

    main(xtile=args.xtile, ytile=args.ytile, model_suffix=args.model_suffix)
