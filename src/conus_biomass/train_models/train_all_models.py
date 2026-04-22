import logging
import time

import numpy as np
import xarray as xr

from conus_biomass.dir_info import dir_processed
from conus_biomass.train_models import (
    train_model_delta_burned,
    train_model_delta_unburned,
    train_model_init_biomass,
    train_models_utils,
)


def load_data(
    fpath=dir_processed + "restructured_FIA/" + "*_FIA_plots_and_PRISM_v8.nc", condprop_filter=True
):
    fia_data = train_models_utils.load_data(fpath)
    if condprop_filter:
        fia_data["biomass"] = fia_data["biomass"].where(fia_data["CONDPROP_UNADJ"] > 0.1)
    fia_data["biomass_end"] = fia_data["biomass"].sel(year=fia_data["measyear_2"])

    fia_data["biomass_delta"] = fia_data["biomass"].sel(year=fia_data["measyear_2"]) - fia_data[
        "biomass"
    ].sel(year=fia_data["measyear_1"])

    fia_data["biomass_delta_per_year"] = fia_data["biomass_delta"] / (
        fia_data["measyear_2"] - fia_data["measyear_1"]
    )

    return fia_data


def split_test_train(
    fia_data, random_seed: int = 42, model_suffix: str = "", save_splits: bool = True
):

    plotids = np.unique(fia_data.plotid.values)

    # Randomly select 20% for testing
    if random_seed is not None:
        # option to set random seed for reproducibility
        np.random.seed(random_seed)
    test_plotids = np.random.choice(plotids, size=int(0.2 * len(plotids)), replace=False)

    # Create boolean masks for selection
    is_test_np = np.isin(fia_data.plotid.values, test_plotids)
    is_train_np = ~is_test_np
    # Convert to DataArray masks with correct dimension
    is_train = xr.DataArray(is_train_np, dims=["plotid"])

    # Split the dataset
    fia_data_test = fia_data.where(fia_data["plotid"].isin(test_plotids), drop=True)
    fia_data_train = fia_data.where(is_train, drop=True)

    if save_splits:
        fname_out_train = dir_processed + "models/train_plotids" + model_suffix + ".nc"

        is_train.to_dataset(name="is_train").to_netcdf(fname_out_train)

    return fia_data_test, fia_data_train


def split_subcomponents(fia_data):
    fia_data_burned = fia_data.where((fia_data["fire_between_measurements"] > 0).load(), drop=True)

    fia_data_undisturbed = fia_data.where(
        (fia_data["fire_between_measurements"] == 0).load(),
        drop=True,
    )

    return fia_data_burned, fia_data_undisturbed


def train_all_models(model_suffix="", random_seed: int = 42, split_test_train_bool=True):
    start_time = time.time()

    logging.info("Loading data")
    fia_data = load_data()

    logging.info("Subsetting data to west")
    fia_data = fia_data.where(
        fia_data["STATECD"].isin([4, 6, 8, 16, 30, 32, 35, 41, 49, 53, 56]).load(), drop=True
    )

    logging.info("Splitting testing and training data")
    if split_test_train_bool:
        fia_data_test, fia_data_train = split_test_train(
            fia_data, random_seed=random_seed, model_suffix=model_suffix
        )
    else:
        fia_data_test = fia_data
        fia_data_train = fia_data

    fia_data_train_burned, fia_data_train_undisturbed = split_subcomponents(fia_data_train)
    fia_data_test_burned, fia_data_test_undisturbed = split_subcomponents(fia_data_test)

    # fia_data_all_burned["biomass_delta"] = fia_data_all_burned["biomass_delta"].where(
    #    fia_data_all_burned["biomass_delta"] < 0, fia_data_all_burned["biomass_delta_per_year"]
    # )

    # Initialization ---------------------------------- #
    logging.info("######### INITIAL BIOMASS ############")
    train_models_utils.main(
        fia_variable_predictors=train_model_init_biomass.FIA_VARIABLE_PREDICTORS,
        predictors_meas1=train_model_init_biomass.PREDICTOR_VARIABLES_MEAS1,
        predictors_meas2=train_model_init_biomass.PREDICTOR_VARIABLES_MEAS2,
        output_variable=train_model_init_biomass.OUTPUT_VARIABLE,
        path_model=train_model_init_biomass.FPATH_MODEL + model_suffix + ".pkl",
        path_input_variable_names=train_model_init_biomass.FPATH_PREDICTORS,
        fia_data_train=fia_data_train,
        fia_data_test=fia_data_test,
        max_depth=40,
        n_estimators=200,
        min_child_weight=1,
    )

    # Fire ---------------------------------- #
    logging.info("######### FIRE ############")
    train_models_utils.main(
        fia_variable_predictors=train_model_delta_burned.FIA_VARIABLE_PREDICTORS,
        predictors_meas1=train_model_delta_burned.PREDICTOR_VARIABLES_MEAS1,
        predictors_meas2=train_model_delta_burned.PREDICTOR_VARIABLES_MEAS2,
        output_variable=train_model_delta_burned.OUTPUT_VARIABLE,
        path_model=train_model_delta_burned.FPATH_MODEL + model_suffix + ".pkl",
        path_input_variable_names=train_model_delta_burned.FPATH_PREDICTORS,
        fia_data_train=fia_data_train_burned,
        fia_data_test=fia_data_test_burned,
        max_depth=15,
        n_estimators=500,
        min_child_weight=3,
        max_cat_threshold=10,
    )

    # Undisturbed ---------------------------------- #
    logging.info("######### UNDISTURBED ############")
    train_models_utils.main(
        fia_variable_predictors=train_model_delta_unburned.FIA_VARIABLE_PREDICTORS,
        predictors_meas1=train_model_delta_unburned.PREDICTOR_VARIABLES_MEAS1,
        predictors_meas2=train_model_delta_unburned.PREDICTOR_VARIABLES_MEAS2,
        output_variable=train_model_delta_unburned.OUTPUT_VARIABLE,
        path_model=train_model_delta_unburned.FPATH_MODEL + model_suffix + ".pkl",
        path_input_variable_names=train_model_delta_unburned.FPATH_PREDICTORS,
        fia_data_train=fia_data_train_undisturbed,
        fia_data_test=fia_data_test_undisturbed,
        max_depth=30,
        n_estimators=300,
        min_child_weight=1,
        max_cat_threshold=10,
    )

    end_time = time.time()
    logging.info(f"Elapsed time for all training: {end_time - start_time:.2f} seconds")


def main():
    train_all_models(model_suffix="", random_seed=42, split_test_train_bool=True)


if __name__ == "__main__":
    main()
