import logging
import time

import joblib
import numpy as np
import pandas as pd
import shap
import xarray as xr
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from conus_biomass.dir_info import dir_lookups, dir_processed
from conus_biomass.make_figures import plot_model_evaluation

logging.basicConfig(level=logging.INFO)


def encode_categorical(da, category: str):
    lookups = pd.read_csv(dir_lookups + "ecosection_lookup.csv")

    if category == "ecosection":
        lookups_unique = lookups.drop_duplicates(subset="Ecosection")
        mapping = dict(zip(lookups_unique["Ecosection"], lookups_unique["Ecosection_code"]))
    elif category == "ecoprovince":
        lookups_unique = lookups.drop_duplicates(subset="Ecoprovince")
        mapping = dict(zip(lookups_unique["Ecoprovince"], lookups_unique["Ecoprovince_code"]))
    else:
        raise ValueError("category must be 'ecosection' or 'ecoprovince'")

    # Bind mapping in the lambda using a default argument
    def block_mapper(block):
        return np.vectorize(lambda x, mapping=mapping: mapping.get(x, -1))(block)

    dask_array = da.data

    mapped = dask_array.map_blocks(block_mapper, dtype=int)

    encoded_xr = xr.DataArray(mapped, coords=da.coords, dims=da.dims, name=f"{category}_code")

    return encoded_xr


def get_ecosection_lists():
    ds = pd.read_csv(dir_lookups + "ecosection_lookup.csv")
    ecosection_cats = ds["Ecosection"].unique().tolist()
    ecoprovince_cats = ds["Ecoprovince"].unique().tolist()
    return ecosection_cats, ecoprovince_cats


def load_data(
    fpath: str = dir_processed + "restructured_FIA/*_FIA_plots_and_PRISM_v7.nc",
) -> xr.Dataset:
    """Load the FIA dataset and calculate a couple of additional variables.

    Args:
        fpath (str, optional): File path to the FIA dataset. Defaults to dir_processed + "*_FIA_plots_and_PRISM_v5.nc".

    Returns:
        xr.Dataset: The loaded FIA dataset.
    """
    fia_data = xr.open_mfdataset(fpath, combine="nested", concat_dim="plotid")

    fia_data["fire_between_measurements"] = (
        fia_data["fires_occurred"]
        .where(
            (fia_data["fires_occurred"].year >= fia_data["measyear_1"])
            * (fia_data["fires_occurred"].year <= fia_data["measyear_2"])
            * (fia_data["measyear_1"] < fia_data["measyear_2"])
        )
        .sum(dim="year")
    )

    fia_data["harvest_between_measurements"] = (
        fia_data["harvest_occurred"]
        .where(
            (fia_data["harvest_occurred"].year >= fia_data["measyear_1"])
            * (fia_data["harvest_occurred"].year <= fia_data["measyear_2"])
            * (fia_data["measyear_1"] < fia_data["measyear_2"])
        )
        .sum(dim="year")
    )

    return fia_data


def calculate_change_in_var(fia_data: xr.Dataset, var: str) -> xr.DataArray:
    """Calculate the change in a variable between two measurement years.

    Args:
        fia_data (xr.Dataset): The FIA dataset.
        var (str, optional): The variable to calculate the change for. Defaults to "LIVE_CANOPY_CVR_PCT".

    Returns:
        xr.DataArray: The change in the variable between the two measurement years.
    """
    var_at_measyear2 = fia_data["LIVE_CANOPY_CVR_PCT"].sel(year=fia_data["measyear_2"])

    var_at_measyear1 = fia_data["LIVE_CANOPY_CVR_PCT"].sel(year=fia_data["measyear_1"])

    delta_var = var_at_measyear2 - var_at_measyear1

    if var == "LIVE_CANOPY_CVR_PCT":
        # delta_var = delta_var.where(~np.isnan(delta_var), 0)
        delta_var = delta_var.where(~np.isinf(delta_var), 0)

    return delta_var


def calculate_new_fia_variables(fia_data: xr.Dataset) -> xr.Dataset:
    """
    Calculate new variables for the FIA data.

    Args:
        fia_data: xarray dataset containing the FIA data

    Returns:
        fia_data: xarray dataset with new variables added
    """

    if "delta_live_canopy_cvr_pct" not in fia_data.data_vars:
        fia_data["delta_live_canopy_cvr_pct"] = calculate_change_in_var(
            fia_data, var="LIVE_CANOPY_CVR_PCT"
        ).load()

    if "delta_live_canopy_cvr_pct_per_year" not in fia_data.data_vars:
        fia_data["delta_live_canopy_cvr_pct_per_year"] = fia_data["delta_live_canopy_cvr_pct"] / (
            fia_data["measyear_2"] - fia_data["measyear_1"]
        )

    if "biomass_delta_per_year" not in fia_data.data_vars:
        fia_data["biomass_delta_per_year"] = fia_data["biomass_delta"] / (
            fia_data["measyear_2"] - fia_data["measyear_1"]
        )

    if "biomass_start" not in fia_data.data_vars:
        fia_data["biomass_start"] = fia_data["biomass"].sel(year=fia_data["measyear_1"])

    if "years_after_fire" not in fia_data.data_vars:
        fia_data["years_after_fire"] = fia_data["years_after_fire"].where(
            fia_data["years_after_fire"] < 100, 70
        )

    if "biomass_delta_frac" not in fia_data.data_vars:
        fia_data["biomass_delta_frac"] = fia_data["biomass_delta"] / fia_data["biomass_start"]
        fia_data["biomass_delta_frac"] = fia_data["biomass_delta_frac"].where(
            ~np.isinf(fia_data["biomass_delta_frac"]), 0
        )

    if "pct_own_private" not in fia_data.data_vars:
        fia_data["pct_own_private"] = (fia_data["OWNGRPCD"] == 40) * 100

    if "pct_own_statelocal" not in fia_data.data_vars:
        fia_data["pct_own_statelocal"] = (fia_data["OWNGRPCD"] == 30) * 100

    if "pct_own_federal" not in fia_data.data_vars:
        fia_data["pct_own_federal"] = (
            (fia_data["OWNGRPCD"] == 10) + (fia_data["OWNGRPCD"] == 20)
        ) * 100

    if "pct_own_public" not in fia_data.data_vars:
        fia_data["pct_own_public"] = fia_data["pct_own_federal"] + fia_data["pct_own_statelocal"]

    if "Ecosection_code" not in fia_data.data_vars:
        fia_data["Ecosection_code"] = encode_categorical(fia_data["Ecosection"], "ecosection")

    if "Ecoprovince_code" not in fia_data.data_vars:
        fia_data["Ecoprovince_code"] = encode_categorical(fia_data["Ecoprovince"], "ecoprovince")

    return fia_data


def prepare_input_data(
    fia_data: xr.Dataset,
    fia_variable_predictors: list[str],
    predictors_meas1: list[str],
    predictors_meas2: list[str],
    yearmatch=None,
) -> list[pd.DataFrame]:
    """
    Prepare the input data for the model training.

    Args:
        fia_data: xarray dataset containing the FIA data
        fia_variable_predictors: list of FIA variable predictors
        predictors_meas1: list of initial FIA variable predictors
        predictors_meas2: list of PRISM variable predictors

    Returns:
        df_inputs: pandas dataframe containing the input data
    """

    fia_data = calculate_new_fia_variables(fia_data)

    if yearmatch == "nearest":
        df_inputs = (
            fia_data["year"]
            .sel(year=fia_data["measyear_1"], method="nearest")
            .to_dataframe(name="year")
        )
    else:
        df_inputs = fia_data["year"].sel(year=fia_data["measyear_1"]).to_dataframe(name="year")

    for var in fia_variable_predictors:
        df_inputs[var] = fia_data[var]

    for var in predictors_meas1:
        df_inputs[var + "_start"] = fia_data[var].sel(year=fia_data["measyear_1"])

    for var in predictors_meas2:
        df_inputs[var] = fia_data[var].sel(year=fia_data["measyear_2"])

    df_inputs = df_inputs.drop(columns=["year"])

    categorical_variables_codes = [
        "FORTYPCD",
        "FORTYPCD_GRP",
        "OWNGRPCD",
        "fire_count",
        "harvest_count",
    ]
    for var in categorical_variables_codes:
        if var in df_inputs.columns:
            df_inputs[var] = df_inputs[var].astype("category")

    ecosection_cats, ecoprovince_cats = get_ecosection_lists()
    categorical_variables_strings = [
        # "ECOSUBCD",
        "Ecosection",
        "Ecoprovince",
        "Ecosection_code",
        "Ecoprovince_code",
    ]
    for var in categorical_variables_strings:
        if var in df_inputs.columns:
            if var == "Ecosection":
                df_inputs[var] = pd.Categorical(df_inputs[var], categories=ecosection_cats).codes
            elif var == "Ecoprovince":
                df_inputs[var] = pd.Categorical(df_inputs[var], categories=ecoprovince_cats).codes
            elif var in df_inputs.columns:
                df_inputs[var] = df_inputs[var].astype("category").cat.codes

    return df_inputs


def prepare_output_data(fia_data: xr.Dataset, output_variable: str) -> xr.DataArray:
    """
    Prepare the output data for the model training.

    Args:
        fia_data: xarray dataset containing the FIA data
        output_variable: variable to be used as the output

    Returns:
        y: pandas series containing the output data
    """
    y = fia_data[output_variable].to_pandas()

    return y


def filter_out_nans(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Filter out NaN values from the input and output data.

    Args:
        X: pandas dataframe containing the input data
        y: pandas series containing the output data

    Returns:
        X_filtered: pandas dataframe containing the filtered input data
        y_filtered: pandas series containing the filtered output data
    """

    # Combine X and y into one DataFrame
    df = X.copy()
    df["_target"] = y

    # Drop any row where either X or y has NaNs
    df = df.dropna()

    # Split back out into separate X and y
    y_filtered = df["_target"]
    X_filtered = df.drop(columns=["_target"])

    return X_filtered, y_filtered


def get_X_y(
    fia_data: xr.Dataset,
    fia_variable_predictors: list,
    predictors_meas1: list,
    predictors_meas2: list,
    output_variable,
    yearmatch=None,
):

    X = prepare_input_data(
        fia_data,
        fia_variable_predictors=fia_variable_predictors,
        predictors_meas1=predictors_meas1,
        predictors_meas2=predictors_meas2,
        yearmatch=yearmatch,
    )
    y = prepare_output_data(fia_data, output_variable=output_variable)

    [X, y] = filter_out_nans(X, y)

    return X, y


def construct_model(
    modeltype="random_forest",
    tree_method=None,
    enable_categorical=None,
    n_estimators=None,
    max_depth=None,
    min_child_weight=None,
    max_cat_threshold=None,
    random_state=None,
    subsample=True,
    monotonic_cst=None,
    eval_metric=None,
):

    if modeltype == "random_forest":
        required_vars = [n_estimators, max_depth, random_state]
        assert not any(
            v is None for v in required_vars
        ), "All required variables must be specified and not None"

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            min_samples_split=5,
            min_samples_leaf=2,
            monotonic_cst=monotonic_cst,
        )
    elif modeltype == "xgboost":
        required_vars = [
            tree_method,
            enable_categorical,
            n_estimators,
            max_depth,
            min_child_weight,
            max_cat_threshold,
            random_state,
            subsample,
        ]
        assert not any(
            v is None for v in required_vars
        ), "All required variables must be specified and not None"

        model = xgb.XGBRegressor(
            tree_method=tree_method,
            enable_categorical=enable_categorical,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            max_cat_threshold=max_cat_threshold,
            random_state=random_state,
            subsample=subsample,
            eta=0.03,
            eval_metric=eval_metric,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            gamma=0.6,
            # lambda=2,
            # alpha=0.3
            # scale_pos_weight=0.8,
            # sampling_method='gradient_based',
            # gamma=0
        )
    else:
        raise ValueError(f"Unknown model type: {modeltype}")

    return model


def train_model(
    X: pd.DataFrame,
    y: xr.DataArray,
    modeltype: str = "random_forest",
    save_model: str = None,
    save_input_variable_names: str = None,
    n_estimators: int = 800,
    max_depth: int = 25,
    random_state: int = 94,
    test_size: float = 0.2,
    pre_split: bool = False,
    X_test: pd.DataFrame = None,
    y_test: xr.DataArray = None,
    tree_method: str = "hist",
    eval_metric="rmse",
    enable_categorical: bool = True,
    min_child_weight: int = 3,
    max_cat_threshold: int = 10,
    subsample: float = 0.8,
    monotonic_cst=None,
) -> list:
    """
    Train a Random Forest model on the provided features and target variable.

    Args:
        X: Features (input data)
        y: Target variable (output data)

    Returns:
        model: Trained Random Forest model
        X_train, X_test, y_train, y_test: Split datasets for evaluation
    """

    if pre_split:
        logging.info("Using pre-split training and testing sets.")
        X_train = X
        y_train = y
    else:
        logging.info("Splitting data into training and testing sets.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    logging.info("Training " + str(modeltype) + " model")
    assert modeltype in [
        "random_forest",
        "xgboost",
    ], "modeltype must be 'random_forest' or 'xgboost'"
    if modeltype == "random_forest":
        model = construct_model(
            modeltype=modeltype,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            monotonic_cst=monotonic_cst,
            eval_metric=eval_metric,
        )

        model.fit(X_train, y_train)
    elif modeltype == "xgboost":
        model = construct_model(
            modeltype=modeltype,
            tree_method=tree_method,
            enable_categorical=enable_categorical,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            max_cat_threshold=max_cat_threshold,
            random_state=random_state,
            subsample=subsample,
            eval_metric=eval_metric,
        )

        _ = model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    logging.info("Model training complete")

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    logging.info(f"Model R²: {r2:.4f}")
    logging.info(f"Model RMSE: {rmse:.4f}")

    if save_model is not None:
        logging.info(f"Saving model to {save_model}...")
        joblib.dump(model, save_model)

        logging.info(f"Saving model input variable names to {save_input_variable_names}...")
        pd.Series(X.columns.values).to_csv(save_input_variable_names, index=False, header=False)

    return model, X_train, X_test, y_train, y_test


def do_shap_analysis(
    X_test: pd.DataFrame, X_train: pd.DataFrame, X: pd.DataFrame, model, dir_figures: str
):
    """Do SHAP analysis to interpret model predictions.

    Args:
        X_test (pd.DataFrame): The test dataset features.
        X_train (pd.DataFrame): The training dataset features.
        X (pd.DataFrame): The complete dataset features.
        model: The trained model.
        dir_figures (str): Directory to save figures.

    Returns:
        The SHAP values for the test dataset.
    """
    logging.info("Creating SHAP explainer")
    explainer = shap.TreeExplainer(model, X_train)
    shap_values = explainer(X_test, check_additivity=False)

    logging.info("Generating model evaluation plots")
    plot_model_evaluation.plot_feature_importance(shap_values, X_test, dir_figures)

    plot_model_evaluation.plot_partial_dependencies(shap_values, X, dir_figures)

    return shap_values


def main(
    fia_variable_predictors: list,
    predictors_meas1: list,
    predictors_meas2: list,
    output_variable: str,
    path_model: str,
    path_input_variable_names: str,
    fia_data_train=None,
    fia_data_test=None,
    max_depth: int = 15,
    n_estimators: int = 100,
    min_child_weight: int = 3,
    max_cat_threshold: int = 10,
    yearmatch: str = "nearest",
):

    # --------------------------------------------------
    # Prepare data
    # --------------------------------------------------
    start_time = time.time()
    logging.info(">>> Preparing data")

    if fia_data_train is None:
        print("Error: no fia data provided")

    [X_train, y_train] = get_X_y(
        fia_data=fia_data_train,
        fia_variable_predictors=fia_variable_predictors,
        predictors_meas1=predictors_meas1,
        predictors_meas2=predictors_meas2,
        output_variable=output_variable,
        yearmatch=yearmatch,
    )

    [X_test, y_test] = get_X_y(
        fia_data=fia_data_test,
        fia_variable_predictors=fia_variable_predictors,
        predictors_meas1=predictors_meas1,
        predictors_meas2=predictors_meas2,
        output_variable=output_variable,
        yearmatch=yearmatch,
    )

    end_time = time.time()
    logging.info(f"Elapsed time: {end_time - start_time:.2f} seconds")
    # --------------------------------------------------
    # Train model
    # --------------------------------------------------
    logging.info(">>> Training model")

    model, X_train_out, X_test_out, y_train_out, y_test_out = train_model(
        X=X_train,
        y=y_train,
        modeltype="random_forest",
        save_model=path_model,
        save_input_variable_names=path_input_variable_names,
        max_depth=max_depth,
        n_estimators=n_estimators,
        pre_split=True,
        min_child_weight=min_child_weight,
        max_cat_threshold=max_cat_threshold,
        X_test=X_test,
        y_test=y_test,
    )

    end_time = time.time()
    logging.info(f"Elapsed time: {end_time - start_time:.2f} seconds")
