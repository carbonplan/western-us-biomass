import logging

import numpy as np
import pandas as pd
import xarray as xr

from conus_biomass.dir_info import dir_lookups
from conus_biomass.process_inputs.load_fia import load_data
from conus_biomass.unit_conversion import ACRES_PER_HECTARE, POUNDS_PER_METRIC_TON

INVALID_YEAR_THRESHOLD = 9000
MIN_YEAR = 1981
MAX_YEAR = 2024
RECENT_YEAR_THRESHOLD = 2000
NAN_DSTRBYR = -9999

logging.basicConfig(level=logging.INFO)


def calculate_biomass(
    tree_df: pd.DataFrame,
    var_input: str,
    var_output: str,
    live_tree: bool = True,
) -> pd.Series:
    """Calculate aboveground biomass for each plot-condition using tree-level data.
    Function can be used to calculate live or dead biomass
    by setting the live_tree argument to True or False, respectively.

    Inputs:
    tree_df (pd.DataFrame): DataFrame containing tree data
    live_tree (bool): If True, calculate biomass for live trees; if False, calculate for dead trees
    var_input (str): Column name in tree_df to use for biomass calculation (CARBON_AG for aboveground)

    Outputs:
    biomass (pd.Series): Series containing aboveground biomass for each plot-condition

    """

    if live_tree:
        statuscd = 1
    else:
        statuscd = 2

    tree_df[var_output] = (
        tree_df[var_input] * tree_df["TPA_UNADJ"] / (POUNDS_PER_METRIC_TON / ACRES_PER_HECTARE)
    ).where(tree_df["STATUSCD"] == statuscd)

    biomass = tree_df.groupby("PLT_CN_CONDID")[var_output].sum()

    return biomass


def calculate_qmd(tree_df: pd.DataFrame, live_tree: bool = True) -> pd.Series:
    """Calculate quadratic mean diameter (QMD) for each plot-condition using tree-level data

    Inputs:
    tree_df (pd.DataFrame): DataFrame containing tree data
    live_tree (bool): If True, calculate QMD for live trees; if False, calculate for dead trees

    Outputs:
    QMD (pd.Series): Series containing quadratic mean diameter for each plot-condition

    """

    if live_tree:
        statuscd = 1
    else:
        statuscd = 2

    tree_df["DIA_squared"] = (tree_df["DIA"]) ** 2

    tree_df["DIA_squared"] = tree_df["DIA_squared"].where(tree_df["STATUSCD"] == statuscd)

    dsquared = tree_df.groupby("PLT_CN_CONDID")["DIA_squared"].sum()
    n = tree_df.groupby("PLT_CN_CONDID")["DIA_squared"].count()

    QMD = np.sqrt(dsquared / n)
    QMD.name = "QMD"
    return QMD


def do_tree_level_calculations(tree_df: pd.DataFrame, cond_data: pd.DataFrame) -> pd.DataFrame:
    """Perform tree-level calculations to calculate new variables (biomass and QMD)
    at the plot-condition level.

    Inputs:
    tree_df (pd.DataFrame): DataFrame containing tree data
    cond_data (pd.DataFrame): DataFrame containing condition data

    Outputs:
    cond_data (pd.DataFrame): DataFrame containing condition data with new variables (biomass and QMD)

    """
    QMD = calculate_qmd(tree_df)
    cond_agb = calculate_biomass(
        tree_df=tree_df,
        live_tree=True,
        var_input="CARBON_AG",
        var_output="unadj_ag_biomass_t_per_ha",
    )

    cond_bgb = calculate_biomass(
        tree_df=tree_df,
        live_tree=True,
        var_input="CARBON_BG",
        var_output="unadj_bg_biomass_t_per_ha",
    )

    cond_agb_dead = calculate_biomass(
        tree_df=tree_df,
        live_tree=False,
        var_input="CARBON_AG",
        var_output="unadj_ag_dead_biomass_t_per_ha",
    )

    cond_bgb_dead = calculate_biomass(
        tree_df=tree_df,
        live_tree=False,
        var_input="CARBON_BG",
        var_output="unadj_bg_dead_biomass_t_per_ha",
    )

    cond_data = cond_data.join(
        other=QMD,
        how="left",
        on="PLT_CN_CONDID",
    )
    cond_data = cond_data.join(
        other=cond_agb,
        how="left",
        on="PLT_CN_CONDID",
    )

    cond_data = cond_data.join(
        other=cond_bgb,
        how="left",
        on="PLT_CN_CONDID",
    )

    cond_data = cond_data.join(
        other=cond_agb_dead,
        how="left",
        on="PLT_CN_CONDID",
    )

    cond_data = cond_data.join(
        other=cond_bgb_dead,
        how="left",
        on="PLT_CN_CONDID",
    )

    for var in [
        "ag_biomass_t_per_ha",
        "bg_biomass_t_per_ha",
        "ag_dead_biomass_t_per_ha",
        "bg_dead_biomass_t_per_ha",
    ]:
        cond_data["adj_" + var] = cond_data["unadj_" + var] / cond_data["CONDPROP_UNADJ"]

    return cond_data


def add_forest_group(cond_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add forest group information to the plot-condition data.

    Inputs:
    cond_data (pd.DataFrame): DataFrame containing condition data

    Outputs:
    cond_data (pd.DataFrame): DataFrame containing condition data with new variable (FORTYPCD_GRP)

    """
    try:
        forestgroup_crosswalk = pd.read_csv(dir_lookups + "forest_group_crosswalk.csv")
    except FileNotFoundError:
        raise FileNotFoundError("Forest group crosswalk file not found.")

    # Merge cond_data with forestgroup_crosswalk to get the Forest group code
    cond_data = cond_data.merge(
        forestgroup_crosswalk[["Forest type code", "Forest group code"]],
        how="left",
        left_on="FORTYPCD",
        right_on="Forest type code",
    )

    # Rename the 'Forest group code' column to 'FORTYPCD_GRP'
    cond_data.rename(columns={"Forest group code": "FORTYPCD_GRP"}, inplace=True)

    return cond_data


def add_ecosection(cond_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add ecosection information to the condition data.
    Inputs:
    cond_data (pd.DataFrame): DataFrame containing condition data

    Outputs:
    cond_data (pd.DataFrame): DataFrame containing condition data with new variable (Ecosection)
    """
    try:
        ecosection_crosswalk = pd.read_csv(dir_lookups + "ecosection_lookup.csv")
    except FileNotFoundError:
        raise FileNotFoundError("Ecosection crosswalk file not found.")

    ecosection_crosswalk["Ecosubsection"] = ecosection_crosswalk["Ecosubsection"].astype(str)
    cond_data["ECOSUBCD"] = cond_data["ECOSUBCD"].astype(str)
    # Merge cond_data with forestgroup_crosswalk to get the Forest group code
    cond_data = cond_data.merge(
        ecosection_crosswalk, how="left", left_on="ECOSUBCD", right_on="Ecosubsection"
    )

    return cond_data


def process_fia_data(state) -> pd.DataFrame:
    """Process FIA data for a given state. This function loads the data, performs tree-level calculations,
    identifies disturbances, and merges datasets to calculate new variables at the plot-condition level.

    Inputs:
    state (str): State abbreviation

    Outputs:
    cond_data (pd.DataFrame): DataFrame containing condition data with new variables (biomass, QMD, disturbances)
    """

    # Load data

    logging.info("Loading data")
    try:
        [cond_data, tree_data] = load_data(state)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading data for {state}: {e}")

    # Do tree-level calculations
    logging.info("Doing tree-level calculations data")
    cond_data = do_tree_level_calculations(tree_data, cond_data)

    # Identify disturbances based on disturbance code data
    logging.info("Identifying disturbances")

    cond_data = identify_disturbances(
        cond_data,
        dstrb_cds=[30, 31, 32],
        output_varname="fire_happened",
        output_varname2="Most_recent_fire_year",
    )
    cond_data = identify_disturbances(
        cond_data,
        dstrb_cds=[10, 11, 12],
        output_varname="insects_happened",
        output_varname2="Most_recent_insect_year",
    )
    cond_data = identify_disturbances(
        cond_data,
        dstrb_cds=[20, 21, 22],
        output_varname="disease_happened",
        output_varname2="Most_recent_disease_year",
    )

    cond_data = identify_disturbances(
        cond_data,
        dstrb_cds=[52],
        output_varname="wind_happened",
        output_varname2="Most_recent_wind_year",
    )

    cond_data = identify_disturbances(
        cond_data,
        dstrb_cds=[54],
        output_varname="drought_happened",
        output_varname2="Most_recent_drought_year",
    )

    cond_data = identify_harvest(cond_data, trtcds=[10])

    # Determine higher-level ecosection and forest group data
    cond_data = add_ecosection(cond_data)
    cond_data = add_forest_group(cond_data)

    return cond_data


def identify_disturbances(
    cond_data: pd.DataFrame,
    dstrb_cds: list,
    output_varname: str = "fire_happened",
    output_varname2: str = "Most_recent_fire_year",
) -> pd.DataFrame:
    """Identify disturbances based on disturbance code data.

    Inputs:
    cond_data (pd.DataFrame): DataFrame containing condition data
    dstrb_cds (list): List of disturbance codes for fire disturbances
    * 30, 31, and 32 are fire disturbance codes
    * 10, 11, and 12 are insect disturbance codes
    output_varname (str): Name of the output variable for disturbance occurrence
    output_varname2 (str): Name of the output variable for most recent disturbance year

    Outputs:
    df (pd.DataFrame): DataFrame containing condition data with new variables about disturbances
    """

    # Calculate whether fire disturbance recorded for this plot
    disturbance_happened = (
        (cond_data["DSTRBCD1"].isin(dstrb_cds))
        + (cond_data["DSTRBCD2"].isin(dstrb_cds))
        + (cond_data["DSTRBCD3"].isin(dstrb_cds))
    )
    cond_data[output_varname] = disturbance_happened

    # Process years of disturbance
    cond_data["DSTRBYR1"] = np.where(
        cond_data["DSTRBYR1"] > INVALID_YEAR_THRESHOLD, NAN_DSTRBYR, cond_data["DSTRBYR1"]
    )
    cond_data["DSTRBYR2"] = np.where(
        cond_data["DSTRBYR2"] > INVALID_YEAR_THRESHOLD, NAN_DSTRBYR, cond_data["DSTRBYR2"]
    )
    cond_data["DSTRBYR3"] = np.where(
        cond_data["DSTRBYR3"] > INVALID_YEAR_THRESHOLD, NAN_DSTRBYR, cond_data["DSTRBYR3"]
    )

    # Calculate when most recent fire disturbance occurred
    cond_data[output_varname2] = np.nan

    most_recent_disturbance_years = []
    for i, dstrbcd1 in enumerate(cond_data["DSTRBCD1"]):
        yr = 0
        if disturbance_happened.values[i]:
            if np.isin(dstrbcd1, dstrb_cds):
                yr1 = cond_data["DSTRBYR1"].values[i]
                if yr1 > yr:
                    yr = yr1
            if np.isin(cond_data["DSTRBCD2"].values[i], dstrb_cds):
                yr2 = cond_data["DSTRBYR2"].values[i]
                if yr2 > yr:
                    yr = yr2
            if np.isin(cond_data["DSTRBCD3"].values[i], dstrb_cds):
                yr3 = cond_data["DSTRBYR3"].values[i]
                if yr3 > yr:
                    yr = yr3
        most_recent_disturbance_years.append(yr)

    cond_data[output_varname2] = most_recent_disturbance_years

    return cond_data


def identify_harvest(
    cond_data: pd.DataFrame,
    trtcds: list,
    output_varname: str = "harvest_happened",
    output_varname2: str = "Most_recent_harvest_year",
) -> pd.DataFrame:
    """
    Identify harvest disturbances based on disturbance code data.
    Inputs:
    cond_data (pd.DataFrame): DataFrame containing condition data
    trtcds (list): List of treatment codes for treatments (10 = harvest)
    output_varname (str): Name of the output variable for treatment occurrence
    output_varname2 (str): Name of the output variable for most recent treatment year

    Outputs:
    cond_data (pd.DataFrame): DataFrame containing condition data with new variables

    """

    # Calculate whether fire disturbance recorded for this plot
    treatment_happened = (
        # Fire damage (from crown and ground fire  either prescribed or natural)
        +(cond_data["TRTCD1"].isin(trtcds))
        + (cond_data["TRTCD2"].isin(trtcds))
        + (cond_data["TRTCD3"].isin(trtcds))
    )
    cond_data[output_varname] = treatment_happened

    # Process years of disturbance
    cond_data["TRTYR1"] = np.where(
        cond_data["TRTYR1"] > INVALID_YEAR_THRESHOLD, NAN_DSTRBYR, cond_data["TRTYR1"]
    )
    cond_data["TRTYR2"] = np.where(
        cond_data["TRTYR2"] > INVALID_YEAR_THRESHOLD, NAN_DSTRBYR, cond_data["TRTYR2"]
    )
    cond_data["TRTYR3"] = np.where(
        cond_data["TRTYR3"] > INVALID_YEAR_THRESHOLD, NAN_DSTRBYR, cond_data["TRTYR3"]
    )

    # Calculate when most recent fire disturbance occurred
    cond_data[output_varname2] = np.nan

    most_recent_treatment_years = []
    for i, trtcd1 in enumerate(cond_data["TRTCD1"]):
        yr = 0
        if treatment_happened.values[i]:
            if np.isin(trtcd1, trtcds):
                yr1 = cond_data["TRTYR1"].values[i]
                if yr1 > yr:
                    yr = yr1
            if np.isin(cond_data["TRTCD2"].values[i], trtcds):
                yr2 = cond_data["TRTYR2"].values[i]
                if yr2 > yr:
                    yr = yr2
            if np.isin(cond_data["TRTCD3"].values[i], trtcds):
                yr3 = cond_data["TRTYR3"].values[i]
                if yr3 > yr:
                    yr = yr3
        most_recent_treatment_years.append(yr)

    cond_data[output_varname2] = most_recent_treatment_years

    return cond_data


def calculate_disturbance_over_time(
    ds: xr.Dataset,
    cond_data: pd.DataFrame,
    dstrb_cds: list,
    var_input: str,
) -> xr.DataArray:
    """
    Calculate fire disturbances over time for each plot-condition.
    Inputs:
    ds (pd.DataFrame): DataFrame containing plot information
    cond_data (pd.DataFrame): DataFrame containing condition data
    var_input (str): Name of the input variable for disturbance occurrence
    dstrb_cds (list): List of disturbance codes

    Outputs:
    da (xr.DataArray) indicating years when disturbance occurred
    """
    years = ds["year"]
    disturbance_occurred = np.zeros([np.size(ds["plotid"].values), np.size(years)])
    disturbance_occurred[:] = np.nan
    disturbance_happened = np.zeros([np.size(ds["plotid"].values)])

    for i, plot in enumerate(ds["plotid"].values):
        # Was a fire disturbance code EVER recorded for this plot?
        disturbance_happened[i] = cond_data[(cond_data["STATE_PLOT"] == plot)][var_input].sum() > 0

        # When fire disturbance code was recorded
        plot_data_with_disturbance = cond_data[
            (cond_data["STATE_PLOT"] == plot) & (cond_data[var_input])
        ]

        if len(plot_data_with_disturbance) > 0:
            dstrbcd1_yr = (plot_data_with_disturbance["DSTRBCD1"].isin(dstrb_cds)) * (
                plot_data_with_disturbance["DSTRBYR1"]
            )
            dstrbcd2_yr = (plot_data_with_disturbance["DSTRBCD2"].isin(dstrb_cds)) * (
                plot_data_with_disturbance["DSTRBYR2"]
            )
            dstrbcd3_yr = (plot_data_with_disturbance["DSTRBCD3"].isin(dstrb_cds)) * (
                plot_data_with_disturbance["DSTRBYR3"]
            )
            disturbance_years = np.unique(
                np.concatenate([dstrbcd1_yr.values, dstrbcd2_yr.values, dstrbcd3_yr.values])
            )
            disturbance_years = disturbance_years[~np.isnan(disturbance_years)]
            disturbance_occurred[i, :] = ds["year"].isin(disturbance_years)

    dstrb_cds_str = ",".join(map(str, dstrb_cds))

    fires_da = xr.DataArray(
        data=disturbance_occurred,
        dims=["plotid", "year"],
        coords=dict(
            plotid=ds["plotid"],
            year=years,
        ),
    ).assign_attrs(
        description="Indicates if " + dstrb_cds_str + " disturbance code was recorded in that year"
    )

    return fires_da


def calculate_treatment_over_time(
    ds: xr.Dataset,
    cond_data: pd.DataFrame,
    trtcds: list,
    input_var: str,
) -> xr.DataArray:
    """
    Calculate harvest disturbances over time for each plot-condition.
    Inputs:
    ds (pd.DataFrame): DataFrame containing plot information
    cond_data (pd.DataFrame): DataFrame containing condition data
    trtcds (list): List of treatment codes

    Outputs:
    da (xr.DataArray) indicating years when treatment occurred
    1 if harvest disturbance code was recorded in that year
    0 if harvest disturbance code was not recorded in that year
    NaN if harvest disturbance code was not recorded for that plot
    """
    years = ds["year"]
    harvest_occurred = np.zeros([np.size(ds["plotid"].values), np.size(years)])
    harvest_occurred[:] = np.nan

    for i, plot in enumerate(ds["plotid"].values):
        plot_data_with_harvest = cond_data[
            (cond_data["STATE_PLOT"] == plot) & (cond_data[input_var])
        ]
        if len(plot_data_with_harvest) > 0:
            trtcd1 = plot_data_with_harvest["TRTCD1"]
            trtcd2 = plot_data_with_harvest["TRTCD2"]
            trtcd3 = plot_data_with_harvest["TRTCD3"]

            trtcd1_yr = (trtcd1.isin(trtcds)) * (plot_data_with_harvest["TRTYR1"])
            trtcd2_yr = (trtcd2.isin(trtcds)) * (plot_data_with_harvest["TRTYR2"])
            trtcd3_yr = (trtcd3.isin(trtcds)) * (plot_data_with_harvest["TRTYR3"])
            trt_years = np.unique(
                np.concatenate([trtcd1_yr.values, trtcd2_yr.values, trtcd3_yr.values])
            )  # +dstrbcd2.values+dstrbcd3.values
            trt_years = trt_years[~np.isnan(trt_years)]
            harvest_occurred[i, :] = ds["year"].isin(trt_years)

    harvest_da = xr.DataArray(
        data=harvest_occurred,
        dims=["plotid", "year"],
        coords=dict(
            plotid=ds["plotid"],
            year=years,
        ),
    )

    da = harvest_da.assign_attrs(
        description="Indicates if harvest treatment code was recorded in that year"
    )

    return da


def create_restructured_dataset(cond_data: pd.DataFrame = None) -> xr.Dataset:
    """Create a restructured dataset with plot information.
    Inputs:
    cond_data (pd.DataFrame): DataFrame containing condition data

    Outputs:
    ds (xarray.Dataset): Dataset containing plot-level information (latitude and longitude)"""

    all_plots = np.unique(cond_data["STATE_PLOT"])

    lats = cond_data.groupby("STATE_PLOT")["LAT"].mean()
    lats_da = xr.DataArray(
        data=lats,
        dims=["plotid"],
        coords=dict(
            plotid=all_plots,
        ),
    )
    lons = cond_data.groupby("STATE_PLOT")["LON"].mean()
    lons_da = xr.DataArray(
        data=lons,
        dims=["plotid"],
        coords=dict(
            plotid=all_plots,
        ),
    )

    ds = lats_da.to_dataset(name="lat")
    ds["lon"] = lons_da

    return ds


def sum_var_over_time(
    ds: xr.Dataset,
    cond_data: pd.DataFrame,
    var: str = "unadj_ag_biomass_t_per_ha",
    years_range: list = [MIN_YEAR, MAX_YEAR],
    var_description=None,
    area_weighted=False,
) -> xr.DataArray:
    """
    Sum a variable over time for each plot-condition.
    Inputs:
    ds (pd.DataFrame): DataFrame containing plot information
    cond_data (pd.DataFrame): DataFrame containing condition data
    var (str): Variable to sum over time
    years_range (list): Range of years to consider
    var_description (str): Description of the variable
    area_weighted (bool): If True, calculate area-weighted sum

    Outputs:
    var_da (xarray.DataArray): DataArray containing the summed variable for each plot
    """

    # Create empty array to hold the variable data
    temp = cond_data.groupby("STATE_PLOT")["LON"].mean().to_frame(name="LON")

    years = np.arange(years_range[0], years_range[1])
    var_array = np.zeros([np.size(ds["plotid"].values), np.size(years)])
    var_array[:] = np.nan

    # Fill array with the variable data for each year
    if area_weighted:
        for i, year in enumerate(years):
            cond_data_1yr = cond_data[cond_data["MEASYEAR"] == year]
            var_1yr = (
                (cond_data_1yr[var] * cond_data_1yr["COND_FRAC_FOREST"])
                .groupby(cond_data_1yr["STATE_PLOT"])
                .sum()
            )
            var_1yr.name = var + "_area_weighted"
            temp2 = temp.join(
                other=var_1yr,
                how="left",
                on="STATE_PLOT",
            )
            var_array[:, i] = temp2[var + "_area_weighted"].values
    else:
        for i, year in enumerate(years):
            var_1yr = cond_data[cond_data["MEASYEAR"] == year].groupby("STATE_PLOT")[var].sum()
            temp2 = temp.join(
                other=var_1yr,
                how="left",
                on="STATE_PLOT",
            )
            var_array[:, i] = temp2[var].values

    # Turn array into xarray with labeled coordinates
    var_da = xr.DataArray(
        data=var_array,
        dims=["plotid", "year"],
        coords=dict(
            plotid=ds["plotid"],
            year=years,
        ),
    )

    # Add metadatadata and add to plot dataset
    if var_description is None:
        var_description = var

    var_da = var_da.assign_attrs(description=var_description)

    return var_da


def calculate_num_conditions(
    ds: xr.Dataset,
    cond_data: pd.DataFrame,
    var="PLT_CN",
    years_range=[RECENT_YEAR_THRESHOLD, MAX_YEAR],
) -> xr.DataArray:
    """
    Calculate the number of conditions for each plot over time.
    Inputs:
    ds (pd.DataFrame): DataFrame containing plot information
    cond_data (pd.DataFrame): DataFrame containing condition data
    var (str): Variable to count
    years_range (list): Range of years to consider

    Outputs:
    var_da (xarray.DataArray): DataArray containing counts for each plot
    """

    temp = cond_data.groupby("STATE_PLOT")["LON"].mean().to_frame(name="LON")

    years = np.arange(years_range[0], years_range[1])
    var_array = np.zeros([np.size(ds["plotid"].values), np.size(years)])
    var_array[:] = np.nan

    for i, year in enumerate(years):
        var_1yr = cond_data[cond_data["MEASYEAR"] == year].groupby("STATE_PLOT")[var].count()
        temp2 = temp.join(
            other=var_1yr,
            how="left",
            on="STATE_PLOT",
        )
        var_array[:, i] = temp2[var].values

    var_da = xr.DataArray(
        data=var_array,
        dims=["plotid", "year"],
        coords=dict(
            plotid=ds["plotid"],
            year=years,
        ),
    ).assign_attrs(description=var)

    return var_da


def choose_categorical_vars(cond_data: pd.DataFrame, var: str = "FORTYPCD_GRP") -> xr.DataArray:
    """
    Selects which categorical variable to use for each plot-condition. This deals with the fact
    that sometimes these categorical variables change over time, or differ across multiple conditions
    at a certain plot. We deal with this by selecting the selecting the categorical variable for the
    largest condition in the earliest year measured.

    We have one exception to this rule: if the categorical variable is FORTYPCD_GRP, we will not select
    the value if it is equal to 999. This is because this value indicates that the forest type code
    was "unstocked," so it doesn't give any information about what kind of forest would otherwise be there.

    Inputs:
    cond_data (pd.DataFrame): DataFrame containing condition data
    var (str): Variable to choose

    Outputs:
    result_da (xarray.DataArray): DataArray containing the chosen variable for each plot"""

    # Group by STATE_PLOT and find the earliest MEASYEAR for each group
    earliest_year = cond_data.groupby("STATE_PLOT")["MEASYEAR"].min()

    # Broadcast the earliest year back to the original dataframe
    cond_data = cond_data.assign(earliest_year=cond_data["STATE_PLOT"].map(earliest_year))

    # Filter the data to only include rows corresponding to the earliest year for each STATE_PLOT
    filtered_data = cond_data[cond_data["MEASYEAR"] == cond_data["earliest_year"]]

    if var == "FORTYPCD_GRP":
        filtered_data = filtered_data[filtered_data[var] != 999]

    # For each STATE_PLOT, select the row with the largest COND_FRAC_FOREST for the earliest year
    result = filtered_data.loc[
        filtered_data.groupby("STATE_PLOT")["COND_FRAC_FOREST"].idxmax()
    ].set_index("STATE_PLOT")[var]

    result_da = xr.DataArray(result, coords=[result.index.values], dims=["plotid"], name=var)

    return result_da


def count_categorical_vars(cond_data: pd.DataFrame, var: str = "FORTYPCD_GRP") -> xr.DataArray:
    """
    Counts the number of unique values for a categorical variable for each plot-condition. This is useful
    for identifying when assumptions in the function choose_categorical_vars() actually matter. Most of the time,
    we expect that the categorical variable will be the same for all conditions at a plot and for all years.

    Inputs:
    cond_data (pd.DataFrame): DataFrame containing condition data
    var (str): Variable to count

    Outputs:
    nunique (pd.Series): Series containing the number of unique values for the categorical variable for each plot
    """

    if var == "FORTYPCD_GRP":
        nunique = cond_data[cond_data[var] != 999].groupby("STATE_PLOT")[var].nunique()
    else:
        nunique = cond_data.groupby("STATE_PLOT")[var].nunique()

    result_da = xr.DataArray(nunique, coords=[nunique.index.values], dims=["plotid"], name=var)

    return result_da


def calculate_years_since_disturbance(ds: xr.Dataset, disturbance_var: str) -> xr.DataArray:
    """
    Calculate the number of years relative to the year in which fire occurred for each plot-condition.

    Inputs:
    ds (xr.Dataset): DataFrame containing plot information with fire disturbance data
    disturbance_var (str): Name of the variable indicating fire disturbance occurrence

    Outputs:
    years_after_disturbance (xr.DataArray): DataArray containing years since fire occurred

    """
    # Initialize years_after_disturbance with 100 for the first year
    years_after_disturbance = xr.full_like(ds[disturbance_var], fill_value=100)

    # Iterate over years to calculate years_after_disturbance
    for year_idx in range(1, len(ds["year"])):
        # Increment the years_after_disturbance from the previous year
        years_after_disturbance[:, year_idx] = years_after_disturbance[:, year_idx - 1] + 1

        # Reset to 0 where fires_occurred == 1
        years_after_disturbance[:, year_idx] = xr.where(
            ds[disturbance_var][:, year_idx] == 1, 0, years_after_disturbance[:, year_idx]
        )

    return years_after_disturbance


def calculate_biomass_pre_post(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate biomass before and after fire for each plot-condition.
    Inputs:
    ds (xr.Dataset): DataFrame containing plot information with fire disturbance data

    Outputs:
    ds (xr.Dataset): DataFrame containing plot information with new variables (postfire_biomass, prefire_biomass, postfire_measyear, prefire_measyear, fire_impact)
    """

    # Calculate when first and last biomass measurements occurred, relative to fire year
    first_biomass_obs = ds["years_after_fire"].where(~np.isnan(ds["biomass"])).min(dim="year")
    last_biomass_obs = ds["years_after_fire"].where(~np.isnan(ds["biomass"])).max(dim="year")

    # Filter to plots with at least one biomass obs before fire and one biomass obs after
    ds_2measurements = ds.where(first_biomass_obs < 0, drop=True).where(
        last_biomass_obs >= 0, drop=True
    )

    # Identify first measurement year after fire and most recent measurement year before fire
    years_with_measurements = ds_2measurements["years_after_fire"].where(
        ~np.isnan(ds_2measurements["biomass"])
    )
    postfire_measyear = years_with_measurements.where(
        ds_2measurements["years_after_fire"] >= 0
    ).min(dim="year")
    prefire_measyear = years_with_measurements.where(ds_2measurements["years_after_fire"] < 0).max(
        dim="year"
    )

    # Calculate biomass at those two measurement years pre/post fire
    postfire_biomass = (
        ds_2measurements["biomass"]
        .where(ds_2measurements["years_after_fire"] == postfire_measyear)
        .mean(dim="year")
    )
    prefire_biomass = (
        ds_2measurements["biomass"]
        .where(ds_2measurements["years_after_fire"] == prefire_measyear)
        .mean(dim="year")
    )

    ds_2measurements["postfire_biomass"] = postfire_biomass
    ds_2measurements["prefire_biomass"] = prefire_biomass
    ds_2measurements["postfire_measyear"] = postfire_measyear
    ds_2measurements["prefire_measyear"] = prefire_measyear

    ds_2measurements["fire_impact"] = (
        ds_2measurements["postfire_biomass"] / ds_2measurements["prefire_biomass"]
    ) * 100 - 100

    return ds_2measurements


def calculate_biomass_deltas(ds: xr.Dataset, biomass_var="biomass") -> xr.Dataset:
    biomass = ds[biomass_var]

    last_index = biomass.notnull().cumsum("year").argmax("year").load()

    # Create a masked version to remove the last non-NaN element (for second-to-last)
    masked_biomass = biomass.copy()
    masked_biomass = masked_biomass.where(masked_biomass.year != biomass.year[last_index], np.nan)

    # Get the second-to-last non-NaN index
    second_to_last_index = masked_biomass.notnull().cumsum("year").argmax("year").load()

    biomass_end = biomass[:, last_index]
    biomass_start = biomass[:, second_to_last_index]
    biomass_delta = biomass_end.values - biomass_start.values

    # Get year at second-to-last index
    measyear_1 = biomass["year"].values[second_to_last_index]
    measyear_2 = biomass["year"].values[last_index]

    ds["biomass_delta"] = (("plotid"), biomass_delta)
    ds["biomass_most_recent"] = (("plotid"), biomass_end.values)

    ds["measyear_2"] = (("plotid"), measyear_2)
    ds["measyear_1"] = (("plotid"), measyear_1)

    return ds


def filter_cond_data(cond_data: pd.DataFrame) -> pd.DataFrame:
    """Filter data to only include forested plots and recent measurements

    # Criteria for including a plot:
    * Plot has two measurements after RECENT_YEAR_THRESHOLD
    * Plot has only one condition code
    * Forest type code did not change between measurement intervals

    Inputs:
    cond_data (pd.DataFrame): DataFrame containing condition data

    Outputs:
    cond_data_filtered (pd.DataFrame): DataFrame containing filtered condition data with new variable (COND_FRAC_FOREST)
    """

    # Filter: Accessible forest land
    is_forest = cond_data["COND_STATUS_CD"] == 1
    frac_nonforest = len(cond_data[~is_forest]) / len(cond_data)

    # Filter: Measured after 1981
    is_recent = cond_data["MEASYEAR"] >= MIN_YEAR  # RECENT_YEAR_THRESHOLD
    frac_old_measurements = len(cond_data[~is_recent]) / len(cond_data)

    # Filter: Plot was also measured after systematic inventory started
    recent_plots = np.unique(
        cond_data[cond_data["MEASYEAR"] >= RECENT_YEAR_THRESHOLD]["STATE_PLOT"]
    )
    measured_recently = cond_data["STATE_PLOT"].isin(recent_plots)

    logging.info("Fraction nonforest: " + str(frac_nonforest))
    logging.info("Fraction old measurements: " + str(frac_old_measurements))

    # Filter: Plot is the right kind of inventory plot
    filter_kind_code = (cond_data["KINDCD"] != 0) * (cond_data["KINDCD"] != 4)

    cond_data_filtered = cond_data[is_forest & is_recent & measured_recently & filter_kind_code]

    # After filtering, calculate the fraction of total forest area in each condition
    frac_total = cond_data_filtered.groupby(["STATE_PLOT", "MEASYEAR"])["CONDPROP_UNADJ"].transform(
        "sum"
    )

    cond_data_filtered = cond_data_filtered.assign(
        COND_FRAC_FOREST=cond_data_filtered["CONDPROP_UNADJ"] / frac_total
    )

    test1 = cond_data_filtered["LAT"].groupby(cond_data_filtered["STATE_PLOT"]).nunique()
    if test1.max() > 1:
        raise ValueError(
            str(test1[test1 > 1].count()) + " plots have more than one latitude recorded."
        )

    test1 = cond_data_filtered["LON"].groupby(cond_data_filtered["STATE_PLOT"]).nunique()
    if test1.max() > 1:
        raise ValueError(
            str(test1[test1 > 1].count()) + " plots have more than one longitude recorded."
        )

    return cond_data_filtered


def run_through_restructuring(cond_data_recent: pd.DataFrame) -> xr.Dataset:
    """
    Run through the restructuring process for the condition data.

    Inputs:
    cond_data_recent (pd.DataFrame): DataFrame containing condition data

    Outputs:
    ds (xarray.Dataset): Dataset containing plot-level information with new variables. Dimensions are [plotid, year]
    """
    ds = create_restructured_dataset(cond_data_recent)

    ds["num_forest_conditions_recorded"] = calculate_num_conditions(
        ds, cond_data_recent, years_range=[MIN_YEAR, MAX_YEAR]
    )

    # Calculate disturbances over time
    ds["harvest_occurred"] = calculate_treatment_over_time(
        ds,
        cond_data_recent,
        trtcds=[10],
        input_var="harvest_happened",
    )

    ds["fires_occurred"] = calculate_disturbance_over_time(
        ds,
        cond_data_recent,
        var_input="fire_happened",
        dstrb_cds=[30, 31, 32],
    )

    ds["insects_occurred"] = calculate_disturbance_over_time(
        ds,
        cond_data_recent,
        var_input="insects_happened",
        dstrb_cds=[10, 11, 12],
    )

    ds["disease_occurred"] = calculate_disturbance_over_time(
        ds,
        cond_data_recent,
        var_input="disease_happened",
        dstrb_cds=[20, 21, 22],
    )

    ds["drought_occurred"] = calculate_disturbance_over_time(
        ds,
        cond_data_recent,
        var_input="drought_happened",
        dstrb_cds=[54],
    )

    ds["wind_occurred"] = calculate_disturbance_over_time(
        ds,
        cond_data_recent,
        var_input="wind_happened",
        dstrb_cds=[52],
    )

    # Calculate variables over time
    ds["biomass_unadj"] = sum_var_over_time(
        ds,
        cond_data_recent,
        var="unadj_ag_biomass_t_per_ha",
        var_description="Plot-level biomass",
        years_range=[MIN_YEAR, MAX_YEAR],
        area_weighted=True,
    )

    ds["biomass"] = sum_var_over_time(
        ds,
        cond_data_recent,
        var="adj_ag_biomass_t_per_ha",
        var_description="Plot-level biomass",
        years_range=[MIN_YEAR, MAX_YEAR],
        area_weighted=True,
    )

    # Calculate timeseries for variables which can change over time
    varlist = [
        "unadj_bg_biomass_t_per_ha",
        "unadj_ag_dead_biomass_t_per_ha",
        "unadj_bg_dead_biomass_t_per_ha",
        "QMD",
        "STDAGE",
        "BALIVE",
        "LIVE_CANOPY_CVR_PCT",
        "LIVE_MISSING_CANOPY_CVR_PCT",
        "LAND_COVER_CLASS_CD",
        "NBR_LIVE_STEMS",
        "STDSZCD",
        "SITECLCD",
        "CARBON_UNDERSTORY_AG",
        "CARBON_UNDERSTORY_BG",
        "CARBON_DOWN_DEAD",
        "CONDPROP_UNADJ",
    ]

    for var in varlist:
        ds[var] = sum_var_over_time(ds, cond_data_recent, var=var, area_weighted=True)

    # Calculate continuous variables which do not change over time
    # Some plots have these quantities change over time. TK need to  have a way of dealing with this systematically, but for now just take the first value
    varlist = ["SLOPE", "ASPECT", "ELEV"]

    for var in varlist:
        ds[var + "_preliminary"] = xr.DataArray(
            data=cond_data_recent.groupby("STATE_PLOT")[var].first(),
            dims=["plotid"],
            coords=dict(
                plotid=ds["plotid"],
            ),
        )

    # Calculate categorical variables which (mostly) do not change over time
    varlist = [
        "FORTYPCD",
        "RESERVCD",
        "FORTYPCD_GRP",
        "OWNGRPCD",
        "ECOSUBCD",
        "Ecosection",
        "Ecoprovince",
        "STDORGCD",
        "PHYSCLCD",
        "STATECD",
    ]

    for var in varlist:
        ds[var] = choose_categorical_vars(cond_data_recent, var=var)
        ds[var + "_counts"] = count_categorical_vars(cond_data_recent, var=var)

    # Calculate most recent fire and harvest years
    ds["most_recent_fire_year"] = xr.DataArray(
        data=cond_data_recent.groupby("STATE_PLOT")["Most_recent_fire_year"].max(),
        dims=["plotid"],
        coords=dict(
            plotid=ds["plotid"],
        ),
    ).assign_attrs(description="Most recent fire year")

    ds["most_recent_harvest_year"] = xr.DataArray(
        data=cond_data_recent.groupby("STATE_PLOT")["Most_recent_harvest_year"].max(),
        dims=["plotid"],
        coords=dict(
            plotid=ds["plotid"],
        ),
    ).assign_attrs(description="Most recent harvest year")

    ds["years_after_fire"] = calculate_years_since_disturbance(ds, disturbance_var="fires_occurred")
    ds["years_after_harvest"] = calculate_years_since_disturbance(
        ds, disturbance_var="harvest_occurred"
    )
    ds["years_after_drought"] = calculate_years_since_disturbance(
        ds, disturbance_var="drought_occurred"
    )
    ds["years_after_insects"] = calculate_years_since_disturbance(
        ds, disturbance_var="insects_occurred"
    )
    ds["years_after_disease"] = calculate_years_since_disturbance(
        ds, disturbance_var="disease_occurred"
    )
    ds["years_after_wind"] = calculate_years_since_disturbance(ds, disturbance_var="wind_occurred")

    ds = calculate_biomass_deltas(ds)

    return ds
