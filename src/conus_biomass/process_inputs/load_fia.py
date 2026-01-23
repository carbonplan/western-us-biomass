import os

import pandas as pd

from conus_biomass.dir_info import dir_fia_csvs

try:
    ref_species_df = pd.read_csv(dir_fia_csvs + "fiadb_reference/REF_SPECIES.csv")
except FileNotFoundError:
    raise FileNotFoundError("REF_SPECIES.csv file not found.")


def load_fia_table(state: str = "PA", table: str = "TREE") -> pd.DataFrame:
    """Load a table from the FIA database for a given state.
    Inputs:
    state (str): State abbreviation
    table (str): Table name

    Outputs:
    df (pd.DataFrame): DataFrame containing the table data"""

    fname = os.path.join(dir_fia_csvs, "csv_fiadb_by_state", state,  f"{state}_{table}.csv")
    if state == "entire":
        fname = os.path.join(dir_fia_csvs, "csv_fiadb_entire", f"ENTIRE_{table}.csv")

    try:
        df = pd.read_csv(fname, low_memory=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"{table} table not found for {state}")

    return df


def load_data(state: str = "CA") -> list[pd.DataFrame]:
    """Load data from FIA database for a given state. This function works when using plot,
    condition, and tree CSVs downloaded from FIADB.

    Inputs:
    state (str): State abbreviation

    Outputs:
    cond_data (pd.DataFrame): Condition data with plot information
    tree_data (pd.DataFrame): Tree data with plot information

    """

    tree_cols = [
        "PLT_CN",
        "CONDID",
        # "SUBP",
        # "TREE",
        "STATUSCD",
        # "DRYBIO_AG",
        "CARBON_AG",
        "CARBON_BG",
        "TPA_UNADJ",
        "DIA",
        # "SPCD",
    ]

    cond_cols = [
        "ADFORCD",
        "ASPECT",
        "BALIVE",
        "CARBON_DOWN_DEAD",
        "CARBON_UNDERSTORY_AG",
        "CARBON_UNDERSTORY_BG",
        "CONDID",
        "CONDPROP_UNADJ",
        "COND_STATUS_CD",
        "DSTRBCD1",
        "DSTRBCD2",
        "DSTRBCD3",
        "DSTRBYR1",
        "DSTRBYR2",
        "DSTRBYR3",
        "DWM_FUELBED_TYPCD",
        "FLDTYPCD",
        "FORTYPCD",
        "LAND_COVER_CLASS_CD",
        "LIVE_CANOPY_CVR_PCT",
        "LIVE_MISSING_CANOPY_CVR_PCT",
        "MAPDEN",
        "NBR_LIVE_STEMS",
        "OWNCD",
        "OWNGRPCD",
        "PHYSCLCD",
        "PLT_CN",
        "PROP_BASIS",
        "RESERVCD",
        "SITECLCD",
        "SLOPE",
        "STDAGE",
        "STDORGCD",
        "STDSZCD",
        "TRTCD1",
        "TRTCD2",
        "TRTCD3",
        "TRTYR1",
        "TRTYR2",
        "TRTYR3",
    ]

    plot_cols = [
        "CN",
        "COUNTYCD",
        "ELEV",
        "INVYR",
        "KINDCD",
        "LAT",
        "LON",
        "MACRO_BREAKPOINT_DIA",
        "MEASYEAR",
        "PLOT",
        "PLOT_NONSAMPLE_REASN_CD",
        "PLOT_STATUS_CD",
        "STATECD",
        "UNITCD",
    ]

    plotgeom_cols = [
        "CN",
        "ECOSUBCD",
    ]

    # pop_stratum_cols = [
    #    "CN",
    #    "ADJ_FACTOR_MACR",
    #    "ADJ_FACTOR_SUBP",
    # ]

    # pop_plot_stratum_cols = ["PLT_CN", "STRATUM_CN"]

    # Load data
    tree_df = load_fia_table(state=state, table="TREE")[tree_cols]
    cond_df = load_fia_table(state=state, table="COND")[cond_cols]
    plot_df = load_fia_table(state=state, table="PLOT")[plot_cols]
    plotgeom_df = load_fia_table(state=state, table="PLOTGEOM")[plotgeom_cols]
    # pop_stratum_df = load_fia_table(state=state, table="POP_STRATUM")[pop_stratum_cols]
    # pop_plot_stratum_df = load_fia_table(state=state, table="POP_PLOT_STRATUM_ASSGN")[
    #    pop_plot_stratum_cols
    # ]

    cond_df["PLT_CN_CONDID"] = (
        cond_df["PLT_CN"].astype(int).astype(str) + "_" + cond_df["CONDID"].astype(int).astype(str)
    )

    tree_df["PLT_CN_CONDID"] = (
        tree_df["PLT_CN"].astype(int).astype(str) + "_" + tree_df["CONDID"].astype(int).astype(str)
    )

    # Merge plot and plot geometry data
    plot_df = pd.merge(plot_df, plotgeom_df, how="left", left_on="CN", right_on="CN")

    # Merge condition data with plot data
    cond_data = pd.merge(plot_df, cond_df, how="left", left_on="CN", right_on="PLT_CN").drop(
        columns="CN"
    )

    # Merge stratum data with plot data
    # cond_data = pd.merge(
    #    cond_data, pop_plot_stratum_df, how="left", left_on="PLT_CN", right_on="PLT_CN"
    # )

    # cond_data = pd.merge(
    #    cond_data, pop_stratum_df, how="left", left_on="STRATUM_CN", right_on="CN"
    # ).drop(columns="CN")

    # Create a new plotid index to deal with some states have different plots with the same plot ids
    cond_data["STATE_PLOT"] = (
        cond_data["STATECD"].astype(str)
        + "_"
        + cond_data["COUNTYCD"].astype(str)
        + "_"
        + cond_data["PLOT"].astype(str)
    )

    return [cond_data, tree_df]
