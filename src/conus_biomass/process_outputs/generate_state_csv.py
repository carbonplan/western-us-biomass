import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import xarray as xr

from conus_biomass import dir_info
from conus_biomass.process_outputs.postprocess_output import get_fname_processed_biomass
from conus_biomass.settings import STATE_LIST

logging.basicConfig(level=logging.INFO)

SHP = gpd.read_file(dir_info.dir_shp + "cb_2018_us_state_20m.shp")


def get_crs(dir_model_input: str = dir_info.dir_model_input):
    ds = xr.open_zarr(dir_model_input + "aspect.zarr")
    crs_this_res = ds.spatial_ref.crs_wkt

    res_y = np.abs(ds["y"].values[1] - ds["y"].values[0])
    res_x = np.abs(ds["x"].values[1] - ds["x"].values[0])

    if res_y == res_x:
        grid_res = res_y
    else:
        grid_res = np.nan

    return crs_this_res, grid_res


def get_output_biomass(
    year: int,
    crs: str,
    dir_processed_model_output: str = dir_info.dir_model_output[:-1] + "_processed/",
):

    fname = get_fname_processed_biomass(
        dir_processed_model_output=dir_processed_model_output, year=year
    )

    da = xr.open_dataset(fname)["predicted_biomass"]

    # Normalize the CRS so GDAL can parse it consistently
    clean_crs = pyproj.CRS.from_wkt(crs).to_wkt()

    da = da.rio.write_crs(clean_crs)

    return da


def get_FRF_area(year: int, crs: str, dir_model_input: str = dir_info.dir_model_input):
    da = xr.open_dataset(dir_model_input + "all_variables.nc")["forest_remaining_forest"].sel(
        year=year
    )
    da = da.rio.write_crs(crs)
    return da


def calculate_total_carbon_stock(biomass_forest, grid_res: int = 1000):
    if grid_res == 100:
        gridcell_ha = 1
    elif grid_res == 1000:
        gridcell_ha = 100

    MMT = (biomass_forest * gridcell_ha).sum().load().values.flatten()[0] / 1e6
    return MMT


def clip_to_shape(da, shp):
    if shp.crs != da.rio.crs:
        shp = shp.to_crs(da.rio.crs)
    clipped = da.rio.clip(shp.geometry, shp.crs, drop=True)
    return clipped


def calculate_state_stocks_from_gridded(
    year: int, crs_this_res: str, grid_res: int, state_list: list = STATE_LIST
):
    biomass_states = []
    biomass_forest = get_output_biomass(year=year, crs=crs_this_res)
    for state in state_list:
        logging.info(state)
        shp_state = SHP[SHP["STUSPS"] == state]
        biomass_forest_state = clip_to_shape(da=biomass_forest, shp=shp_state)

        MMT = calculate_total_carbon_stock(biomass_forest_state, grid_res=grid_res)
        logging.info(MMT)
        biomass_states.append(MMT)

    df_year = pd.DataFrame(
        {
            "state_abbr": state_list,
            "live_biomass_MMT": biomass_states,
        }
    )
    df_year["year"] = year

    return df_year


def main(
    dir_output_csv: str = dir_info.dir_model_output[:-1] + "_processed/",
    years: np.array = np.arange(2005, 2023),
):
    crs_this_res, grid_res = get_crs()

    for i, year in enumerate(years):
        logging.info(year)
        df_year = calculate_state_stocks_from_gridded(
            year=year, crs_this_res=crs_this_res, grid_res=grid_res
        )

        if i == 0:
            df_all = df_year
        else:
            df_all = pd.concat([df_all, df_year])
        df_all["estimate_type"] = "our_study"
        df_all.to_csv(dir_output_csv + "MMTC_our_study.csv")


if __name__ == "__main__":
    main()
