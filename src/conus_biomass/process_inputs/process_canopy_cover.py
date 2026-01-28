import gc

import numpy as np
import rioxarray as rioxr
import xarray as xr
from rasterio.enums import Resampling

from conus_biomass import dir_info


def get_tree_cover_fname(dir_base=dir_info.dir_canopy_cover, year=2005):
    fname = f"{dir_base}nlcd_tcc_CONUS_{year}_v2023-5_wgs84/nlcd_tcc_conus_wgs84_v2023-5_{year}0101_{year}1231.tif"
    return fname


def process_one_year(year: int, dir_out: str, ds_ref_grid: xr.DataArray):
    fname = get_tree_cover_fname(year=year)

    with rioxr.open_rasterio(fname) as tcc:  # , chunks={"x": 1024, "y": 1024}) as tcc:
        tcc = tcc[0, :, :]
        tcc_on_ref_grid = tcc.rio.reproject_match(ds_ref_grid, resampling=Resampling.average)
        tcc_on_ref_grid = tcc_on_ref_grid.rename("LIVE_CANOPY_CVR_PCT")
        tcc_on_ref_grid.to_zarr(dir_out + f"TREE_CANOPY_COVER/NLCD_TCC_{year}.zarr", mode="w")

        del tcc_on_ref_grid, tcc
        gc.collect()


def process_all_years(dir_out: str, ds_ref_grid: xr.DataArray, year_range=np.arange(1990, 2005)):

    for year in year_range:
        print(year)
        process_one_year(year=year, dir_out=dir_out, ds_ref_grid=ds_ref_grid)
