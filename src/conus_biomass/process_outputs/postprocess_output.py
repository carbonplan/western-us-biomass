import logging

import numpy as np
import xarray as xr

from conus_biomass import dir_info

logging.basicConfig(level=logging.INFO)


def get_fname_processed_biomass(
    year: int,
    dir_processed_model_output: str = dir_info.dir_model_output[:-1] + "_processed/",
    varname: str = "predicted_biomass",
):
    fname = dir_processed_model_output + varname + "_filtered_" + str(year) + ".nc"
    return fname


def process_model_output(
    dir_processed_model_output: str = dir_info.dir_model_output[:-1] + "_processed/",
    dir_model_input: str = dir_info.dir_model_input,
    dir_model_output: str = dir_info.dir_model_output,
    chunk_size: int = 200,
    year_range: np.array = np.arange(2005, 2023),
    varname_file: str = "predicted_biomass",
    varname_array: str = "predicted_biomass",
):
    logging.info("Model input: " + dir_model_input)
    logging.info("Model output: " + dir_model_output)
    logging.info("Postprocessed model output: " + dir_processed_model_output)

    def get_frf_mask(year: int):
        if year > 2021:
            frf_mask = inputs["forest_remaining_forest"].sel(year=2021)
            frf_mask = frf_mask.chunk({"x": chunk_size, "y": chunk_size})
        else:
            frf_mask = inputs["forest_remaining_forest"].sel(year=year)
            frf_mask = frf_mask.chunk({"x": chunk_size, "y": chunk_size})

        return frf_mask / 100

    inputs_fname = dir_model_input + "all_variables.nc"
    inputs = xr.open_dataset(inputs_fname)

    for year in year_range:
        ds = xr.open_mfdataset(
            dir_model_output + varname_file + "_unfiltered_" + str(year) + "_*_.nc",
            join="outer",
        )

        for var in ds.data_vars:
            if var in ["x", "y"]:
                ds[var] = ds[var].chunk({var: chunk_size})
            elif var not in ["band", "spatial_ref", "year"]:
                ds[var] = ds[var].chunk({"x": chunk_size, "y": chunk_size})
        ds = ds.chunk({"x": chunk_size, "y": chunk_size})

        frf_mask = get_frf_mask(year)
        ds_filtered = ds[varname_array] * (frf_mask)
        ds_filtered = ds_filtered.where(frf_mask > 0)
        ds_filtered = ds_filtered.chunk({"x": chunk_size, "y": chunk_size})

        fname_out = get_fname_processed_biomass(
            year=year,
            dir_processed_model_output=dir_processed_model_output,
            varname="predicted_biomass",
        )
        logging.info(fname_out)
        ds_filtered.to_dataset(name=varname_array).to_netcdf(fname_out, mode="w")


def main():
    process_model_output()


if __name__ == "__main__":
    main()
