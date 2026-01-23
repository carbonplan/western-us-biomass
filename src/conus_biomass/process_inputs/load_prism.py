import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
import xarray as xr

from conus_biomass.dir_info import dir_prism


def get_prism_dir(var="tmean"):
    return f"{dir_prism}PRISM_{var}_stable_4kmM3_198101_202408_bil/"


def extract_temperature_data(latitudes, longitudes, location_ids, tmean_folder):
    """
    Extract mean temperature data for given latitudes, longitudes, and location IDs.

    Parameters:
        latitudes (list): List of latitude values.
        longitudes (list): List of longitude values.
        location_ids (list): List of location IDs corresponding to the lat/lon pairs.

    Returns:
        xr.DataArray: An xarray DataArray containing the mean temperature data.
    """
    # Initialize a dictionary to store data for each lat/lon point
    results = {"lat": [], "lon": [], "time": [], "mean_temperature": []}

    # Loop through all files in the folder
    for file_name in sorted(os.listdir(tmean_folder)):
        if file_name.endswith(".bil"):  # Check if the file is a .bil file
            # Construct the full file path
            file_path = os.path.join(tmean_folder, file_name)

            # Extract the date from the file name
            year_month = file_name.split("_")[4]  # Extract YYYYMM
            date = pd.to_datetime(year_month, format="%Y%m")

            # Open the .bil file using rasterio
            with rasterio.open(file_path) as src:
                # Read the data into a NumPy array
                data_array = src.read(1)
                # Get the affine transform from the metadata
                transform = src.transform

                # Loop through each latitude and longitude
                for lat, lon in zip(latitudes, longitudes):
                    # Convert latitude and longitude to row and column indices
                    row, col = rowcol(transform, lon, lat)
                    # Extract the value at the specified location
                    value = data_array[row, col]

                    # Append the data to the results dictionary
                    results["lat"].append(lat)
                    results["lon"].append(lon)
                    results["time"].append(date)
                    results["mean_temperature"].append(value)

    # Reshape the data to match the dimensions
    unique_times = sorted(set(results["time"]))
    num_locations = len(latitudes)
    reshaped_data = np.array(results["mean_temperature"]).reshape(len(unique_times), num_locations)

    # Create an xarray DataArray from the results
    data_array = xr.DataArray(
        data=reshaped_data,
        dims=["time", "location_id"],
        coords={
            "time": unique_times,
            "location_id": location_ids,
            "lat": ("location_id", latitudes),
            "lon": ("location_id", longitudes),
        },
        name="mean_temperature",
    )

    return data_array


def load_prism_data_for_all_plots(ds):
    prism_tmean = extract_temperature_data(
        latitudes=ds["lat"].values,
        longitudes=ds["lon"].values,
        location_ids=ds["plotid"].values,
        tmean_folder=get_prism_dir(var="tmean"),
    )

    prism_ds = prism_tmean.to_dataset(name="tmean")

    # Load all the other variables
    # This is not efficient and requires a lot of memory but makes the data easy to work with all at once
    # Takes about 2.5 mins on Claire's computer

    # PRISM variable names
    varlist = [
        "tmean",
        "ppt",
        "tdmean",
        "tmax",
        "tmin",
        "vpdmax",
        "vpdmin",
    ]

    # Descriptions of the variables
    varlist_longnames = [
        "daily mean temperature (averaged over all days in the month) [C]",
        "monthly total precipitation (rain+ melted snow) [mm]",
        "mean dewpoint temperature (averaged over all days in the month) [C]",
        "daily maximum temperature (averaged over all days in the month) [C]",
        "daily minimum temperature (averaged over all days in the month) [C]",
        "daily maximum vapor pressure deficit (averaged over all days in the month) [hPa]",
        "daily minimum vapor pressure deficit (averaged over all days in the month) [hPa]",
    ]

    # Variable units
    varlist_units = ["C", "mm", "C", "C", "C", "hPa", "hPa"]

    for i, var in enumerate(varlist):
        prism_ds[var] = extract_temperature_data(
            latitudes=ds["lat"].values,
            longitudes=ds["lon"].values,
            location_ids=ds["plotid"].values,
            tmean_folder=get_prism_dir(var=var),
        )
        prism_ds[var].attrs["units"] = varlist_units[i]
        prism_ds[var].attrs["long_name"] = varlist_longnames[i]
        prism_ds[var].attrs["name"] = var
    return prism_ds


def combine_two_datasets(ds, prism_ds):
    ds_combined = ds

    prism_ds = prism_ds.rename({"location_id": "plotid"}).drop(["lat", "lon"])

    varlist = [
        "tmean",
        "ppt",
        "tdmean",
        "tmax",
        "tmin",
        "vpdmax",
        "vpdmin",
    ]
    for i, var in enumerate(varlist):
        ds_combined[var] = prism_ds[var]

    return ds_combined
