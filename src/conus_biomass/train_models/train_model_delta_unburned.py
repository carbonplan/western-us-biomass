import logging

from conus_biomass.dir_info import dir_processed

logging.basicConfig(level=logging.INFO)

# Static predictors to use
FIA_VARIABLE_PREDICTORS = [
    # FIA and FIA-derived variables
    "Ecosection_code",
    "Ecoprovince_code",
    "SLOPE_preliminary",
    "ASPECT_preliminary",
    "ELEV_preliminary",
    # "FORTYPCD_GRP",
    # "lat",
    # "lon",
    "pct_own_public",
    "biomass_start",
    "delta_live_canopy_cvr_pct_per_year",
    # PRISM variables
    # "tmean_clim_minseason",
    # "tmean_clim_maxseason",
    "tmean_clim_mean",
    # "ppt_clim_minseason",
    "ppt_clim_maxseason",
    "ppt_clim_mean",
    # "tmax_clim_minseason",
    "tmax_clim_maxseason",
    # "tmax_clim_mean",
    "tmin_clim_minseason",
    # "tmin_clim_maxseason",
    # "tmin_clim_mean",
    # "vpdmax_clim_minseason",
    "vpdmax_clim_maxseason",
    # "vpdmax_clim_mean",
]

# Time-varying predictors to use from the first measurement time
PREDICTOR_VARIABLES_MEAS1 = [
    # "LIVE_CANOPY_CVR_PCT",
]

# Time-varying predictors to use from the second measurement time
PREDICTOR_VARIABLES_MEAS2 = [
    # PRISM variables
    # "LIVE_CANOPY_CVR_PCT",
    # "tmean_minseason_anom",
    # "tmeanmaxseason_anom",
    # "tmean_mean_anom",
    # "ppt_minseason_anom",
    # "pptmaxseason_anom",
    # "ppt_mean_anom",
    # "tmax_minseason_anom",
    # "tmaxmaxseason_anom",
    # "tmax_mean_anom",
    # "tmin_minseason_anom",
    # "tminmaxseason_anom",
    # "tmin_mean_anom",
    # "vpdmax_minseason_anom",
    # "vpdmaxmaxseason_anom",
    # "vpdmax_mean_anom",
    # FIA variables
    # "years_after_harvest",
    "years_after_fire",
    # "years_since_disturbance_Uconn",
    # "years_after_drought",
]

OUTPUT_VARIABLE = "biomass_delta_per_year"

FPATH_MODEL = dir_processed + "models/biomass_delta_undisturbed_random_forest_model"

FPATH_PREDICTORS = (
    dir_processed + "models/biomass_delta_undisturbed_random_forest_model_variable_names.csv"
)

FPATH_MODEL_BACKWARDS = (
    dir_processed + "models/biomass_delta_undisturbed_random_forest_model_backwards"
)

FPATH_PREDICTORS_BACKWARDS = (
    dir_processed
    + "models/biomass_delta_undisturbed_random_forest_model_variable_names_backwards.csv"
)
