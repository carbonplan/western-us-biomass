import logging

from western_us_biomass.dir_info import dir_processed

logging.basicConfig(level=logging.INFO)

# Static predictors to use
FIA_VARIABLE_PREDICTORS = [
    # FIA and FIA-derived variables
    "SLOPE_preliminary",
    "ASPECT_preliminary",
    # "ELEV_preliminary",
    "Ecosection_code",
    "Ecoprovince_code",
    "pct_own_public",
    "delta_live_canopy_cvr_pct_per_year",
    "biomass_start",
]

# Time-varying predictors to use from the first measurement time
PREDICTOR_VARIABLES_MEAS1 = [
    # "LIVE_CANOPY_CVR_PCT",
]

# Time-varying predictors to use from the second measurement time
PREDICTOR_VARIABLES_MEAS2 = [
    # PRISM variables
    # "LIVE_CANOPY_CVR_PCT",
    # FIA variables
    # "years_after_harvest",
    "years_after_fire",
    "tmean_mean_10yr",
    "ppt_mean_10yr",
    "tmax_maxseason_10yr",
    "tmin_minseason_10yr",
    "vpdmax_maxseason_10yr",
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
