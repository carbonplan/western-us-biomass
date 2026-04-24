import logging

from western_us_biomass.dir_info import dir_processed

logging.basicConfig(level=logging.INFO)

# Static predictors to use
FIA_VARIABLE_PREDICTORS = [
    # FIA variables
    # "FORTYPCD_GRP",
    "SLOPE_preliminary",
    "ASPECT_preliminary",
    # "ELEV_preliminary",
    "Ecosection_code",
    "Ecoprovince_code",
    "pct_own_public",
    "delta_live_canopy_cvr_pct",
    "biomass_start",
]

# Time-varying predictors to use from the first measurement time
PREDICTOR_VARIABLES_MEAS1 = [
    # "QMD",
    # "STDAGE",
    # "LIVE_CANOPY_CVR_PCT",
    # "LIVE_MISSING_CANOPY_CVR_PCT",
    # "NBR_LIVE_STEMS",
    # "STDSZCD",
    # "SITECLCD",
    # "LAND_COVER_CLASS_CD",
]

# Time-varying predictors to use from the second measurement time
PREDICTOR_VARIABLES_MEAS2 = [
    # "LIVE_CANOPY_CVR_PCT",
    # PRISM variables
    "tmean_mean_10yr",
    "ppt_mean_10yr",
    "tmax_maxseason_10yr",
    "tmin_minseason_10yr",
    "vpdmax_maxseason_10yr",
    # "vpdmax_mean_anom",
    # FIA variables
    # "years_after_harvest",
    # "years_after_fire",
    # "years_after_harvest",
    # "years_after_drought",
]

OUTPUT_VARIABLE = "biomass_delta"

FPATH_MODEL = dir_processed + "models/biomass_delta_burned_random_forest_model"

FPATH_PREDICTORS = (
    dir_processed + "models/biomass_delta_burned_random_forest_model_variable_names.csv"
)

FPATH_MODEL_BACKWARDS = dir_processed + "models/biomass_delta_burned_random_forest_model_backwards"

FPATH_PREDICTORS_BACKWARDS = (
    dir_processed + "models/biomass_delta_burned_random_forest_model_variable_names_backwards.csv"
)
