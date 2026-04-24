import logging

from western_us_biomass.dir_info import dir_processed

logging.basicConfig(level=logging.INFO)

FIA_VARIABLE_PREDICTORS = [
    # FIA and FIA-derived variables
    "Ecosection_code",
    "Ecoprovince_code",
    "SLOPE_preliminary",
    "ASPECT_preliminary",
    "ELEV_preliminary",
    "lat",
    "lon",
    "pct_own_public",
    # PRISM variables
    #"ppt_clim_maxseason",
    #"tmin_clim_minseason",
    #"tmean_clim_minseason",
    #"tmax_clim_minseason",
    #"vpdmax_clim_minseason",
    #"ppt_clim_minseason",
    #"tmin_clim_mean",
    #"vpdmax_clim_maxseason",
    #"tmean_clim_maxseason",
    #"tmax_clim_mean",
    #"tmax_clim_maxseason",
    #
    # Decided not to include
    # "Ecosubsection",
    # "bigmap_agb_mean",
    #  "bigmap_agb_max",
]

PREDICTOR_VARIABLES_MEAS1 = [
    "LIVE_CANOPY_CVR_PCT",
    "years_after_fire",
    # "years_after_harvest",
    "STDAGE",
    "tmean_mean_10yr",
    "ppt_mean_10yr",
    "tmax_maxseason_10yr",
    "tmin_minseason_10yr",
    "vpdmax_maxseason_10yr",
    # "CONDPROP_UNADJ",
    # "years_since_disturbance_Uconn",
]

PREDICTOR_VARIABLES_MEAS2 = []

OUTPUT_VARIABLE = "biomass_start"  # "biomass_end"

FPATH_MODEL = dir_processed + "models/absolute_biomass_random_forest_model"

FPATH_PREDICTORS = dir_processed + "models/absolute_biomass_random_forest_model_variable_names.csv"
