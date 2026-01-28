dir_raw_data = "/dfs8/jranders_lab1/users/czarakas/uncertain_land_sink_data/raw_data/"
dir_processed = "/dfs8/jranders_lab1/users/czarakas/uncertain_land_sink_data/processed_data/"
dir_lookups = "/data/homezvol3/czarakas/conus-biomass/data/"

################ Raw data inputs ################
dir_mtbs_raw = dir_raw_data + "MTBS/mtbs_perimeter_data/"
dir_landuse_raw = dir_raw_data + "LCMS_Landuse/"
dir_prism = dir_raw_data + "PRISM/"
dir_canopy_cover = dir_raw_data + "NLCD_Tree_Canopy_Cover/"

dir_ecosubsections = dir_raw_data + "USFS_ecosubsections/data/commondata/sections2007"
dir_fia_csvs = dir_raw_data + "fiadb/"
dir_shp = dir_raw_data + "Geographic boundary shapefiles/cb_2018_us_state_20m/"
dir_shp_states = dir_raw_data + "Geographic boundary shapefiles/"

################ Intermediate processed data ################
dir_mtbs = dir_processed + "MBTS_raster/"
dir_landuse = dir_processed + "landuse/"
dir_years_post_disturbance = dir_processed + "years_post_disturbance/"
dir_carbon_densities = dir_processed + "carbon_densities/"
dir_forest_growth_rates_pct = dir_processed + "forest_growth_rates_pct/"

dir_frf = dir_processed + "forest_remaining_forest/"
dir_years_forest = dir_processed + "years_as_forest/"

################ Model inputs and outputs ################
dir_data_on_ref_grid = dir_processed + "data_on_ref_grid/"
dir_model_output = dir_processed + "model_results/1000m_noharvestmodel_2026jan27/"
dir_model_input = dir_processed + "data_on_ref_grid/1000m/"

dir_figures = "/Users/clairezarakas/Documents/Science/conus-biomass/figures/"
dir_QAQC = "/data/homezvol3/czarakas/conus-biomass/figures/"

################ Other ################
# Directories to processed_data
dir_walters = "/dfs8/jranders_lab1/users/czarakas/US_FIA_based_reports/Walters_et_al_2024/By_State/"
dir_EPA_2024 = "/Users/clairezarakas/Documents/Science/data/raw_data/EPA_GHG_emissions/allstateghgdata90-22_v082924/"
dir_biomass = dir_processed + "output_biomass/"
dir_forest_pct = dir_processed + "forest_pct/"
dir_greenbook = dir_processed + "ecosection-greenbook-biomass/"

################ S3 directories ################
# dir_landuse_raw = "s3://carbonplan-forests/USFS LCMS/CONUS/"

# dir_lookups = "s3://carbonplan-forests/fiadb/crosswalks/"
# dir_fia_csvs = "s3://carbonplan-forests/fiadb/" #"s3://carbonplan-forests/fia_csvs/"
