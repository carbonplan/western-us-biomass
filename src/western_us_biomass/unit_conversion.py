import numpy as np

M2_PER_HA = 10000
GRAMS_PER_MEGAGRAM = 1000000

ACRES_PER_HECTARE = 2.47105
POUNDS_PER_METRIC_TON = 2204.62

GRIDCELL_AREA_FINE_M2 = 30 * 30  # m2
GRIDCELL_AREA_COARSE_M2 = (30 * 80) * (30 * 40)  # m2


def convert_CECS_to_Mg_per_ha(da):
    # CECS biomass data is in 1/10 g biomass/m2

    da_gB_per_m2 = da * 10  # convert to g biomass/m2
    da_gB_per_ha = da_gB_per_m2 * M2_PER_HA  # convert to g biomass/ha
    da_MgB_per_ha = da_gB_per_ha / GRAMS_PER_MEGAGRAM  # convert to Mg biomass / ha
    da_MgC_per_ha = da_MgB_per_ha / 2  # convert to Mg carbon / ha
    da_MgC_all_per_ha = (
        da_MgC_per_ha * 1.2
    )  # convert Mg aboveground live carbon to all (AGB+BGB) carbon

    return da_MgC_all_per_ha


def calculate_area(da, gridcell_area_m2):
    num_gridcells = (~np.isnan(da)).sum()

    region_area_m2 = num_gridcells * gridcell_area_m2

    region_area_ha = region_area_m2 / M2_PER_HA

    return region_area_ha
