import logging

import numpy as np

from conus_biomass.train_models import train_all_models

for i in np.arange(0, 3):
    model_suffix = f"{i:04d}"
    logging.info("Training model " + model_suffix)

    train_all_models.main(model_suffix=model_suffix, random_seed=None)
