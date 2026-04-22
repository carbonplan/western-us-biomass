# conus-biomass

## Set up the environment
```bash
# Clone the repository
git clone https://github.com/carbonplan/western-us-biomass.git
cd western-us-biomass

# Install dependencies with pixi
pixi install

# Activate the environment
pixi shell
```

## Train model components
To train a single ensemble member:
```bash
pixi run python src/western_us_biomass/train_models/train_all_models.py
```

To train multiple ensemble members:
```bash
sbatch src/western_us_biomass/train_models/submit_training_jobs.sh
```

## Run the model
To run multiple ensemble members:
```bash
sbatch src/western_us_biomass/run_model/submit_all_tiles_ensemble.sh
```

To run a single ensemble member:
```bash
sbatch src/western_us_biomass/run_model/submit_all_tiles.sh
```

## Postprocess the model output
```bash
pixi run python src/western_us_biomass/process_outputs/postprocess_output.py
```

## Make figures
Navigate to ```nbs/4_make_figures/``` and run notebooks corresponding to each figure. Each figure is split into two notebooks: one notebook that processes all data for the figure into smaller intemediate data files, and another notebook that generates the figure from that data.
