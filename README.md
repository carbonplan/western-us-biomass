# conus-biomass

## Setting up environment
```bash
# Clone the repository
git clone https://github.com/carbonplan/conus-biomass.git
cd conus-biomass

# Install dependencies with pixi
pixi install

# Activate the environment
pixi shell
```

## Processing input data


## Run the workflow for an ensemble

#### Step 1: Train model components
```bash
sbatch src/western_us_biomass/train_models/submit_training_jobs.sh
```

#### Step 2: Run model
```bash
sbatch src/western_us_biomass/run_model/submit_all_tiles_ensemble.sh
```

#### Step 3: Postprocess data
```bash
pixi run python src/western_us_biomass/process_outputs/postprocess_output.py
```

#### Step 4: Make figures

## Running the workflow (single run)
#### Step 1: Train model components
```bash
pixi run python src/western_us_biomass/train_models/train_all_models.py
```

When running ensemble:
```bash
sbatch src/western_us_biomass/train_models/submit_training_jobs.sh
```


#### Step 2: Run model
```bash
sbatch src/western_us_biomass/run_model/submit_all_tiles.sh
```

#### Step 3: Postprocess data
```bash
pixi run python src/western_us_biomass/process_outputs/postprocess_output.py
```

#### Step 4: Make figures
