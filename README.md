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

## Running the workflow
#### Step 0: Process data
##### 0a: process plot-level data (for training)
##### 0b: process gridded data (for running model)

#### Step 1: Train model components
```bash
pixi run python src/conus_biomass/train_models/train_all_models.py
```

#### Step 2: Run model
```bash
sbatch src/conus_biomass/run_model/submit_all_tiles.sh
```

#### Step 3: Postprocess data
```bash
pixi run python src/conus_biomass/process_outputs/postprocess_output.py
```

#### Step 4: Make figures

