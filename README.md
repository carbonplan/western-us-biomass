<p align="left" >
<a href='https://carbonplan.org'>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://carbonplan-assets.s3.amazonaws.com/monogram/light-small.png">
  <img alt="CarbonPlan monogram." height="48" src="https://carbonplan-assets.s3.amazonaws.com/monogram/dark-small.png">
</picture>
</a>
</p>

# conus-biomass

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Code analyzing changes in live aboveground forest biomass across the Western United States

## Usage
This repository consists of code that reproduce the figures and numbers in our analysis of trends in live aboveground biomass in forests across the Western United States. First, you'll want to download a copy of the supporting data from [Zenodo](https://doi.org/10.5281/zenodo.19698817). Second, you'll want to update references in `src/western_us_biomass/dir_info.py` to refer to the location where you save the data files. From there, you're set to explore the analysis on your own. 

## Instructions

#### Set up the environment
```bash
# Clone the repository
git clone https://github.com/carbonplan/western-us-biomass.git
cd western-us-biomass

# Install dependencies with pixi
pixi install

# Activate the environment
pixi shell
```

#### Train model components
To train a single ensemble member:
```bash
sbatch src/western_us_biomass/train_models/submit_training_jobs_one_run.sh
```

To train multiple ensemble members:
```bash
sbatch src/western_us_biomass/train_models/submit_training_jobs_ensemble.sh
```

#### Run the model
To run multiple ensemble members:
```bash
sbatch src/western_us_biomass/run_model/submit_all_tiles_ensemble.sh
```

To run a single ensemble member:
```bash
sbatch src/western_us_biomass/run_model/submit_all_tiles_single_run.sh
```

#### Postprocess the model output
```bash
cd src/western_us_biomass/process_outputs
sbatch postprocess_output.sh
```

#### Make figures
Navigate to ```nbs/4_make_figures/``` and run notebooks corresponding to each figure. Each figure is split into two notebooks: one notebook that processes all data for the figure into smaller intemediate data files, and another notebook that generates the figure from that data.

## License

All the code in this repository is [MIT](https://choosealicense.com/licenses/mit/)-licensed, but we request that you please provide attribution if reusing any of our digital content (graphics, logo, articles, etc.).

## About us

CarbonPlan is a nonprofit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of climate solutions with open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by opening an issue or [sending us an email](mailto:hello@carbonplan.org).

