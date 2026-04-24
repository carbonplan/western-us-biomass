#!/bin/bash
#SBATCH --job-name=tile1
#SBATCH --account=jranders_lab
#SBATCH --output=logs/tile1_%A_%a.out
#SBATCH --error=logs/tile1_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1

#source /opt/apps/anaconda/2024.06/etc/profile.d/conda.sh
#conda activate uncertain_land_sink
#python src/western_us_biomass/run_model/run_model_spatially.py --xtile 11 --ytile 2

export PATH="$HOME/.pixi/bin:$PATH"
pixi run python src/western_us_biomass/run_model/run_model_spatially.py --xtile 2 --ytile 1 --model-suffix ""
