#!/bin/bash
#SBATCH --job-name=tile
#SBATCH --account=jranders_lab
#SBATCH --output=logs/tile_%A_%a.out
#SBATCH --error=logs/tile_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=15G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-165

# Ensure Pixi is on PATH (adjust if installed elsewhere)
export PATH="$HOME/.pixi/bin:$PATH"

nx=10
ny=15

xtile=$(( SLURM_ARRAY_TASK_ID / ny ))
ytile=$(( SLURM_ARRAY_TASK_ID % ny ))

pixi run python src/conus_biomass/run_model/run_model_spatially.py --xtile $xtile --ytile $ytile --model-suffix ""
