#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --account=jranders_lab
#SBATCH --output=logs/train_model_%A.out
#SBATCH --error=logs/train_model_%A.err
#SBATCH --time=06:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1

export PATH="$HOME/.pixi/bin:$PATH"

pixi run python src/western_us_biomass/train_models/train_all_models.py

