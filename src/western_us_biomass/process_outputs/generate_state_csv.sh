#!/bin/bash
#SBATCH --job-name=generate-state-csv
#SBATCH --output=logs/state_csv_%j.out
#SBATCH --error=logs/state_csv_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=14
#SBATCH --mem=10G

pixi run python generate_state_csv.py