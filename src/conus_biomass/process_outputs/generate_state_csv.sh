#!/bin/bash
#SBATCH --job-name=state_csv
#SBATCH --output=logs/state_csv_%j.out
#SBATCH --error=logs/state_csv_%j.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=14
#SBATCH --mem=80G

pixi run python generate_state_csv.py
