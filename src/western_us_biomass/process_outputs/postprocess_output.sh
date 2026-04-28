#!/bin/bash
#SBATCH --job-name=postprocess-output
#SBATCH --output=logs/postprocess_output_%j.out
#SBATCH --error=logs/postprocess_output_%j.err
#SBATCH --time=06:00:00
#SBATCH --account=jranders_lab
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G

pixi run python postprocess_output.py --start 320 --end 340
