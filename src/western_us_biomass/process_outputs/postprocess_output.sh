#!/bin/bash
#SBATCH --job-name=postprocess-output
#SBATCH --output=logs/postprocess_output_%j.out
#SBATCH --error=logs/postprocess_output_%j.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=14
#SBATCH --mem=80G

pixi run python postprocess_output.py
