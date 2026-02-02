#!/bin/bash

# Script to submit array jobs for multiple model suffixes

# Define the range of model suffixes
START=0
END=3

# Loop through model suffixes and submit array jobs
for i in $(seq -f "%04g" $START $END); do
    suffix="_${i}"

    # Submit the array job
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tile_${i}
#SBATCH --account=jranders_lab
#SBATCH --output=logs/tile_${i}_%A_%a.out
#SBATCH --error=logs/tile_${i}_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-165

export PATH="\$HOME/.pixi/bin:\$PATH"

nx=10
ny=15
xtile=\$(( SLURM_ARRAY_TASK_ID / ny ))
ytile=\$(( SLURM_ARRAY_TASK_ID % ny ))

pixi run python src/conus_biomass/run_model/run_model_spatially.py --xtile \$xtile --ytile \$ytile --model-suffix "$suffix"
EOF

    echo "Submitted array job for model suffix: $suffix (257 tiles per job)"

    # Optional: Add a small delay to avoid overwhelming the scheduler
    sleep 0.1
done

echo "All array jobs submitted!"
