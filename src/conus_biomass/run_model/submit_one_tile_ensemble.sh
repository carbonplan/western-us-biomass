#!/bin/bash

# Submit multiple model runs for a single tile, with an ensemble of different model suffixes

# Define the range of model suffixes
START=0
END=3

# Loop through model suffixes and submit jobs
for i in $(seq -f "%04g" $START $END); do
    suffix="_${i}"

    # Submit the job
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tile1_${i}
#SBATCH --account=jranders_lab
#SBATCH --output=logs/tile1_${i}_%A_%a.out
#SBATCH --error=logs/tile1_${i}_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1

export PATH="\$HOME/.pixi/bin:\$PATH"
pixi run python src/conus_biomass/run_model/run_model_spatially.py --xtile 11 --ytile 2 --model-suffix "$suffix"
EOF

    echo "Submitted job for model suffix: $suffix"

    sleep 0.1
done

echo "All jobs submitted!"
