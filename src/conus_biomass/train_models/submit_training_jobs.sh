#!/bin/bash
# Submit each ensemble member training as a separate SLURM job
# Define the range of model suffixes

START=0
END=99

# Loop through model suffixes and submit jobs
for i in $(seq -f "%04g" $START $END); do
    suffix="_${i}"

    # Submit the job
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=train_ensemble_${i}
#SBATCH --account=jranders_lab
#SBATCH --output=logs/train_ensemble_${i}_%A.out
#SBATCH --error=logs/train_ensemble_${i}_%A.err
#SBATCH --time=06:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1

export PATH="\$HOME/.pixi/bin:\$PATH"
pixi run python -c "
from conus_biomass.train_models import train_all_models
train_all_models.train_all_models(model_suffix='${suffix}', random_seed=None)
"
EOF
    echo "Submitted job for model suffix: $suffix"
    sleep 0.1
done
echo "All jobs submitted!"
