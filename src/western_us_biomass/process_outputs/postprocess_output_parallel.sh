#!/bin/bash

# Usage: bash postprocess_output_parallel.sh 10 20

START=$1
END=$2

for i in $(seq $START $((END-1))); do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=postprocess-${i}
#SBATCH --account=jranders_lab
#SBATCH --output=logs/postprocess_output_%j.out
#SBATCH --error=logs/postprocess_output_%j.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G

pixi run python postprocess_output.py --start ${i} --end $((i+1))
EOF
    echo "Submitted job for range ${i} to $((i+1))"
done
