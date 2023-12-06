#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8GB
#SBATCH --time=48:00:00
#SBATCH --job-name=bootstrap_sizes_experiment_0_8
#SBATCH --output=bootstrap_sizes_experiment_0_8.out

module purge
source $SCRATCH/fairness-variance/.venv/bin/activate

python ./scripts/bootstrap_sizes_experiment_income_GA/Folk_GA_2018_bootstrap_experiment.py \
    --run_nums "[1, 2, 3, 4, 5]" --bootstrap_fraction 0.8 --uuid ""
