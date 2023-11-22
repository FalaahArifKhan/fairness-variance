#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --job-name=rich_income_DIR_LR_70K_fair_run_1

module purge
source $SCRATCH/fairness-variance/.venv/bin/activate

python ./notebooks/out_of_domain_uncertainty_rich_income_LR/rich_income_DIR_LR_70K.py \
    --run_nums \[1\] --fairness_intervention_params \[0.7\]
