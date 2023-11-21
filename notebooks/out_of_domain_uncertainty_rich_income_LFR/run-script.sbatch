#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --job-name=rich_income_DIR_RF_5K

module purge
source $SCRATCH/fairness-variance/.venv/bin/activate

python ./notebooks/out_of_domain_uncertainty_rich_income_LFR/rich_income_DIR_RF_5K.py