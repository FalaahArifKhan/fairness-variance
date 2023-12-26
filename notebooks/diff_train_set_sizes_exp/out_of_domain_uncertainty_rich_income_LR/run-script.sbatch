#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2GB
#SBATCH --time=48:00:00
#SBATCH --job-name=rich_income_DIR_LR_70K_fair_run_1

module purge
source $SCRATCH/projects/fairness-variance/exp_venv/bin/activate
pip install -r ./requirements.txt
pip install BlackBoxAuditing==0.1.54
pip install git+https://github.com/DataResponsibly/Virny.git@development

python ./notebooks/out_of_domain_uncertainty_rich_income_LR/rich_income_DIR_LR_70K.py \
    --run_nums \[1\] --fairness_intervention_params \[0.7\]
