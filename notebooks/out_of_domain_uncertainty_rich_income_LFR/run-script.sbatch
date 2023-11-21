#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --job-name=rich_income_DIR_RF_5K

module purge
source $SCRATCH/fairness-variance/.venv/bin/activate

python ./notebooks/out_of_domain_uncertainty_rich_income_LFR/rich_income_DIR_RF_5K.py \
    --run_nums \[1\] --fairness_intervention_params \[0.0,0.7\]

python ./notebooks/out_of_domain_uncertainty_rich_income_LFR/rich_income_DIR_RF_5K.py \
    --run_nums \[2,3\] --fairness_intervention_params \[0.0,0.7\] \
    --tuned_params_filenames \[\"tuning_results_Folktables_2018_Income_alpha_0.0_exp_iter_1_20231121__142702.csv\",\"tuning_results_Folktables_2018_Income_alpha_0.7_exp_iter_1_20231121__142812.csv\"\]
