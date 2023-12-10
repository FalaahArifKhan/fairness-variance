#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80GB
#SBATCH --time=12:00:00
#SBATCH --job-name=postprocessing_credit_card_sex_run_1
#SBATCH --output=postprocessing_credit_card_sex_run_1.out

module purge
source $SCRATCH/fairness-variance/.venv/bin/activate

python ./scripts/postprocessing_intervention_credit_card_sex/Credit_Card.py \
    --run_nums "[1]" --bootstrap_fraction 0.8 --uuid "2398ee13-56ee-475b-a010-f20656c4559c"
