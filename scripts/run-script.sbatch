#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --job-name=postprocessing-folk-ga-2018-race
#SBATCH --output=postprocessing-folk-ga-2018-race.out

module purge
source $SCRATCH/fairness-variance/.venv/bin/activate

python ./scripts/postprocessing_intervention_income_GA_race/Folk_GA_2018_Batch1.py
