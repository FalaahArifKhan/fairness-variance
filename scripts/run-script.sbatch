#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --job-name=postprocessing-race
#SBATCH --output=postprocessing-race.out

module purge
source $SCRATCH/fairness-variance/.venv/bin/activate

python ./postprocessing_intervention_income_GA/Folk_GA_2018_Batch1.py