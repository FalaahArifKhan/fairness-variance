#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --job-name=postprocessing_law_school_race_batch5
#SBATCH --output=postprocessing_law_school_race_batch5.out

module purge
source $SCRATCH/fairness-variance/.venv/bin/activate

python ./scripts/postprocessing_law_school_race/Law_School_Batch5.py
