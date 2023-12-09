#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80GB
#SBATCH --time=12:00:00
#SBATCH --job-name=postprocessing_law_school_race_run_1
#SBATCH --output=postprocessing_law_school_race_run_1.out

module purge
source $SCRATCH/fairness-variance/.venv/bin/activate

python ./scripts/postprocessing_law_school_race/Law_School.py \
    --run_nums "[1]" --bootstrap_fraction 0.8 --uuid "eff514bd-bf12-4bab-80da-d45780aee27d"
