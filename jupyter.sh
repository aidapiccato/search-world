#!/bin/sh
#SBATCH -p jazayeri
#SBATCH -n 1                    # two cores
#SBATCH --mem=8G
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=apiccato
#SBATCH --mail-type=FAIL

hostname
jupyter notebook
