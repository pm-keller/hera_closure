#!/bin/bash
#SBATCH -J test
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user pmk46@cam.ac.uk

pythonpath="/lustre/aoc/projects/hera/pkeller/anaconda3/bin/python3"

for trclass in EQ28
do 
    for field in B
    do
        "${pythonpath}" test.py
    done
done