#!/bin/bash
#SBATCH -p hera
#SBATCH -J trcov
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user pmk46@cam.ac.uk

pythonpath="/lustre/aoc/projects/hera/pkeller/anaconda3/bin/python3"

for trclass in EQ14 EQ28
do 
    for field in A B C E
    do  
        echo $trclass $field
        path="/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/sample/${trclass}_F${field}_B2_AVG.h5"
        "${pythonpath}" trcov.py -p "${path}"
    done
done

