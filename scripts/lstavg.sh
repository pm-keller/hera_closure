#!/bin/bash
#SBATCH -p hera
#SBATCH -J lstavg
#SBATCH -t 3:00:00
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user pmk46@cam.ac.uk

pythonpath="/lustre/aoc/projects/hera/pkeller/anaconda3/bin/python3"

for trclass in EQ14 EQ28
do 
    for field in A B C D E
    do
        inpath="/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/sample/${trclass}_F${field}_B2.h5"
        outpath="/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/sample/${trclass}_F${field}_B2_AVG.h5"
        "${pythonpath}" lstavg.py -i "${inpath}" -o "${outpath}" -n 16
    done
done

