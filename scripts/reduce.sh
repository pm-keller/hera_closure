#!/bin/bash
#SBATCH -p hera
#SBATCH -J reduce
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user pmk46@cam.ac.uk

pythonpath="/lustre/aoc/projects/hera/pkeller/anaconda3/bin/python3"

for trclass in EQ14 EQ28
do 
    for field in A B C D E
    do
        inpath="/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/sample/${trclass}_F${field}.h5"
        outpath="/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/sample/${trclass}_F${field}_B2.h5"
        if [ -f "${inpath}" ]
        then
            "${pythonpath}" reduce.py -i "${inpath}" -o "${outpath}" --fmin 152.25 --fmax 167.97
        fi
        "${pythonpath}" trmed.py -p "${outpath}"
    done
done