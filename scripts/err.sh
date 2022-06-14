#!/bin/bash
#SBATCH -p hera
#SBATCH -J err
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user pmk46@cam.ac.uk

pythonpath="/lustre/aoc/projects/hera/pkeller/anaconda3/bin/python3"

for trclass in EQ14 EQ28
do 
    for field in A B C E
    do  
        echo $trclass $field
        cppath="/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/sample/${trclass}_F${field}_B2_AVG.h5"
        xpspath="/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/sample/${trclass}_F${field}_B2_XPS.h5"
        scaling="/users/pkeller/code/H1C_IDR3.2/data/scaling_${trclass}_F${field}B2.dat"
        "${pythonpath}" err.py -c "${cppath}" -x "${xpspath}" -s "${scaling}"
    done
done

