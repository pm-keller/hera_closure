#!/bin/bash
#SBATCH -p hera
#SBATCH -J lstavg
#SBATCH -t 00:10:00
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user pmk46@cam.ac.uk


pythonpath="/lustre/aoc/projects/hera/pkeller/anaconda3/bin/python3"

for trclass in EQ14 EQ28
do 
    for field in A B C
    do
        inpath="/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/sample/${trclass}_F${field}_B2.h5"
        outpath="/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/sample/${trclass}_F${field}_B2_AVG.h5"
        scaling="/users/pkeller/code/H1C_IDR3.2/data/scaling_${trclass}_F${field}B2_model.dat"
        veff="/users/pkeller/code/H1C_IDR3.2/data/veff_cal_${trclass}_F${field}B2.dat"
        flags="/users/pkeller/code/H1C_IDR3.2/data/lst_flags_${trclass}_model.dat"
        "${pythonpath}" lstavg.py -i "${inpath}" -o "${outpath}" -s "${scaling}" -v "${veff}" -f "${flags}" -n 16 -m
    done
done

