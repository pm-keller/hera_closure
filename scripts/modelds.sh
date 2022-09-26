#!/bin/bash
#SBATCH -p hera
#SBATCH -J modelds
#SBATCH -t 1:00:00
#SBATCH --mem=1288G
#SBATCH --mail-type=ALL
#SBATCH --mail-user pmk46@cam.ac.uk


pythonpath="/lustre/aoc/projects/hera/pkeller/anaconda3/bin/python3"

for trclass in EQ14 EQ28
do 
    for field in A B C
    do
        gleam="/users/pkeller/code/ClosureSim/data/vis_{trclass}_F{field}B2.h5"
        bright="/users/pkeller/code/ClosureSim/data/vis_bright_{trclass}_F{field}B2.h5"
        data="/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/sample/{trclass}_F{field}_B2.h5"
        std="/users/pkeller/code/H1C_IDR3.2/data/noise_std.h5"
        "${pythonpath}" modelds.py -g "${gleam}" -b "${bright}" -d "${data}" -s "${std}"
    done
done