#!/bin/bash
#SBATCH -J microscopy_unet
#SBATCH -o out/unet_64_seres50-%j.out         # output and error file name (%j expands to jobID)
#SBATCH -N 1              # total number of nodes
#SBATCH -n 1
#SBATCH -p gtx           # queue (partition) -- normal, development, etc.
#SBATCH -t 24:00:00        # run time (hh:mm:ss) - 4 hours


ibrun -np 1 sh run.sh

