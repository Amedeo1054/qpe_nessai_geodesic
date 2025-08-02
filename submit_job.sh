#!/bin/bash
#PBS -q gpuq                    # <-- Use the GPU queue (ask your admin if the name is different)
#PBS -l walltime=300:00:00
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb  # Request 1 GPU, 8 CPUs, and memory (adjust as needed)
#PBS -N run_objG

module load gcc/10.3.0 gsl/2.6 lapack/3.10.0 python/3.9.16
module load cuda/11.8              # <-- Load CUDA module if you're using GPU (adjust version)

cd $PBS_O_WORKDIR

python3 mc_binary_nessai.py runfit_test.cfg
