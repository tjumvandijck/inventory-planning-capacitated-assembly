#!/bin/bash
module load 2024
module load CMake/3.29.3-GCCcore-13.3.0
module load OpenMPI/5.0.3-GCC-13.3.0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
conda activate envDP

#CUDA not currently tested. 
#module load cuDNN/8.4.1.50-CUDA-11.7.0
#module load cuDNN/8.6.0.163-CUDA-11.8.0

module list

chmod +x exeWrapper.sh
