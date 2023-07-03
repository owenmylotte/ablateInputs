#!/bin/bash
#### See https://hpc.llnl.gov/training/tutorials/slurm-and-moab#LC

##### These lines are for Slurm
#SBATCH -n 18432
#SBATCH --use-min-nodes
#SBATCH -J slab
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH --mail-type=NONE
#SBATCH -A sunyb
#SBATCH --mail-user=owenmylo@buffalo.edu

##### Load Required modules
# gcc
module load clang/14.0.6-magic
module load gcc/10.3.1-magic
module load cmake/3.25.2

# Load PETSC ENV
export PETSC_DIR="/usr/workspace/mylotte1/petsc"
export PETSC_ARCH="arch-ablate-opt" # arch-ablate-debug or arch-ablate-opt
export PKG_CONFIG_PATH="${PETSC_DIR}/${PETSC_ARCH}/lib/pkgconfig:$PKG_CONFIG_PATH"
export HDF5_ROOT="${PETSC_DIR}/${PETSC_ARCH}"
# Include the bin directory to access mpi commands
export PATH="${PETSC_DIR}/${PETSC_ARCH}/bin:$PATH"

# Make a temp directory so that tchem has a place to vomit its files
mkdir tmp_$SLURM_JOBID
cd tmp_$SLURM_JOBID

export UPWIND=leastsquares

echo "Configuration: _15_18432_[190,45,45]"

##### Launch parallel job using srun
srun -n18432 /usr/workspace/mylotte1/ablateOpt/ablate \
  --input /p/lustre1/mylotte1/ablateInputs/ScalingTests/slab.3D/input.yaml \
  -yaml::timestepper::domain::faces [190,45,45] \
  -yaml::solvers::[5]::radiation::rays 15 \
  -log_view :slab.3D_15_18432_[190,45,45].csv:ascii_csv

echo "End Time " $(date +%s)
