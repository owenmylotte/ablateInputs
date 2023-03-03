#!/bin/bash
#SBATCH -n 16
#SBATCH -N $(nnodes16)
#SBATCH -J radScale
#SBATCH -t $(time16)
#SBATCH -p pbatch
#SBATCH --mail-type=NONE
#SBATCH -A sunyb
#SBATCH --mail-user=owenmylo@buffalo.edu
#SBATCH --output=volumetricSF_5_16_[297,42,42]

##### Load Required modules
# gcc
module load mkl/2019.0
module load valgrind/3.16.1
module load gcc/10.2.1
module load cmake/3.21.1

# Load PETSC ENV
export PETSC_DIR="/usr/workspace/mylotte1/petsc"
export PETSC_ARCH="arch-ablate-opt-gcc" # arch-ablate-debug or arch-ablate-opt
export PKG_CONFIG_PATH="${PETSC_DIR}/${PETSC_ARCH}/lib/pkgconfig:$PKG_CONFIG_PATH"
export HDF5_ROOT="${PETSC_DIR}/${PETSC_ARCH}"
# Include the bin directory to access mpi commands
export PATH="${PETSC_DIR}/${PETSC_ARCH}/bin:$PATH"

# Make a temp directory so that tchem has a place to vomit its files
mkdir tmp_$SLURM_JOBID
cd tmp_$SLURM_JOBID

export UPWIND=leastsquares

##### Launch parallel job using srun
srun -n16 /usr/workspace/mylotte1/ablateOpt/ablate \
  --input $(inputfile[297,42,42]) \
  -yaml::environment::title volumetricSFScaling_5_16_[297,42,42] \
  -yaml::timestepper::domain::faces [297,42,42] \
  -yaml::solvers::[4]::radiation::rays 5 \
  -yaml::timestepper::domain::fields::[0]::conservedFieldOptions::petscfv_type $UPWIND \
  -log_view :volumetricSFScaling_5_16_[297,42,42].csv:ascii_csv

echo "End Time " $(date +%s)
