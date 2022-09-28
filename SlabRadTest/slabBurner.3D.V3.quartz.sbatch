#!/bin/bash
#### See https://hpc.llnl.gov/training/tutorials/slurm-and-moab#LC

##### These lines are for Slurm
#SBATCH -N 12
#SBATCH -J slabV3
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH --mail-type=ALL
#SBATCH -A sunyb
#SBATCH --mail-user=owenmylo@buffalo.edu

##### Load Required modules
# gcc
module load mkl/2019.0
module load valgrind/3.16.1
module load gcc/10.2.1
module load  cmake/3.21.1

# Load PETSC ENV
export PETSC_DIR="/g/g15/mcgurn4/petsc"
export PETSC_ARCH="arch-ablate-opt-gcc" # arch-ablate-debug or arch-ablate-opt
export PKG_CONFIG_PATH="${PETSC_DIR}/${PETSC_ARCH}/lib/pkgconfig:$PKG_CONFIG_PATH"
export HDF5_ROOT="${PETSC_DIR}/${PETSC_ARCH}"
# Include the bin directory to access mpi commands
export PATH="${PETSC_DIR}/${PETSC_ARCH}/bin:$PATH"

# Make a temp directory so that tchem has a place to vomit its files
mkdir tmp_$SLURM_JOBID
cd tmp_$SLURM_JOBID

export TITLE=slabRadTest
export FACES=100,20,20
export UPWIND=leastsquares

##### Launch parallel job using srun
srun -n432 /g/g20/mylotte1/ablateOpt/ablate \
	--input /p/lustre1/mylotte1/ablateInputs/SlabRadTest/slabBurner.3D.V3.yaml \
	-yaml::environment::title $TITLE \
	-yaml::timestepper::domain::faces [$FACES] \
	-yaml::timestepper::domain::fields::[0]::conservedFieldOptions::petscfv_type $UPWIND

echo 'Done'