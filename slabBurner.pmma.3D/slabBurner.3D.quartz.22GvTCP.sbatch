#!/bin/bash
#### See https://hpc.llnl.gov/training/tutorials/slurm-and-moab#LC

##### These lines are for Slurm
#SBATCH -N 20
#SBATCH -J 22GvTCP
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH --mail-type=ALL
#SBATCH -A sunyb
#SBATCH --mail-user=mylotte1@buffalo.edu

##### Load Required modules
# gcc
module load mkl/2019.0
module load valgrind/3.16.1
module load gcc/10.2.1
module load  cmake/3.21.1 

# Load PETSC ENV
export PETSC_DIR="/p/lustre1/mylotte1/petsc"
export PETSC_ARCH="arch-ablate-opt-gcc" # arch-ablate-debug or arch-ablate-opt
export PKG_CONFIG_PATH="${PETSC_DIR}/${PETSC_ARCH}/lib/pkgconfig:$PKG_CONFIG_PATH"
export HDF5_ROOT="${PETSC_DIR}/${PETSC_ARCH}"  
# Include the bin directory to access mpi commands
export PATH="${PETSC_DIR}/${PETSC_ARCH}/bin:$PATH"

# Make a temp directory so that tchem has a place to vomit its files
mkdir tmp_$SLURM_JOBID
cd tmp_$SLURM_JOBID

export TITLE=6G-186x40x40-pmma-rad-$SLURM_JOBID
export FACES=186,40,40
export FILE=/p/lustre1/mylotte1/ablateInputs/slabBurner.pmma.3D/slabBurner.3D.22G.pmma.virtualTCP.yaml

# export TITLE=6G-186x40x40-pmma-radLoss-$SLURM_JOBID
# export FACES=186,40,40
# export FILE=/p/lustre2/ubchrest/ablateInputs/slabBurner.pmma.3D/slabBurner.3D.6G.pmma.radLoss.yaml

##### Launch parallel job using srun
srun -n720 /p/lustre1/mylotte1/ablateOpt \
   --input $FILE \
   -yaml::environment::title $TITLE \
   -yaml::timestepper::domain::faces [$FACES]

echo 'Done'
