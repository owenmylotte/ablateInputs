#!/bin/sh
#SBATCH --partition=general-compute --qos=general-compute
#SBATCH --time=24:00:00
#SBATCH --nodes=24
#SBATCH --ntasks-per-node=16
##SBATCH --constraint=IB
#SBATCH --mem=256000
#SBATCH --job-name="radiationSlab"
#SBATCH --output=radiationSlab.out
#SBATCH --mail-user=owenmylo@buffalo.edu
#SBATCH --mail-type=ALL

# Print the current environment
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR

# Load the required modules.
module load intel/20.2
module load intel-mpi/2020.2
module load gcc/11.2.0
module load cmake/3.22.3
module load valgrind/3.14.0
module load gdb/7.8

# setup petsc
export PETSC_DIR=/projects/academic/chrest/owenmylo/petsc
export PETSC_ARCH=arch-ablate-opt
export PKG_CONFIG_PATH="${PETSC_DIR}/${PETSC_ARCH}/lib/pkgconfig:$PKG_CONFIG_PATH"
export HDF5_ROOT="${PETSC_DIR}/${PETSC_ARCH}"

# The initial srun will trigger the SLURM prologue on the compute nodes.
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS=$NPROCS

# The PMI library is necessary for srun
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so

# Make a temp directory so that tchem has a place to vomit its files
mkdir tmp_$SLURM_JOBID
cd tmp_$SLURM_JOBID

export TITLE=slabRadTest
export FACES=100,20,20
export UPWIND=leastsquares

srun -n $NPROCS /projects/academic/chrest/owenmylo/ablateOpt/ablate --input /panasas/scratch/grp-chrest/owenmylo/ablateInputs/SlabRadTest/slabBurner.3D.V3.yaml

echo 'Done'