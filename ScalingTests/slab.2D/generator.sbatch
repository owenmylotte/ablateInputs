#!/bin/bash
#### See https://hpc.llnl.gov/training/tutorials/slurm-and-moab#LC

# Array for determining the allocated time of the job
declare -A time # Replace this array with however much time it actually takes + margin
time[36]="24:00:00"
time[72]="24:00:00"
time[144]="12:00:00"
time[288]="12:00:00"
time[576]="12:00:00"
time[1152]="06:00:00"
time[2304]="06:00:00"
time[4608]="06:00:00"
time[9216]="05:00:00"
time[18432]="04:00:00"
time[36864]="04:00:00"
time[73728]="04:00:00"
time[147456]="04:00:00"
time[294912]="04:00:00"

##### These lines are for Slurm
#SBATCH -n {{processes}}
#SBATCH --use-min-nodes
#SBATCH -J slab.vol
#SBATCH -t ${time[{{processes}}]}
#SBATCH -p pbatch
#SBATCH --mail-type=NONE
#SBATCH -A sunyb
#SBATCH --mail-user=owenmylo@buffalo.edu
#SBATCH --output=volumetricSF_{{rays}}_{{processes}}_{{faces}}

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
srun -n{{processes}} /usr/workspace/mylotte1/ablateOpt/ablate \
  --input ../input.yaml \
  -yaml::environment::title volumetricSFScaling_{{rays}}_{{processes}}_{{faces}} \
  -yaml::timestepper::domain::faces {{faces}} \
  -yaml::solvers::[4]::radiation::rays {{rays}} \
  -yaml::timestepper::domain::fields::[0]::conservedFieldOptions::petscfv_type $UPWIND \
  -log_view :volumetricSFScaling_{{rays}}_{{processes}}_{{faces}}.csv:ascii_csv

echo "End Time " $(date +%s)
