#!/bin/bash
#### See https://hpc.llnl.gov/training/tutorials/slurm-and-moab#LC

# Array for determining the allocated time of the job
declare -A time # Replace this array with however much time it actually takes + margin
time[1]="24:00:00"
time[2]="24:00:00"
time[4]="12:00:00"
time[8]="12:00:00"
time[16]="12:00:00"
time[32]="08:00:00"
time[64]="08:00:00"
time[128]="08:00:00"
time[256]="05:00:00"
time[512]="05:00:00"
time[1024]="05:00:00"
time[2048]="05:00:00"
time[4096]="05:00:00"
time[8192]="05:00:00"
time[16384]="05:00:00"
time[32768]="05:00:00"

# Array representing the dimensionality of the input file that should be called
declare -A dimensions # Add entries for additional meshes as needed
dimensions["[297,42]"]="2D"
dimensions["[594,85]"]="2D"
dimensions["[297,42,42]"]="3D"
dimensions["[594,85,85]"]="3D"

##### These lines are for Slurm
#SBATCH -n 2
#SBATCH --use-min-nodes
#SBATCH -J radScale
#SBATCH -t ${time[2]}
#SBATCH -p pbatch
#SBATCH --mail-type=NONE
#SBATCH -A sunyb
#SBATCH --mail-user=owenmylo@buffalo.edu
#SBATCH --output=volumetricSF_5_2_[297,42,42]

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
srun -n2 /usr/workspace/mylotte1/ablateOpt/ablate \
  --input /p/lustre1/mylotte1/ablateInputs/ScalingTests/volumetricSFScaling/volumetricSF${dimensions[[297,42,42]]}.yaml \
  -yaml::environment::title volumetricSFScaling_5_2_[297,42,42] \
  -yaml::timestepper::domain::faces [297,42,42] \
  -yaml::solvers::[4]::radiation::rays 5 \
  -yaml::timestepper::domain::fields::[0]::conservedFieldOptions::petscfv_type $UPWIND \
  -log_view :volumetricSFScaling_5_2_[297,42,42].csv:ascii_csv

echo "End Time " $(date +%s)
