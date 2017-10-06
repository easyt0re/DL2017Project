#!/bin/bash 

# Set the allocation to be charged for this job
# not required if you have set a default allocation

# The name of the script is myjob
#SBATCH -J myjob

# Only 1 hour wall-clock time will be given to this job
#SBATCH -t 1:00:00

# Number of nodes
#SBATCH --nodes=4
# Number of MPI processes per node (24 is recommended for most cases)
# 48 is the default to allow the possibility of hyperthreading
#SBATCH --ntasks-per-node=24
# Number of MPI processes.

#SBATCH -e error_file.e
#SBATCH -o output_file.o

# load the compiler module
module load i-compilers/17.0.1
    # alternatively the module for the GNU compiler if
    # it has been used for compilation.
 
# load the intel mpi module
module load intelmpi/17.0.1
 
# Run the executable named myexe 
# and write the output into my_output_file
mpirun -np 96 ./myexe