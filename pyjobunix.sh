#!/bin/bash -l
# salloc -t 30 -A edu17.DD2424 -N 8 # --gres=gpu:K80:2

# load other modules
module add cudnn/5.1-cuda-8.0
# load the anaconda module
module load anaconda/py35/4.2.0
# if you need the custom conda environment:
source activate tensorflow
# source activate my_tensorflow1.1

# install python dependence
# conda install -c anaconda scipy=0.19.0

# load mpirun
module load i-compilers/17.0.1
module load intelmpi/17.0.1

# execute the program
# (on Beskow use aprun instead)
# SBATCH -J TESTJOB
# SBATCH -A edu17.DD2424
# SBATCH -t 30:00
# SBATCH --nodes=1

salloc -t 30 -A edu17.DD2424 -N 1 mpirun -np 1 python inception_FCN.py
# to deactivate the Anaconda environment
source deactivate
# copy files
# "C:\Program Files\PuTTY with GSS-API Key Exchange\pscp.exe" "C:\Nobackup\Local YW\git\DL2017Project\pyjob.sh" tzh@t04n28.pdc.kth.se:/afs/pdc.kth.se/home/t/tzh/workspace
