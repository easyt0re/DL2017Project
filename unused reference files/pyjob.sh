#!/bin/bash -l
salloc -t 00:00:01 -A edu17.DD2424 #--gres=gpu:K80:2
# load other modules
module add cudnn/5.1-cuda-8.0
# load the anaconda module
module load anaconda/py35/4.2.0
# if you need the custom conda environment:
source activate tensorflow1.1
# execute the program
# (on Beskow use aprun instead)
mpirun -np 1 python some_script.py
# to deactivate the Anaconda environment
source deactivate
# copy files
# "C:\Program Files\PuTTY with GSS-API Key Exchange\pscp.exe" "C:\Nobackup\Local YW\git\DL2017Project\pyjob.sh" tzh@t04n28.pdc.kth.se:/afs/pdc.kth.se/home/t/tzh/workspace
# "C:\Program Files\PuTTY with GSS-API Key Exchange\pscp.exe" "C:\Nobackup\wang5\Box Sync\Deep Learning\Assignment4\Assignment4.m" tzh@t04n28.pdc.kth.se:/afs/pdc.kth.se/home/t/tzh/workspace
# "C:\Program Files\PuTTY with GSS-API Key Exchange\pscp.exe" tzh@t04n28.pdc.kth.se:/afs/pdc.kth.se/home/t/tzh/workspace/test.txt "C:\Nobackup\wang5\Box Sync\Deep Learning\Assignment4\nothing"
# salloc -t 30 -A edu17.DD2424 matlab -nosplash -nodesktop -r 'Assignment4; quit'
# "C:\Program Files\PuTTY with GSS-API Key Exchange\pscp.exe" "C:\Nobackup\wang5\Box Sync\Deep Learning\Assignment4\epoch5\learnMat376.mat" tzh@t04n28.pdc.kth.se:/afs/pdc.kth.se/home/t/tzh/workspace
# "C:\Program Files\PuTTY with GSS-API Key Exchange\pscp.exe" tzh@t04n28.pdc.kth.se:/cfs/klemming/nobackup/t/tzh/DL2017/pyjobunix.sh "C:\Nobackup\wang5\Box Sync\Deep Learning\Assignment4\nothing"
# "C:\Program Files\PuTTY with GSS-API Key Exchange\pscp.exe" "C:\Nobackup\Local YW\download\hello.cu" tzh@t04n28.pdc.kth.se:/cfs/klemming/nobackup/t/tzh/
# "C:\Program Files\PuTTY with GSS-API Key Exchange\pscp.exe" -r tzh@t04n28.pdc.kth.se:/cfs/klemming/nobackup/t/tzh/DL2017/logs "C:\Nobackup\Local YW\download"
# "C:\Program Files\PuTTY with GSS-API Key Exchange\pscp.exe" -r "C:\Nobackup\Local YW\git\DL2017Project\Data_zoo\MIT_SceneParsing" tzh@t04n28.pdc.kth.se:/cfs/klemming/nobackup/t/tzh/DL2017/Data_zoo/
# "C:\Program Files\PuTTY with GSS-API Key Exchange\pscp.exe" -r "C:\Nobackup\Local YW\git\DL2017Project\inception_FCN.py" tzh@t04n28.pdc.kth.se:/cfs/klemming/nobackup/t/tzh/DL2017/
# "C:\Program Files\PuTTY with GSS-API Key Exchange\pscp.exe" -r "C:\Nobackup\wang5\Box Sync\Deep Learning\DL2017Project" tzh@t04n28.pdc.kth.se:/afs/pdc.kth.se/home/t/tzh/Public
# "C:\Program Files\PuTTY with GSS-API Key Exchange\pscp.exe" -r "C:\Nobackup\Local YW\download\logs 100k\checkpoint" tzh@t04n28.pdc.kth.se:/cfs/klemming/nobackup/t/tzh/DL2017/logs
# cp -a /afs/pdc.kth.se/home/t/tzh/Public/DL2017Project ./