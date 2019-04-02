#!/bin/bash

########### Please run this file using 'source'command ################


# ROOT ENV SET using CMSSW
echo "$HOSTNAME"
TopDir=`pwd`
export SCRAM_ARCH=slc6_amd64_gcc530
export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
echo "$VO_CMS_SW_DIR $SCRAM_ARCH"
source $VO_CMS_SW_DIR/cmsset_default.sh
cd /home/jwkim/CMSSW_8_0_26_patch1/src
eval `scramv1 runtime -sh`
cd $TopDir
export LD_LIBRARY_PATH=/x5/cms/jwkim/Delphes-3.4.1:$LD_LIBRARY_PATH


# Set Path

Dirname=$1
ABSpath=$TopDir/gridpack/Data/$Dirname/ROOTfiles
libfile=/x5/cms/jwkim/Delphes-3.4.1/libDelphes.so

datatype=$2
xsec=$3
genN=$4
maxfile=$5


# Make working directory
if [ ! -d condorOut_hist/$Dirname ]; then mkdir -p condorOut_hist/$Dirname; fi
cp sampling.sh condorOut_hist/$Dirname/

# Read ntuples with absolute path and write a list
ls -1v $ABSpath/*.root  > condorOut_hist/$Dirname/data.list

chmod 777 condorOut_hist/$Dirname/data.list

# copy running 

nfile=`cat condorOut_hist/$Dirname/data.list | wc -l`
nfile=`expr $maxfile + $nfile - 1`
nJob=`expr $nfile / $maxfile`




cat << EOF > condorOut_hist/$Dirname/job.jds
executable = sampling.sh
universe = vanilla
output   = condorOut_\$(Process).out
error    = condorErr_\$(Process).err
log      = condor_logfile.log
should_transfer_files = yes
transfer_input_files = data.list
transfer_output_files = condorOut
when_to_transfer_output = ON_EXIT
arguments = \$(Process) $maxfile data.list $datatype $xsec $genN Out_\$(Cluster)_\$(Process).root
queue $nJob
EOF


cd condorOut_hist/$Dirname

condor_submit job.jds

