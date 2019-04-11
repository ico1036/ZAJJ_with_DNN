#!/bin/bash


echo "$HOSTNAME"
TopDir=`pwd`
export SCRAM_ARCH=slc6_amd64_gcc530
export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
echo "$VO_CMS_SW_DIR $SCRAM_ARCH"
source $VO_CMS_SW_DIR/cmsset_default.sh
cd /home/jwkim/CMSSW_8_0_26_patch1/src
eval `scramv1 runtime -sh`
cd $TopDir


#g++ `root-config --cflags` `root-config --libs` /x5/cms/jwkim/MG5_aMC_v2_6_4/MLzajj/makeHist.C -I/x5/cms/jwkim/Delphes-3.4.1 -L/x5/cms/jwkim/Delphes-3.4.1 -lDelphes -o /x5/cms/jwkim/MG5_aMC_v2_6_4/MLzajj/makeHist.exe

export LD_LIBRARY_PATH=/x5/cms/jwkim/Delphes-3.4.1:$LD_LIBRARY_PATH



thisIndex=$1
maxFile=$2
listFile=$3
datatype=$4
xsec=$5
genN=$6
outname=$7

firstIndex=`expr $thisIndex \* $maxFile + 1`
lastIndex=`expr $firstIndex + $maxFile - 1`

inFiles=""
index=0
for file in `cat $listFile`
do
    ((index++))
    if [ $index -ge $firstIndex ] && [ $index -le $lastIndex ]; then
        #echo "$index $file"
        inFiles="$inFiles $file"
    fi
done
if [ ! -d condorOut ]; then mkdir condorOut; fi




/x5/cms/jwkim/MG5_aMC_v2_6_4/MLzajj/makeHist.exe condorOut/$outname $datatype $xsec $genN $inFiles



