#!/bin/bash

#for i in `seq 13982 19999`
#do
#ls zajjQED/condorOut/events_46204_$i.lhe >> remain.txt
#done


lists=`cat remain.txt`
count=1
num=$1
dirname=zajjQED


for file in $lists
do


if [ $count = `expr $num + 49` ]; then break; fi



if [ $count -ge $num ]
then

## --Make and split directories
filename=`basename $file`
filename="${filename%.*}"
if [ ! -d Data/$dirname/CMND ]; then mkdir -p Data/$dirname/CMND; fi
if [ ! -d Data/$dirname/ROOTfiles ]; then mkdir -p Data/$dirname/ROOTfiles; fi


echo $filename

## --cmnd file gen: set 1) # of gen events 2) LHEpath
cat << EOF > Data/$dirname/CMND/$filename.cmnd
! 1) Settings used in the main program.
Main:numberOfEvents = 500          ! number of events to generate
#Main:timesAllowErrors = 3          ! how many aborts before run stops
! 2) Settings related to output in init(), next() and stat().
Init:showChangedSettings = on      ! list changed settings
Init:showChangedParticleData = off ! list changed particle data
Next:numberCount = 200             ! print message every n events
Next:numberShowInfo = 1            ! print event information n times
Next:numberShowProcess = 1         ! print process record n times
Next:numberShowEvent = 0           ! print event record n times
! 3) Set the input LHE file
Beams:frameType = 4
Beams:LHEF = /gridpack/$file
EOF

## --Run
../Delphes-3.4.1/DelphesPythia8 ../Delphes-3.4.1/cards/delphes_card_CMS.tcl Data/$dirname/CMND/$filename.cmnd Data/$dirname/ROOTfiles/$filename.root &
fi

count=$(($count+1))

done

