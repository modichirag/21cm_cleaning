#!/bin/bash

cc=0
for wopt in opt pess;
do
    for topt in opt pess reas;
    #for topt in reas;
    do
	for aa in  0.1429 0.2000 0.3333;
	do
	    for spread in  1;
	    do
		echo
		echo Doing for $wopt $topt $aa
		echo
		cc=$((cc+1))
		
		cat ./multi/cscriptwedzauprsd.sb >  ./multi/cscriptwedzauprsd-$cc.sb

		#sed -i "/-J/s/.*/#SBATCH -J wedzauprsd$cc /" ./multi/cscriptwedzauprsd-$cc.sb
		#sed -i "/srun/s/.*/time srun -n 256 python -u recon-wedgeHIup.py scripts/multi/paramswedzauprsd-$cc.sb/" ./multi/cscriptwedzauprsd-$cc.sb
		sed -i "s/cchere/$cc/g" ./multi/cscriptwedzauprsd-$cc.sb
		echo
		cat ./multi/cscriptwedzauprsd-$cc.sb;
		sbatch ./multi/cscriptwedzauprsd-$cc.sb;
		echo
		echo
		#mpirun -n 8 python -u recon-wedgeHI.py ./multi/cscriptwedzauprsd-$cc.sb
		done;
	 done;
    done;
done;
       
