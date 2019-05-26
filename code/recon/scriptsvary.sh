#!/bin/bash

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
		sed -i "/wopt/s/.*/    wopt : $wopt/" ./params/paramsvary.yml
		sed -i "/stage2/s/.*/    stage2 : $topt/" ./params/paramsvary.yml
		sed -i "/aa/s/.*/    aa : $aa/" ./params/paramsvary.yml
		sed -i "/spread/s/.*/    spread : $spread/" ./params/paramsvary.yml
		echo
		cat ./params/paramsvary.yml;
		echo
		echo
		mpirun -n 8 python -u recon-wedgeHI.py ./params/paramsvary.yml
		done;
	 done;
    done;
done;
       
