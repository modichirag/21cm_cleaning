#!/bin/bash

for wopt in opt pess;
do
    for topt in opt pess reas;
    do
	for aa in 0.3333 0.1429 0.2000;
	do
	    echo
	    echo Doing for $wopt $topt $aa
	    echo
	    sed -i "/wopt/s/.*/    wopt : $wopt/" ./params/paramvary.yml
	    sed -i "/stage2/s/.*/    stage2 : $topt/" ./params/paramvary.yml
	    sed -i "/aa/s/.*/    aa : $aa/" ./params/paramvary.yml
	    #cat ./params/paramvary.yml;
	    mpirun -n 8 python -u recon-wedgeHI.py ./params/paramvary.yml
	 done;
    done;
done;
       
