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
		
		cat ./multi/paramswedzauprsd.yml >  ./multi/paramswedzauprsd-$cc.yml

		sed -i "/wopt/s/.*/    wopt : $wopt/" ./multi/paramswedzauprsd-$cc.yml
		sed -i "/stage2/s/.*/    stage2 : $topt/" ./multi/paramswedzauprsd-$cc.yml
		sed -i "/aa/s/.*/    aa : $aa/" ./multi/paramswedzauprsd-$cc.yml
		sed -i "/spread/s/.*/    spread : $spread/" ./multi/paramswedzauprsd-$cc.yml
		echo
		cat ./multi/paramswedzauprsd-$cc.yml;
		echo
		echo
		#mpirun -n 8 python -u recon-wedgeHI.py ./multi/paramswedzauprsd-$cc.yml
		done;
	 done;
    done;
done;
       
