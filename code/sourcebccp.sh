#!/bin/bash
##salloc -N 1 -C haswell -q interactive -t 01:00:00 -L SCRATCH -A m3127

source /usr/common/contrib/bccp/conda-activate.sh 3.6
#bcast-pip -U --no-deps https://github.com/bccp/nbodykit/archive/master.zip                                                                                                     
bcast-pip -U --no-deps https://github.com/bccp/nbodykit/archive/master.zip                                                                                                     
bcast-pip  -U --no-deps https://github.com/bccp/simplehod/archive/master.zip

export OMP_NUM_THREADS=1
