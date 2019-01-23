import numpy as np
from pmesh.pm import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower, ArrayCatalog
from nbodykit.source.mesh.field import FieldMesh
import os

import sys
sys.path.append('./utils')
from time import time
from mpi4py import MPI

#Global, fixed things
scratch = '/global/cscratch1/sd/yfeng1/m3127/'
project = '/project/projectdirs/m3127/21cm_cleaning/'
myscratch = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/'

cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
aafiles = [0.1429, 0.1538, 0.1667, 0.1818, 0.2000, 0.2222, 0.2500, 0.2857, 0.3333]
#aafiles = aafiles[:1]
#aafiles = [0.3333]
atoz = lambda a: 1/a-1
zzfiles = [round(atoz(aa), 2) for aa in aafiles]

#Paramteres
#Maybe set up to take in as args file?
bs, nc = 256, 256
ncsim, sim, prefix = 256, 'lowres/%d-9100-fixed'%256, 'lowres'
ncsim, sim, prefix = 2560, 'highres/%d-9100-fixed'%2560, 'highres'




if __name__=="__main__":

    cube_size = 256
    shift = cube_size
    ncube = int(ncsim/shift)
    cube_length = cube_size*bs/ncsim


    aa = 0.2000
    path = myscratch +  sim + '/fastpm_%0.4f/'%aa + '/HImesh_N%04d'%(ncsim)
    kmin = 0.1
    R = 1.0
    #print('Input path = ', path)
    pm = ParticleMesh(BoxSize = bs, Nmesh = [ncsim, ncsim, ncsim])
    rank = pm.comm.rank
    wsize = pm.comm.size
    comm = pm.comm


    savepath = myscratch +  sim + '/fastpm_%0.4f/voxels-%d/'%(aa, cube_size)
    try: 
        os.makedirs(savepath)
    except: pass
    
    pmsmall = ParticleMesh(BoxSize = bs/ncube, Nmesh = [cube_size, cube_size, cube_size], dtype=np.float32, comm=MPI.COMM_SELF)
    gridsmall = pmsmall.generate_uniform_particle_grid(shift=0)

    dataset = ['HI', 'kpar%dp%d'%(int(kmin), kmin*10), 'kpar%dp%d-R%dp%d'%(int(kmin), (kmin*10)%10, int(R), (R*10)%10)]
    meshes = []
    vals = []
    for i in range(3):
        meshes.append(BigFileMesh(path, dataset=dataset[i], mode='real').paint())
        #vals.append(meshes[i].readout(grid, resampler = 'nearest', layout=layout).astype(np.float32))
    #mesh = BigFileMesh(path, dataset='HI', mode='real').paint()
    if rank == 0 : print('Meshes read')
    
    bindexes = np.arange(ncube**3)
    bindexsplit = np.array_split(bindexes, wsize)

    maxload = max(np.array([len(i) for i in bindexsplit]))


    for ibindex in range(maxload):
        if rank == 0: print('For index = %d'%(int(ibindex)))
        try:
            bindex = bindexsplit[rank][ibindex]
            bi, bj, bk = bindex//ncube**2, (bindex%ncube**2)//ncube, (bindex%ncube**2)%ncube
            bi *= cube_length
            bj *= cube_length
            bk *= cube_length
            print('For rank & index : ', rank, bindex, '-- x, y, z : ', bi, bj, bk)
            poslarge = gridsmall + np.array([bi, bj, bk])
            save = True
        except IndexError:
            poslarge = np.empty((1, 3))
            save = False

        print(rank, 'Position created')
        layout = pm.decompose(poslarge)
        if rank == 0 : print(rank, 'Layout decomposed')

        for i in range(3):
            vals = meshes[i].readout(poslarge, layout=layout, resampler='nearest').astype(np.float32)
            if save:
                savemesh = pmsmall.paint(gridsmall, mass = vals, resampler='nearest')
                savemesh = FieldMesh(savemesh)
                savemesh.save(savepath + '%04d'%bindex, dataset[i])
        if save: print('Rank %d Saved '%rank, bindex)
