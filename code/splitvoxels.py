import numpy as np
from pmesh.pm import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower, ArrayCatalog
from nbodykit.source.mesh.field import FieldMesh
import os

import sys
sys.path.append('./utils')
from time import time

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

pm = ParticleMesh(BoxSize = bs, Nmesh = [nc, nc, nc])
rank = pm.comm.rank
size = pm.comm.size


def splitvoxels(ftlist, cube_size, shift=None, ncube=None):
    '''Split the meshes in ftlist in voxels of 'cube_size' in a regular fashion by
    shifting with 'shift' over the range of (0, ncp) on the mesh
    '''
    if type(ftlist) is not list: ftlist = [ftlist]
    ncp = ftlist[0].shape[0]
    if shift is None: shift = cube_size
    if ncube is None: ncube = int(ncp/shift)
    print(cube_size, ncp, shift, ncube)
    
    inp = []
    for i in range(ncube):
        for j in range(ncube):
            for k in range(ncube):
                x1, y1, z1 = i*shift, j*shift, k*shift
                x2, y2, z2 = x1+cube_size, y1+cube_size, z1+cube_size
                fts = np.stack([ar[x1:x2, y1:y2, z1:z2] for ar in ftlist], axis=-1)
                inp.append(fts)

    inp = np.stack(inp, axis=0)
    return inp



if __name__=="__main__":

    aa = 0.3333
    path = myscratch +  sim + '/fastpm_%0.4f/'%aa + '/HImesh_N%04d'%(ncsim)
    print(path)

    kmin = 0.1
    R = 1.0
    cube_size = 256
    shift = cube_size
    ncsim = 2560
    ncube = int(ncsim/shift)
    inp = []
    counter = 0 
    savepath = myscratch +  sim + '/fastpm_%0.4f/voxels-%d/'%(aa, cube_size)
    try: 
        os.makedirs(savepath)
    except: pass
    for i in range(ncube):
        for j in range(ncube):
            for k in range(ncube):
          
                print(counter)
                x1, y1, z1 = i*shift, j*shift, k*shift
                x2, y2, z2 = x1+cube_size, y1+cube_size, z1+cube_size
                #mesh = BigFileMesh(path, dataset='HI', mode='real').paint()
                #print(mesh.shape)
                mesh = BigFileMesh(path, dataset='HI', mode='real').paint()[x1:x2, y1:y2, z1:z2].astype(np.float32)
                print(x1, y1, z1)
                print(x2, y2, z2)
                print(mesh.shape)
#                np.save(savepath + 'HI-%04d.f4'%counter, mesh)
#
                dataset ='kpar%dp%d'%(int(kmin), kmin*10) # 
                meshkpar = BigFileMesh(path, dataset=dataset, mode='real').paint()[x1:x2, y1:y2, z1:z2].astype(np.float32)
#                np.save(savepath + dataset + '-%04d.f4'%counter, meshkpar)
#                
                dataset ='kpar%dp%d-R%dp%d'%(int(kmin), (kmin*10)%10, int(R), (R*10)%10) # 
                meshkparG = BigFileMesh(path, dataset=dataset, mode='real').paint()[x1:x2, y1:y2, z1:z2].astype(np.float32)
                #np.save(savepath + dataset + '-%04d.f4'%counter, meshkparG)

                tosave = np.stack([mesh, meshkpar, meshkparG], axis=-1)
                dataset ='mesh-kpar%dp%d-R%dp%d'%(int(kmin), (kmin*10)%10, int(R), (R*10)%10) # 
                np.save(savepath + dataset + '-%04d.f4'%counter, tosave)
                
                counter += 1
