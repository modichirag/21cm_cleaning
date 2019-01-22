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
#aafiles = [0.2222]
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

def readincatalog(aa, matter=False):

    if matter: dmcat = BigFileCatalog(scratch + sim + '/fastpm_%0.4f/'%aa, dataset='1')
    halocat = BigFileCatalog(scratch + sim + '/fastpm_%0.4f/'%aa, dataset='LL-0.200')
    mp = halocat.attrs['MassTable'][1]*1e10
    #print('Mass of particle is = %0.2e'%mp)
    halocat['Mass'] = halocat['Length'] * mp
    halocat['Position'] = halocat['Position']%bs # Wrapping positions assuming periodic boundary conditions
    if matter:
        return dmcat, halocat
    else: return halocat



def HI_hod(mhalo,aa,mcut=2e9):
    """Returns the 21cm "mass" for a box of halo masses."""
    zp1 = 1.0/aa
    zz  = zp1-1
    alp = 1.0
    alp = (1+2*zz)/(2+2*zz)
    norm= 3e5*(1+(3.5/zz)**6)
    xx  = mhalo/mcut+1e-10
    mHI = xx**alp * np.exp(-1/xx)
    mHI*= norm
    return(mHI)
    #


def assignHImass(aa, save=False):
    '''assign HI masses to halos'''
    zz = atoz(aa)
    if rank == 0 :print('Redshift = %0.2f'%zz)

    halocat = readincatalog(aa)
    hmass = halocat['Mass'].compute()
    hpos = halocat['Position'].compute()
    #Do hod
    ofolder = myscratch + '/%s/fastpm_%0.4f/'%(sim, aa)
    try : os.make_dirs(ofolder)
    except: pass
    h1mass = HI_hod(hmass, aa)
    halocat['HImass'] = h1mass

    if save:
        colsave = [cols for cols in halocat.columns]
        colsave = ['ID', 'Position', 'Mass', 'HImass']
        if rank == 0: print(colsave)
        halocat.save(ofolder+'halocat', colsave)
        if rank == 0: print('Halos saved at path\n%s'%ofolder)

    return halocat


def removemodes(mesh, kmin=0.1, R=1):
    '''Remove k_par modes below kmin. Also smooth the box with Gaussian of scale R
    '''
    cmesh = mesh.r2c()
    mask = [np.ones_like(ki) for ki in cmesh.x]
    mask[2] *= abs(cmesh.x[2]) > kmin
    mask = np.prod(mask)
    cmesh *= mask
    mesh2 = cmesh.c2r()
    mesh3 = cmesh.apply(lambda k, v: v*np.exp(-sum(ki**2 for ki in k)*R**2)).c2r()
    return mesh2, mesh3
    


def savemesh(aa, bs=bs, nc=nc, kmin=0.1, R=1, savecat=False):
    '''save HI mesh with modes removed as in the function "removemodes"
    '''

    pm = ParticleMesh(BoxSize = bs, Nmesh = [nc, nc, nc])
    hcat = assignHImass(aa, save=savecat)
    #mesh = pm.paint(hcat['Position'], mass=hcat['HImass'])
    mesh  = hcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc], position='Position',value='HImass').to_real_field()
    path = myscratch +  sim + '/fastpm_%0.4f/'%aa + '/HImesh_N%04d'%(nc)
    try : os.make_dirs(path)
    except: pass
    if rank == 0: print('Save mesh to path \n', path)
    meshkpar, meshkparG = removemodes(mesh, kmin, R)
    
    mesh = FieldMesh(mesh)
    mesh.save(path, dataset='HI', mode='real')

    meshkpar = FieldMesh(meshkpar)
    dataset ='kpar%dp%d'%(int(kmin), kmin*10) # 
    meshkpar.save(path, dataset=dataset, mode='real')

    meshkparG = FieldMesh(meshkparG)
    dataset ='kpar%dp%d-R%dp%d'%(int(kmin), (kmin*10)%10, int(R), (R*10)%10) # 
    meshkparG.save(path, dataset=dataset, mode='real')

    
#

if __name__=="__main__":

    for aa in aafiles[:]:
        if rank == 0: print(aa)
        savemesh(aa, nc=ncsim, savecat=True)
