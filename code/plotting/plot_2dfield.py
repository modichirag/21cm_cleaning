#!/usr/bin/env python3
#
# Plots the power spectra and Fourier-space biases for the HI.
#
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from pmesh.pm import ParticleMesh
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from nbodykit.lab import BigFileMesh, BigFileCatalog, FFTPower
from nbodykit.cosmology import Planck15, EHPower, Cosmology

sys.path.append('../utils/')
sys.path.append('../recon/')
sys.path.append('../recon/cosmo4d/')
from lab import mapbias as mapp
from lab import report as rp
from lab import dg
from getbiasparams import getbias
import tools
#

from matplotlib import rc, rcParams, font_manager
rcParams['font.family'] = 'serif'
fsize = 12
fontmanage = font_manager.FontProperties(family='serif', style='normal',
    size=fsize, weight='normal', stretch='normal')
font = {'family': fontmanage.get_family()[0],
        'style':  fontmanage.get_style(),
        'weight': fontmanage.get_weight(),
        'size': fontmanage.get_size(),
        }

print(font)


#
import argparse
parser = argparse.ArgumentParser()
#parser.add_argument('-m', '--model', help='model name to use')
parser.add_argument('-a', '--aa', help='scale factor', default=0.3333, type=float)
parser.add_argument('-l', '--bs', help='boxsize', default=1024, type=float)
parser.add_argument('-n', '--nmesh', help='nmesh', default=256, type=int)
parser.add_argument('-t', '--angle', help='angle of the wedge', default=50, type=float)
parser.add_argument('-k', '--kmin', help='kmin of the wedge', default=0.03, type=float)
parser.add_argument('-r', '--rsdpos', help='kmin of the wedge', default=True, type=bool)
parser.add_argument('--pp', help='upsample', default=0)
args = parser.parse_args()

figpath = './figs/'

#bs, nc, aa = args.bs, args.nmesh, args.aa
#kmin = args.kmin
#ang = args.angle

aa = args.aa
bs, nc = 1024, 256
zz = 1/aa- 1
ang = 'opt'
pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])

#dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_ang%0.1f/'%(aa, kmin, ang)
#dpath += 'L%04d-N%04d/stage2/'%(bs, nc)
dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_%s/'%(aa, 0.03, ang)
dpath += 'L%04d-N%04d-R/thermal-reas-hex/'%(bs, nc)

################
def make_rep_plot():
    """Does the work of making the real-space xi(r) and b(r) figure."""
    

    fpath = 'ZA/opt_s999_h1massA_fourier'
    if args.rsdpos : fpath += '_rsdpos/'
    if args.pp: 
        dataprsd = mapp.Observable.load(dpath+fpath+'/datap_up').mapp[...]
        dataprsdw = mapp.Observable.load(dpath+fpath+'/dataw_up').mapp[...]
    else:
        dataprsd = mapp.Observable.load(dpath+fpath+'/datap').mapp[...]
        dataprsdw = mapp.Observable.load(dpath+fpath+'/dataw').mapp[...]
    basepath = dpath+fpath+'/%d-0.00/'%(nc)
    if args.pp: basepath = dpath+fpath+'upsample2/%d-0.00/'%(nc)
    bpaths = [basepath+'/best-fit'] + [basepath + '/%04d/fit_p/'%i for i in range(100, -1, -20)]
    for path in bpaths:
        if os.path.isdir(path): break
    print(path)
    bfit = mapp.Observable.load(path).mapp[...]
    
    fig, ax = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)

    for i, f  in enumerate([dataprsd, dataprsdw, bfit]):
        i0, i1 = 145, 150
        cmap = 'RdBu_r'
        #cmap = 'viridis'

        off = 5
        vmin, vmax = None, None
        #vmin, vmax = dataprsd[i0:i1,...].sum(axis=0).min(), dataprsd[i0:i1,...].sum(axis=0).max()

        vmin, vmax = dataprsd[i0:i1,...].sum(axis=0).min()-off, dataprsd[i0:i1,...].sum(axis=0).max()+off
        im = ax[0, i].imshow(f[i0:i1,...].sum(axis=0), cmap=cmap, vmin=vmin, vmax=vmax)
        #plt.colorbar(im, ax=ax[0, i])

        vmin, vmax = dataprsd[:,i0:i1,:].sum(axis=1).min()-off, dataprsd[:,i0:i1,:].sum(axis=1).max()+off
        im = ax[1, i].imshow(f[:,i0:i1,:].sum(axis=1), cmap=cmap, vmin=vmin, vmax=vmax)
        #plt.colorbar(im, ax=ax[1, i])

        vmin, vmax = dataprsd[...,i0:i1].sum(axis=2).min()-off, dataprsd[...,i0:i1].sum(axis=2).max()+off
        im = ax[2, i].imshow(f[...,i0:i1].sum(axis=2), cmap=cmap, vmin=vmin, vmax=vmax)
        #plt.colorbar(im, ax=ax[2, i])

    ax[0, 0].set_title('Truth', fontdict=font)
    ax[0, 1].set_title('Data', fontdict=font)
    ax[0, 2].set_title('Recon', fontdict=font)
    ax[0, 0].set_ylabel('X', fontdict=font)
    ax[1, 0].set_ylabel('Y', fontdict=font)
    ax[2, 0].set_ylabel('Z', fontdict=font)

    if args.pp: plt.savefig(figpath + '/map_L%04d_%04d-up.pdf'%(bs, aa*10000))
    else: plt.savefig(figpath + '/map_L%04d_%04d.pdf'%(bs, aa*10000))



################


if __name__=="__main__":
    make_rep_plot()
    #
