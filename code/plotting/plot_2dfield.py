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
parser.add_argument('-l', '--bs', help='boxsize', default=256, type=float)
parser.add_argument('-n', '--nmesh', help='nmesh', default=128, type=int)
parser.add_argument('-t', '--angle', help='angle of the wedge', default=50, type=float)
parser.add_argument('-k', '--kmin', help='kmin of the wedge', default=0.01, type=float)
parser.add_argument('-r', '--rsdpos', help='kmin of the wedge', default=False, type=bool)
args = parser.parse_args()

figpath = './figs/'

bs, nc, aa = args.bs, args.nmesh, args.aa
zz = 1/aa- 1
kmin = args.kmin
ang = args.angle
pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])

dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_ang%0.1f/'%(aa, kmin, ang)
dpath += 'L%04d-N%04d/stage2/'%(bs, nc)

################
def make_rep_plot():
    """Does the work of making the real-space xi(r) and b(r) figure."""
    

    noises = np.loadtxt('/global/u1/c/chmodi/Programs/21cm/21cm_cleaning/data/summaryHI.txt').T
    for i in range(noises[0].size):
        if noises[0][i] == np.round(1/aa-1, 2): noise = noises[3][i]
    print(noise)

    fpath = 'ZA/opt_s999_h1massA_fourier'
    if args.rsdpos : fpath += '_rsdpos/'
    dataprsd = mapp.Observable.load(dpath+fpath+'/datap').mapp[...]
    dataprsdw = mapp.Observable.load(dpath+fpath+'/dataw').mapp[...]
    basepath = dpath+fpath+'/%d-0.00/'%(nc)
    bpaths = [basepath+'/best-fit'] + [basepath + '/%04d/fit_p/'%i for i in range(100, -1, -20)]
    for path in bpaths:
        if os.path.isdir(path): break
    print(path)
    bfit = mapp.Observable.load(path).mapp[...]
    
    fig, ax = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)

    vmin, vmax = dataprsd.sum(axis=0).min(), dataprsd.sum(axis=0).max()
    for i, f  in enumerate([dataprsd, dataprsdw, bfit]):
        for j in range(3):
            #ax[j, i].imshow(f.sum(axis=j), vmin=vmin, vmax=vmax)
            ax[j, i].imshow(f.sum(axis=j))
        
    ax[0, 0].set_title('Truth', fontdict=font)
    ax[0, 1].set_title('Data', fontdict=font)
    ax[0, 2].set_title('Recon', fontdict=font)
    ax[0, 0].set_ylabel('X', fontdict=font)
    ax[1, 0].set_ylabel('Y', fontdict=font)
    ax[2, 0].set_ylabel('Z', fontdict=font)
##
##
##
##    # Put on some more labels.
##    for axis in ax:
##        axis.set_xscale('log')
##        for tick in axis.xaxis.get_major_ticks():
##            tick.label.set_fontproperties(fontmanage)
##        for tick in axis.yaxis.get_major_ticks():
##            tick.label.set_fontproperties(fontmanage)
##    ##and finish
##    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(figpath + '/map_L%04d_%04d.pdf'%(bs, aa*10000))



################


if __name__=="__main__":
    make_rep_plot()
    #
