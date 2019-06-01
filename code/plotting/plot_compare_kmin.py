#!/usr/bin/env python3
#
# Plots the power spectra and Fourier-space biases for the HI.
#
import warnings
from mpi4py import MPI
rank = MPI.COMM_WORLD.rank
#warnings.filterwarnings("ignore")            
if rank!=0: warnings.filterwarnings("ignore")


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
from lab import mapnoise as mapn

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



#
import argparse
parser = argparse.ArgumentParser()
#parser.add_argument('-m', '--model', help='model name to use')
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--bs', help='boxsize', default=1024, type=float)
parser.add_argument('-n', '--nmesh', help='nmesh', default=256, type=int)

args = parser.parse_args()

figpath = './figs/'

bs, nc = args.bs, args.nmesh
pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank


################
def make_rep_plot():
    """Does the work of making the real-space xi(r) and b(r) figure."""
    
   
    fig, axar = plt.subplots(1, 2, figsize=(9, 4), sharex=True)

    #fits
    linestyle=['-', '--']
    colors=['C%d'%i for i in range(7)]
    lww = 2
    
    wopt = 'opt'
    thopt = 'reas'
    lss = '-'
    for ia, aa  in enumerate([0.2000]):
        zz = 1/aa-1
        for ik, kmin in enumerate([0.001, 0.01, 0.03, 0.05]):
            cc = colors[ik]
            angle = np.round(mapn.wedge(zz, att=wopt, angle=True), 0)
            #dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_ang%0.1f/'%(aa, 0.03, angle)
            dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_%s/'%(aa, kmin, wopt)
            dpath += 'L%04d-N%04d-R//thermal-%s-hex/ZA/opt_s999_h1massA_fourier_rsdpos/'%(bs, nc, thopt)
            datapp = mapp.Observable.load(dpath+'/datap')
            bpaths = [dpath+'%d-0.00//best-fit'%nc] + [dpath + '%d-0.00//%04d/fit_p/'%(nc,i) for i in range(100, 30, -20)]
            for path in bpaths:
                if os.path.isdir(path): 
                    break
            if rank == 0: print(path)
            bfit = mapp.Observable.load(path)
            rpfit = rp.evaluate1(bfit, datapp, field='mapp')[:-2]
            lbl = '$k_\parallel = %.2f$'%kmin
            axar[0].plot(rpfit[0]['k'], rpfit[0]['power']/(rpfit[1]['power']*rpfit[2]['power'])**0.5, ls=lss, lw=lww, color=cc, label=lbl)
            axar[1].plot(rpfit[0]['k'], (rpfit[1]['power']/rpfit[2]['power'])**0.5, ls=lss, lw=lww, color=cc)


    ik += 1
    cc = colors[ik]
    dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_ang%0.2f/'%(aa, 0, 0)
    dpath += 'L%04d-N%04d-R//thermal-%s-hex/ZA/opt_s999_h1massA_fourier_rsdpos/'%(bs, nc, thopt)
    datapp = mapp.Observable.load(dpath+'/datap')
    bpaths = [dpath+'%d-0.00//best-fit'%nc] + [dpath + '%d-0.00//%04d/fit_p/'%(nc,i) for i in range(100, 30, -20)]
    for path in bpaths:
        if os.path.isdir(path): 
            break
    if rank == 0: print(path)
    bfit = mapp.Observable.load(path)
    rpfit = rp.evaluate1(bfit, datapp, field='mapp')[:-2]
    lbl = 'All modes'
    axar[0].plot(rpfit[0]['k'], rpfit[0]['power']/(rpfit[1]['power']*rpfit[2]['power'])**0.5, ls='--', lw=lww, color=cc, label=lbl)
    axar[1].plot(rpfit[0]['k'], (rpfit[1]['power']/rpfit[2]['power'])**0.5, ls=lss, lw=lww, color=cc)

    axis = axar[0]
    axis.set_ylabel('$r_{cc}$', fontdict=font)
    axis.set_ylim(-0.05, 1.1)
    #for axis in axar[:, 1]: axis.set_ylabel(r'$\sqrt{P_{\rm mod}/P_{hh}}$', fontdict=font)
    axis = axar[1]
    axis.set_ylabel(r'$T_f$', fontdict=font)
    axis.set_ylim(-0.05, 2)
    for axis in axar[:]: axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    for axis in axar.flatten():
        axis.axhline(1, color='k', ls=':')
        axis.set_xscale('log')
        axis.grid(which='both', lw=0.2, alpha=0.2, color='gray')
        axis.legend(prop=fontmanage)

    # Put on some more labels.
    for axis in axar.flatten():
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
    ##and finish
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if rank  == 0: plt.savefig(figpath + '/kmin_L%04d-hex.pdf'%(bs))



################


if __name__=="__main__":
    make_rep_plot()
    #
