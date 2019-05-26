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
parser.add_argument('-a', '--aa', help='scale factor', default=0.3333, type=float)
parser.add_argument('-l', '--bs', help='boxsize', default=256, type=float)
parser.add_argument('-n', '--nmesh', help='nmesh', default=128, type=int)
parser.add_argument('-t', '--angle', help='angle of the wedge', default=50, type=float)
parser.add_argument('-k', '--kmin', help='kmin of the wedge', default=0.01, type=float)
args = parser.parse_args()

figpath = './figs/'

bs, nc, aa = args.bs, args.nmesh, args.aa
zz = 1/aa- 1
kmin = args.kmin
ang = args.angle
pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank

################
def make_rep_plot():
    """Does the work of making the real-space xi(r) and b(r) figure."""
    
   
    fig, axar = plt.subplots(3, 2, figsize=(9, 9), sharex=True)

    #fits
    linestyle=['-', '--']
    colors=['C0', 'C1', 'C2']
    lww = 2
    
    for ia, aa  in enumerate([0.3333, 0.2000, 0.1429]):
        zz = 1/aa-1
        for iw, wopt in enumerate(['opt', 'pess']):
            lss = linestyle[iw]
            for it, thopt in enumerate(['opt', 'pess', 'reas']):
                if rank == 0: print(aa, wopt, thopt)
                cc = colors[it]
                ax = axar[ia]
                try:
                    angle = np.round(mapn.wedge(zz, att=wopt, angle=True), 0)
                    #dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_ang%0.1f/'%(aa, 0.03, angle)
                    dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_%s/'%(aa, 0.03, wopt)
                    dpath += 'L%04d-N%04d-R//thermal-%s-hex/ZA/opt_s999_h1massA_fourier_rsdpos/'%(bs, nc, thopt)
                    datapp = mapp.Observable.load(dpath+'/datap')
                    bpaths = [dpath+'%d-0.00//best-fit'%nc] + [dpath + '%d-0.00//%04d/fit_p/'%(nc,i) for i in range(100, 50, -20)]
                    for path in bpaths:
                        if os.path.isdir(path): 
                            break
                    if rank == 0: print(path)
                    bfit = mapp.Observable.load(path)
                    #bfit = mapp.Observable.load(dpath+'ZA/opt_s999_h1massA_fourier_rsdpos/best-fit/')
                    rpfit = rp.evaluate1(bfit, datapp, field='mapp')[:-2]
                    if ia == 0 and iw == 0:
                        if thopt == 'reas': thopt = 'fid'
                        lbl = 'Noise - %s'%thopt
                    elif ia == 1 and it == 0:
                        lbl = 'Wedge = %s'%wopt
                    else: lbl = None
                    ax[0].plot(rpfit[0]['k'], rpfit[0]['power']/(rpfit[1]['power']*rpfit[2]['power'])**0.5, ls=lss, lw=lww, color=cc, label=lbl)
                    ax[1].plot(rpfit[0]['k'], (rpfit[1]['power']/rpfit[2]['power'])**0.5, ls=lss, lw=lww, color=cc)
                    ax[1].text(0.02, 2, r'$z = %.2f$'%zz,color='black',ha='left',va='bottom', fontdict=font)
                except Exception as e: 
                    if rank == 0: print(e)



    for axis in axar[:, 0]: 
        axis.set_ylabel('$r_{cc}$', fontdict=font)
        axis.set_ylim(-0.05, 1.1)
    #for axis in axar[:, 1]: axis.set_ylabel(r'$\sqrt{P_{\rm mod}/P_{hh}}$', fontdict=font)
    for axis in axar[:, 1]: 
        axis.set_ylabel(r'$T_f$', fontdict=font)
        axis.set_ylim(-0.05, 3)
    for axis in axar[2, :]: axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
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
    if rank  == 0: plt.savefig(figpath + '/allcompare_L%04d-hex.pdf'%(bs))



################
def make_repwd_plot():
    """Does the work of making the real-space xi(r) and b(r) figure."""
    
   
    fig, axar = plt.subplots(3, 2, figsize=(9, 9), sharex=True)

    #fits
    linestyle=['-', '--']
    colors=['C0', 'C1', 'C2']
    lww = 2
    
    for ia, aa  in enumerate([0.3333, 0.2000, 0.1429]):
        zz = 1/aa-1
        for iw, wopt in enumerate(['opt', 'pess']):
            lss = linestyle[iw]
            for it, thopt in enumerate(['opt', 'pess', 'reas']):
                if rank == 0: print(aa, wopt, thopt)
                cc = colors[it]
                ax = axar[ia]
                try:
                    angle = np.round(mapn.wedge(zz, att=wopt, angle=True), 0)
                    #dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_ang%0.1f/'%(aa, 0.03, angle)
                    dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_%s/'%(aa, 0.03, wopt)
                    dpath += 'L%04d-N%04d-R//thermal-%s-hex/ZA/opt_s999_h1massA_fourier_rsdpos/'%(bs, nc, thopt)
                    datapp = mapp.Observable.load(dpath+'/datap')
                    bpaths = [dpath+'%d-0.00//best-fit'%nc] + [dpath + '%d-0.00//%04d/fit_p/'%(nc,i) for i in range(100, 50, -20)]
                    for path in bpaths:
                        if os.path.isdir(path): 
                            break
                    if rank == 0: print(path)
                    bfit = mapp.Observable.load(path)
                    #bfit = mapp.Observable.load(dpath+'ZA/opt_s999_h1massA_fourier_rsdpos/best-fit/')
                    rpfit = rp.evaluate1(bfit, datapp, field='mapp')[:-2]
                    if ia == 0 and iw == 0:
                        if thopt == 'reas': thopt = 'fid'
                        lbl = 'Noise - %s'%thopt
                    elif ia == 1 and it == 0:
                        lbl = 'Wedge = %s'%wopt
                    else: lbl = None
                    ax[0].plot(rpfit[0]['k'], rpfit[0]['power']/(rpfit[1]['power']*rpfit[2]['power'])**0.5, ls=lss, lw=lww, color=cc, label=lbl)
                    ax[1].plot(rpfit[0]['k'], (rpfit[1]['power']/rpfit[2]['power'])**0.5, ls=lss, lw=lww, color=cc)
                except Exception as e: 
                    if rank == 0: print(e)



    for ia, aa  in enumerate([0.3333, 0.2000, 0.1429]):
        zz = 1/aa-1
        for iw, wopt in enumerate(['opt', 'pess']):
            lss = linestyle[iw]
            for it, thopt in enumerate(['opt', 'pess', 'reas']):
                if rank == 0: print(aa, wopt, thopt)
                cc = colors[it]
                ax = axar[ia]
                try:
                    angle = np.round(mapn.wedge(zz, att=wopt, angle=True), 0)
                    #dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_ang%0.1f/'%(aa, 0.03, angle)
                    dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_%s/'%(aa, 0.03, wopt)
                    dpath += 'L%04d-N%04d-R//thermal-%s-hex/ZA/opt_s999_h1massA_fourier_rsdpos/'%(bs, nc, thopt)
                    datapp = mapp.Observable.load(dpath+'/datap')
                    bfit = mapp.Observable.load(dpath+'/dataw')
                    #bfit = mapp.Observable.load(dpath+'ZA/opt_s999_h1massA_fourier_rsdpos/best-fit/')
                    rpfit = rp.evaluate1(bfit, datapp, field='mapp')[:-2]
                    if ia == 0 and iw == 0:
                        lbl = 'Noise - %s'%thopt
                    elif ia == 1 and it == 0:
                        lbl = 'Wedge = %s'%wopt
                    else: lbl = None
                    ax[0].plot(rpfit[0]['k'], rpfit[0]['power']/(rpfit[1]['power']*rpfit[2]['power'])**0.5, ls=lss, lw=1, color=cc, alpha=0.5)
                    ax[1].plot(rpfit[0]['k'], (rpfit[1]['power']/rpfit[2]['power'])**0.5, ls=lss, lw=1, color=cc, alpha=0.5)
                    ax[1].text(0.02, 2, r'$z = %.2f$'%zz,color='black',ha='left',va='bottom', fontdict=font)
                except Exception as e: 
                    if rank == 0: print(e)


    for axis in axar[:, 0]: 
        axis.set_ylabel('$r_{cc}$', fontdict=font)
        axis.set_ylim(-0.05, 1.1)
    for axis in axar[:, 1]: 
        axis.set_ylabel(r'$T_f$', fontdict=font)
        axis.set_ylim(-0.05, 3)
    for axis in axar[2, :]: axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    for axis in axar.flatten():
        axis.axhline(1, color='k', ls=':')
        axis.set_xscale('log')
        axis.grid(which='both', lw=0.2, alpha=0.2, color='gray')
        axis.legend(prop=fontmanage, loc=4)

    # Put on some more labels.
    for axis in axar.flatten():
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
    ##and finish
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if rank  == 0: plt.savefig(figpath + '/allcomparewd_L%04d-hex.pdf'%(bs))


################


if __name__=="__main__":
    #make_rep_plot()
    make_repwd_plot()
    #
