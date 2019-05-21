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
args = parser.parse_args()

figpath = './figs/'

bs, nc, aa = args.bs, args.nmesh, args.aa
zz = 1/aa- 1
kmin = args.kmin
ang = args.angle
pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank

#dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_ang%0.1f/'%(aa, kmin, ang)
#dpath += 'L%04d-N%04d/'%(bs, nc)

################
def make_rep_plot():
    """Does the work of making the real-space xi(r) and b(r) figure."""
    



    fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    def makeplot(bfit, datapp, lss, lww, cc, lbl=None):
        rpfit = rp.evaluate1(bfit, datapp, field='mapp')[:-2]
        ax[0].plot(rpfit[0]['k'], rpfit[0]['power']/(rpfit[1]['power']*rpfit[2]['power'])**0.5, ls=lss, lw=lww, color=cc, label=lbl)
        ax[1].plot(rpfit[0]['k'], (rpfit[1]['power']/rpfit[2]['power'])**0.5, ls=lss, lw=lww, color=cc)
        

    #fits
    try:        
        aa = 0.1429
        dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_ang%0.1f/'%(aa, kmin, ang)
        dpath += 'L%04d-N%04d//'%(bs, nc)
        dataprsd = mapp.Observable.load(dpath+'ZA/opt_s999_h1massA_fourier_rsdpos/datap')
        basepath = dpath+'ZA/opt_s999_h1massA_fourier_rsdpos/%d-0.00/'%(nc)
        bpaths = [basepath+'/best-fit'] + [basepath + '/%04d/fit_p/'%i for i in range(100, -1, -20)]
        for path in bpaths:
            if os.path.isdir(path): break
        print(path)
        bfit = mapp.Observable.load(path)
        datapp = dataprsd
        lss, lww, cc, lbl = '-', 2, 'C0', 'rsd z=%.2f'%(1/aa-1)
        makeplot(bfit, datapp, lss, lww, cc, lbl)
        print('%s done'%lbl)
    except Exception as e: print(e)
            
    try:        
        aa = 0.3333
        dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_ang%0.1f/'%(aa, kmin, ang)
        dpath += 'L%04d-N%04d//'%(bs, nc)
        dataprsd = mapp.Observable.load(dpath+'ZA/opt_s999_h1massA_fourier_rsdpos/datap')
        basepath = dpath+'ZA/opt_s999_h1massA_fourier_rsdpos/%d-0.00/'%(nc)
        bpaths = [basepath+'/best-fit'] + [basepath + '/%04d/fit_p/'%i for i in range(100, -1, -20)]
        for path in bpaths:
            if os.path.isdir(path): break
        print(path)
        bfit = mapp.Observable.load(path)
        datapp = dataprsd
        lss, lww, cc, lbl = '-', 2, 'C1', 'rsd z=%.2f'%(1/aa-1)
        makeplot(bfit, datapp, lss, lww, cc, lbl)
        print('%s done'%lbl)
    except Exception as e: print(e)

    try:        
        aa = 0.1429
        dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_ang%0.1f/'%(aa, kmin, ang)
        dpath += 'L%04d-N%04d/stage2/'%(bs, nc)
        dataprsd = mapp.Observable.load(dpath+'ZA/opt_s999_h1massA_fourier_rsdpos/datap')
        basepath = dpath+'ZA/opt_s999_h1massA_fourier_rsdpos/%d-0.00/'%(nc)
        bpaths = [basepath+'/best-fit'] + [basepath + '/%04d/fit_p/'%i for i in range(100, -1, -20)]
        for path in bpaths:
            if os.path.isdir(path): break
        print(path)
        bfit = mapp.Observable.load(path)
        datapp = dataprsd
        lss, lww, cc, lbl = '--', 2, 'C0', 'Thermal, rsd z=%.2f'%(1/aa-1)
        makeplot(bfit, datapp, lss, lww, cc, lbl)
        print('%s done'%lbl)
    except Exception as e: print(e)
            
    try:        
        aa = 0.3333
        dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_ang%0.1f/'%(aa, kmin, ang)
        dpath += 'L%04d-N%04d/stage2/'%(bs, nc)
        dataprsd = mapp.Observable.load(dpath+'ZA/opt_s999_h1massA_fourier_rsdpos/datap')
        basepath = dpath+'ZA/opt_s999_h1massA_fourier_rsdpos/%d-0.00/'%(nc)
        bpaths = [basepath+'/best-fit'] + [basepath + '/%04d/fit_p/'%i for i in range(100, -1, -20)]
        for path in bpaths:
            if os.path.isdir(path): break
        print(path)
        bfit = mapp.Observable.load(path)
        datapp = dataprsd
        lss, lww, cc, lbl = '--', 2, 'C1', 'Thermal, rsd z=%.2f'%(1/aa-1)
        makeplot(bfit, datapp, lss, lww, cc, lbl)
        print('%s done'%lbl)
    except Exception as e: print(e)


    ax[0].set_ylabel('$r_{cc}$', fontdict=font)
    ax[1].set_ylabel(r'$\sqrt{P_{\rm mod}/P_{hh}}$', fontdict=font)
    for axis in ax:
        axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
        axis.set_xscale('log')
        axis.grid(which='both')
        axis.legend(prop=fontmanage)

    # Put on some more labels.
    for axis in ax:
        axis.set_xscale('log')
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
    ##and finish
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if rank  == 0: plt.savefig(figpath + '/zcompare_L%04d.pdf'%(bs))



################


if __name__=="__main__":
    make_rep_plot()
    #
