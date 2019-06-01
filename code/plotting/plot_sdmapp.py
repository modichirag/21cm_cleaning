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
parser.add_argument('-a', '--aa', help='scale factor', default=0.2000, type=float)
parser.add_argument('-l', '--bs', help='boxsize', default=1024, type=float)
parser.add_argument('-n', '--nmesh', help='nmesh', default=256, type=int)
parser.add_argument('-t', '--angle', help='angle of the wedge', default=50, type=float)
parser.add_argument('-k', '--kmin', help='kmin of the wedge', default=0.01, type=float)
args = parser.parse_args()

figpath = './figs/'

bs, nc, aa = args.bs, args.nmesh, args.aa
zz = 1/aa- 1
kmin = args.kmin
ang = args.angle
pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])

#dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_ang%0.1f/'%(aa, kmin, ang)
#dpath += 'L%04d-N%04d/'%(bs, nc)
wopt = 'opt'
thopt = 'reas'
dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_%s/'%(aa, 0.03, wopt)
dpath += 'L%04d-N%04d-R//thermal-%s-hex/'%(bs, nc, thopt)
################
def make_rep_plot():
    """Does the work of making the real-space xi(r) and b(r) figure."""
    

    noises = np.loadtxt('/global/u1/c/chmodi/Programs/21cm/21cm_cleaning/data/summaryHI.txt').T
    for i in range(noises[0].size):
        if noises[0][i] == np.round(1/aa-1, 2): noise = noises[3][i]
    print(noise)

    dataprsd = mapp.Observable.load(dpath+'ZA/opt_s999_h1massA_fourier_rsdpos/datap')
    dataprsdup = mapp.Observable.load(dpath+'ZA/opt_s999_h1massA_fourier_rsdpos/datap_up')
    #except Exception as e: print(e)

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    def makeplot(bfit, datapp, lss, lww, cc, lbs, al=1):
        rpfit = rp.evaluate(bfit, datapp)[:-2]
        #lbls = ['HI', 'Initial', 'Final']
        
        for ii in [0, 1, 2]:
            ax[0].plot(rpfit[0]['k'], rpfit[ii]['power']/(rpfit[2*ii+3]['power']*rpfit[2*ii+4]['power'])**0.5, ls=lss, lw=lww, color='C%d'%ii, label=lbls[ii], alpha=al)
            ax[1].plot(rpfit[0]['k'], (rpfit[2*ii+3]['power']/rpfit[2*ii+4]['power'])**0.5, ls=lss, lw=lww, color='C%d'%ii,alpha=al)
        

    try:
        basepath = dpath+'ZA/opt_s999_h1massA_fourier_rsdpos/upsample2/%d-0.00/'%(2*nc)
        bpaths = [basepath+'/best-fit'] + [basepath + '/%04d/fit_p/'%i for i in range(100, -1, -20)]
        for path in bpaths:
            if os.path.isdir(path): break
        print(path)
        bfit = mapp.Observable.load(path)
        datapp = dataprsdup
        lss, lww, cc, lbl = '-', 2, 'C1', 'rsd up'
        lbls = ['HI', 'Initial', 'Final']
        makeplot(bfit, datapp, lss, lww, cc, lbl)
        print('%s done'%lbl)
    except Exception as e: print(e)

    #rsd
    try:
        basepath = dpath+'ZA/opt_s999_h1massA_fourier_rsdpos/%d-0.00/'%(nc)
        bpaths = [basepath+'/best-fit'] + [basepath + '/%04d/fit_p/'%i for i in range(100, -1, -20)]
        for path in bpaths:
            if os.path.isdir(path): break
        print(path)
        bfit = mapp.Observable.load(path)
        datapp = dataprsd
        lss, lww, cc, lbl, al = '--', 3, 'C0', 'rsd', 0.5
        lbls = ['Fid Recon', None, None]
        makeplot(bfit, datapp, lss, lww, cc, lbls, al)
        print('%s done'%lbl)
    except Exception as e: print(e)

    ax[0].set_ylabel('$r_{cc}$', fontdict=font)
    ax[0].set_ylim(-0.05, 1.1)
    ax[1].set_ylabel(r'$T_f$', fontdict=font)
    ax[1].set_ylim(-0.05, 3)
    for axis in ax:
        axis.axhline(1, color='k', ls='--')
        axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
        axis.set_xscale('log')
        axis.grid(which='both', alpha=0.2, lw=0.2, color='gray')
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
    plt.savefig(figpath + '/sdmap_L%04d_%04d.pdf'%(bs, aa*10000))



################


if __name__=="__main__":
    make_rep_plot()
    #
