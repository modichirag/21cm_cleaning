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

print(font)


#
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--bs', help='boxsize', default=1024, type=float)
parser.add_argument('-n', '--nmesh', help='nmesh', default=256, type=int)

args = parser.parse_args()

figpath = './figs/'

bs, nc = args.bs, args.nmesh
pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank


################
def make_rep_plot(Nmu=4, nx=2, ny=2):
    """Does the work of making the real-space xi(r) and b(r) figure."""
    

    linestyle = ['-', '--']
    colors = ['C0', 'C1', 'C2']
    fig, ax = plt.subplots(nx, ny, figsize=(ny*3+2, nx*3+0), sharex=True, sharey=True)
    for ia, aa  in enumerate([0.3333, 0.2000, 0.1429]):
        zz = 1/aa-1
        for iw, wopt in enumerate(['opt', 'pess']):
        #for iw, wopt in enumerate(['opt']):
            lss = linestyle[iw]
            #for it, thopt in enumerate(['opt', 'pess', 'reas']):
            for it, thopt in enumerate([ 'reas']):
                if rank == 0: print(aa, wopt, thopt)
                cc = colors[ia]
                try:
                    angle = np.round(mapn.wedge(zz, att=wopt, angle=True), 0)
                    dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_%s/'%(aa, 0.03, wopt)
                    dpath += 'L%04d-N%04d-R//thermal-%s-hex/ZA/opt_s999_h1massA_fourier_rsdpos/'%(bs, nc, thopt)
                    datapp = mapp.Observable.load(dpath+'/datap')
                    bpaths = [dpath+'%d-0.00//best-fit'%nc] + [dpath + '%d-0.00//%04d/fit_p/'%(nc,i) for i in range(100, 50, -20)]
                    for path in bpaths:
                        if os.path.isdir(path): 
                            break
                    if rank == 0: print(path)
                    bfit = mapp.Observable.load(path)
                    rpfit = rp.evaluate2d1(bfit, datapp, Nmu=Nmu, field='mapp')
                    #
                    mus = rpfit[0]['mu'][5:].mean(axis=0)
                    for i in range(10):
                        if np.isnan(mus).sum(): mus = rpfit[0]['mu'][i:].mean(axis=0)
                    k = rpfit[0]['k'].mean(axis=1)
                    for i in range(mus.size):
                        axis = ax.flatten()[i]
                        if i == 0 and iw==0:
                            lbl = 'z = %.1f'%(1/aa-1)
                        elif i == 1 and ia == 0:
                            lbl = 'Wedge = %s'%wopt
                        else: lbl = None
                        rcc = rpfit[0]['power'][:, i]/(rpfit[1]['power'][:, i]*rpfit[2]['power'][:, i])**0.5
                        axis.plot(k, rcc, ls=lss, lw=2, color=cc, label=lbl)
                        if Nmu == 4: axis.text(0.02, 0.41, r'$\mu = %.2f$'%mus[i],color='black',ha='left',va='bottom', fontdict=font)
                        if Nmu == 8: axis.text(0.05, 0.41, r'$\mu = %.2f$'%mus[i],color='black',ha='left',va='bottom', fontdict=font)
                except Exception as e: 
                    if rank == 0: print(e)


    ##
    for axis in ax[:, 0]: axis.set_ylabel('$r_{cc}$', fontdict=font)
    for axis in ax[1]: axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    for axis in ax.flatten():
        axis.set_xscale('log')
        axis.set_xlim(-0.05, 1.1)
        axis.grid(which='both', lw=0.2, alpha=0.2, color='gray')
        axis.legend(prop=fontmanage)
        #axis.set_ylim(0.5, 1.05)
    # Put on some more labels.
    for axis in ax.flatten():
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
    ##and finish
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if rank == 0: plt.savefig(figpath + '/rep2d_L%04d_mu%d.pdf'%(bs, Nmu))



################


if __name__=="__main__":
    #make_rep_plot()
    make_rep_plot(Nmu=8, nx=2, ny=4)
    #
