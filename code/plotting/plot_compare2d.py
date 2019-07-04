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
parser.add_argument('-l', '--bs', help='boxsize', default=1024, type=float)
parser.add_argument('-n', '--nmesh', help='nmesh', default=256, type=int)
parser.add_argument( '--pp', help='upsample', default=0) 

args = parser.parse_args()

figpath = './figs/'

bs, nc = args.bs, args.nmesh
nc2 = nc*2
if args.pp: pm = ParticleMesh(BoxSize=bs, Nmesh=[nc2, nc2, nc2])
else: pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank



################
def make_rep_plot(Nmu=4, nx=2, ny=2):
    """Does the work of making the real-space xi(r) and b(r) figure."""
    

    mub = np.linspace(0, 1, Nmu+1)
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
                    fname = './tmpdata/mapp%d-L%04d_%0.4f_kmin%0.2f_%s-th%shex'%(Nmu, bs, aa, 0.03, wopt, thopt)
                    if args.pp : fname = fname[:] + '-up'
                    try:
                        rpfit = [{'k':np.loadtxt(fname+'-k.txt'), 'power':np.loadtxt(fname+'-p%d.txt'%i)} for i in range(3)]
                    except:
                        if args.pp:
                            datapp = mapp.Observable.load(dpath+'/datap_up')
                            bpaths = [dpath+'upsample2/%d-0.00//best-fit'%nc2] + [dpath + 'upsample2/%d-0.00//%04d/fit_p/'%(nc2,i) for i in range(100, 50, -20)]
                        else:
                            datapp = mapp.Observable.load(dpath+'/datap')
                            bpaths = [dpath+'%d-0.00//best-fit'%nc] + [dpath + '%d-0.00//%04d/fit_p/'%(nc,i) for i in range(100, 50, -20)]
                        for path in bpaths:
                            if os.path.isdir(path): 
                                break
                        if rank == 0: print(path)
                        bfit = mapp.Observable.load(path)
                        rpfit = rp.evaluate2d1(bfit, datapp, Nmu=Nmu, field='mapp')
                        if rank == 0: 
                            np.savetxt(fname + '-k.txt', rpfit[0]['k'])
                            for ip in range(3): np.savetxt(fname + '-p%d.txt'%ip, rpfit[ip]['power'].real)

                    #
                    k = rpfit[0]['k'].mean(axis=1)
                    for i in range(Nmu):
                        axis = ax.flatten()[i]
                        if i == 0 and iw==0:
                            lbl = 'z = %.1f'%(1/aa-1)
                        elif i == 1 and ia == 0:
                            lbl = 'Wedge = %s'%wopt
                        else: lbl = None
                        rcc = rpfit[0]['power'][:, i]/(rpfit[1]['power'][:, i]*rpfit[2]['power'][:, i])**0.5
                        axis.plot(k, rcc, ls=lss, lw=2, color=cc, label=lbl)
                        if Nmu == 4: axis.text(0.1, 0.3, r'$\mu = %.2f-%.2f$'%(mub[i], mub[i+1]),color='black',ha='left',va='bottom', fontdict=font)
                        #if Nmu == 8: axis.text(0.05, 0.41, r'$\mu = %.2f$'%mus[i],color='black',ha='left',va='bottom', fontdict=font)
                        if Nmu == 8: axis.text(0.1, 0.3, r'$\mu = %.2f-%.2f$'%(mub[i], mub[i+1]),color='black',ha='left',va='bottom', fontdict=font)
                except Exception as e: 
                    if rank == 0: print(e)


    ##
    for axis in ax[:, 0]: axis.set_ylabel('$r_{cc}$', fontdict=font)
    for axis in ax[1]: axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    for axis in ax.flatten():
        axis.set_xscale('log')
        axis.axhline(1, ls="--", color='k')
        axis.set_xlim(0.02, 1.1)
        axis.grid(which='both', lw=0.2, alpha=0.2, color='gray')
        axis.legend(loc='center left', prop=fontmanage)
        #axis.set_ylim(0.5, 1.05)
    # Put on some more labels.
    for axis in ax.flatten():
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
    ##and finish
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if rank  == 0 and not args.pp:  plt.savefig(figpath + '/rep2d_L%04d_mu%d.pdf'%(bs, Nmu))
    if rank  == 0 and args.pp:  plt.savefig(figpath + '/rep2d_L%04d_mu%d_up.pdf'%(bs, Nmu))


################
def make_mu_plot(Nmu=8, nx=2, ny=2):
    """Does the work of making the real-space xi(r) and b(r) figure."""
    

    mub = np.linspace(0, 1, Nmu+1)
    linestyle = ['-', '--']
    markers = ['o', 'x']
    colors = ['C0', 'C1', 'C2']
    fig, ax = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    for ia, aa  in enumerate([0.3333, 0.2000, 0.1429]):
        zz = 1/aa-1
        for iw, wopt in enumerate(['opt', 'pess']):
        #for iw, wopt in enumerate(['opt']):
            lss = linestyle[iw]
            mm = markers[iw]
            #for it, thopt in enumerate(['opt', 'pess', 'reas']):
            for it, thopt in enumerate([ 'reas']):
                if rank == 0: print(aa, wopt, thopt)
                cc = colors[ia]
                try:
                    angle = np.round(mapn.wedge(zz, att=wopt, angle=True), 0)
                    dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_%s/'%(aa, 0.03, wopt)
                    dpath += 'L%04d-N%04d-R//thermal-%s-hex/ZA/opt_s999_h1massA_fourier_rsdpos/'%(bs, nc, thopt)
                    fname = './tmpdata/mapp%d-L%04d_%0.4f_kmin%0.2f_%s-th%shex'%(Nmu, bs, aa, 0.03, wopt, thopt)
                    if args.pp : fname = fname[:] + '-up'
                    try:
                        rpfit = [{'k':np.loadtxt(fname+'-k.txt'), 'power':np.loadtxt(fname+'-p%d.txt'%i)} for i in range(3)]
                    except:
                        if args.pp:
                            datapp = mapp.Observable.load(dpath+'/datap_up')
                            bpaths = [dpath+'upsample2/%d-0.00//best-fit'%nc2] + [dpath + 'upsample2/%d-0.00//%04d/fit_p/'%(nc2,i) for i in range(100, 50, -20)]
                        else:
                            datapp = mapp.Observable.load(dpath+'/datap')
                            bpaths = [dpath+'%d-0.00//best-fit'%nc] + [dpath + '%d-0.00//%04d/fit_p/'%(nc,i) for i in range(100, 50, -20)]
                        for path in bpaths:
                            if os.path.isdir(path): break
                        if rank == 0: print(path)
                        bfit = mapp.Observable.load(path)
                        rpfit = rp.evaluate2d1(bfit, datapp, Nmu=Nmu, field='mapp')
                    #
                    mus = np.linspace(0, 1, Nmu)
                    k = rpfit[0]['k'].mean(axis=1)
                    k[np.isnan(k)] = -1
                    kk = np.logspace(np.log10(0.05), 0, 6)
                    for i in range(kk.size):
                        axis = ax.flatten()[i]
                        if i == 0 and iw==0:
                            lbl = 'z = %.1f'%(1/aa-1)
                        elif i == 1 and ia == 0:
                            lbl = 'Wedge = %s'%wopt
                        else: lbl = None
                        ik = np.where(k > kk[i])[0][0]
                        #if rank == 0: print(k, kk[i], ik)
                        #ik = ik[0][0]
                        rcc = (rpfit[0]['power'][ik, :]/(rpfit[1]['power'][ik, :]*rpfit[2]['power'][ik, :])**0.5).real
                        axis.plot(mus, rcc, ls=lss, lw=2, color=cc, label=lbl, marker=mm)
                        if args.pp: axis.text(0.7, 0.41, r'$k = %.3f$'%(kk[i]),color='black',ha='left',va='bottom', fontdict=font)
                        else: axis.text(0.75, 0.15, r'$k = %.3f$'%(kk[i]),color='black',ha='left',va='bottom', fontdict=font)
                        amu = np.sin(angle*np.pi/180.) 
                        print(angle, amu)
                        axis.axvline(amu, color=cc, ls=lss, lw=1, alpha=0.5)
                except Exception as e: 
                    if rank == 0: print(e)


    ##
    for axis in ax[:, 0]: axis.set_ylabel('$r_{cc}$', fontdict=font)
    for axis in ax[1]: axis.set_xlabel(r'$\mu$', fontdict=font)
    for axis in ax.flatten():
        axis.axhline(1, ls="--", color='k')
        axis.set_xlim(-0.05, 1.05)
        axis.grid(which='both', lw=0.2, alpha=0.2, color='gray')
        axis.legend(loc=3, prop=fontmanage)
        #axis.set_ylim(0.5, 1.05)
    # Put on some more labels.
    for axis in ax.flatten():
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
    ##and finish
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if rank  == 0 and not args.pp:  plt.savefig(figpath + '/rep2dmu_L%04d_mu%d.pdf'%(bs, Nmu))
    if rank  == 0 and args.pp:  plt.savefig(figpath + '/rep2dmu_L%04d_mu%d_up.pdf'%(bs, Nmu))


################


if __name__=="__main__":
    #make_rep_plot()
    #make_rep_plot(Nmu=8, nx=2, ny=4)
    make_mu_plot()
    #
