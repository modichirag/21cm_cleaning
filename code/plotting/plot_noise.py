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
parser.add_argument( '--pp', help='upsample', default=1) 

args = parser.parse_args()

figpath = './figs/'

bs, nc = args.bs, args.nmesh
nc2 = nc*2
if args.pp: pm = ParticleMesh(BoxSize=bs, Nmesh=[nc2, nc2, nc2])
else: pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank



################
def savenoisep():
    for ia, aa  in enumerate([0.3333, 0.2000, 0.1429]):
        zz = 1/aa-1
        for iw, wopt in enumerate(['opt', 'pess']):
        #for iw, wopt in enumerate(['opt']):
            for it, thopt in enumerate(['opt', 'pess', 'reas']):
            #for it, thopt in enumerate([ 'reas']):
                if rank == 0: print(aa, wopt, thopt)
                dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_%s/'%(aa, 0.03, wopt)
                dpath += 'L%04d-N%04d-R//thermal-%s-hex/ZA/opt_s999_h1massA_fourier_rsdpos/'%(bs, nc, thopt)
                ofolder = '../../data/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/'%(wopt, thopt)
                try: os.makedirs(ofolder)
                except: pass
                fname = ofolder + 'pnoise-L%04d_%0.4f.txt'%(bs, aa)
                if args.pp : fname = fname[:-4] + '-up.txt'
                try:
                    rep = np.loadtxt(fname).T
                    rpfit = [{'k':rep[0], 'power':rep[i+1]} for i in range(3)]
                except:
                    if args.pp:
                        ivar = BigFileMesh(dpath+'/ivarmesh_up', 'ivar').paint()
                    else:
                        ivar = BigFileMesh(dpath+'/ivarmesh', 'ivar').paint()

                    svar = (ivar.r2c()**-0.5).c2r()
                    p0 = FFTPower(svar, mode='1d').power
                    if rank == 0: np.savetxt(fname, np.stack([p0['k']]+ [p0['power'].real]).T, header='k, p0')
    
def savenoisep2d(Nmu=4):
    for ia, aa  in enumerate([0.3333, 0.2000, 0.1429]):
        zz = 1/aa-1
        for iw, wopt in enumerate(['opt', 'pess']):
        #for iw, wopt in enumerate(['opt']):
            for it, thopt in enumerate(['opt', 'pess', 'reas']):
            #for it, thopt in enumerate([ 'reas']):
                if rank == 0: print(aa, wopt, thopt)
                dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_%s/'%(aa, 0.03, wopt)
                dpath += 'L%04d-N%04d-R//thermal-%s-hex/ZA/opt_s999_h1massA_fourier_rsdpos/'%(bs, nc, thopt)
                ofolder = '../../data/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/Nmu%d/'%(wopt, thopt, Nmu)
                try: os.makedirs(ofolder)
                except: pass
                fname = ofolder + 'pnoise-L%04d_%0.4f.txt'%(bs, aa)
                if args.pp : fname = fname[:-4] + '-up.txt'
                try:
                    rep = np.loadtxt(fname).T
                    rpfit = [{'k':rep[0], 'power':rep[i+1]} for i in range(3)]
                except:
                    pass
                if args.pp:
                    ivar = BigFileMesh(dpath+'/ivarmesh_up', 'ivar').paint()
                else:
                    ivar = BigFileMesh(dpath+'/ivarmesh', 'ivar').paint()

                svar = (ivar.r2c()**-0.5).c2r()
                p0 = FFTPower(svar, mode='2d', Nmu=Nmu).power
                if rank == 0: np.savetxt(fname, p0['power'].real)



def make_rep_plot(Nmu=4,):
    """Does the work of making the real-space xi(r) and b(r) figure."""
    

    mub = np.linspace(0, 1, Nmu+1)
    linestyle = ['-', '-.', ':', '--']
    colors = ['C0', 'C1', 'C2']
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for ia, aa  in enumerate([0.3333, 0.2000, 0.1429]):
        zz = 1/aa-1
        #for iw, wopt in enumerate(['opt', 'pess']):
        for iw, wopt in enumerate(['opt']):
            for it, thopt in enumerate(['opt', 'pess', 'reas']):
            #for it, thopt in enumerate([ 'reas']):
                if rank == 0: print(aa, wopt, thopt)
                cc = colors[it]
                try:
                    angle = np.round(mapn.wedge(zz, att=wopt, angle=True), 0)
                    dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_%s/'%(aa, 0.03, wopt)
                    dpath += 'L%04d-N%04d-R//thermal-%s-hex/ZA/opt_s999_h1massA_fourier_rsdpos/'%(bs, nc, thopt)
                    ofolder = '../../data/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/Nmu%d/'%(wopt, thopt, Nmu)
                    data = ofolder + 'dataw-L%04d_%0.4f.txt'%(bs, aa)
                    noise = ofolder + 'pnoise-L%04d_%0.4f.txt'%(bs, aa)
                    if args.pp : 
                        data = data[:-4] + '-up'
                        noise = noise[:-4] + '-up'
                    k = np.loadtxt(data+'-k.txt').T
                    dfit = np.loadtxt(data+'-pm2.txt').T
                    nfit = np.loadtxt(noise+'.txt').T
                    
                    for i in range(0, k.shape[0]):
                        lss = linestyle[i%len(linestyle)]
                        axis = ax.flatten()[ia]
                        if ia==2 and it==0:
                            lbl = r'$\mu = %.2f-%.2f$'%(mub[i], mub[i+1])                            
                        elif ia == 1 and i == 1:
                            if thopt == 'reas': thopt = 'fid'
                            lbl = 'Noise = %s'%thopt
                        else: lbl = None
                        axis.plot(k[i], dfit[i]/(dfit[i] + nfit[i]), ls=lss, lw=2, color=cc, label=lbl)
                        #axis.plot(k[i], nfit[i], ls=lss, lw=2, color=cc, label=lbl)
                        #axis.set_yscale('log')
                        #axis.set_ylim(10, 1e4)
                        #
                        axis.text(0.22, 0.0, 'z = %.1f'%(1/aa-1),color='black',ha='left',va='bottom', fontdict=font)


                except Exception as e: 
                    if rank == 0: print(e)


    ##
    ax[0].set_ylabel(r'$S/(S+N)$', fontdict=font)
    for axis in ax[:]: axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    for axis in ax.flatten():
        axis.axhline(1, ls="--", color='k')
        #axis.set_xscale('log')
        axis.set_xlim(0.008, 1.3)
        #axis.set_yscale('log')
        #axis.set_ylim(1e-3, 1.3)
        axis.grid(which='both', lw=0.2, alpha=0.2, color='gray')
        #axis.legend(loc='center left', prop=fontmanage)
        axis.legend(loc=0, prop=fontmanage)
    # Put on some more labels.
    for axis in ax.flatten():
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
    ##and finish
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if rank  == 0 and not args.pp:  plt.savefig(figpath + '/noise2d_L%04d_mu%d.pdf'%(bs, Nmu))
    if rank  == 0 and args.pp:  plt.savefig(figpath + '/noise2d_L%04d_mu%d_up.pdf'%(bs, Nmu))
    for axis in ax.flatten():
        axis.set_xscale('log')
    if rank  == 0 and not args.pp:  plt.savefig(figpath + '/noise2d_L%04d_mu%d_log.pdf'%(bs, Nmu))
    if rank  == 0 and args.pp:  plt.savefig(figpath + '/noise2d_L%04d_mu%d_up_log.pdf'%(bs, Nmu))
        
##

################


if __name__=="__main__":
    #print('1d')
    #savenoisep()
    #print('2d, Nmu4')
    #savenoisep2d(Nmu=4)
    #print('2d, Nmu3')
    #savenoisep2d(Nmu=3)

    make_rep_plot()
    make_rep_plot(Nmu=3)
    #
