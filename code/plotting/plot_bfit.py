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
from getbiasparams import getbias, getbiask
import tools, za
import features as ft
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


cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)

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

dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_ang%0.1f/'%(aa, kmin, ang)
dpath += 'L%04d-N%04d/'%(bs, nc)

################
def make_bias_plot():
    """Does the work of making the real-space xi(r) and b(r) figure."""
    
    hmesh = BigFileMesh(dpath+'ZA/opt_s999_h1massA_fourier/datap', 'mapp').paint()

    noises = np.loadtxt('/global/u1/c/chmodi/Programs/21cm/21cm_cleaning/data/summaryHI.txt').T
    for i in range(noises[0].size):
        if noises[0][i] == np.round(1/aa-1, 2): noise = noises[3][i]
    print(noise)


    zamod = BigFileMesh(dpath+'ZA/opt_s999_h1massA_fourier/fitp/', 'mapp').paint()
    pmmod = BigFileMesh(dpath+'T05-B1/opt_s999_h1massA_fourier/fitp/', 'mapp').paint()
    fin = BigFileMesh(dpath+'ZA/opt_s999_h1massA_fourier/datap/', 'd').paint()
    fin /= fin.cmean()
    fin -= 1
    finsm = ft.smooth(fin, 3, 'gauss')
    grid = pm.mesh_coordinates()*bs/nc
    params, finmod = getbias(pm, basemesh=finsm, hmesh=hmesh, pos=grid, grid=grid)
    models = [zamod, pmmod, finmod, fin.copy()]
    lbls = ['ZA shift', 'PM shift', 'Eulerian (R=3)', 'Final Matter']
    lss = ['-', '-', '-', '--']

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for ii, mod in enumerate(models):

        pmod = FFTPower(mod, mode='1d').power
        k, pmod = pmod['k'], pmod['power']
        ph = FFTPower(hmesh, mode='1d').power['power']
        pxmodh = FFTPower(hmesh, second=mod, mode='1d').power['power']
        perr = FFTPower(hmesh -mod, mode='1d').power['power']
         
        ax[0].plot(k, pxmodh/(pmod*ph)**0.5, label=lbls[ii], lw=2, ls=lss[ii])
        ax[0].set_ylabel('$r_{cc}$', fontdict=font)

        ax[1].plot(k,(pmod/ph)**0.5, lw=2, ls=lss[ii])
        #ax[1].set_ylabel(r'$\sqrt{P_{\rm mod}/P_{hh}}$', fontdict=font)
        ax[1].set_ylabel(r'$T_f$', fontdict=font)

        ax[2].plot(k, perr, lw=2, ls=lss[ii])
        ax[2].set_ylabel(r'$P_{\delta_{\rm mod}-\delta_h}$', fontdict=font)

    ax[2].set_yscale('log')
    ax[2].axhline(noise)
    for axis in ax:
        axis.set_xscale('log')
        axis.grid(which='both')
        axis.legend(prop=fontmanage)
        axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
        axis.set_xscale('log')

    # Put on some more labels.
    for axis in ax:
        axis.set_xscale('log')
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
    ##and finish
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(figpath + '/bfit_L%04d_%04d.pdf'%(bs, aa*10000))






################
def make_bias2p_plot():
    """Does the work of making the real-space xi(r) and b(r) figure."""
    
    hmesh = BigFileMesh(dpath+'ZA/opt_s999_h1massA_fourier/datap', 'mapp').paint()
    grid = pm.mesh_coordinates()*bs/nc

    noises = np.loadtxt('/global/u1/c/chmodi/Programs/21cm/21cm_cleaning/data/summaryHI.txt').T
    for i in range(noises[0].size):
        if noises[0][i] == np.round(1/aa-1, 2): noise = noises[3][i]
    print(noise)

    zamod = BigFileMesh(dpath+'ZA/opt_s999_h1massA_fourier/fitp/', 'mapp').paint()
    try: pmmod = BigFileMesh(dpath+'T05-B1/opt_s999_h1massA_fourier/fitp/', 'mapp').paint()
    except:
        lin = BigFileMesh(dpath+'ZA/opt_s999_h1massA_fourier/datap/', 's').paint()
        dyn = BigFileCatalog('/global/cscratch1/sd/chmodi/m3127/cm_lowres/5stepT-B1/%d-%d-9100-fixed/fastpm_%0.4f/1'%(bs, nc, aa))
        fpos = dyn['Position']
        _, pmod = getbias(pm, basemesh=lin, hmesh=hmesh, pos=fpos, grid=grid, fpos=fpos)

    fin = BigFileMesh(dpath+'ZA/opt_s999_h1massA_fourier/datap/', 'd').paint()
    fin /= fin.cmean()
    fin -= 1
    finsm = ft.smooth(fin, 3, 'gauss')
    params, finmod = getbias(pm, basemesh=finsm, hmesh=hmesh, pos=grid, grid=grid)
    models = [zamod, pmmod, finmod, finsm.copy()]
    lbls = ['ZA shift', 'PM shift', 'Eulerian (R=3)', 'Eulerian (R=3), $b_1$']
    lss = ['-', '-', '-', '--']

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for ii, mod in enumerate(models):

        pmod = FFTPower(mod, mode='1d').power
        k, pmod = pmod['k'], pmod['power']
        ph = FFTPower(hmesh, mode='1d').power['power']
        if ii == 3: 
            mod *= (ph[1]/pmod[1]).real**0.5 
            pmod *= (ph[1]/pmod[1]).real 
        pxmodh = FFTPower(hmesh, second=mod, mode='1d').power['power']
        perr = FFTPower(hmesh -mod, mode='1d').power['power']
 
        ax[0].plot(k, pxmodh/(pmod*ph)**0.5, label=lbls[ii], lw=2, ls=lss[ii])
        ax[0].set_ylabel('$r_{cc}$', fontdict=font)

        #if ii == 3: pmod *= ph[1]/pmod[1]
        ax[1].plot(k,(pmod/ph)**0.5, lw=2, ls=lss[ii])
        ax[1].set_ylabel(r'$T_f$', fontdict=font)
        #ax[1].set_ylabel(r'$\sqrt{P_{\rm mod}/P_{hh}}$', fontdict=font)

    for axis in ax:
        axis.set_xscale('log')
        axis.grid(which='both')
        axis.legend(prop=fontmanage)
        axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
        axis.set_xscale('log')

    # Put on some more labels.
    for axis in ax:
        axis.set_xscale('log')
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
    ##and finish
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(figpath + '/bfit2p_L%04d_%04d.pdf'%(bs, aa*10000))



################

def make_rsdbias_plot():
    """Does the work of making the real-space xi(r) and b(r) figure."""
    
    hmesh = BigFileMesh(dpath+'ZA/opt_s999_h1massA_fourier_rsdpos/datap', 'mapp').paint()

    noises = np.loadtxt('/global/u1/c/chmodi/Programs/21cm/21cm_cleaning/data/summaryHI.txt').T
    for i in range(noises[0].size):
        if noises[0][i] == np.round(1/aa-1, 2): noise = noises[3][i]
    print(noise)

    dgrow = cosmo.scale_independent_growth_factor(zz)
    rsdfac = 0 
    with open('/global/cscratch1/sd/chmodi/m3127/H1mass/highres/2560-9100-fixed/fastpm_%0.4f/Header/attr-v2'%aa) as ff:
        for line in ff.readlines():
            if 'RSDFactor' in line: rsdfac = float(line.split()[-2])
    print(rsdfac)            
    los = np.array([0, 0, 1]).reshape(1, -1)
    
    grid = pm.mesh_coordinates()*bs/nc
    lin = BigFileMesh(dpath+'ZA/opt_s999_h1massA_fourier/datap/', 's').paint()
    dyn = BigFileCatalog('/global/cscratch1/sd/chmodi/m3127/cm_lowres/5stepT-B1/%d-%d-9100-fixed/fastpm_%0.4f/1'%(bs, nc, aa))
    zapos = za.doza(lin.r2c(), grid, z=zz, dgrow=dgrow)
    zdisp = za.doza(lin.r2c(), grid, z=zz, displacement=True, dgrow=dgrow)
    zarsd = cosmo.scale_independent_growth_rate(zz)*zdisp*los
    pmrsd = dyn['Velocity']*rsdfac*los
    fpos = dyn['Position']

    params, fitrsd = getbias(pm, basemesh=lin, hmesh=hmesh, pos=zapos+zarsd, grid=grid, fpos=zapos+zarsd)
    params, fitreal = getbias(pm, basemesh=lin, hmesh=hmesh, pos=zapos, grid=grid, fpos=zapos+zarsd)
    models = [fitrsd, fitreal]
    lbls = ['Fit Redshift', 'Fit Real']
    lss = ['-', '-']

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for ii, mod in enumerate(models):

        pmod = FFTPower(mod, mode='1d').power
        k, pmod = pmod['k'], pmod['power']
        ph = FFTPower(hmesh, mode='1d').power['power']
        pxmodh = FFTPower(hmesh, second=mod, mode='1d').power['power']
        perr = FFTPower(hmesh -mod, mode='1d').power['power']
         
         
        ax[0].plot(k, pxmodh/(pmod*ph)**0.5, label=lbls[ii], lw=2, ls=lss[ii])
        ax[0].set_ylabel('$r_{cc}$', fontdict=font)

        ax[1].plot(k,(pmod/ph)**0.5, lw=2, ls=lss[ii])
        ax[1].set_ylabel(r'$\sqrt{P_{\rm mod}/P_{hh}}$', fontdict=font)

        ax[2].plot(k, perr, lw=2, ls=lss[ii])
        ax[2].set_ylabel(r'$P_{\delta_{\rm mod}-\delta_h}$', fontdict=font)

    ax[2].set_yscale('log')
    ax[2].axhline(noise, color='k', ls="--")
    for axis in ax:
        axis.set_xscale('log')
        axis.grid(which='both')
        axis.legend(prop=fontmanage)
        axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
        axis.set_xscale('log')

    # Put on some more labels.
    for axis in ax:
        axis.set_xscale('log')
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
    ##and finish
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(figpath + '/bfitrsd_L%04d_%04d.pdf'%(bs, aa*10000))


################

def make_biask_plot():
    """Does the work of making the real-space xi(r) and b(r) figure."""
    
    hmesh = BigFileMesh(dpath+'ZA/opt_s999_h1massA_fourier/datap', 'mapp').paint()

    noises = np.loadtxt('/global/u1/c/chmodi/Programs/21cm/21cm_cleaning/data/summaryHI.txt').T
    for i in range(noises[0].size):
        if noises[0][i] == np.round(1/aa-1, 2): noise = noises[3][i]
    print(noise)

    zamod = BigFileMesh(dpath+'ZA/opt_s999_h1massA_fourier/fitp/', 'mapp').paint()
    pmmod = BigFileMesh(dpath+'T05-B1/opt_s999_h1massA_fourier/fitp/', 'mapp').paint()
    fin = BigFileMesh(dpath+'ZA/opt_s999_h1massA_fourier/datap/', 'd').paint()
    fin /= fin.cmean()
    fin -= 1
    finsm = ft.smooth(fin, 3, 'gauss')
    grid = pm.mesh_coordinates()*bs/nc
    params, finmod = getbias(pm, basemesh=finsm, hmesh=hmesh, pos=grid, grid=grid)
    models = [zamod, pmmod, finmod, finsm.copy()]
    lbls = ['ZA shift', 'PM shift', 'Eulerian (R=3)', 'Eulerian (R=3), $b_1$']
    lss = ['-', '-', '-', '--']

    lin = BigFileMesh(dpath+'ZA/opt_s999_h1massA_fourier/datap/', 's').paint()
    dyn = BigFileCatalog('/global/cscratch1/sd/chmodi/m3127/cm_lowres/5stepT-B1/%d-%d-9100-fixed/fastpm_%0.4f/1'%(bs, nc, aa))
    dgrow = cosmo.scale_independent_growth_factor(zz)
    zapos = za.doza(lin.r2c(), grid, z=zz, dgrow=dgrow)
    fpos = dyn['Position']

    params, _ = getbias(pm, basemesh=lin, hmesh=hmesh, pos=zapos, grid=grid)
    kk, paramsk, _ = getbiask(pm, basemesh=lin, hmesh=hmesh, pos=zapos, grid=grid)

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for ii, mod in enumerate(models):

        pmod = FFTPower(mod, mode='1d').power
        k, pmod = pmod['k'], pmod['power']
        ph = FFTPower(hmesh, mode='1d').power['power']
        if ii == 3: 
            mod *= (ph[1]/pmod[1]).real **0.5
            pmod *= (ph[1]/pmod[1]).real 
        pxmodh = FFTPower(hmesh, second=mod, mode='1d').power['power']
        perr = FFTPower(hmesh -mod, mode='1d').power['power']
         
        ax[0].plot(k, perr, lw=2, ls=lss[ii], label=lbls[ii])
        ax[0].set_ylabel(r'$P_{\rm err}$', fontdict=font)
        ax[0].set_yscale('log')
    ax[0].axhline(noise, color='k', ls="--")
    ax[0].set_ylim(2, 5e2)

         
    lbls = ['$b_1$', '$b_2$', '$b_s$']
    for ii in range(3):
        ax[1].plot(kk, paramsk[ii], lw=2, color='C%d'%ii, label=lbls[ii])
        ax[1].axhline(params[ii], lw=1, color='C%d'%ii, ls="--")
        ax[1].set_ylabel(r'$b(k)$', fontdict=font)

    for axis in ax:
        axis.set_xscale('log')
        axis.grid(which='both')
        axis.legend(prop=fontmanage, ncol=2)
        axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
        axis.set_xscale('log')

    # Put on some more labels.
    for axis in ax:
        axis.set_xscale('log')
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
    ##and finish
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(figpath + '/bfitk_L%04d_%04d.pdf'%(bs, aa*10000))





def make_bsum_plot():
    """bias rcc, tf, error, bk"""
    

    noises = np.loadtxt('/global/u1/c/chmodi/Programs/21cm/21cm_cleaning/data/summaryHI.txt').T
    for i in range(noises[0].size):
        if noises[0][i] == np.round(1/aa-1, 2): noise = noises[3][i]
    print(noise)

    hmesh = BigFileMesh(dpath+'ZA/opt_s999_h1massA_fourier/datap', 'mapp').paint()
    grid = pm.mesh_coordinates()*bs/nc

    noises = np.loadtxt('/global/u1/c/chmodi/Programs/21cm/21cm_cleaning/data/summaryHI.txt').T
    for i in range(noises[0].size):
        if noises[0][i] == np.round(1/aa-1, 2): noise = noises[3][i]
    print(noise)

    zamod = BigFileMesh(dpath+'ZA/opt_s999_h1massA_fourier/fitp/', 'mapp').paint()
    try: pmmod = BigFileMesh(dpath+'T05-B1/opt_s999_h1massA_fourier/fitp/', 'mapp').paint()
    except Exception as e:
        print(e)
        lin = BigFileMesh(dpath+'ZA/opt_s999_h1massA_fourier/datap/', 's').paint()
        dyn = BigFileCatalog('/global/cscratch1/sd/chmodi/m3127/cm_lowres/5stepT-B1/%d-%d-9100-fixed/fastpm_%0.4f/1'%(bs, nc, aa))
        fpos = dyn['Position']
        _, pmmod = getbias(pm, basemesh=lin, hmesh=hmesh, pos=fpos, grid=grid, fpos=fpos)

    fin = BigFileMesh(dpath+'ZA/opt_s999_h1massA_fourier/datap/', 'd').paint()
    fin /= fin.cmean()
    fin -= 1
    finsm = ft.smooth(fin, 3, 'gauss')
    params, finmod = getbias(pm, basemesh=finsm, hmesh=hmesh, pos=grid, grid=grid)
    models = [zamod, pmmod, finmod, finsm.copy()]
    lbls = ['ZA shift', 'PM shift', 'Eulerian (R=3)', 'Eulerian (R=3), $b_1^E$']
    lss = ['-', '-', '-', '--']


    lin = BigFileMesh(dpath+'ZA/opt_s999_h1massA_fourier/datap/', 's').paint()
    dyn = BigFileCatalog('/global/cscratch1/sd/chmodi/m3127/cm_lowres/5stepT-B1/%d-%d-9100-fixed/fastpm_%0.4f/1'%(bs, nc, aa))
    dgrow = cosmo.scale_independent_growth_factor(zz)
    zapos = za.doza(lin.r2c(), grid, z=zz, dgrow=dgrow)
    fpos = dyn['Position']

    params, _ = getbias(pm, basemesh=lin, hmesh=hmesh, pos=zapos, grid=grid)
    kk, paramsk, _ = getbiask(pm, basemesh=lin, hmesh=hmesh, pos=zapos, grid=grid)

    print('Setup done')
    #####
    fig, ax = plt.subplots(2, 2, figsize=(8, 6), sharex=True)

    for ii, mod in enumerate(models):

        pmod = FFTPower(mod, mode='1d').power
        k, pmod = pmod['k'], pmod['power']
        ph = FFTPower(hmesh, mode='1d').power['power']
        if ii == 3: 
            mod *= (ph[1]/pmod[1]).real**0.5 
            pmod *= (ph[1]/pmod[1]).real 
        pxmodh = FFTPower(hmesh, second=mod, mode='1d').power['power']
        perr = FFTPower(hmesh -mod, mode='1d').power['power']
 
        ax[0, 0].plot(k, pxmodh/(pmod*ph)**0.5, label=lbls[ii], lw=2, ls=lss[ii])
        ax[0, 0].set_ylabel('$r_{cc}$', fontdict=font)

        #if ii == 3: pmod *= ph[1]/pmod[1]
        ax[0, 1].plot(k,(pmod/ph)**0.5, lw=2, ls=lss[ii])
        ax[0, 1].set_ylabel(r'$T_f$', fontdict=font)
        #ax[1].set_ylabel(r'$\sqrt{P_{\rm mod}/P_{hh}}$', fontdict=font)
         
        ax[1, 0].plot(k, perr, lw=2, ls=lss[ii])
        ax[1, 0].set_ylabel(r'$P_{\rm err}$', fontdict=font)
        ax[1, 0].set_yscale('log')
    ax[1, 0].axhline(noise, color='k', ls="--")
    ax[1, 0].set_ylim(2, 5e2)

         
    lbls = ['$b_1$', '$b_2$', '$b_s$']
    for ii in range(3):
        ax[1, 1].plot(kk, paramsk[ii], lw=2, color='C%d'%ii, label=lbls[ii])
        ax[1, 1].axhline(params[ii], lw=1, color='C%d'%ii, ls="--")
        ax[1, 1].set_ylabel(r'$b(k)$', fontdict=font)

    for axis in ax.flatten():
        axis.set_xscale('log')
        axis.grid(which='both', alpha=0.2, color='gray', lw=0.2)
        axis.legend(prop=fontmanage)
        axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
        axis.set_xscale('log')

    # Put on some more labels.
    for axis in ax.flatten():
        axis.set_xscale('log')
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)

    ##and finish
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(figpath + '/bsum_L%04d_%04d.pdf'%(bs, aa*10000))


if __name__=="__main__":
    #make_bias_plot()
    #make_bias2p_plot()
    #make_rsdbias_plot()
    #make_biask_plot()
    make_bsum_plot()
    #
