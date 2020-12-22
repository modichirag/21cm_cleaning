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
parser.add_argument('-n', '--nmesh', help='nmesh', default=256, type=int)
parser.add_argument('-t', '--angle', help='angle of the wedge', default=50, type=float)
parser.add_argument('-k', '--kmin', help='kmin of the wedge', default=0.03, type=float)
parser.add_argument( '--up', help='upsample', default=0) 
args = parser.parse_args()

figpath = './figs/'

bs, nc, aa = args.bs, args.nmesh, args.aa
nc2 = nc*2
zz = 1/aa- 1
kmin = args.kmin
ang = args.angle
if args.up: pm = ParticleMesh(BoxSize=bs, Nmesh=[nc2, nc2, nc2])
else: pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank



def getps(f1, f2=None, p2=False):
    p1 = FFTPower(f1, mode='1d').power['power']
    if f2 is not None:
        px = FFTPower(f1, second=f2,  mode='1d').power['power']
        if p2:
            p2 = FFTPower(f2,  mode='1d').power['power']
            return p1, p2, px
        else: return p1, px
    else: return p1



def getps2D(f1, f2=None, p2=False):
    p1 = FFTPower(f1, mode='2d').power['power']
    if f2 is not None:
        px = FFTPower(f1, second=f2,  mode='2d').power['power']
        if p2:
            p2 = FFTPower(f2,  mode='2d').power['power']
            return p1, p2, px
        else: return p1, px
    else: return p1

    
def plot1D(nc=nc):

    aas = [0.2000, 0.5000, 0.5000]
    lsstns = [0.0035, 0.0500, 0.0500]
    elgns = [None, 0.0010, 0.0010]


    fig, axar = plt.subplots(3, 2, figsize = (9, 12), sharex=True, sharey=False)

    for i in range(3):
        aa, lsstn, elgn = aas[i], lsstns[i], elgns[i]
        ax = axar[i]
        
        if i== 0:dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/reconlsstv2//fastpm_0.2000/wedge_kmin0.03_pess/L1024-N0256-R/thermal-reas-hex/ZA/opt_s777_h1massD_%s_rsdpos/'
        if i==1: dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/reconlsstv2//fastpm_0.5000/wedge_kmin0.03_pess/L1024-N0256-R/thermal-reas-hex/ZA/opt_s777_h1massD_%s_rsdpos/'
        if i==2 :
            dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/reconlsstv2//fastpm_0.5000/wedge_kmin0.03_pess/L1024-N0256-R/thermal-reas-hex-hirax/ZA/opt_s777_h1massD_%s_rsdpos/'
        
        if nc == 256:
            hmesh = BigFileMesh(dpath%('lwt0') + '/datap', 'mapp').paint()
            lmesh = BigFileMesh(dpath%('lsst_ln%04d'%(lsstn*1e4)) + '/datap', 'mapp2').paint()
            ph = FFTPower(hmesh, mode='1d').power
            k, ph = ph['k'], ph['power']
            pl = FFTPower(lmesh, mode='1d').power['power']

            if i == 0: suff = 'lwt0'
            else: suff = 'lwt0-nob2'
            print(dpath%suff + '256-0.00/best-fit/')
            h0, l0  = BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp2').paint()
            ph0, ph0x = getps(h0, hmesh)
            pl0, pl0x = getps(l0, lmesh)

            if i == 0: suff = 'lsst_ln%04d'%(lsstn * 1e4)
            else: suff = 'lsst-nob2_ln%04d'%(lsstn * 1e4)
            h1, l1  = BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp2').paint()
            ph1, ph1x = getps(h1, hmesh)
            pl1, pl1x = getps(l1, lmesh)

            if i: 
                suff = 'elg-nob2_ln%04d'%(elgn * 1e4)
                emesh = BigFileMesh(dpath%('elg_ln%04d'%(elgn*1e4)) + '/datap', 'mapp2').paint()
                pe = FFTPower(emesh, mode='1d').power['power']
                h2, l2  = BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp2').paint()
                ph2, ph2x = getps(h2, hmesh)
                pe2, pe2x = getps(l2, emesh)

            if i==1: 
                suff = 'elg_noh1_ln%04d'%(elgn * 1e4)
                h3, l3  = BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp2').paint()
                ph3, ph3x = getps(h3, hmesh)
                pe3, pe3x = getps(l3, emesh)

                
        elif nc == 512:
            hmesh = BigFileMesh(dpath%('lwt0') + '/datap_up', 'mapp').paint()
            lmesh = BigFileMesh(dpath%('lsst_ln%04d'%(lsstn*1e4)) + '/datap_up', 'mapp2').paint()
            ph = FFTPower(hmesh, mode='1d').power
            k, ph = ph['k'], ph['power']
            pl = FFTPower(lmesh, mode='1d').power['power']

            if i == 0: suff = 'lwt0'
            else: suff = 'lwt0-nob2'
            h0, l0  = BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp2').paint()
            ph0, ph0x = getps(h0, hmesh)
            pl0, pl0x = getps(l0, lmesh)

            if i == 0: suff = 'lsst_ln%04d'%(lsstn * 1e4)
            else: suff = 'lsst-nob2_ln%04d'%(lsstn * 1e4)
            h1, l1  = BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp2').paint()
            ph1, ph1x = getps(h1, hmesh)
            pl1, pl1x = getps(l1, lmesh)

            if i :                
                suff = 'elg-nob2_ln%04d'%(elgn * 1e4)
                emesh = BigFileMesh(dpath%('elg_ln%04d'%(elgn*1e4)) + '/datap_up', 'mapp2').paint()
                pe = FFTPower(emesh, mode='1d').power['power']
                h2, l2  = BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp2').paint()
                ph2, ph2x = getps(h2, hmesh)
                pe2, pe2x = getps(l2, emesh)

            if i==1: 
                suff = 'elg_noh1_ln%04d'%(elgn * 1e4)
                h3, l3  = BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + 'upsample2//512-0.00/best-fit/', 'mapp2').paint()
                ph3, ph3x = getps(h3, hmesh)
                pe3, pe3x = getps(l3, emesh)





        ax[0].plot(k, ph0x/(ph0*ph)**0.5, 'C0', label='HI data')
        ax[0].plot(k, ph1x/(ph1*ph)**0.5, 'C1', label='HI + LSST data')
        if i : ax[0].plot(k, ph2x/(ph2*ph)**0.5, 'C2', label='HI + DESI data')
        ax[0].plot(k, pl1x/(pl1*pl)**0.5, 'C1--')
        if i : ax[0].plot(k, pe2x/(pe2*pe)**0.5, 'C2:')
        if i == 1 : ax[0].plot(k, ph3x/(ph3*ph)**0.5, 'C3', label='DESI data')
        if i == 1: ax[0].plot(k, pe3x/(pe3*pe)**0.5, 'C3:')

        
        ax[1].plot(k, (ph0/ph)**0.5, 'C0', label='HI')
        ax[1].plot(k, (ph1/ph)**0.5, 'C1')
        if i: ax[1].plot(k, (ph2/ph)**0.5, 'C2')
        ax[1].plot(k, (pl1/pl)**0.5, 'C1--', label='LSST')
        if i : ax[1].plot(k, (pe2/pe)**0.5, 'C2:', label='DESI')
        if i == 1 : ax[1].plot(k, (ph3/ph)**0.5, 'C3')
        if i == 1 : ax[1].plot(k, (pe3/pe)**0.5, 'C3:')


        ax[0].set_ylabel('$r_{cc}$', fontdict=font)
        ax[1].set_ylabel('$t_{f}$', fontdict=font)
        ax[0].set_ylim(0.5, 1.07)
        ax[1].set_ylim(0.5, 2.05)
        if i == 2:
            for axis in ax: axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
        if i == 1:
            for axis in ax: axis.legend(prop=fontmanage)

        if i < 2: ax[1].text(1e-2, 1.82, 'z=%0.2f, PUMA'%(1/aa - 1), fontsize=fsize)
        else: ax[1].text(1e-2, 1.82, 'z=%0.2f, HIRAX'%(1/aa - 1), fontsize=fsize)
                
        for axis in ax.flatten():
            axis.set_xscale('log')
            axis.grid(which='both')
        # Put on some more labels.
        for axis in ax.flatten():
            axis.axhline(1, color='k', ls="--")
            for tick in axis.xaxis.get_major_ticks():
                tick.label.set_fontproperties(fontmanage)
            for tick in axis.yaxis.get_major_ticks():
                tick.label.set_fontproperties(fontmanage)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if rank == 0: plt.savefig(figpath + '/lsst_N%04d.pdf'%(nc))





def plot2D(nc=nc):

    aas = [0.2000, 0.5000, 0.5000]
    lsstns = [0.0035, 0.0500, 0.0500]
    elgns = [None, 0.0010, 0.0010]


    fig, axar = plt.subplots(3, 2, figsize = (9, 12), sharex=True, sharey=True)

    for i in range(3):
        aa, lsstn, elgn = aas[i], lsstns[i], elgns[i]
        ax = axar[i]
        
        if i== 0:dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/reconlsstv2//fastpm_0.2000/wedge_kmin0.03_pess/L1024-N0256-R/thermal-reas-hex/ZA/opt_s777_h1massD_%s_rsdpos/'
        if i==1: dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/reconlsstv2//fastpm_0.5000/wedge_kmin0.03_pess/L1024-N0256-R/thermal-reas-hex/ZA/opt_s777_h1massD_%s_rsdpos/'
        if i==2 :
            dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/reconlsstv2//fastpm_0.5000/wedge_kmin0.03_pess/L1024-N0256-R/thermal-reas-hex-hirax/ZA/opt_s777_h1massD_%s_rsdpos/'
            
        if nc == 256:
            hmesh = BigFileMesh(dpath%('lwt0-nob2') + '/datap', 'mapp').paint()
            lmesh = BigFileMesh(dpath%('lsst_ln%04d'%(lsstn*1e4)) + '/datap', 'mapp2').paint()
            ph = FFTPower(hmesh, mode='2d').power
            k, ph = ph['k'], ph['power']
            pl = FFTPower(lmesh, mode='2d').power['power']

            if i == 0: suff = 'lwt0'
            else: suff = 'lwt0-nob2'
            h0, l0  = BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp2').paint()
            ph0, ph0x = getps2D(h0, hmesh)
            pl0, pl0x = getps2D(l0, lmesh)

            if i == 0: suff = 'lsst_ln%04d'%(lsstn * 1e4)
            else: suff = 'lsst-nob2_ln%04d'%(lsstn * 1e4)
            h1, l1  = BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp2').paint()
            ph1, ph1x = getps2D(h1, hmesh)
            pl1, pl1x = getps2D(l1, lmesh)

            if i :
                suff = 'elg-nob2_ln%04d'%(elgn * 1e4)
                emesh = BigFileMesh(dpath%('elg_ln%04d'%(elgn*1e4)) + '/datap', 'mapp2').paint()
                pe = FFTPower(emesh, mode='2d').power['power']
                h2, l2  = BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp2').paint()
                ph2, ph2x = getps2D(h2, hmesh)
                pe2, pe2x = getps2D(l2, emesh)

            if i==1: 
                suff = 'elg_noh1_ln%04d'%(elgn * 1e4)
                h3, l3  = BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp2').paint()
                ph3, ph3x = getps2D(h3, hmesh)
                pe3, pe3x = getps2D(l3, emesh)


        elif nc == 512:
            hmesh = BigFileMesh(dpath%('lwt0') + '/datap_up', 'mapp').paint()
            lmesh = BigFileMesh(dpath%('lsst_ln%04d'%(lsstn*1e4)) + '/datap_up', 'mapp2').paint()
            ph = FFTPower(hmesh, mode='2d').power
            k, ph = ph['k'], ph['power']
            pl = FFTPower(lmesh, mode='2d').power['power']

            if i == 0: suff = 'lwt0'
            else: suff = 'lwt0-nob2'
            h0, l0  = BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp2').paint()
            ph0, ph0x = getps2D(h0, hmesh)
            pl0, pl0x = getps2D(l0, lmesh)

            if i == 0: suff = 'lsst_ln%04d'%(lsstn * 1e4)
            else: suff = 'lsst-nob2_ln%04d'%(lsstn * 1e4)
            h1, l1  = BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp2').paint()
            ph1, ph1x = getps2D(h1, hmesh)
            pl1, pl1x = getps2D(l1, lmesh)

            if i :
                suff = 'elg-nob2_ln%04d'%(elgn * 1e4)
                emesh = BigFileMesh(dpath%('elg_ln%04d'%(elgn*1e4)) + '/datap_up', 'mapp2').paint()
                pe = FFTPower(emesh, mode='2d').power['power']
                h2, l2  = BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp2').paint()
                ph2, ph2x = getps2D(h2, hmesh)
                pe2, pe2x = getps2D(l2, emesh)
        

            if i==1: 
                suff = 'elg_noh1_ln%04d'%(elgn * 1e4)
                h3, l3  = BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp2').paint()
                #h3, l3  = BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp2').paint()
                ph3, ph3x = getps2D(h3, hmesh)
                pe3, pe3x = getps2D(l3, emesh)


        ax[0].plot(k[:, 0], ph0x[:, 0]/(ph0[:, 0]*ph[:, 0])**0.5, 'C0', label='HI data')
        ax[0].plot(k[:, 0], ph1x[:, 0]/(ph1[:, 0]*ph[:, 0])**0.5, 'C1', label='HI + LSST data')
        if i: ax[0].plot(k[:, 0], ph2x[:, 0]/(ph2[:, 0]*ph[:, 0])**0.5, 'C2', label='HI + DESI data')
        ax[0].plot(k[:, 0], pl1x[:, 0]/(pl1[:, 0]*pl[:, 0])**0.5, 'C1--')
        if i: ax[0].plot(k[:, 0], pe2x[:, 0]/(pe2[:, 0]*pe[:, 0])**0.5, 'C2:')
        if i == 1 : ax[0].plot(k[:, 0], (ph3x/(ph3*ph)**0.5)[:, 0], 'C3', label='DESI data')
        if i == 1: ax[0].plot(k[:, 0], (pe3x/(pe3*pe)**0.5)[:, 0], 'C3:')


        ax[-1].plot(k[:, 0], ph0x[:, -1]/(ph0[:, -1]*ph[:, -1])**0.5, 'C0', label='HI')
        ax[-1].plot(k[:, 0], ph1x[:, -1]/(ph1[:, -1]*ph[:, -1])**0.5, 'C1')
        if i: ax[-1].plot(k[:, 0], ph2x[:, -1]/(ph2[:, -1]*ph[:, -1])**0.5, 'C2')
        ax[-1].plot(k[:, 0], pl1x[:, -1]/(pl1[:, -1]*pl[:, -1])**0.5, 'C1--', label='LSST')
        if i: ax[-1].plot(k[:, 0], pe2x[:, -1]/(pe2[:, -1]*pe[:, -1])**0.5, 'C2:', label='DESI')
        if i == 1 : ax[1].plot(k[:, 0], (ph3x/(ph3*ph)**0.5)[:, -1], 'C3')
        if i == 1: ax[1].plot(k[:, 0], (pe3x/(pe3*pe)**0.5)[:, -1], 'C3:')

    #
        ax[0].set_ylabel('$r_{cc}$', fontdict=font)
        #ax[1].set_ylabel('$r_{cc},\, \mu=0.9$', fontdict=font)

        if i < 2: ax[1].text(0.15, 1.02, 'z=%0.2f, PUMA'%(1/aa - 1), fontsize=fsize)
        else: ax[1].text(0.15, 1.02, 'z=%0.2f, HIRAX'%(1/aa - 1), fontsize=fsize)
        
        if i == 0:
            ax[0].text(1e-2, 0.55, '$\mu=0.1$', fontsize=fsize)
            ax[1].text(1e-2, 0.55, '$\mu=0.9$', fontsize=fsize)
        
        for axis in ax: axis.set_ylim(0.3, 1.1)
        if i == 2:
            for axis in ax:   axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
        if i ==1 :
            for axis in ax: axis.legend(prop=fontmanage)
            
        for axis in ax.flatten():
            axis.set_xscale('log')
            axis.grid(which='both')
            
        # Put on some more labels.
        for axis in ax.flatten():
            for tick in axis.xaxis.get_major_ticks():
                tick.label.set_fontproperties(fontmanage)
            for tick in axis.yaxis.get_major_ticks():
                tick.label.set_fontproperties(fontmanage)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if rank == 0:
        plt.savefig(figpath + '/lsst_N%04d-2D.pdf'%(nc))

################


if __name__=="__main__":
    #plot1D()
    plot2D()
