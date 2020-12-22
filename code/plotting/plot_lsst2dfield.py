#!/usr/bin/env python3
#
# Plots the power spectra and Fourier-space biases for the HI.
#
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
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
parser.add_argument('-a', '--aa', help='scale factor', default=0.5000, type=float)
parser.add_argument('-l', '--bs', help='boxsize', default=1024, type=float)
parser.add_argument('-n', '--nmesh', help='nmesh', default=256, type=int)
parser.add_argument('-t', '--angle', help='angle of the wedge', default='opt')
parser.add_argument('-k', '--kmin', help='kmin of the wedge', default=0.03, type=float)
parser.add_argument('-r', '--rsdpos', help='kmin of the wedge', default=True, type=bool)
parser.add_argument('--pp', help='upsample', default=0)
args = parser.parse_args()

figpath = './figs/'


aa = args.aa
bs, nc = 1024, args.nmesh
zz = 1/aa- 1
ang = args.angle
pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])

#dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_ang%0.1f/'%(aa, kmin, ang)
#dpath += 'L%04d-N%04d/stage2/'%(bs, nc)
dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_%s/'%(aa, 0.03, ang)
dpath += 'L%04d-N%04d-R/thermal-reas-hex/'%(bs, nc)

################
def makeplot(fig, ax, meshes, nc, aa, checkbase):

    hmesh = meshes[0]
    for cmap in ['Oranges']:
        for i, f  in enumerate(meshes):
            i0, i1 = 100, 140
            j0, j1 = 0, nc
            #i0, i1 = 0, nc
            #j0, j1 = 0, nc
            off = 0
            vmin, vmax = None, None
            norm = None
            #norm = SymLogNorm(1)
            
            vmin, vmax = hmesh[i0:i1,j0:j1, j0:j1].sum(axis=0).min()-off, hmesh[i0:i1,j0:j1, j0:j1].sum(axis=0).max()+off
            if i == 1: vmin, vmax = None, None
            im = ax[0, i].imshow(f[i0:i1,j0:j1, j0:j1].sum(axis=0), cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
            
            vmin, vmax = hmesh[j0:j1,i0:i1,j0:j1].sum(axis=1).min()-off, hmesh[j0:j1,i0:i1,j0:j1].sum(axis=1).max()+off
            if i == 1: vmin, vmax = None, None
            im = ax[1, i].imshow(f[j0:j1,i0:i1,j0:j1].sum(axis=1), cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
            #plt.colorbar(im, ax=ax[1, i])

            vmin, vmax = hmesh[j0:j1, j0:j1,i0:i1].sum(axis=2).min()-off, hmesh[j0:j1, j0:j1,i0:i1].sum(axis=2).max()+off
            if i == 1: vmin, vmax = None, None
            im = ax[2, i].imshow(f[j0:j1, j0:j1,i0:i1].sum(axis=2), cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
            #plt.colorbar(im, ax=ax[2, i])
            print(vmin, vmax)

        ax[0, 0].set_ylabel('Truth', fontdict=font)
        ax[0, 1].set_ylabel('Data', fontdict=font)
        ax[0, 2].set_ylabel('Recon - HI only', fontdict=font)
        ax[0, 3].set_ylabel('Recon - HI + LSST', fontdict=font)
        try: ax[0, 4].set_ylabel('Recon - HI + ELG', fontdict=font)
        except: pass
        #ax[0, 0].set_ylabel('X', fontdict=font)
        #ax[1, 0].set_ylabel('Y', fontdict=font)
        #ax[2, 0].set_ylabel('Z', fontdict=font)

        x0, y0, dxy = 15, 40, 10
        if nc == 512: fac = 2
        else: fac = 1
        coords = [['Z', 'Y'], ['Z', 'X'], ['Y', 'X']]
        for i in range(3):
            ax[i, 0].arrow(x0*fac, y0*fac, dxy*fac, 0, width=1, color='k')
            ax[i, 0].text((x0+dxy+6)*fac, (y0+2)*fac, coords[i][0], fontsize=fsize)
            ax[i, 0].arrow(x0*fac, y0*fac, 0, -1*dxy*fac, width=1, color='k')
            ax[i, 0].text((x0-7)*fac, (y0-dxy-7)*fac, coords[i][1], fontsize=fsize)

        for axis in ax.flatten():
            axis.axes.xaxis.set_ticks([])
            axis.axes.yaxis.set_ticks([])


        if cmap != 'viridis': ang = args.angle +'-' + cmap 
        else: ang = args.angle
        plt.tight_layout(h_pad=0.01, w_pad=0.05)
        #plt.tight_layout()
        if not checkbase: 
            if i1 - i0 == nc :         plt.savefig(figpath + '/lsst_N%04d_%04d-fullmap-nocheckbase.pdf'%(nc, aa*10000))
            else: plt.savefig(figpath + '/lsst_N%04d_%04d-map-nocheckbase.pdf'%(nc, aa*10000))
        else: 
            if i1 - i0 == nc :         plt.savefig(figpath + '/lsst_N%04d_%04d-fullmap.pdf'%(nc, aa*10000))
            else: plt.savefig(figpath + '/lsst_N%04d_%04d-map.pdf'%(nc, aa*10000))




            
def make_implotz1(checkbase=True):
    """Does the work of making the real-space xi(r) and b(r) figure."""
    

    aa = 0.5000
    lsstn = 0.0500
    elgn = 0.0010
    dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/reconlsstv2//fastpm_0.5000/wedge_kmin0.03_pess/L1024-N0256-R/thermal-reas-hex/ZA/opt_s777_h1massD_%s_rsdpos/'
    dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/reconlsstv2//fastpm_0.5000/wedge_kmin0.03_pess/L1024-N0256-R/thermal-reas-hex/ZA/opt_s777_h1massD_%s_rsdpos/'
    if not checkbase : dwpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/reconlsstv2//fastpm_0.5000/wedge_kmin0.03_pess/L1024-N0256-R/thermal-reas-hex/ZA/opt_s777_h1massD_%s_rsdpos-nocheckbase/'
    else: dwpath = dpath
        
    if nc == 256:
        #wmesh = BigFileMesh(dpath%('lwt0') + '/dataw', 'mapp').paint(mode='real')
        wmesh = BigFileMesh(dpath%('lwt0') + '/dataw', 'mapp').paint(mode='real')
        hmesh = BigFileMesh(dpath%('lwt0') + '/datap', 'mapp').paint(mode='real')
        #lmesh = BigFileMesh(dpath%('lwt0') + '/datap', 'mapp2').paint(mode='real')

        suff = 'lwt0-nob2'
        h0, l0  = BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp2').paint()
        
        suff = 'lsst-nob2_ln%04d'%(lsstn * 1e4)
        h1, l1  = BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp2').paint()
       
        suff = 'elg-nob2_ln%04d'%(elgn * 1e4)
        h2, l2  = BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp2').paint()
        
    elif nc == 512:
        wmesh = BigFileMesh(dpath%('lwt0-nob2') + '/dataw_up', 'mapp').paint()
        hmesh = BigFileMesh(dpath%('lwt0-nob2') + '/datap_up', 'mapp').paint()
        #lmesh = BigFileMesh(dpath%('lwt0-nob2') + '/datap_up', 'mapp2').paint()

        suff = 'lwt0-nob2'
        #h0, l0  = BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp2').paint()
        h0  = BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp').paint()
        
        suff = 'lsst-nob2_ln%04d'%(lsstn * 1e4)
        #h1, l1  = BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp2').paint()
        h1  = BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp').paint()
        
        suff = 'elg-nob2_ln%04d'%(elgn * 1e4)
        #h2, l2  = BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp2').paint()
        h2  = BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp').paint()
        
    fig, ax = plt.subplots(5, 3, figsize=(6, 10), sharex=True, sharey=True)
    ax = ax.T

    print(hmesh[...].min(), hmesh[...].max(), wmesh[...].min(), wmesh[...].max())
    #cmap = 'RdBu_r'
    cmap = 'viridis'
    #for cmap in ['viridis', 'RdBu_r', 'Reds', 'gist_heat', 'magma', 'cividis', 'Oranges', 'autumn', 'inferno']:
    #for cmap in ['viridis', 'Oranges', 'inferno']:

    makeplot(fig, ax, [hmesh, wmesh, h0, h1, h2], nc, aa, checkbase)
        

def make_implotz4(checkbase=True):
    """Does the work of making the real-space xi(r) and b(r) figure."""
    

    aa = 0.2000
    lsstn = 0.0035
    dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/reconlsstv2//fastpm_0.2000/wedge_kmin0.03_pess/L1024-N0256-R/thermal-reas-hex/ZA/opt_s777_h1massD_%s_rsdpos/'
    if not checkbase : dwpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/reconlsstv2//fastpm_0.5000/wedge_kmin0.03_pess/L1024-N0256-R/thermal-reas-hex/ZA/opt_s777_h1massD_%s_rsdpos-nocheckbase/'
    else: dwpath = dpath
    
    if nc == 256:
        wmesh = BigFileMesh(dwpath%('lwt0') + '/dataw', 'mapp').paint()
        hmesh = BigFileMesh(dpath%('lwt0') + '/datap', 'mapp').paint()
        lmesh = BigFileMesh(dpath%('lwt0') + '/datap', 'mapp2').paint()

        suff = 'lwt0'
        h0, l0  = BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp2').paint()
        
        suff = 'lsst_ln%04d'%(lsstn * 1e4)
        h1, l1  = BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + '256-0.00/best-fit/', 'mapp2').paint()
       
        
    elif nc == 512:
        wmesh = BigFileMesh(dpath%('lwt0') + '/dataw_up', 'mapp').paint()
        hmesh = BigFileMesh(dpath%('lwt0') + '/datap_up', 'mapp').paint()
        lmesh = BigFileMesh(dpath%('lwt0') + '/datap_up', 'mapp2').paint()

        suff = 'lwt0'
        h0, l0  = BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp2').paint()
        
        suff = 'lsst_ln%04d'%(lsstn * 1e4)
        h1, l1  = BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp').paint(), BigFileMesh(dpath%suff + 'upsample2/512-0.00/best-fit/', 'mapp2').paint()


        
    fig, ax = plt.subplots(4, 3, figsize=(9, 12), sharex=True, sharey=True)
    ax = ax.T

    print(hmesh[...].min(), hmesh[...].max(), wmesh[...].min(), wmesh[...].max())
    #cmap = 'RdBu_r'
    cmap = 'viridis'
    #for cmap in ['viridis', 'RdBu_r', 'Reds', 'gist_heat', 'magma', 'cividis', 'Oranges', 'autumn', 'inferno']:
    #for cmap in ['viridis', 'Oranges', 'inferno']:
    #makeplot(fig, ax, [hmesh, wmesh, h0, h1, h2], nc, aa)
    makeplot(fig, ax, [hmesh, wmesh, h0, h1], nc, aa, checkbase)
    


################


if __name__=="__main__":
    checkbase = True
    make_implotz1(checkbase)
    #make_implotz4(checkbase)
    #
