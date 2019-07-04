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
#import matplotlib
#matplotlib.use('pdf')

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
parser.add_argument('-m', '--mode', help='save files or make plot', default='plot')
parser.add_argument('-a', '--aa', help='scale factor', default=0.3333, type=float)
parser.add_argument('-l', '--bs', help='boxsize', default=1024, type=float)
parser.add_argument('-n', '--nmesh', help='nmesh', default=256, type=int)
parser.add_argument('-t', '--angle', help='angle of the wedge', default=50, type=float)
parser.add_argument('-k', '--kmin', help='kmin of the wedge', default=0.03, type=float)
parser.add_argument( '--pp', help='upsample', default=1) 

args = parser.parse_args()

figpath = './figs/'

bs, nc, aa = args.bs, args.nmesh, args.aa
nc2 = nc*2
zz = 1/aa- 1
kmin = args.kmin
ang = args.angle
if args.pp: pm = ParticleMesh(BoxSize=bs, Nmesh=[nc2, nc2, nc2])
else: pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank

################
def savestd():
    #for ia, aa  in enumerate([0.3333, 0.2000, 0.1429]):
    for ia, aa  in enumerate([0.1429]):
    #for ia, aa  in enumerate([0.3333, 0.2000, 0.1429]):
        zz = 1/aa-1
        for iw, wopt in enumerate(['opt', 'pess']):
        #for iw, wopt in enumerate(['opt']):
            for it, thopt in enumerate(['opt', 'pess', 'reas']):
            #for it, thopt in enumerate([ 'pess']):
                if rank == 0: print(aa, wopt, thopt)
                dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_%s/'%(aa, 0.03, wopt)
                dpath += 'L%04d-N%04d-R//thermal-%s-hex/ZA/opt_s999_h1massA_fourier_rsdpos/'%(bs, nc, thopt)
                ofolder = '../../data/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/'%(wopt, thopt)
                fname = ofolder + 'std-L%04d_%0.4f.txt'%(bs, aa)
                fxname = ofolder + 'xstd-L%04d_%0.4f.txt'%(bs, aa)
                if args.pp : fname = fname[:-4] + '-up.txt'
                if args.pp : fxname = fxname[:-4] + '-up.txt'
                try:
                    rep = np.loadtxt(fname+'s').T
                except:
                    try:
                        if args.pp:
                            std = BigFileMesh(dpath+'/stdrecon_up-noise', 'std').paint()
                            ss = BigFileMesh(dpath+'/datap_up', 's').paint()
                        else:
                            std = BigFileMesh(dpath+'/stdrecon-noise', 'std').paint()
                            ss = BigFileMesh(dpath+'/datap', 's').paint()

                        p0 = FFTPower(std, mode='1d').power
                        px = FFTPower(std, second=ss, mode='1d').power
                        if rank == 0: np.savetxt(fname, np.stack([p0['k']]+ [p0['power'].real]).T, header='k, p0')
                        if rank == 0: np.savetxt(fxname, np.stack([px['k']]+ [px['power'].real]).T, header='k, px')
                    except Exception as e:
                        print(e)

                        
def savestd2d(Nmu=4):
    #for ia, aa  in enumerate([0.3333, 0.2000, 0.1429]):
    for ia, aa  in enumerate([0.1429]):
        zz = 1/aa-1
        for iw, wopt in enumerate(['opt', 'pess']):
        #for iw, wopt in enumerate(['opt']):
            for it, thopt in enumerate(['opt', 'pess', 'reas']):
            #for it, thopt in enumerate([ 'reas']):
                if rank == 0: print(aa, wopt, thopt)
                dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_%s/'%(aa, 0.03, wopt)
                dpath += 'L%04d-N%04d-R//thermal-%s-hex/ZA/opt_s999_h1massA_fourier_rsdpos/'%(bs, nc, thopt)
                ofolder = '../../data/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/Nmu%d/'%(wopt, thopt, Nmu)
                fname = ofolder + 'std-L%04d_%0.4f.txt'%(bs, aa)
                fxname = ofolder + 'xstd-L%04d_%0.4f.txt'%(bs, aa)
                if args.pp : fname = fname[:-4] + '-up.txt'
                if args.pp : fxname = fxname[:-4] + '-up.txt'
                try:
                    rep = np.loadtxt(fname+'s').T
                except:
                    try:
                        if args.pp:
                            std = BigFileMesh(dpath+'/stdrecon_up-noise', 'std').paint()
                            ss = BigFileMesh(dpath+'/datap_up', 's').paint()
                        else:
                            std = BigFileMesh(dpath+'/stdrecon-noise', 'std').paint()
                            ss = BigFileMesh(dpath+'/datap', 's').paint()

                        p0 = FFTPower(std, mode='2d', Nmu=Nmu).power
                        px = FFTPower(std, second=ss, mode='2d', Nmu=Nmu).power
                        if rank == 0: np.savetxt(fname, p0['power'].real)
                        if rank == 0: np.savetxt(fxname, px['power'].real)
                    except Exception as e:
                        print(e)

def fishintrep1d():
    '''1D'''

    A0 = 0.4529
    silk = 7.76
    vol = bs**3
    fac = vol*A0**2

    
    for ia, aa  in enumerate([0.3333, 0.2000, 0.1429]):
        zz = 1/aa-1
        for iw, wopt in enumerate(['opt', 'pess']):
        #for iw, wopt in enumerate([ 'opt']):
            #for it, thopt in enumerate(['opt', 'pess', 'reas']):
            for it, thopt in enumerate(['reas']):
                if rank == 0: print(aa, wopt, thopt)


                angle = np.round(mapn.wedge(zz, att=wopt, angle=True), 0)
                dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_%s/'%(aa, 0.03, wopt)
                dpath += 'L%04d-N%04d-R//thermal-%s-hex/ZA/opt_s999_h1massA_fourier_rsdpos/'%(bs, nc, thopt)
                ofolder = '../../data/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/'%(wopt, thopt)
                if args.pp: 
                    suff = '-up'
                else: 
                    suff = ''
                recon = np.loadtxt(ofolder + 'recon-L%04d_%0.4f%s.txt'%(bs, aa, suff))
                k = recon[:, 0]
                pxr, pr, ps = recon[:, 2], recon[:, 6], recon[:, 7]
                rccr, tfr = pxr/(pr * ps)**0.5, (pr/ps)**0.5
                biasr = tfr[1:5].mean()

                pstd = np.loadtxt(ofolder + 'std-L%04d_%0.4f%s.txt'%(bs, aa, suff))[:, 1]
                pxstd = np.loadtxt(ofolder + 'xstd-L%04d_%0.4f%s.txt'%(bs, aa, suff))[:, 1]
                rccstd, tfstd = pxstd/(pstd * ps)**0.5, (pstd/ps)**0.5
                biasstd = tfstd[1:5].mean()

                #biasr, biasstd = 1, 1
                #print(biasr, biasstd)
                fid, std = [rccr, tfr, biasr, pr], [rccstd, tfstd, biasstd, pstd]
                sig = []
                for rcc, tf, bias, pkr in [fid, std]:

                    p02 = np.interp(0.2, k, ps).real
                    ck = rcc*tf/bias
                    #     pmc = pkr - ck**2 * pkd                
                    pmc = pkr - bias**2* ck**2 * ps
                    #     den = (pkd * ck**2 + pmc)/p02 
                    den = (ps * (bias* ck)**2 + pmc)/p02
                    num = np.exp(-2*(k*silk)**1.4)*(bias*ck)**4
                    
                    #integ = fac*np.array([np.trapz(k[:i]**2 * num[:i]/den[:i]**2,  k[:i]) for i in np.arange(k.size)])
                    fish = fac*np.trapz(k[1:]**2 * num[1:]/den[1:]**2,  k[1:]) 
                    sig.append(fish**-0.5)
                
                #print(sig, sig[1]/sig[0])
                print(sig[1]/sig[0])



################
def make_1d_plot():
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
                    dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_%s/'%(aa, 0.03, wopt)
                    dpath += 'L%04d-N%04d-R//thermal-%s-hex/ZA/opt_s999_h1massA_fourier_rsdpos/'%(bs, nc, thopt)
                    ofolder = '../../data/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/'%(wopt, thopt)
                    if args.pp: suff = '-up'
                    else: suff = ''
                    recon = np.loadtxt(ofolder + 'recon-L%04d_%0.4f%s.txt'%(bs, aa, suff))
                    k = recon[:, 0]
                    pxr = recon[:, 2]
                    pr = recon[:, 6]
                    ps = recon[:, 7]
                    pstd = np.loadtxt(ofolder + 'std-L%04d_%0.4f%s.txt'%(bs, aa, suff))[:, 1]
                    pxstd = np.loadtxt(ofolder + 'xstd-L%04d_%0.4f%s.txt'%(bs, aa, suff))[:, 1]

                    if ia == 0 and iw == 0:
                        if thopt == 'reas': thopt = 'fid'
                        lbl = 'Noise - %s'%thopt
                    elif ia == 1 and it == 0:
                        lbl = 'Wedge = %s'%wopt
                    else: lbl = None
                    ax[0].plot(k, pxr/(pr*ps)**0.5, ls=lss, lw=lww, color=cc, label=lbl)
                    ax[1].plot(k, pxstd/(pstd*ps)**0.5, ls=lss, lw=lww, color=cc)
                    ax[1].text(0.6, 0.8, r'$z = %.2f$'%zz,color='black',ha='left',va='bottom', fontdict=font)
                except Exception as e: 
                    if rank == 0: print(e)


    axar[0, 0].set_title('Iterative', fontdict=font)
    axar[0, 1].set_title('Standard', fontdict=font)
    for axis in axar[:, 0]: 
        axis.set_ylabel('$r_{cc}$', fontdict=font)
        axis.set_ylim(-0.05, 1.1)
    for axis in axar[:, 1]: 
        axis.set_ylabel(r'$T_f$', fontdict=font)
        axis.set_ylim(-0.05, 1.1)
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
    if rank  == 0 and not args.pp: plt.savefig(figpath + '/rccstd_L%04d-hex.pdf'%(bs))
    if rank  == 0 and args.pp: plt.savefig(figpath + '/rccstd_L%04d-hexup.pdf'%(bs))




def make_2d_plot(Nmu=4):
    """Does the work of making the real-space xi(r) and b(r) figure."""
    
  
    mubins = np.linspace(0, 1, Nmu+1)
    mus = 0.5*(mubins[1:] + mubins[:-1])
    fig, axar = plt.subplots(3, 2, figsize=(9, 9), sharex=True)

    #fits
    linestyle=['-', '--']
    colors=['C0', 'C1', 'C2']
    lww = 2
    
    for ia, aa  in enumerate([0.3333, 0.2000, 0.1429]):
        zz = 1/aa-1
        for iw, wopt in enumerate(['opt', 'pess']):
            lss = linestyle[iw]
            #for it, thopt in enumerate(['opt', 'pess', 'reas']):
            for it, thopt in enumerate(['reas']):
                if rank == 0: print(aa, wopt, thopt)
                cc = colors[it]
                ax = axar[ia]
                try:
                    angle = np.round(mapn.wedge(zz, att=wopt, angle=True), 0)
                    dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_%s/'%(aa, 0.03, wopt)
                    dpath += 'L%04d-N%04d-R//thermal-%s-hex/ZA/opt_s999_h1massA_fourier_rsdpos/'%(bs, nc, thopt)
                    ofolder = '../../data/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/Nmu%d/'%(wopt, thopt, Nmu)
                    if args.pp: suff = '-up'
                    else: suff = ''
                    k = np.loadtxt(ofolder + 'recon-L%04d_%0.4f%s-k.txt'%(bs, aa, suff))
                    pxr = np.loadtxt(ofolder + 'recon-L%04d_%0.4f%s-xs.txt'%(bs, aa, suff))
                    pr = np.loadtxt(ofolder + 'recon-L%04d_%0.4f%s-ps1.txt'%(bs, aa, suff))
                    ps = np.loadtxt(ofolder + 'recon-L%04d_%0.4f%s-ps2.txt'%(bs, aa, suff))
                    pstd = np.loadtxt(ofolder + 'std-L%04d_%0.4f%s.txt'%(bs, aa, suff))
                    pxstd = np.loadtxt(ofolder + 'xstd-L%04d_%0.4f%s.txt'%(bs, aa, suff))
                    rcc = pxr/(pr*ps)**0.5
                    rccstd = pxstd/(pstd*ps)**0.5
                    for imu, mu in enumerate([0, Nmu//2, Nmu-1]):
                        cc = 'C%d'%imu
                        if ia == 0 and iw == 0:
                            if thopt == 'reas': thopt = 'fid'
                            lbl = r'$\mu = %.3f-%0.3f$'%(mubins[mu], mubins[mu+1])
                        elif ia == 1 and mu == 0:
                            lbl = 'Wedge = %s'%wopt
                        else: lbl = None
                        ax[0].plot(k[:, mu], rcc[:, mu], ls=lss, lw=lww, color=cc, label=lbl)
                        ax[1].plot(k[:, mu], rccstd[:, mu], ls=lss, lw=lww, color=cc)
                    ax[1].text(0.6, 0.8, r'$z = %.2f$'%zz,color='black',ha='left',va='bottom', fontdict=font)

                except Exception as e: 
                    if rank == 0: print(e)


    axar[0, 0].set_title('Iterative', fontdict=font)
    axar[0, 1].set_title('Standard', fontdict=font)

    for axis in axar[:, 0]: 
        axis.set_ylabel('$r_{cc}$', fontdict=font)
        axis.set_ylim(-0.05, 1.1)
    for axis in axar[:, 1]: 
        axis.set_ylabel(r'$r_{cc}$', fontdict=font)
        axis.set_ylim(-0.05, 1.1)
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
    if rank  == 0 and not args.pp: plt.savefig(figpath + '/rccstd_L%04d-Nmu%d-hex.pdf'%(bs, Nmu))
    if rank  == 0 and args.pp: plt.savefig(figpath + '/rccstd_L%04d-Nmu%d-hexup.pdf'%(bs, Nmu))





def make_plot(Nmu=4):
    """Does the work of making the real-space xi(r) and b(r) figure."""
    
  
    mubins = np.linspace(0, 1, Nmu+1)
    mus = 0.5*(mubins[1:] + mubins[:-1])
    fig, axar = plt.subplots(2, 3, figsize=(9, 6), sharex=True, sharey=True)

    #fits
    linestyle=['-', '--']
    colors=['C0', 'C1', 'C2']
    lww = 2
    
    for ia, aa  in enumerate([0.3333, 0.2000, 0.1429]):
        zz = 1/aa-1
        for iw, wopt in enumerate(['opt', 'pess']):
            lss = linestyle[iw]
            #for it, thopt in enumerate(['opt', 'pess', 'reas']):
            for it, thopt in enumerate(['reas']):
                if rank == 0: print(aa, wopt, thopt)
                cc = colors[ia]
                try:
                    angle = np.round(mapn.wedge(zz, att=wopt, angle=True), 0)
                    dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_%s/'%(aa, 0.03, wopt)
                    dpath += 'L%04d-N%04d-R//thermal-%s-hex/ZA/opt_s999_h1massA_fourier_rsdpos/'%(bs, nc, thopt)
                    ofolder = '../../data/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/Nmu%d/'%(wopt, thopt, Nmu)
                    if args.pp: suff = '-up'
                    else: suff = ''
                    k2d = np.loadtxt(ofolder + 'recon-L%04d_%0.4f%s-k.txt'%(bs, aa, suff))
                    pxr = np.loadtxt(ofolder + 'recon-L%04d_%0.4f%s-xs.txt'%(bs, aa, suff))
                    pr = np.loadtxt(ofolder + 'recon-L%04d_%0.4f%s-ps1.txt'%(bs, aa, suff))
                    ps = np.loadtxt(ofolder + 'recon-L%04d_%0.4f%s-ps2.txt'%(bs, aa, suff))
                    pstd = np.loadtxt(ofolder + 'std-L%04d_%0.4f%s.txt'%(bs, aa, suff))
                    pxstd = np.loadtxt(ofolder + 'xstd-L%04d_%0.4f%s.txt'%(bs, aa, suff))
                    rcc2d = pxr/(pr*ps)**0.5
                    rccstd2d = pxstd/(pstd*ps)**0.5

                    ofolder = '../../data/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/'%(wopt, thopt)
                    if args.pp: suff = '-up'
                    else: suff = ''
                    recon = np.loadtxt(ofolder + 'recon-L%04d_%0.4f%s.txt'%(bs, aa, suff))
                    pstd = np.loadtxt(ofolder + 'std-L%04d_%0.4f%s.txt'%(bs, aa, suff))[:, 1]
                    pxstd = np.loadtxt(ofolder + 'xstd-L%04d_%0.4f%s.txt'%(bs, aa, suff))[:, 1]
                    k = recon[:, 0]
                    rcc = recon[:, 2]/(recon[:, 6]*recon[:, 7])**0.5
                    rccstd = pxstd/(pstd*recon[:, 7])**0.5
                    
                    lbl, lbl1, lbl2 = None, None, None
                    if iw == 0:
                        lbl1 = 'z = %0.1f'%zz
                    if ia == 0:
                        lbl2 = 'Wedge = %s'%wopt

                    ax = axar.T
                    ax[0, 0].plot(k, rcc, ls=lss, lw=lww, color=cc, label=lbl1)
                    ax[0, 1].plot(k, rccstd, ls=lss, lw=lww, color=cc, label=lbl2)
                    ax[1, 0].plot(k, rcc2d[:, 0], ls=lss, lw=lww, color=cc, label=lbl)
                    ax[1, 1].plot(k, rccstd2d[:, 0], ls=lss, lw=lww, color=cc, label=lbl)
                    ax[2, 0].plot(k, rcc2d[:, -1], ls=lss, lw=lww, color=cc, label=lbl)
                    ax[2, 1].plot(k, rccstd2d[:, -1], ls=lss, lw=lww, color=cc, label=lbl)

                    ax[1, 0].text(0.01, 0.4, 'Iterative\nRecon',color='black',ha='left',va='top', fontdict=font)
                    ax[1, 1].text(0.01, 0.8, 'Standard\nRecon',color='black',ha='left',va='top', fontdict=font)
                    ax[0, 0].text(1.5, 1, r'1D',color='black',ha='right',va='bottom', fontdict=font)
                    ax[1, 0].text(1.5, 1, r'$\mu=%0.3f-%0.3f$'%(mubins[0], mubins[1]),color='black',ha='right',va='bottom', fontdict=font)
                    ax[2, 0].text(1.5, 1, r'$\mu=%0.3f-%0.3f$'%(mubins[-2], mubins[-1]),color='black',ha='right',va='bottom', fontdict=font)


                except Exception as e: 
                    if rank == 0: print(e)


    #axar[0, 0].set_title('Iterative', fontdict=font)
    #axar[0, 1].set_title('Standard', fontdict=font)

    for axis in axar[:, 0]: 
        axis.set_ylabel('$r_{cc}$', fontdict=font)
        axis.set_ylim(-0.1, 1.15)
        axis.legend(prop=fontmanage, loc=0)
    for axis in axar[1, :]: axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    for axis in axar.flatten():
        axis.axhline(1, color='k', ls=':', lw=0.5)
        axis.set_xscale('log')
        axis.grid(which='both', lw=0.2, alpha=0.2, color='gray')

    # Put on some more labels.
    for axis in axar.flatten():
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
    ##and finish
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if rank  == 0 and not args.pp: plt.savefig(figpath + '/rccstd12d_L%04d-Nmu%d-hex.pdf'%(bs, Nmu))
    if rank  == 0 and args.pp: plt.savefig(figpath + '/rccstd12d_L%04d-Nmu%d-hexup.pdf'%(bs, Nmu))


################


if __name__=="__main__":
    if args.mode == 'save':
        savestd()
        savestd2d(Nmu=4)
        savestd2d(Nmu=8)
    elif args.mode =='plot':
        #make_1d_plot()
        #make_2d_plot(Nmu=4)
        #make_2d_plot(Nmu=8)
        make_plot(Nmu=4)
        make_plot(Nmu=8)
    elif args.mode == 'fish':
        fishintrep1d()
    #
    #
