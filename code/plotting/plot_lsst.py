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
from cosmo4d.pmeshengine import nyquist_mask
from lab import mapbias as mapp
from lab import mapnoise as mapn
from lab import report as rp
from lab import dg
from getbiasparams import getbias
import tools
#

from matplotlib import rc, rcParams, font_manager
rcParams['font.family'] = 'serif'
fsize = 12-1
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
parser.add_argument('-a', '--aa', help='scale factor', default=0.5000, type=float)
parser.add_argument('-l', '--bs', help='boxsize', default=1024, type=float)
parser.add_argument('-n', '--nmesh', help='nmesh', default=256, type=int)
parser.add_argument('-t', '--angle', help='angle of the wedge', default=50, type=float)
parser.add_argument('-k', '--kmin', help='kmin of the wedge', default=0.03, type=float)
parser.add_argument( '--pp', help='upsample', default=1) 

args = parser.parse_args()

figpath = './figs/'
dpath = '../../data/'
bs, nc, aa = args.bs, args.nmesh, args.aa
nc2 = nc*2
zz = 1/aa- 1
kmin = args.kmin
ang = args.angle
if args.pp: pm = ParticleMesh(BoxSize=bs, Nmesh=[nc2, nc2, nc2])
else: pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank

##





def save2dphoto(Nmu=4, numd=1e-2, aa=None, scatter=False):
    
    if numd > 1e-2: 
        print('Too high number density')
        sys.exit()
    num = int(numd*bs**3)

    if aa is None: aas = [0.3333, 0.2000, 0.1429]
    else: aas = [aa]
    for ia, aa  in enumerate(aas):
        zz = 1/aa-1
        sigz = lambda z : 120*((1+z)/5)**-0.5

        ##
        cat = BigFileCatalog('/global/cscratch1/sd/chmodi/m3127/H1mass/highres/10240-9100/fastpm_%0.4f/Hcat-Numd-%04d/'%(aa, 1e-2*1e4))
        if scatter:
            pos = cat['Position'][:num].compute()
            dz = np.random.normal(0, sigz(zz), size=pos[:, -1].size)
            pos[:, -1] += dz
            layout = pm.decompose(pos)
            hmesh = pm.paint(pos, layout=layout)
        else:
            pos = cat['Position'][:num].compute()
            layout = pm.decompose(pos)
            hmesh = pm.paint(pos, layout=layout)

        def tf(k): #Photoz smoothing
            kmesh = sum(ki ** 2 for ki in k)**0.5
            kmesh[kmesh == 0] = 1
            mumesh = k[2]/kmesh
            weights = np.exp(-kmesh**2 * mumesh**2 * sigz(zz)**2/2.)
            return weights

        hmesh /= hmesh.cmean()
        if not scatter:
            hmeshc = hmesh.r2c()
            hmeshc.apply(lambda k, v: nyquist_mask(tf(k), v) * v, out=Ellipsis)
            hmesh = hmeshc.c2r()
        ph = FFTPower(hmesh, mode='2d', Nmu=Nmu).power

        #
        for iw, wopt in enumerate(['opt', 'pess']):
        #for iw, wopt in enumerate(['opt']):
            for it, thopt in enumerate(['opt', 'pess', 'reas']):
            #for it, thopt in enumerate([ 'reas']):
                if rank == 0: print(aa, wopt, thopt)

                angle = np.round(mapn.wedge(zz, att=wopt, angle=True), 0)
                #dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_ang%0.1f/'%(aa, 0.03, angle)
                dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_%s/'%(aa, 0.03, wopt)
                dpath += 'L%04d-N%04d-R//thermal-%s-hex/ZA/opt_s999_h1massA_fourier_rsdpos/'%(bs, nc, thopt)
                if scatter: ofolder = '../../data/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/photog-Numd%04d-Nmu%d/'%(wopt, thopt, numd*1e4, Nmu)
                else: ofolder = '../../data/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/photo-Numd%04d-Nmu%d/'%(wopt, thopt, numd*1e4, Nmu)
                try: os.makedirs(ofolder)
                except: pass
                if rank == 0: print(ofolder)

                if args.pp:
                    datapp = BigFileMesh(dpath+'/dataw_up/', 'mapp').paint()
                    bpaths = [dpath+'upsample2/%d-0.00//best-fit'%nc2] + [dpath + 'upsample2/%d-0.00//%04d/fit_p/'%(nc2,i) for i in range(100, 50, -20)]
                else:
                    datapp = BigFileMesh(dpath+'/dataw/', 'mapp').paint()
                    bpaths = [dpath+'%d-0.00//best-fit'%nc] + [dpath + '%d-0.00//%04d/fit_p/'%(nc,i) for i in range(100, 50, -20)]
                for path in bpaths:
                    if os.path.isdir(path): 
                        break
                if rank == 0: print(path)
                bfit = BigFileMesh(path, 'mapp').paint()

                pxrh = FFTPower(hmesh, second=bfit, mode='2d', Nmu=Nmu).power
                pxwh = FFTPower(hmesh, second=datapp, mode='2d', Nmu=Nmu).power

                fname = ofolder + 'photo-L%04d_%0.4f.txt'%(bs, aa)
                if args.pp : fname = fname[:-4] + '-up.txt'
                np.savetxt(fname, ph['power'].real)

                fname = ofolder + 'xdataw-L%04d_%0.4f.txt'%(bs, aa)
                if args.pp : fname = fname[:-4] + '-up.txt'
                np.savetxt(fname, pxwh['power'].real)

                fname = ofolder + 'xrecon-L%04d_%0.4f.txt'%(bs, aa)
                if args.pp : fname = fname[:-4] + '-up.txt'
                np.savetxt(fname, pxrh['power'].real)


def make_plot(Nmu=4, wopt='opt', thopt='reas'):

    sigz = lambda z : 120*((1+z)/5)**-0.5
    nbar = 10**-2.5
    b = 3.2
    Dphoto = lambda k, mu, z: np.exp(-k**2 * mu**2 * sigz(z)**2/2.)
    
   
    kk = np.loadtxt(dpath + '/ZArecon-rsd/kmin-003_wedge-opt/thermal-reas-hex/Nmu%d/recon-L%04d_%0.4f-up-k.txt'%(Nmu, bs, aa))
    try: modes = np.loadtxt(dpath + '/ZArecon-rsd/kmin-003_wedge-opt/thermal-reas-hex/Nmu%d/recon-L%04d_%0.4f-up-modes.txt'%(Nmu, bs, aa))
    except:
        datap = mapp.Observable.load('/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin0.03_opt/L%04d-N0256-R/thermal-reas-hex/ZA/opt_s999_h1massA_fourier_rsdpos/datap_up/'%(aa, bs))
        tmp = FFTPower(datap.mapp, mode='2d', Nmu=Nmu).power
        modes = tmp['modes'].astype('float64')
        np.savetxt(dpath + '/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/Nmu%d/recon-L%04d_%0.4f-up-modes.txt'%(wopt, thopt, Nmu, bs, aa), modes)
    pm1 = np.loadtxt(dpath + '/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/Nmu%d/recon-L%04d_%0.4f-up-pm1.txt'%(wopt, thopt, Nmu, bs, aa))
    pm2 = np.loadtxt(dpath + '/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/Nmu%d/recon-L%04d_%0.4f-up-pm2.txt'%(wopt, thopt, Nmu, bs, aa))
    xm = np.loadtxt(dpath + '/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/Nmu%d/recon-L%04d_%0.4f-up-xm.txt'%(wopt, thopt, Nmu, bs, aa))
    xmw = np.loadtxt(dpath + '/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/Nmu%d/dataw-L%04d_%0.4f-up-xm.txt'%(wopt, thopt, Nmu, bs, aa))
    pm1w = np.loadtxt(dpath + '/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/Nmu%d/dataw-L%04d_%0.4f-up-pm1.txt'%(wopt, thopt, Nmu, bs, aa))

    mubins = np.linspace(0, 1, kk.shape[1]+1)
    mu = (mubins[1:] + mubins[:-1])*0.5
    pkd = np.loadtxt(dpath + '/pklin_%0.4f.txt'%aa)
    # pk = np.loadtxt(dpath + '/pklin_1.0000.txt')
    ipkd = ius(pkd[:, 0], pkd[:, 1])


    rr = xm/(pm1*pm2)**0.5
    rrw = xmw/(pm1w*pm2)**0.5
    pkd = ipkd(kk)

    fac = b**2*Dphoto(kk, mu, zz)**2 *nbar*pkd
    rhosq = rr**2*fac/(1+fac)
    rhosqw = rrw**2*fac/(1+fac)


    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    for i in range(mu.size):
        lbl1, lbl2 = None, None
        if i < mu.size//2: lbl1 = '$\mu$=%0.3f-%0.3f'%(mubins[i], mubins[i+1])
        else: lbl2 = '$\mu$=%0.3f-%0.3f'%(mubins[i], mubins[i+1])
        ax[0].plot(kk[:, i], rhosq[:, i], 'C%d'%i, label=lbl1, lw=2)
        ax[1].plot(kk[:, i], modes[:, i]**-1*(1+rhosq[:, i]**-1), 'C%d'%i, label=lbl2, lw=2)

        ax[0].plot(kk[:, i], rhosqw[:, i], 'C%d--'%i, alpha=0.5)
        ax[1].plot(kk[:, i], modes[:, i]**-1*(1+rhosqw[:, i]**-1), 'C%d--'%i, alpha=0.5)
        ax[0].plot(kk[:, 0], Dphoto(kk[:, 0], mu[i], zz)**2, 'C%d'%i, lw=1, alpha=1, ls=":")

    ax[1].set_ylim(1e-3, 100)
    ax[1].set_yscale('log')
    ax[1].axhline(1, color='k', ls="--")

    ax[0].set_ylabel(r'$\rho^2$', fontdict=font)
    #ax[1].set_ylabel(r'$N^{-1}(1+\rho^{-2})$', fontsize=14)
    ax[1].set_ylabel(r'Var$(P_\times)/P_\times^2$', fontdict=font)

    ax[0].legend(prop=fontmanage, loc=1)
    ax[1].legend(prop=fontmanage, loc=4)

    for axis in ax[:]: axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    for axis in ax.flatten():
        #axis.axhline(1, color='k', ls=':')
        axis.set_xscale('log')
        axis.grid(which='both', lw=0.2, alpha=0.2, color='gray')

    # Put on some more labels.
    for axis in ax.flatten():
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
    ##and finish
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if rank  == 0 and not args.pp: plt.savefig(figpath + '/photo_z%d_L%04d-Nmu%d.pdf'%(zz*10, bs, Nmu))
    if rank  == 0 and args.pp: plt.savefig(figpath + '/photo_z%d_L%04d-Nmu%d-up.pdf'%(zz*10, bs, Nmu))


                


                
def make_plot_data(aa, numd, Nmu=8, wopt='opt', thopt='reas', scatter=False):


    #
    mubins = np.linspace(0, 1, Nmu+1)
    mu = (mubins[1:] + mubins[:-1])*0.5
    kk = np.loadtxt(dpath + '/ZArecon-rsd/kmin-003_wedge-opt/thermal-reas-hex/Nmu%d/recon-L%04d_%0.4f-up-k.txt'%(Nmu, bs, aa))
    modes = np.loadtxt(dpath + '/ZArecon-rsd/kmin-003_wedge-opt/thermal-reas-hex/Nmu%d/recon-L%04d_%0.4f-up-modes.txt'%(Nmu, bs, aa))

    pr = np.loadtxt(dpath + '/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/Nmu%d/recon-L%04d_%0.4f-up-pm1.txt'%(wopt, thopt, Nmu, bs, aa))
    pw = np.loadtxt(dpath + '/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/Nmu%d/dataw-L%04d_%0.4f-up-pm1.txt'%(wopt, thopt, Nmu, bs, aa))

    pm1 = np.loadtxt(dpath + '/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/Nmu%d/recon-L%04d_%0.4f-up-pm1.txt'%(wopt, thopt, Nmu, bs, aa))
    pm2 = np.loadtxt(dpath + '/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/Nmu%d/recon-L%04d_%0.4f-up-pm2.txt'%(wopt, thopt, Nmu, bs, aa))
    xm = np.loadtxt(dpath + '/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/Nmu%d/recon-L%04d_%0.4f-up-xm.txt'%(wopt, thopt, Nmu, bs, aa))
    rr = xm/(pm1*pm2)**0.5

    pkd = np.loadtxt(dpath + '/pklin_%0.4f.txt'%aa)
    ipkd = ius(pkd[:, 0], pkd[:, 1])
    pkd = ipkd(kk)

    if scatter : ofolder = '../../data/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/photog-Numd%04d-Nmu%d/'%(wopt, thopt, numd*1e4, Nmu)
    else: ofolder = '../../data/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/photo-Numd%04d-Nmu%d/'%(wopt, thopt, numd*1e4, Nmu)
    print(ofolder)

    #get data
    fname = ofolder + 'photo-L%04d_%0.4f.txt'%(bs, aa)
    if args.pp : fname = fname[:-4] + '-up.txt'
    ph = np.loadtxt(fname)
    ph += 1/numd
    fname = ofolder + 'xrecon-L%04d_%0.4f.txt'%(bs, aa)
    if args.pp : fname = fname[:-4] + '-up.txt'
    pxrh = np.loadtxt(fname)
    fname = ofolder + 'xdataw-L%04d_%0.4f.txt'%(bs, aa)
    if args.pp : fname = fname[:-4] + '-up.txt'
    pxwh = np.loadtxt(fname)
    
    rhosq = pxrh**2/ph/pr 
    rhosqw = pxwh**2/ph/pw 

    #get theory
    sigz = lambda z : 120*((1+z)/5)**-0.5
    Dphoto = lambda k, mu, z: np.exp(-k**2 * mu**2 * sigz(z)**2/2.)
    nbar = 10**-2.5
    b = 3.2

    def iget(ii, k=1):
        yy = rr[ii]
        mask = ~np.isnan(yy)
        return ius(mu[mask], yy[mask], k=k)

    mus = np.linspace(0, 1, 500)
    rhosqmu = np.zeros((kk.shape[0],  mus.size))
    for ik, kv in enumerate(kk[:, -1]):
        fac = b**2*Dphoto(kv, mus, zz)**2 *nbar*ipkd(kv)
        try: rhosqmu[ik] = iget(ik)(mus)**2*fac/(1+fac)
        except Exception as e: print(ik, e)
    
    rhosqav = np.zeros((kk.shape[0], mu.size))
    for i in range(mu.size):
        mask = (mus > mubins[i]) & (mus < mubins[i+1])
        rhosqav[: ,i] = np.trapz(rhosqmu[:, mask], mus[mask])/(mubins[i+1]-mubins[i])


    #make figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    for i in range(mu.size):
        lbl1, lbl2 = None, None
        if i <= mu.size: lbl1 = '$\mu$=%0.3f-%0.3f'%(mubins[i], mubins[i+1])
        #else: lbl2 = '$\mu$=%0.3f-%0.3f'%(mubins[i], mubins[i+1])
        if i ==0: lbl2 = r'Recon$_{\rm Sim}$'
        ax[0].plot(kk[:, i], rhosq[:, i], 'C%d'%i, label=lbl1)
        ax[1].plot(kk[:, i], modes[:, i]**-1*(1+rhosq[:, i]**-1), 'C%d'%i, label=lbl2)

        #ax[0].plot(kk[:, i], rhosqw[:, i], 'C%d--'%i, alpha=0.4)
        if i ==0: lbl2 = r'Noisy$_{\rm Sim}$'
        ax[1].plot(kk[:, i], modes[:, i]**-1*(1+rhosqw[:, i]**-1), 'C%d:'%i, alpha=1, lw=0.5, label=lbl2)

        ax[0].plot(kk[:, i], rhosqav[:, i], 'C%d--'%i, alpha=1, lw=1)
        if i ==0: lbl2 = r'Recon$_{\rm Pred}$'
        ax[1].plot(kk[:, i], modes[:, i]**-1*(1+rhosqav[:, i]**-1), 'C%d--'%i, alpha=1, lw=1, label=lbl2)

        if i ==0: lbl0 = r'$D_{\rm  photo}^2$'
        else: lbl0 = None
        ax[0].plot(kk[:, 0], Dphoto(kk[:, 0], mu[i], zz)**2, 'C%d'%i, lw=0.5, alpha=1, ls=":", label=lbl0)

    #
    ax[0].set_ylim(-.05, 1.1)

    ax[1].set_ylim(9e-4, 100)
    ax[1].set_yscale('log')
    ax[1].axhline(1, color='k', ls="--")

    ax[0].set_ylabel(r'$\rho^2$', fontdict=font)
    #ax[1].set_ylabel(r'$N^{-1}(1+\rho^{-2})$', fontsize=14)
    ax[1].set_ylabel(r'Var$(P_\times)/P_\times^2$', fontdict=font)

    ax[0].legend(prop=fontmanage, loc=1, ncol=1)
    ax[1].legend(prop=fontmanage, loc=3, ncol=1)

    for axis in ax[:]: axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    for axis in ax.flatten():
        #axis.axhline(1, color='k', ls=':')
        axis.set_xscale('log')
        axis.grid(which='both', lw=0.2, alpha=0.2, color='gray')

    # Put on some more labels.
    for axis in ax.flatten():
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
    # and finish
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if rank  == 0 and not args.pp: plt.savefig(figpath + '/photod_z%d_L%04d-Nmu%d.pdf'%(zz*10, bs, Nmu))
    if rank  == 0 and args.pp: 
        if scatter : plt.savefig(figpath + '/photodg_z%d_L%04d-Nmu%d-up.pdf'%(zz*10, bs, Nmu))
        else : plt.savefig(figpath + '/photod_z%d_L%04d-Nmu%d-up.pdf'%(zz*10, bs, Nmu))


################

if __name__=="__main__":
    #save2dphoto(Nmu=4, numd=10**-2.5, aa=0.2000)
    #save2dphoto(Nmu=8, numd=10**-2.5, aa=0.2000)
    #save2dphoto(Nmu=4, numd=10**-2.5, aa=0.2000, scatter=True)
    #save2dphoto(Nmu=8, numd=10**-2.5, aa=0.2000, scatter=True)
    #make_plot(Nmu=4)
    #make_plot(Nmu=8)
    make_plot_data(aa=0.2000, numd=10**-2.5, Nmu=8)
    make_plot_data(aa=0.2000, numd=10**-2.5, Nmu=8, scatter=True)
    make_plot_data(aa=0.2000, numd=10**-2.5, Nmu=4)
    make_plot_data(aa=0.2000, numd=10**-2.5, Nmu=4, scatter=True)
    #

