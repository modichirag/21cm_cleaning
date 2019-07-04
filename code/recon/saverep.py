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
#parser.add_argument('-m', '--model', help='model name to use')
parser.add_argument('-a', '--aa', help='scale factor', default=0.3333, type=float)
parser.add_argument('-l', '--bs', help='boxsize', default=1024, type=float)
parser.add_argument('-n', '--nmesh', help='nmesh', default=256, type=int)
parser.add_argument('-t', '--angle', help='angle of the wedge', default=50, type=float)
parser.add_argument('-k', '--kmin', help='kmin of the wedge', default=0.03, type=float)
parser.add_argument( '--pp', help='upsample', default=0) 

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
def make_repwd_plot():
    """Does the work of making the real-space xi(r) and b(r) figure."""
    
   
    
    for ia, aa  in enumerate([0.3333, 0.2000, 0.1429]):
        zz = 1/aa-1
        for iw, wopt in enumerate(['opt', 'pess']):
            for it, thopt in enumerate(['opt', 'pess', 'reas']):
                if rank == 0: print(aa, wopt, thopt)

                angle = np.round(mapn.wedge(zz, att=wopt, angle=True), 0)
                #dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_ang%0.1f/'%(aa, 0.03, angle)
                dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_%s/'%(aa, 0.03, wopt)
                dpath += 'L%04d-N%04d-R//thermal-%s-hex/ZA/opt_s999_h1massA_fourier_rsdpos/'%(bs, nc, thopt)
                ofolder = '../../data/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/'%(wopt, thopt)
                try: os.makedirs(ofolder)
                except: pass
                fname = ofolder + 'recon-L%04d_%0.4f.txt'%(bs, aa)
                if args.pp : fname = fname[:-4] + '-up.txt'
                header = 'k, xm.power, xs.power, xd.power,  pm1.power, pm2.power, ps1.power, ps2.power, pd1.power, pd2.power'
                try:
                    rep = np.loadtxt(fname).T
                    rpfit = [{'k':rep[0], 'power':rep[i+1]} for i in range(3)]
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
                    #bfit = mapp.Observable.load(dpath+'ZA/opt_s999_h1massA_fourier_rsdpos/best-fit/')
                    rpfit = rp.evaluate(bfit, datapp)[:-2]
                    if rank == 0: np.savetxt(fname, np.stack([rpfit[0]['k']]+ [rpfit[i]['power'].real for i in range(len(rpfit))]).T, header=header)


                fname = ofolder + 'dataw-L%04d_%0.4f.txt'%(bs, aa)
                if args.pp : fname = fname[:-4] + '-up.txt'
                try:
                    rep = np.loadtxt(fname).T
                    rpfit = [{'k':rep[0], 'power':rep[i+1]} for i in range(3)]
                except:
                    if args.pp:
                        datapp = mapp.Observable.load(dpath+'/datap_up')
                        bfit = mapp.Observable.load(dpath+'/dataw_up')
                    else:
                        datapp = mapp.Observable.load(dpath+'/datap')
                        bfit = mapp.Observable.load(dpath+'/dataw')
                    #bfit = mapp.Observable.load(dpath+'ZA/opt_s999_h1massA_fourier_rsdpos/best-fit/')
                    rpfit = rp.evaluate(bfit, datapp)[:-2]
                    if rank == 0: np.savetxt(fname, np.stack([rpfit[0]['k']]+ [rpfit[i]['power'].real for i in range(len(rpfit))]).T, header=header)




def make_repwd2d_plot(Nmu=8):
    """Does the work of making the real-space xi(r) and b(r) figure."""
    
   
    
    for ia, aa  in enumerate([0.3333, 0.2000, 0.1429]):
        zz = 1/aa-1
        for iw, wopt in enumerate(['opt', 'pess']):
            for it, thopt in enumerate(['opt', 'pess', 'reas']):
                if rank == 0: print(aa, wopt, thopt)

                angle = np.round(mapn.wedge(zz, att=wopt, angle=True), 0)
                #dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_ang%0.1f/'%(aa, 0.03, angle)
                dpath = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%0.2f_%s/'%(aa, 0.03, wopt)
                dpath += 'L%04d-N%04d-R//thermal-%s-hex/ZA/opt_s999_h1massA_fourier_rsdpos/'%(bs, nc, thopt)
                ofolder = '../../data/ZArecon-rsd/kmin-003_wedge-%s/thermal-%s-hex/Nmu%d/'%(wopt, thopt, Nmu)
                try: os.makedirs(ofolder)
                except: pass
                fname = ofolder + 'recon-L%04d_%0.4f.txt'%(bs, aa)
                if args.pp : fname = fname[:-4] + '-up.txt'
                header = 'k, xm.power, xs.power, xd.power, pm1.power, pm2.power, ps1.power, ps2.power, pd1.power, pd2.power'
                try:
                    rep = np.loadtxt(fname).T
                    rpfit = [{'k':rep[0], 'power':rep[i+1]} for i in range(3)]
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
                    #bfit = mapp.Observable.load(dpath+'ZA/opt_s999_h1massA_fourier_rsdpos/best-fit/')
                    rpfit = rp.evaluate2d(bfit, datapp, Nmu=Nmu)
                    if rank == 0: 
                        np.savetxt(fname[:-4] + '-k.txt', rpfit[0]['k'])
                        for ip in range(len(rpfit)): 
                            sf = header.split(',')[ip+1]
                            ff = fname[:-4] + '-%s.txt'%(sf.split('.')[0][1:])
                            print(sf, ff)
                            np.savetxt(ff, rpfit[ip]['power'].real)


                fname = ofolder + 'dataw-L%04d_%0.4f.txt'%(bs, aa)
                if args.pp : fname = fname[:-4] + '-up.txt'
                try:
                    rep = np.loadtxt(fname).T
                    rpfit = [{'k':rep[0], 'power':rep[i+1]} for i in range(3)]
                except:
                    if args.pp:
                        datapp = mapp.Observable.load(dpath+'/datap_up')
                        bfit = mapp.Observable.load(dpath+'/dataw_up')
                    else:
                        datapp = mapp.Observable.load(dpath+'/datap')
                        bfit = mapp.Observable.load(dpath+'/dataw')
                    #bfit = mapp.Observable.load(dpath+'ZA/opt_s999_h1massA_fourier_rsdpos/best-fit/')
                    rpfit = rp.evaluate2d(bfit, datapp, Nmu=Nmu)
                    if rank == 0: 
                        np.savetxt(fname[:-4] + '-k.txt', rpfit[0]['k'])
                        for ip in range(len(rpfit)): 
                            sf = header.split(',')[ip+1]
                            ff = fname[:-4] + '-%s.txt'%(sf.split('.')[0][1:])
                            print(sf, ff)
                            np.savetxt(ff, rpfit[ip]['power'].real)


if __name__=="__main__":
    #make_rep_plot()
    #make_repwd_plot()
    #make_repwd2d_plot()
    make_repwd2d_plot(Nmu=3)
    #
