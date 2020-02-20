import warnings
from mpi4py import MPI
rank = MPI.COMM_WORLD.rank
#warnings.filterwarnings("ignore")
if rank!=0: warnings.filterwarnings("ignore")

import numpy
import numpy as np

#from cosmo4d.lab import mapbias as map
from cosmo4d import lab
from cosmo4d.lab import mapnoise

from nbodykit.lab import BigFileMesh, FFTPower


#


Nmu = 10
mus = np.linspace(0, 1, Nmu)


def getmask(k, angle, kmin=0.03):
    kmesh = sum(ki ** 2 for ki in k)**0.5
    mask = [numpy.ones_like(ki) for ki in k]
    mask[2] *= abs(k[2]) >= kmin
    mask = numpy.prod(mask)

    mask2 = numpy.ones_like(mask)
    if angle > 0:
        kperp = (k[0]**2 + k[1]**2)**0.5
        kangle = abs(k[2])/(kperp+1e-10)
        angles = (numpy.arctan(kangle)*180/numpy.pi)
        mask2[angles < angle] = 0
    fgmask = mask*mask2
    return fgmask



recfolder = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/'
nc = 512
kk = None

ff = open('../../data/lostmodes-N%d.txt'%nc, 'w')
ff.write('#%4s    %5s   %5s   %6s    %6s  %6s\n'%('a', 'wedge', 'noise', 'angle', 'lost-w', 'lost-n'))

for aa in [0.1429, 0.2000, 0.3333]:
#for aa in [0.1429]:
    zz = 1/aa-1
    for wed in ['opt', 'pess']:
        for th in ['opt', 'pess', 'reas']:

            optfolder = recfolder + '/fastpm_%0.4f/wedge_kmin0.03_%s/L1024-N0256-R/thermal-%s-hex/ZA/opt_s999_h1massA_fourier_rsdpos/'%(aa, wed,th)
            if nc == 256:
                dd = BigFileMesh(optfolder + 'datap/', 'mapp').paint()
                dn = BigFileMesh(optfolder + 'datan/', 'mapp').paint()
            else:
                dd = BigFileMesh(optfolder + 'datap_up/', 'mapp').paint()
                dn = BigFileMesh(optfolder + 'datan_up/', 'mapp').paint()
            dn -= dd
            if kk is None: kk = dd.r2c().x
            pd2 = FFTPower(dd, mode='2d', Nmu=Nmu).power
            k2, modes2, pd2 = pd2['k'], pd2['modes'], pd2['power']
            pn2 = FFTPower(dn, mode='2d', Nmu=Nmu).power['power']

            #Lost due to wedge
            wedang = mapnoise.wedge(1/aa-1,att=wed,angle=True)
            wedmu = mapnoise.wedge(1/aa-1,att=wed,angle=False)
            mask = getmask(kk, wedang, kmin=0.03)
            lostw = 1-mask.sum()/mask.size

            #lost due to noise:
            noisedom = 0
            k0 = 10

            for ii in range(Nmu-1):
                if mus[ii] < wedmu: pass
                else: noisedom += modes2[k0:][pn2[k0:, ii] > 2*pd2[k0:, ii], ii].sum()
            lostn = noisedom/modes2.sum()

            print(aa, wed, th, wedang, lostw, lostn)
            ff.write('%0.4f    %4s    %4s    %02.2f    %02.2f    %02.2f\n'%(aa, wed, th, wedang, lostw, lostn))

ff.close()
                     
