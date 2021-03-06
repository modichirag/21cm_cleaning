import numpy
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from pmesh.pm import ParticleMesh
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from nbodykit.lab import BigFileMesh, BigFileCatalog, FFTPower
from nbodykit.cosmology import Planck15, EHPower, Cosmology

import sys
sys.path.append('../utils/')
import tools, za
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)


#########################################
def diracdelta(i, j):
    if i == j: return 1
    else: return 0

#def shear(pm, base):          
#    '''Takes in a PMesh object in real space. Returns am array of shear'''          
#    s2 = pm.create(mode='real', value=0)                                                  
#    kk = base.r2c().x
#    k2 = sum(ki**2 for ki in kk)                                                                          
#    k2[0,0,0] = 1  
#    for i in range(3):
#        for j in range(i, 3):                                                       
#            basec = base.r2c()
#            basec *= (kk[i]*kk[j] / k2 - diracdelta(i, j)/3.)              
#            baser = basec.c2r()                                                                
#            s2[...] += baser**2                                                        
#            if i != j:                                                              
#                s2[...] += baser**2                                                    
#    return s2  
#

def shear(pm, base):                                                                                                                                          
    '''Takes in a PMesh object in real space. Returns am array of shear'''          
    s2 = pm.create(mode='real', value=0)                                                  
    kk = base.r2c().x
    k2 = sum(ki**2 for ki in kk)                                                                          
    k2[0,0,0] =  1                                                                  
    for i in range(3):
        for j in range(i, 3):                                                       
            basec = base.r2c()
            basec *= (kk[i]*kk[j] / k2 - diracdelta(i, j)/3.)              
            baser = basec.c2r()                                                                
            s2[...] += baser**2                                                        
            if i != j:                                                              
                s2[...] += baser**2                                                    
                                                                                    
    return s2  



def getbias(pm, hmesh, basemesh, pos, grid, doed=False, fpos=None, ik=20, fitshear=True, fitb2=True, retps=False):

    if pm.comm.rank == 0: print('Will fit for bias now')

    try: d0, d2, s2 = basemesh
    except:
        d0 = basemesh.copy()
        d2 = 1.*basemesh**2
        d2 -= d2.cmean()
        s2 = shear(pm, basemesh)
        s2 -= 1.*basemesh**2
        s2 -= s2.cmean()


    ph = FFTPower(hmesh, mode='1d').power
    k, ph = ph['k'], ph['power']

    glay, play = pm.decompose(grid), pm.decompose(pos)
    ed0 = pm.paint(pos, mass=d0.readout(grid, layout = glay, resampler='nearest'), layout=play)
    ed2 = pm.paint(pos, mass=d2.readout(grid, layout = glay, resampler='nearest'), layout=play)
    es2 = pm.paint(pos, mass=s2.readout(grid, layout = glay, resampler='nearest'), layout=play)

    ped0 = FFTPower(ed0, mode='1d').power['power']
    ped2 = FFTPower(ed2, mode='1d').power['power']
    pes2 = FFTPower(es2, mode='1d').power['power']


    pxed0d2 = FFTPower(ed0, second=ed2, mode='1d').power['power']
    pxed0s2 = FFTPower(ed0, second=es2, mode='1d').power['power']
    pxed2s2 = FFTPower(ed2, second=es2, mode='1d').power['power']

    pxhed0 = FFTPower(hmesh, second=ed0, mode='1d').power['power']
    pxhed2 = FFTPower(hmesh, second=ed2, mode='1d').power['power']
    pxhes2 = FFTPower(hmesh, second=es2, mode='1d').power['power']

    if doed:
        ed = pm.paint(pos, mass=ones.readout(grid, resampler='nearest'))
        ped = FFTPower(ed, mode='1d').power['power']
        pxhed = FFTPower(hmesh, second=ed, mode='1d').power['power']
        pxedd0 = FFTPower(ed, second=ed0, mode='1d').power['power']
        pxedd2 = FFTPower(ed, second=ed2, mode='1d').power['power']
        pxeds2 = FFTPower(ed, second=es2, mode='1d').power['power']

    def ftomin(bb, ii=ik, retp = False):
        b1, b2, bs = bb
        if not fitb2:
            b2 = 0
            bs = 0
        if not fitshear: bs = 0
        pred = b1**2 *ped0 + b2**2*ped2 + 2*b1*b2*pxed0d2 
        pred += bs**2 *pes2 + 2*b1*bs*pxed0s2 + 2*b2*bs*pxed2s2
        if doed: pred += ped + 2*b1*pxedd0 + 2*b2*pxedd2 + 2*bs*pxeds2 

        predx = 1*b1*pxhed0 + 1*b2*pxhed2
        predx += 1*bs*pxhes2
        if doed: predx += 1*pxhed

        if retp : return pred, predx
        chisq = (((ph + pred - 2*predx)[1:ii])**2).sum()**0.5.real
        return chisq.real

    if pm.comm.rank == 0: print('Minimize\n')

#     b1, b2, bs2 = minimize(ftomin, [1, 1, 1], method='Nelder-Mead', options={'maxfev':10000}).x
    params =  minimize(ftomin, [1, 0, 0]).x

    b1, b2, bs2 = params

    if pm.comm.rank == 0: print('\nBias fit params are : ', b1, b2, bs2)
    
    if fpos is not None:
        glay, play = pm.decompose(grid), pm.decompose(fpos)
        ed0 = pm.paint(fpos, mass=d0.readout(grid, layout = glay, resampler='nearest'), layout=play)
        ed2 = pm.paint(fpos, mass=d2.readout(grid, layout = glay, resampler='nearest'), layout=play)
        es2 = pm.paint(fpos, mass=s2.readout(grid, layout = glay, resampler='nearest'), layout=play)
        mod = b1*ed0 + b2*ed2 + bs2*es2
    else:
        mod = b1*ed0 + b2*ed2 + bs2*es2
    if doed: mod += ed
    
    if retps:
        pmod = FFTPower(mod, mode='1d').power['power']
        ps = [k, ph, pmod, ped0, ped2, pes2, pxed0d2, pxed0s2, pxed2s2]
        return params, mod, ps
    
    else: return params, mod





def eval_bfit(hmesh, mod, ofolder, noise=None, title=None, fsize=15, suff=None):

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    pmod = FFTPower(mod, mode='1d').power
    k, pmod = pmod['k'], pmod['power']
    ph = FFTPower(hmesh, mode='1d').power['power']
    pxmodh = FFTPower(hmesh, second=mod, mode='1d').power['power']
    perr = FFTPower(hmesh -mod, mode='1d').power['power']

    ax[0].plot(k, pxmodh/(pmod*ph)**0.5)
    ax[0].set_ylabel('$r_{cc}$', fontsize=fsize)
        
    ax[1].plot(k,(pmod/ph)**0.5)
    ax[1].set_ylabel('$\sqrt{P_{mod}/P_{hh}}$', fontsize=fsize)
    
    ax[2].plot(k, perr)
    ax[2].set_yscale('log')
    ax[2].set_ylabel('$P_{\delta{mod}-\delta_h}$', fontsize=fsize)
    if noise is not None: ax[2].axhline(noise)

    if hmesh.pm.comm.rank == 0:
        for axis in ax:
            axis.set_xscale('log')
            axis.grid(which='both')
            axis.set_xlabel('$k$ (h/Mpc)', fontsize=fsize)
            axis.legend(fontsize=fsize)
            
        if title is not None: plt.suptitle(title, fontsize=fsize)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fname = ofolder + 'evalbfit'
        if suff is not None: fname = fname + '%s'%suff
        print(fname)
        fig.savefig(fname+'.png')

    plt.close()

    return k[1:], perr.real[1:]





def getbiask(pm, hmesh, basemesh, pos, grid, fpos=None):

    bs = pm.BoxSize[0]
    nc = pm.Nmesh[0]
    print(bs, nc)
    if pm.comm.rank == 0: print('Will fit for bias now')

    try: d0, d2, s2 = basemesh
    except:
        d0 = basemesh.copy()
        d2 = 1.*basemesh**2
        d2 -= d2.cmean()
        s2 = shear(pm, basemesh)
        s2 -= 1.*basemesh**2
        s2 -= s2.cmean()

    glay, play = pm.decompose(grid), pm.decompose(pos)
    ed0 = pm.paint(pos, mass=d0.readout(grid, layout = glay, resampler='nearest'), layout=play)
    ed2 = pm.paint(pos, mass=d2.readout(grid, layout = glay, resampler='nearest'), layout=play)
    es2 = pm.paint(pos, mass=s2.readout(grid, layout = glay, resampler='nearest'), layout=play)


    dk = 2.0*numpy.pi/bs
    kmin = 2.0*numpy.pi/bs / 2.0
    kmax = 1.5*nc*numpy.pi/bs
#     dk, kmin = None, 0

    ph = FFTPower(hmesh, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power
    k, ph = ph['k'], ph['power']
    kedges = numpy.arange(k[0]-dk/2., k[-1]+dk/2., dk)
    
    #ed = pm.paint(pos, mass=ones.readout(grid, resampler='nearest'))
    ed0 = pm.paint(pos, mass=d0.readout(grid, resampler='nearest'))
    ed2 = pm.paint(pos, mass=d2.readout(grid, resampler='nearest'))
    es2 = pm.paint(pos, mass=s2.readout(grid, resampler='nearest'))

    #ped = FFTPower(ed, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']
    ped0 = FFTPower(ed0, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']
    ped2 = FFTPower(ed2, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']
    pes2 = FFTPower(es2, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']

    #pxedd0 = FFTPower(ed, second=ed0, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']
    #pxedd2 = FFTPower(ed, second=ed2, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']
    #pxeds2 = FFTPower(ed, second=es2, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']

    pxed0d2 = FFTPower(ed0, second=ed2, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']
    pxed0s2 = FFTPower(ed0, second=es2, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']
    pxed2s2 = FFTPower(ed2, second=es2, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']

    #pxhed = FFTPower(hmesh, second=ed, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']
    pxhed0 = FFTPower(hmesh, second=ed0, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']
    pxhed2 = FFTPower(hmesh, second=ed2, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']
    pxhes2 = FFTPower(hmesh, second=es2, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']




    def ftomink(bb, ii, retp = False):
        b1, b2, bs = bb
        pred = b1**2 *ped0 + b2**2*ped2 + 2*b1*b2*pxed0d2 
        pred += bs**2 *pes2 + 2*b1*bs*pxed0s2 + 2*b2*bs*pxed2s2

        predx = 1*b1*pxhed0 + 1*b2*pxhed2
        predx += 1*bs*pxhes2

        if retp : return pred, predx
        chisq = (((ph + pred - 2*predx)[ii])**2).real
        return chisq

    if pm.comm.rank == 0: print('Minimize\n')

    b1k, b2k, bsk = numpy.zeros_like(k), numpy.zeros_like(k), numpy.zeros_like(k)
    for ii in range(k.size):
        tfunc = lambda p: ftomink(p,ii)
        b1k[ii], b2k[ii], bsk[ii] = minimize(tfunc, [1, 1, 1]).x

    paramsk = [b1k, b2k, bsk]

    def transfer(mesh, tk):
        meshc = mesh.r2c()
        kk = meshc.x
        kmesh = sum([i ** 2 for i in kk])**0.5
#         _, kedges = numpy.histogram(kmesh.flatten(), nc)
        kind = numpy.digitize(kmesh, kedges, right=False)
        toret = mesh.pm.create(mode='complex', value=0)

        for i in range(kedges.size):
            mask = kind == i
            toret[mask] = meshc[mask]*tk[i]
        return toret.c2r()

    
    if fpos is not None:
        glay, play = pm.decompose(grid), pm.decompose(fpos)
        ed0 = pm.paint(fpos, mass=d0.readout(grid, layout = glay, resampler='nearest'), layout=play)
        ed2 = pm.paint(fpos, mass=d2.readout(grid, layout = glay, resampler='nearest'), layout=play)
        es2 = pm.paint(fpos, mass=s2.readout(grid, layout = glay, resampler='nearest'), layout=play)
        mod = transfer(ed0, b1k) + transfer(ed2, b2k) + transfer(es2, bsk)        
    else:
        mod = transfer(ed0, b1k) + transfer(ed2, b2k) + transfer(es2, bsk)
    
    return k, paramsk, mod

         


if __name__=="__main__":

    #bs, nc = 256, 128
    bs, nc = 1024, 256
    model = 'ModelA'
    ik = 50
    ii = 50
    ffile = '../../data/bparams-L%04d-N%04d-%s.txt'%(bs, nc, model)
    header = 'b1, b2, bg, b0, bk \nFit bias upto 0.3\nFit tf upto 1.0'
    pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
    rank = pm.comm.rank
    grid = pm.mesh_coordinates()*bs/nc
    
    lin = BigFileMesh('/global/cscratch1/sd/chmodi/m3127/cm_lowres/5stepT-B1/%d-%d-9100/linear'%(bs, nc), 'LinearDensityK').paint()

    tosave = []

    for aa in [0.1429, 0.2000, 0.3333]:
        zz = 1/aa-1
        print(aa)
        dyn = BigFileCatalog('/global/cscratch1/sd/chmodi/m3127/cm_lowres/5stepT-B1/%d-%d-9100/fastpm_%0.4f/1'%(bs, nc, aa))
        hmesh = BigFileMesh('/global/cscratch1/sd/chmodi/m3127/H1mass/highres/%d-9100/fastpm_%0.4f/HImesh-N%04d/'%(bs*10, aa, nc), model).paint()
        fpos = dyn['Position']
        dgrow = cosmo.scale_independent_growth_factor(zz)
        zapos = za.doza(lin.r2c(), grid, z=zz, dgrow=dgrow)

        ph = FFTPower(hmesh, mode='1d').power
        k, ph = ph['k'],  ph['power']
        ik = numpy.where(k>0.3)[0][0]
        
        paramsza, zamod = getbias(pm, basemesh=lin, hmesh=hmesh, pos=zapos, grid=grid, ik=ik)
        pmod = FFTPower(zamod, mode='1d').power['power']
        tf = (pmod/ph)**0.5
        if rank == 0: print(tf)
        if rank == 0: print(paramsza)
        
        plt.figure()
        for i, kk in enumerate([0.4, 0.5, 0.6, 0.7, min(1, k.max()-0.01)]):
            ii = numpy.where(k > kk)[0][0]
            if rank == 0: print(ii)
            def ftomin(bb, ii=ii):
                b0, b1, b2 = bb
                b1 = 0 
                pred = b0 + b1*k*1 + b2*k**2
                chisq = (((tf-pred)[1:ii])**2).sum()**0.5.real
                return chisq.real

            params =  minimize(ftomin, [1, 0, 0]).x
            if rank == 0: print(kk, params)
            plt.plot(k, (params[0]+params[1]*k**2 )/tf, 'C%d--'%i, label=kk)

        tosave.append([aa, *paramsza, *params])
        for i, kk in enumerate([0.4, 0.5, 0.6, 0.7, min(1, k.max()-0.01)]):
            ii = numpy.where(k > kk)[0][0]
            if rank == 0: print(ii)
            def ftomin(bb, ii=ii):
                b0, b1, b2 = bb
                b0 = 1
                b1 = 0
                pred = b0 + b1*k*1 + b2*k**2
                chisq = (((tf-pred)[1:ii])**2).sum()**0.5.real
                return chisq.real

            params =  minimize(ftomin, [1, 0, 0]).x
            if rank == 0: print(kk, params)
            plt.plot(k, (params[0]+params[1]*k**2 )/tf, 'C%d'%i, label=kk)
        
        plt.legend()
        plt.ylim(0.8, 1.2)
        plt.semilogx()
        plt.grid(which='both')
        if rank == 0: plt.savefig('z%0.2f.png'%zz)
        
        if rank == 0: numpy.savetxt(ffile, numpy.array(tosave), header=header, fmt='%0.4f')
