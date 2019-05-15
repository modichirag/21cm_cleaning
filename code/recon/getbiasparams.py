import numpy
from scipy.optimize import minimize
from nbodykit.lab import FFTPower

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import warnings


#########################################
def diracdelta(i, j):
    if i == j: return 1
    else: return 0

def shear(pm, base):          
    '''Takes in a PMesh object in real space. Returns am array of shear'''          
    s2 = pm.create(mode='real', value=0)                                                  
    kk = base.r2c().x
    k2 = sum(ki**2 for ki in kk)                                                                          
    print(pm.comm.rank, k2.shape)
    k2[0,0,0] = 1  
    for i in range(3):
        for j in range(i, 3):                                                       
            basec = base.r2c()
            basec *= (kk[i]*kk[j] / k2 - diracdelta(i, j)/3.)              
            baser = basec.c2r()                                                                
            s2[...] += baser**2                                                        
            if i != j:                                                              
                s2[...] += baser**2                                                    
    return s2  


def getbias(pm, hmesh, basemesh, pos, grid, doed=False):

    if pm.comm.rank == 0: print('Will fit for bias now')

#    d0 = basemesh.copy()
#    d2 = 1.*basemesh**2
#    d2 -= d2.cmean()
#    s2 = shear(pm, basemesh)
#    s2 -= 1.*basemesh**2
#    s2 -= s2.cmean()
    d0, d2, s2 = basemesh

    ph = FFTPower(hmesh, mode='1d').power['power']

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

    def ftomin(bb, ii=20, retp = False):
        b1, b2, bs = bb
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
    
    mod = b1*ed0 + b2*ed2 + bs2*es2
    if doed: mod += ed
    
    return params, mod




def eval_bfit(hmesh, mod, ofolder, noise=None, title=None, fsize=15):

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
        fig.savefig(ofolder + 'evalbfit.png')

    plt.close()

    return k, perr
