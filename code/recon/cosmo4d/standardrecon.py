import numpy
import numpy as np
from nbodykit.algorithms import FFTPower
from cosmo4d.pmeshengine import nyquist_mask
import sys


def decic(pm, n=2):
    def tf(k):
        kny = [np.sinc(k[i]*pm.BoxSize[i]/(2*np.pi*pm.Nmesh[i])) for i in range(3)]
        wts = (kny[0]*kny[1]*kny[2])**(-1*n)
        return wts

    if pm.dtype == 'complex128' or pm.dtype == 'complex64':
        toret = pm.apply(lambda k, v: tf(k)*v).c2r()
    elif pm.dtype == 'float32' or pm.dtype == 'float64':
        toret = pm.r2c().apply(lambda k, v: tf(k)*v).c2r()
    return toret


def gauss(pm, R):
    def tf(k):
        k2 = 0
        for ki in k:
            k2 =  k2 + ki ** 2
        wts = np.exp(-0.5*k2*(R**2))
        return wts

    if pm.dtype == 'complex128' or pm.dtype == 'complex64':
        toret = pm.apply(lambda k, v: tf(k)*v).c2r()
    elif pm.dtype == 'float32' or pm.dtype == 'float64':
        toret = pm.r2c().apply(lambda k, v: tf(k)*v).c2r()
    return toret



##
##def calc_disp(i, mesh, b):
##    dk = mesh.r2c()
##    k2 = 0
##    for ki in dk.x: k2 =  k2 + ki ** 2
##    k2[0, 0, 0] = 1
##    sk = -(0+1j)*dk*dk.x[i]/k2
##    sr = sk.c2r()
##    return sr/b
##

def displace(pm, displist, pos, rsd=False, f=None, beta=None, mass=None):
    dispxmesh, dispymesh, dispzmesh = displist
    dispx = dispxmesh.readout(pos)
    dispy = dispymesh.readout(pos)
    dispz = dispzmesh.readout(pos)
    if rsd:
        dispz = dispz + (f -  beta)/(1 + beta)*dispz
    disp = np.array([dispx, dispy, dispz]).T
    layout = pm.decompose(pos+disp)
    if mass is not None: shiftmesh = pm.paint(pos + disp, mass=mass, layout=layout)
    else: shiftmesh = pm.paint(pos + disp, layout=layout)
    if abs(shiftmesh.cmean()) > 1e-3:
        shiftmesh /= shiftmesh.cmean()
        shiftmesh -= 1
    return shiftmesh


def apply_wedge(pm, kmin, angle):
    def tf(k):
        kmesh = sum(ki ** 2 for ki in k)**0.5
        mask = [numpy.ones_like(ki) for ki in k]                
        if kmin > 0:
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

    if pm.dtype == 'complex128' or pm.dtype == 'complex64':
        toret = pm.apply(lambda k, v: tf(k)*v).c2r()
    elif pm.dtype == 'float32' or pm.dtype == 'float64':
        toret = pm.r2c().apply(lambda k, v: tf(k)*v).c2r()
    return toret


def calc_displist(base, b=1, kmin=0, angle=0):   

    if kmin !=0 or angle !=0:
        base = apply_wedge(base, kmin, angle)

    dk = base.r2c()
    k2 = 0
    for ki in dk.x: k2 =  k2 + ki ** 2
    k2[k2==0] = 1

    dispmesh = []
    for i in range(3):
        sk = -(0+1j)*dk*dk.x[i]/k2
        sr = sk.c2r()
        dispmesh.append(sr/b)
        
    return dispmesh

   

    

##
##def standard(pm, fofcat, datap, mf, kb=6, Rsm = 7, rsd = False, zz=0, M= 0.3175, mass=False, poskey='PeakPosition'):
##
##    if rsd:
##        if pm.comm.rank == 0: print('\n RSD! Key used to get position is --- RSPeakPosition \n\n')            
##        position = fofcat['RS%s'%poskey].compute()
##    else:
##        position = fofcat['%s'%poskey].compute()
##    try: hmass =  fofcat['AMass'].compute()*1e10
##    except: hmass = fofcat['Mass'].compute()*1e10
##
##    pks = FFTPower(datap.s, mode='1d').power['power']
##    pkf = FFTPower(datap.d, mode='1d').power['power']
##    
##    random = pm.generate_uniform_particle_grid()
##    # random = np.random.uniform(0, 400, 3*128**3).reshape(-1, 3)
##    Rbao = Rsm/2**0.5
##    aa = mf.cosmo.ztoa(zz)
##    ff = mf.cosmo.Fomega1(mf.cosmo.ztoa(zz))
##
##    
##    layout = pm.decompose(position)
##    if mass: hmesh = pm.paint(position, mass = hmass, layout=layout)
##    else: hmesh = pm.paint(position, layout=layout)
##    hmesh /= hmesh.cmean()
##    hmesh -= 1
##    hmeshsm =  ft.smooth(hmesh, Rbao, 'gauss')
##
##    #bias
##    layout = pm.decompose(fofcat['%s'%poskey])
##    hrealp = pm.paint(fofcat['%s'%poskey], layout=layout)
##    hrealp /= hrealp.cmean()
##    hrealp -= 1
##
##    pkhp = FFTPower(hrealp, mode='1d').power['power']
##    bias = ((pkhp[1:kb]/pkf[1:kb]).mean()**0.5).real
##    beta = bias/ff
##    print('bias = ', bias)
##
##    displist = calc_displist(pm=pm, base=hmeshsm, b=bias)
##
##    if mass: hpshift = displace(pm, displist, position, rsd=rsd, f=ff, beta=beta, mass = hmass)
##    else: hpshift = displace(pm, displist, position, rsd=rsd, f=ff, beta=beta, mass = None)
##
##    rshift = displace(pm, displist, random, rsd=rsd, f=ff, beta=beta)
##    recon = hpshift - rshift
##    
##    return recon, hpshift, rshift
##


##$
##$def dostd(hdict, numd, pkf, datas, kb=6, Rsm = 7, rsd = False, zz=0, M= 0.3175, mass=False, retfield=False, 
##$          retpower=False, mode='1d', Nmu=5, los=[0, 0, 1], retall=False):
##$    #propogator is divided by bias
##$
##$    hpos = hdict['position']
##$    hmass = hdict['mass']
##$    pks = FFTPower(datas, mode=mode, Nmu=Nmu, los=[0, 0, 1]).power['power']
##$    
##$    aa = mf.cosmo.ztoa(zz)
##$    ff = mf.cosmo.Fomega1(mf.cosmo.ztoa(zz))
##$    rsdfac = 100/(aa**2 * mf.cosmo.Ha(z=zz)**1)
##$    print('rsdfac = ', rsdfac)
##$    hposrsd = hdict['position'] + np.array(los)*hdict['velocity']*rsdfac
##$    
##$    layout = pm.decompose(hpos[:int(numd*bs**3)])
##$    if mass: hpmesh = pm.paint(hpos[:int(numd*bs**3)], mass = hmass[:int(numd*bs**3)], layout=layout)
##$    else: hpmesh = pm.paint(hpos[:int(numd*bs**3)], layout=layout)
##$    hpmesh /= hpmesh.cmean()
##$    hpmesh -= 1
##$
##$    layout = pm.decompose(hposrsd[:int(numd*bs**3)])
##$    if mass: hpmeshrsd = pm.paint(hposrsd[:int(numd*bs**3)], mass = hmass[:int(numd*bs**3)], layout=layout)
##$    else: hpmeshrsd = pm.paint(hposrsd[:int(numd*bs**3)], layout=layout)
##$    hpmeshrsd /= hpmeshrsd.cmean()
##$    hpmeshrsd -= 1
##$
##$    pkhp = FFTPower(hpmesh, mode='1d').power['power']
##$    bias = ((pkhp[1:kb]/pkf[1:kb]).mean()**0.5).real
##$    beta = bias/ff
##$    print('bias = ', bias)
##$
##$    random = pm.generate_uniform_particle_grid()
##$    random = np.random.uniform(0, 400, 3*128**3).reshape(-1, 3)
##$    Rbao = Rsm/2**0.5
##$
##$    if not rsd:
##$        hpmeshsm = ft.smooth(hpmesh, Rbao, 'gauss')
##$        displist = calc_displist(hpmeshsm, b=bias)
##$        if mass: 
##$            hpshift = displace(pm, displist, hpos[:int(numd*bs**3)], mass = hmass[:int(numd*bs**3)])
##$        else: 
##$            hpshift = displace(pm, displist, hpos[:int(numd*bs**3)])
##$        rshift = displace(pm, displist, random)
##$        recon = hpshift - rshift
##$        pksstd = FFTPower(recon, mode=mode, Nmu=Nmu, los=[0, 0, 1]).power['power']
##$        pkxsstd = FFTPower(recon, second=datas, mode=mode, Nmu=Nmu, los=[0, 0, 1]).power['power']
##$        rccstd = pkxsstd / (pks*pksstd)**0.5
##$        cksstd = pkxsstd / pks /bias
##$        
##$    
##$    if rsd:
##$        RSD
##$        hpmeshrsdsm = ft.smooth(hpmeshrsd, Rbao, 'gauss')
##$        displist = calc_displist(hpmeshrsdsm, b=bias)
##$        hpshift = displace(pm, displist, hposrsd[:int(numd*bs**3)], rsd=True, f=ff, beta=bias/ff)
##$        if mass: hpshift = displace(pm, displist, hposrsd[:int(numd*bs**3)], rsd=True, f=ff, beta=bias/ff, 
##$                                    mass = hmass[:int(numd*bs**3)])
##$        else: hpshift = displace(pm, displist, hposrsd[:int(numd*bs**3)], rsd=True, f=ff, beta=bias/ff)
##$        rshift = displace(pm, displist, random, rsd=True, f=ff, beta=beta)
##$        recon = hpshift - rshift
##$        pksstd = FFTPower(recon, mode='1d').power['power']
##$        pkxsstd = FFTPower(recon, second=datap[tkey].s, mode='1d').power['power']
##$        pksstd = FFTPower(recon, mode=mode, Nmu=Nmu, los=[0, 0, 1]).power['power']
##$        pkxsstd = FFTPower(recon, second=datas, mode=mode, Nmu=Nmu, los=[0, 0, 1]).power['power']
##$        rccstd = pkxsstd / (pks*pksstd)**0.5
##$        cksstd = pkxsstd / pks / bias
##$        
##$    if retall: return [rccstd, cksstd], [recon, hpshift, rshift, displist],  [pkxsstd, pksstd, pks], bias
##$    if retfield: return [rccstd, cksstd], [recon, hpshift, rshift, displist]
##$    elif retpower: return [rccstd, cksstd], [pkxsstd, pksstd, pks], bias
##$    else: return rccstd, cksstd
##$
##$    
##$    
##$
