
import numpy
import re, json, warnings, os
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from nbodykit.lab import FieldMesh
from nbodykit.cosmology import Planck15, EHPower, Cosmology
#
from cosmo4d import base
from cosmo4d.engine import Literal
from cosmo4d.pmeshengine import nyquist_mask
from cosmo4d.iotools import save_map, load_map
from cosmo4d.mapbias import Observable
#

package_path = os.path.dirname(os.path.abspath(__file__))+'/'
klin, plin = numpy.loadtxt(package_path + '../../../data/pklin_1.0000.txt', unpack = True)
ipk = interpolate(klin, plin)
#cosmo = Planck15.clone(Omega_cdm = 0.2685, h = 0.6711, Omega_b = 0.049)
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)


def volz(z, dz=0.1, sky=20000):
    chiz = cosmo.comoving_distance(z)
    dchiz = cosmo.comoving_distance(z+dz)-cosmo.comoving_distance(z-dz)
    fsky = sky/41252;
    vol = 4*numpy.pi*chiz**2 * dchiz * fsky
    return vol

def wedge(z, D=6 *0.7**0.5, att='opt', angle=False):
    chiz = cosmo.comoving_distance(z)
    hz = cosmo.efunc(z)*100
    R = chiz*hz/(1+z)/2.99e5
    if att == 'opt': thetafov = 1.22 * 0.211 * (1+z)/D/2
    elif att == 'pess': thetafov = 1.22 * 0.211 * (1+z)/D/2 *3
    elif att == 'nope': thetafov = numpy.pi/2.
    X = numpy.sin(thetafov) * R
    mu = X/(1+X**2)**0.5
    if angle: return 90-numpy.arccos(mu)*180/numpy.pi
    return mu

def visibility(stage2=True):
    fsky21 = 20000/41252;
    Ns = 256
    if not stage2: Ns = 32
    Ds = 6
    n0 = (Ns/Ds)**2
    Ls = Ds* Ns
    def vis(k, mu, z):
        chiz = lambda z : cosmo.comoving_distance(z)
        u =  k *numpy.sqrt(1 - mu**2)* chiz(z) /(2* numpy.pi)#* wavez(z)**2
        D = u
        DLs = D/Ls
        #n =  n0 *(0.4847 - 0.33 *(D/Ls))/(1 + 1.3157 *(D/Ls)**1.5974) * numpy.exp(-(D/Ls)**6.8390)
        n =  n0 *(0.4847 - 0.33 *(DLs))/(1 + 1.3157 *(DLs)**1.5974) * numpy.exp(-(DLs)**6.8390)
        return n
    return vis


def thermalnoise(z, stage2=True, mK=False):
    fsky21 = 20000/41252;
    Ns = 256
    if not stage2: Ns = 32
    Ds = 6
    n0 = (Ns/Ds)**2
    Ls = Ds* Ns
    npol = 2
    S21 = 4 *numpy.pi* fsky21
    t0 = 5*365*24*60*60
    Aeff = numpy.pi* (Ds/2)**2 *0.7 #effective
    nu0 = 1420*1e6
    wavez = lambda z: 0.211 *(1 + z)
    chiz = lambda z : cosmo.comoving_distance(z)
    #defintions
    n = lambda D: n0 *(0.4847 - 0.33 *(D/Ls))/(1 + 1.3157 *(D/Ls)**1.5974) * numpy.exp(-(D/Ls)**6.8390) +1e-7
    Tb = lambda z: 180/(cosmo.efunc(z)) *(4 *10**-4 *(1 + z)**0.6) *(1 + z)**2*cosmodef['h']
    FOV= lambda z: (1.22* wavez(z)/Ds)**2; #why is Ds here
    Ts = lambda z: (55 + 30 + 2.7 + 25 *(1420/400/(1 + z))**-2.75) * 1000;
    u = lambda k, mu, z: k *numpy.sqrt(1 - mu**2)* chiz(z) /(2* numpy.pi)#* wavez(z)**2
    #terms
    d2V = lambda z: chiz(z)**2* 3* 10**5 *(1 + z)**2 /cosmo.efunc(z)/100 
    fac = lambda z: Ts(z)**2 * S21 / Aeff **2 * (wavez(z))**4 /FOV(z) 
    # fac = lambda z: Ts(z)**2 * S21 /Aeff **2 * ((1.22 * wavez(z))**2)**2 #/FOV(z)/1.22**4
    cfac = 1 /t0/ nu0 / npol
    #
    setupz = cfac *fac(z) *d2V(z) / wavez(z)**2
    if  not mK: setupz /= Tb(z)**2
    #if mK:  Pn = lambda k, mu, z: cfac *fac(z) *d2V(z) / (n(u(k, mu, z)) * wavez(z)**2)
    #else: Pn =  lambda k, mu, z: cfac *fac(z) *d2V(z) / (n(u(k, mu, z)) * wavez(z)**2) / Tb(z)**2
    Pn = lambda k, mu, z: setupz / n(u(k, mu, z))
    return Pn


class NoiseModel(base.NoiseModel):
    def __init__(self, pm, mask2d, power, seed):
        self.pm = pm
        self.pm2d = self.pm.resize([self.pm.Nmesh[0], self.pm.Nmesh[1], 1])
        if mask2d is None:
            mask2d = self.pm2d.create(mode='real')
            mask2d[...] = 1.0

        self.mask2d = mask2d
        self.power = power
        self.seed = seed
        self.var= power / (self.pm.BoxSize / self.pm.Nmesh).prod()
        self.ivar2d = mask2d * self.var ** -1

    def downsample(self, pm):
        d = NoiseModel(pm, None, self.power, self.seed)
        d.mask2d = d.pm2d.downsample(self.mask2d)
        return d

    def add_noise(self, obs):
        pm = obs.mapp.pm

        if self.seed is None:
            n = pm.create(mode='real')
            n[...] = 0
        else:
            n = pm.generate_whitenoise(mode='complex', seed=self.seed)
            n = n.apply(lambda k, v : (self.power / pm.BoxSize.prod()) ** 0.5 * v, out=Ellipsis).c2r(out=Ellipsis)
            if pm.comm.rank == 0: print('Noise Variance check', (n ** 2).csum() / n.Nmesh.prod(), self.var)
        return Observable(mapp=obs.mapp + n, s=obs.s, d=obs.d)



class ThermalNoise(base.NoiseModel):
    def __init__(self, pm, aa, seed=100, stage2=True):
        self.pm = pm
        self.aa = aa
        self.zz = 1/aa-1
        self.seed = seed
        self.noise = thermalnoise(self.zz, stage2=stage2)
        #self.var= power / (self.pm.BoxSize / self.pm.Nmesh).prod()
        #self.ivar2d = mask2d * self.var ** -1


    def get_ivarmesh(self, obs, ipk=None):
        pm = obs.mapp.pm
        kk = obs.mapp.r2c().x
        kmesh = sum(i**2 for i in kk)**0.5
        kmesh[kmesh == 0] = 1
        mumesh = kk[2]/kmesh
        kperp = (kk[0]**2 + kk[1]**2)**0.5
        
        #print('eval noise ps', kperp.shape, mumesh.shape)
        noiseth = self.noise(kperp, mumesh, self.zz)
        if ipk is not None: noise = ipk(kmesh)
        else: noise = noiseth * 0
        toret = ((noiseth + noise)/ pm.BoxSize.prod()) ** -1.
        return pm.create(mode='complex', value=toret).c2r()
        
    def add_noise(self, obs):
        
        pm = obs.mapp.pm
        kk = obs.mapp.r2c().x
        kmesh = sum(i**2 for i in kk)**0.5
        kmesh[kmesh == 0] = 1
        mumesh = kk[2]/kmesh
        kperp = (kk[0]**2 + kk[1]**2)**0.5
        
        #print('eval noise ps', kperp.shape, mumesh.shape)
        noisep = self.noise(kperp, mumesh, self.zz)
        #print(numpy.isnan(noisep).sum())

        if self.seed is None:
            n = pm.create(mode='real')
            n[...] = 0
        else:
            n = pm.generate_whitenoise(mode='complex', seed=self.seed)
            n = (n * (noisep / pm.BoxSize.prod()) ** 0.5 ).c2r(out=Ellipsis)
            #if pm.comm.rank == 0: print('Noise Variance check', (n ** 2).csum() / n.Nmesh.prod(), self.var)
        return Observable(mapp=obs.mapp + n, s=obs.s, d=obs.d)


class WedgeNoiseModel(base.NoiseModel):
    def __init__(self, pm, power, seed, kmin, angle):
        self.pm = pm
        self.kmin = kmin
        self.angle = angle
        self.power = power
        self.seed = seed
        self.var= power / (self.pm.BoxSize / self.pm.Nmesh).prod()


    def add_noise(self, obs):
        pm = obs.mapp.pm
        field = obs.mapp

        def tf(k):
            kmesh = sum(ki ** 2 for ki in k)**0.5
            mask = [numpy.ones_like(ki) for ki in k]                
            mask[2] *= abs(k[2]) >= self.kmin
            mask = numpy.prod(mask)

            mask2 = numpy.ones_like(mask)
            if self.angle > 0:
                kperp = (k[0]**2 + k[1]**2)**0.5
                kangle = abs(k[2])/(kperp+1e-10)
                angles = (numpy.arctan(kangle)*180/numpy.pi)
                mask2[angles < self.angle] = 0
            fgmask = mask*mask2
            return fgmask


        fieldc = field.r2c()
        fieldc.apply(lambda k, v: nyquist_mask(tf(k), v) * v, out=Ellipsis)
        field = fieldc.c2r()
        obsret = Observable(mapp=field, s=obs.s, d=obs.d)

        return obsret
##
