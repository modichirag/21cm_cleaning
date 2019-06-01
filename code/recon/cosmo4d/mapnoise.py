
import numpy
import numpy as np
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
from   astropy.cosmology import FlatLambdaCDM


package_path = os.path.dirname(os.path.abspath(__file__))+'/'
klin, plin = numpy.loadtxt(package_path + '../../../data/pklin_1.0000.txt', unpack = True)
ipk = interpolate(klin, plin)
#cosmo = Planck15.clone(Omega_cdm = 0.2685, h = 0.6711, Omega_b = 0.049)
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)
cc = FlatLambdaCDM(H0=67.7,Om0=0.309167)


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
        u =  k *numpy.sqrt(1 - mu**2)* chiz(z) /(2* numpy.pi)
        D = u *0.21*(1+z)
        DLs = D/Ls
        #n =  n0 *(0.4847 - 0.33 *(D/Ls))/(1 + 1.3157 *(D/Ls)**1.5974) * numpy.exp(-(D/Ls)**6.8390)
        n =  n0 *(0.4847 - 0.33 *(DLs))/(1 + 1.3157 *(DLs)**1.5974) * numpy.exp(-(DLs)**6.8390) *(0.21*(1+z))**2
        return n
    return vis


def thermalnoise(z, stage2=True, mK=False, retn=False):
    eta =0.7
    fsky21 = 20000/41252;
    Ns = 256
    if not stage2: Ns = 32
    Ds = 6
    n0 = (Ns/Ds)**2
    Ls = Ds* Ns
    npol = 2
    S21 = 4 *numpy.pi* fsky21
    t0 = 5*365*24*60*60
    Aeff = numpy.pi* (Ds/2)**2 *eta #effective
    nu0 = 1420*1e6
    wavez = lambda z: 0.211 *(1 + z)
    chiz = lambda z : cosmo.comoving_distance(z)
    #defintions
    Tb = lambda z: 180/(cosmo.efunc(z)) *(4 *10**-4 *(1 + z)**0.6) *(1 + z)**2*cosmodef['h']
    #FOV= lambda z: (1.22* wavez(z)/Ds)**2/eta; #why is Ds here
    FOV= lambda z: ( wavez(z)/Ds)**2/eta; #why is Ds here
    Ts = lambda z: (55 + 30 + 2.7 + 25 *(1420/400/(1 + z))**-2.75) * 1000;

    #u = lambda k, mu, z: k *numpy.sqrt(1 - mu**2)* chiz(z) /(2* numpy.pi)#* wavez(z)**2
    #n = lambda D: n0 *(0.4847 - 0.33 *(D/Ls))/(1 + 1.3157 *(D/Ls)**1.5974) * numpy.exp(-(D/Ls)**6.8390) +1e-7
    u = lambda k, mu, z: k *numpy.sqrt(1 - mu**2)* chiz(z) /(2* numpy.pi)* wavez(z)**2
    n = lambda D: n0 *(0.4847 - 0.33 *(D/Ls))/(1 + 1.3157 *(D/Ls)**1.5974) * numpy.exp(-(D/Ls)**6.8390)* wavez(z)**2 +1e-7
   #terms
    d2V = lambda z: chiz(z)**2* 3* 10**5 *(1 + z)**2 /cosmo.efunc(z)/100 
    fac = lambda z: Ts(z)**2 * S21 / Aeff **2 * (wavez(z))**4 /FOV(z) 
    # fac = lambda z: Ts(z)**2 * S21 /Aeff **2 * ((1.22 * wavez(z))**2)**2 #/FOV(z)/1.22**4
    cfac = 1 /t0/ nu0 / npol
    #
    setupz = cfac *fac(z) *d2V(z) / wavez(z)**2
    if  not mK: setupz /= Tb(z)**2
    print('d2V =', d2V(z))
    print('Omp =', FOV(z))
    #Pn = lambda k, mu, z: setupz / n(u(k, mu, z))
    #if retn: return lambda k, mu, z: n(u(k, mu, z))
    Pn = lambda k, mu: setupz / visibility()(k, mu, z)
    if retn: return lambda k, mu: visibility()(k, mu, z)
    return Pn



def thermal_n(k, mu,zz,D=6.0,Ns=256,att='reas', spread=1, hex=True):
    """The thermal noise for PUMA -- note noise rescaling from 5->5/4 yr."""
    # Some constants.
    Ns *= spread
    etaA = 0.7                          # Aperture efficiency.
    Aeff = etaA*np.pi*(D/2)**2          # m^2
    lam21= 0.21*(1+zz)                  # m
    nuobs= 1420/(1+zz)                  # MHz
    # The cosmology-dependent factors.
    hub  = cc.H(0).value / 100.0
    Ez   = cc.H(zz).value / cc.H(0).value
    chi  = cc.comoving_distance(zz).value * hub         # Mpc/h.
    OmHI = 4e-4*(1+zz)**0.6 / Ez**2
    Tbar = 0.188*hub*(1+zz)**2*Ez*OmHI  # K
    # Eq. (3.3) of Chen++19
    d2V  = chi**2*2997.925/Ez*(1+zz)**2
    # Eq. (3.5) of Chen++19
    kperp = k*numpy.sqrt(1-mu**2)
    if hex:    # Hexagonal array of Ns^2 elements.
        n0,c1,c2,c3,c4,c5 = (Ns/D)**2,0.5698,-0.5274,0.8358,1.6635,7.3177
        uu   = kperp*chi/(2*np.pi)
        xx   = uu*lam21/Ns/D                # Dimensionless.
        nbase= n0*(c1+c2*xx)/(1+c3*xx**c4)*np.exp(-xx**c5) * lam21**2 + 1e-30
        #return nbase
        nbase[nbase < 0] = 1e-30
        nbase[uu<   D/lam21    ]=1e-30
        nbase[uu>Ns*D/lam21*1.3]=1e-30
    else:      # Square array of Ns^2 elements.
        n0,c1,c2,c3,c4,c5 = (Ns/D)**2,0.4847,-0.33,1.3157,1.5974,6.8390
        uu   = kperp*chi/(2*np.pi)
        xx   = uu*lam21/Ns/D                # Dimensionless.
        nbase= n0*(c1+c2*xx)/(1+c3*xx**c4)*np.exp(-xx**c5) * lam21**2 + 1e-30
        nbase[uu<   D/lam21    ]=1e-30
        nbase[uu>Ns*D/lam21*1.4]=1e-30
    # Eq. (3.2) of Chen++19, updated to match PUMA specs:
    npol = 2
    fsky = 0.5
    tobs = 5.*365.25*24.*3600.          # sec.
    if att == 'opt': tobs/= 1.0         # Scale to 1/2-filled array.
    elif att == 'reas': tobs/= 4.0      # Scale to 1/2-filled array
    elif att == 'pess': tobs/= 16.0     # Scale to 1/2-filled array.
    tobs /= spread**2

    # the signal entering OMT is given by eta_dish*T_s + (1-eta_dish)*T_g
    # and after hitting both with eta_omt and adding amplifier noise you get:
    # T_ampl + eta_omt.eta_dish.T_s + eta_omt(1-eta_dish)T_g
    # so normalizing to have 1 in front of Ts we get
    # T_ampl/(eta_omt*eta_dish) + T_g (1-eta_dish)/(eta_dish) + T_sky
    # Putting in T_ampl=50K T_g=300K eta_omt=eta_dish=0.9 gives:
    Tamp = 50.0/0.9**2                  # K
    Tgnd = 300./0.9*(1-0.9)             # K
    Tsky = 2.7 + 25*(400./nuobs)**2.75  # K
    Tsys = Tamp + Tsky + Tgnd
    Omp  = (lam21/D)**2/etaA
    # Return Pth in "cosmological units", with the Tbar divided out.
    Pth  = (Tsys/Tbar)**2*(lam21**2/Aeff)**2 *\
           4*np.pi*fsky/Omp/(npol*1420e6*tobs*nbase) * d2V
    return(Pth)
    #


##def thermal_n(k, mu, zz,D=6.0,Ns=256,att='reas', spread=1.):
##    """The thermal noise for PUMA -- note noise rescaling."""
##    # Some constants.
##    Ns *= spread
##    etaA = 0.7                          # Aperture efficiency.
##    Aeff = etaA*numpy.pi*(D/2)**2          # m^2
##    lam21= 0.21*(1+zz)                  # m
##    nuobs= 1420/(1+zz)                  # MHz
##    # The cosmology-dependent factors.
##    hub  = cc.H(0).value / 100.0
##    Ez   = cc.H(zz).value / cc.H(0).value
##    chi  = cc.comoving_distance(zz).value * hub         # Mpc/h.
##    OmHI = 4e-4*(1+zz)**0.6 / Ez**2
##    Tbar = 0.188*hub*(1+zz)**2*Ez*OmHI  # K
##    # Eq. (3.3) of Chen++19
##    d2V  = chi**2*2997.925/Ez*(1+zz)**2
##    # Eq. (3.5) of Chen++19
##    n0,c1,c2,c3,c4,c5 = (Ns/D)**2,0.4847,-0.33,1.3157,1.5974,6.8390
##    kperp = k*numpy.sqrt(1-mu**2)
##    uu   = kperp*chi/(2*numpy.pi)
##    xx   = uu*lam21/Ns/D                # Dimensionless.
##    nbase= n0*(c1+c2*xx)/(1+c3*xx**c4)*numpy.exp(-xx**c5) * lam21**2 + 1e-30
##    #nbase[uu<   D/lam21    ]=1e-30
##    #nbase[uu>Ns*D/lam21*1.4]=1e-30
##    # Eq. (3.2) of Chen++19
##    npol = 2
##    fsky = 0.5
##    tobs = 5.*365.25*24.*3600.          # sec.
##    tobs /= spread**2
##    if att == 'opt': tobs/= 1.0         # Scale to 1/2-filled array.
##    elif att == 'reas': tobs/= 4.0      # Scale to 1/2-filled array
##    elif att == 'pess': tobs/= 16.0     # Scale to 1/2-filled array.
##    Tamp = 55.0                         # K
##    Tgnd = 30.0                         # K
##    Tsky = 2.7 + 25*(400./nuobs)**2.75  # K
##    Tsys = Tamp + Tsky + Tgnd
##    Omp  = (lam21/D)**2/etaA
##    # Return Pth in "cosmological units", with the Tbar divided out.
##    Pth  = (Tsys/Tbar)**2*(lam21**2/Aeff)**2 *\
##           4*numpy.pi*fsky/Omp/(npol*1420e6*tobs*nbase) * d2V
##    return(Pth)
##    #
##


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
    def __init__(self, pm, aa, seed=100, stage2='reas', limk = 1.1, spread=1., hex=True):
        self.pm = pm
        self.aa = aa
        self.zz = 1/aa-1
        self.seed = seed
        self.noise = lambda k, mu: thermal_n(k, mu, self.zz, att=stage2, spread=spread, hex=hex)
        self.limk = limk
        self.spread = spread
        self.hex = hex
        #self.noise = lambda kp: thermal_n(kp, self.zz, att=stage2)
        #self.noise = thermalnoise(self.zz, stage2=True)
        

    def get_ivarmesh(self, obs, ipk=None):
        pm = obs.mapp.pm
        kk = obs.mapp.r2c().x
        kmesh = sum(i**2 for i in kk)**0.5
        mask = kmesh == 0
        kmesh[kmesh == 0] = 1
        mumesh = kk[2]/kmesh
        #mumesh[mask] = 0
        kperp = (kk[0]**2 + kk[1]**2)**0.5
        kperpmesh = kmesh*(1-mumesh**2)**0.5
        
        noiseth = self.noise(kmesh, mumesh)
        #nlim = self.noise(self.limk, 0)*10
        #noiseth[kperpmesh>self.limk] = nlim
        #noiseth = self.noise(kperp)
        noiseth = noiseth + kmesh*0
        if ipk is not None: noise = ipk(kmesh)
        else: noise = noiseth * 0
        toret = ((noiseth + noise)/ pm.BoxSize.prod()) ** -1.
        #toret = toret*0+1
        return pm.create(mode='complex', value=toret).c2r()
        
    def add_noise(self, obs):
        
        pm = obs.mapp.pm
        kk = obs.mapp.r2c().x
        kmesh = sum(i**2 for i in kk)**0.5
        mask = kmesh == 0
        kmesh[kmesh == 0] = 1
        mumesh = kk[2]/kmesh
        kperp = (kk[0]**2 + kk[1]**2)**0.5
        mumesh = kk[2]/kmesh
        kperpmesh = kmesh*(1-mumesh**2)**0.5
        #mumesh[mask] = 0
        
        noisep = self.noise(kmesh, mumesh)
        #nlim = self.noise(self.limk, 0)*10
        #noisep[kperpmesh>self.limk] = nlim
        #noisep = self.noise(kperp)
        #noisep = noisep + kmesh*0
        #noise = noisep*0 + 1
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
