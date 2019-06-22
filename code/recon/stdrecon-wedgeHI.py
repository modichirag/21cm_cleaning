import warnings
from mpi4py import MPI
rank = MPI.COMM_WORLD.rank
#warnings.filterwarnings("ignore")
if rank!=0: warnings.filterwarnings("ignore")

import numpy
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.interpolate import interp1d 
from cosmo4d.lab import (UseComplexSpaceOptimizer,
                        NBodyModel, LPTModel, ZAModel,
                        LBFGS, ParticleMesh)

#from cosmo4d.lab import mapbias as map
from cosmo4d import lab
from cosmo4d.lab import report, dg, objectives, mapnoise, std
from abopt.algs.lbfgs import scalar as scalar_diag

from nbodykit.cosmology import Planck15, EHPower, Cosmology
from nbodykit.algorithms.fof import FOF
from nbodykit.lab import KDDensity, BigFileMesh, BigFileCatalog, ArrayCatalog, FieldMesh, FFTPower
import sys, os, json, yaml
from solve import solve
from getbiasparams import getbias, eval_bfit
sys.path.append('../')
sys.path.append('../utils/')
import HImodels


klin, plin = numpy.loadtxt('../../data/pklin_1.0000.txt', unpack = True)
ipk = interpolate(klin, plin)
#cosmo = Planck15.clone(Omega_cdm = 0.2685, h = 0.6711, Omega_b = 0.049)
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)

#########################################

#Set parameters here
##
cfname = sys.argv[1]
upsample = bool(float(sys.argv[2]))
with open(cfname, 'r') as ymlfile: cfg = yaml.load(ymlfile)
for i in cfg['basep'].keys(): locals()[i] = cfg['basep'][i]
zz = 1/aa-1


if upsample: ncd = nc*2
else: ncd = nc


truth_pm = ParticleMesh(BoxSize=bs, Nmesh=(ncd, ncd, ncd), dtype='f8')
comm = truth_pm.comm
rank = comm.rank
kmin, angle = cfg['mods']['kmin'], cfg['mods']['angle']
h1model = HImodels.ModelA(aa)
if angle is None:
    angle = numpy.round(mapnoise.wedge(zz, att=cfg['mods']['wopt'], angle=True), 0)
if rank == 0: 
    print(angle)
try: spread
except : spread = 1.

if numd <= 0: num = -1
else: num = int(bs**3 * numd)
if rank == 0: print('Number of objects : ', num)

objfunc = getattr(objectives, cfg['mods']['objective'])
map = getattr(lab, cfg['mods']['map'])

#
proj = '/project/projectdirs/cosmosim/lbl/chmodi/cosmo4d/'
if ray: dfolder = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/%dstepT-B%d/%d-%d-9100/'%(nsteps, B, bs, nc)
else: dfolder = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/%dstepT-B%d/%d-%d-9100-fixed/'%(nsteps, B, bs, nc)

if cfg['mods']['angle'] is not None: ofolder = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%.2f_ang%0.2f/L%04d-N%04d/'%(aa, kmin, cfg['mods']['angle'], bs, nc)
else: ofolder = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%.2f_%s/L%04d-N%04d/'%(aa, kmin, cfg['mods']['wopt'], bs, nc)
if ray: ofolder = ofolder[:-1]+'-R/'
#Experimental config
if stage2 is not None:
    ofolder += 'thermal-%s/'%stage2
if hex: ofolder = ofolder[:-1] + '-hex/'
if spread != 1: ofolder = ofolder[:-1] + '-sp%.1f/'%spread
if hirax : 
    ofolder = ofolder[:-1] + '-hirax/'
    Ndish = 32
else: Ndish = 256
#Dynamics config
if pmdisp: 
    ofolder += 'T%02d-B%01d/'%(nsteps, B)
else: ofolder += 'ZA/'
if prefix is None:
    prefix = '_fourier'
    if rsdpos: prefix += "_rsdpos"
#
fname = 's999_h1massA%s'%prefix
optfolder = ofolder + 'opt_%s/'%fname
if truth_pm.comm.rank == 0: print('Output Folder is %s'%optfolder)


for folder in [ofolder, optfolder]:
    try: os.makedirs(folder)
    except:pass

####################################
#initiate

#
if ray: hmeshreal = BigFileMesh('/global/cscratch1/sd/chmodi/m3127/H1mass/highres/%d-9100/fastpm_%0.4f/HImesh-N%04d/'%(bs*10, aa, ncd), 'ModelA').paint()
else : hmeshreal = BigFileMesh('/global/cscratch1/sd/chmodi/m3127/H1mass/highres/%d-9100-fixed/fastpm_%0.4f/HImesh-N%04d/'%(bs*10, aa, ncd), 'ModelA').paint()
hmeshreal /= hmeshreal.cmean()
hmeshreal -= 1

if rsdpos:
    if ray: hmesh = BigFileMesh('/global/cscratch1/sd/chmodi/m3127/H1mass/highres/%d-9100/fastpm_%0.4f/HImeshz-N%04d/'%(bs*10, aa, ncd), 'ModelA').paint()
    else: hmesh = BigFileMesh('/global/cscratch1/sd/chmodi/m3127/H1mass/highres/%d-9100-fixed/fastpm_%0.4f/HImeshz-N%04d/'%(bs*10, aa, ncd), 'ModelA').paint()
    hmesh /= hmesh.cmean()
    hmesh -= 1.
    if rsdpos: 
        if ray: pp = '/global/cscratch1/sd/chmodi/m3127/H1mass/highres/10240-9100/fastpm_%0.4f/Header/attr-v2'%aa
        else: pp = '/global/cscratch1/sd/chmodi/m3127/H1mass/highres/10240-9100-fixed/fastpm_%0.4f/Header/attr-v2'%aa
        with open(pp) as ff:
            for line in ff.readlines(): 
                if 'RSDFactor' in line: rsdfac = float(line.split()[-2])
else: 
    hmesh = hmeshreal
    rsdfac = 0
rsdfac *= 100./aa ##Add hoc factor due to incorrect velocity dimensions in nbody.py



###########################################
###dynamics

if upsample: 
    data_pf4 = map.Observable.load(optfolder+'/datap_up')
    #data_n = map.Observable.load(optfolder+'/datan_up')
    #data_w = map.Observable.load(optfolder+'/dataw_up')
else:
    data_pf4 = map.Observable.load(optfolder+'/datap')
    #data_n = map.Observable.load(optfolder+'/datan')
    #data_w = map.Observable.load(optfolder+'/dataw')


meshm = truth_pm.create(mode='real', value=data_pf4.mapp)
meshd = truth_pm.create(mode='real', value=data_pf4.s)
meshs = truth_pm.create(mode='real', value=data_pf4.s)
data_p = map.Observable(meshm, meshd, meshs)

##Get bias
pkd = FFTPower(data_p.d, mode='1d').power
pkh = FFTPower(hmeshreal, mode='1d').power
#pkx = FFTPower(hmeshreal, second=data_p.d, mode='1d').power
bias = ((pkh['power'].real/pkd['power'].real)[1:6]**0.5).mean()
if rank == 0: print('Bias = %0.2f'%bias)

Rsm = 8
Rbao = Rsm/2**0.5
ff = cosmo.scale_independent_growth_rate(zz)
beta = bias/ff

usenoise = True
if usenoise:
    if rank == 0: print('Use noise')
    truth_noise_model = mapnoise.ThermalNoise(truth_pm, seed=100, aa=aa, att=stage2,spread=spread, hex=hex, limk=2, Ns=Ndish, checkbase=False)
    data_p = truth_noise_model.add_noise(data_p)

position = truth_pm.generate_uniform_particle_grid(shift=0)
layout = truth_pm.decompose(position)

rho = data_p.mapp
rho = std.decic(rho)
rho = std.apply_wedge(rho, kmin, angle)
rhomass = rho.readout(position, layout=layout)
if rank == 0: print(rhomass)

random = truth_pm.paint(position, layout=layout)
randommass = random.readout(position, layout=layout)
if rank == 0: print(randommass) 

basemesh =  std.gauss(data_p.mapp, Rbao)
dispmesh = std.calc_displist(base=basemesh, b=bias, kmin=kmin, angle=angle)
if rank == 0: print([d[...].std() for d in dispmesh])

hpshift = std.displace(truth_pm, dispmesh, position, rsd=True, f=ff, beta=beta, mass = rhomass)
rshift = std.displace(truth_pm, dispmesh, position, rsd=True, f=ff, beta=beta, mass = randommass)
recon = hpshift - rshift
if upsample: fname = 'stdrecon_up'
else: fname = 'stdrecon'
if usenoise: fname += '-noise'
FieldMesh(recon).save(optfolder+fname, dataset='std', mode='real')


