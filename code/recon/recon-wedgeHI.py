import numpy
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.interpolate import interp1d 
from cosmo4d.lab import (UseComplexSpaceOptimizer,
                        NBodyModel, LPTModel, ZAModel,
                        LBFGS, ParticleMesh)

#from cosmo4d.lab import mapbias as map
from cosmo4d import lab
from cosmo4d.lab import report, dg, objectives
from abopt.algs.lbfgs import scalar as scalar_diag

from nbodykit.cosmology import Planck15, EHPower, Cosmology
from nbodykit.algorithms.fof import FOF
from nbodykit.lab import KDDensity, BigFileMesh, BigFileCatalog, ArrayCatalog
import sys, os, json, yaml
from solve import solve
from getbiasparams import getbias, eval_bfit
sys.path.append('../')
sys.path.append('../utils/')
import HImodels


#########################################

#Set parameters here
##
cfname = sys.argv[1]
with open(cfname, 'r') as ymlfile: cfg = yaml.load(ymlfile)
for i in cfg['basep'].keys(): locals()[i] = cfg['basep'][i]
kmin, angle = cfg['mods']['kmin'], cfg['mods']['angle']
h1model = HImodels.ModelA(aa)

truth_pm = ParticleMesh(BoxSize=bs, Nmesh=(nc, nc, nc), dtype='f4')
comm = truth_pm.comm
rank = comm.rank

if numd <= 0: num = -1
else: num = int(bs**3 * numd)
if rank == 0: print('Number of objects : ', num)

objfunc = getattr(objectives, cfg['mods']['objective'])
map = getattr(lab, cfg['mods']['map'])

#
proj = '/project/projectdirs/cosmosim/lbl/chmodi/cosmo4d/'
dfolder = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/%dstepT-B%d/%d-%d-9100-fixed/'%(nsteps, B, bs, nc)

ofolder = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%.2f_ang%.1f/L%04d-N%04d/'%(aa, kmin, angle, bs, nc)
if pmdisp: 
    ofolder += 'T%02d-B%01d/'%(nsteps, B)
else: ofolder += 'ZA/'
prefix = '_fourier'
if rsdpos: prefix += "_rsdpos"
fname = 's999_h1massA%s'%prefix
optfolder = ofolder + 'opt_%s/'%fname
if truth_pm.comm.rank == 0: print('Output Folder is %s'%optfolder)


for folder in [ofolder, optfolder]:
    try: os.makedirs(folder)
    except:pass


#initiate

klin, plin = numpy.loadtxt('../../data/pklin_1.0000.txt', unpack = True)
ipk = interpolate(klin, plin)
#cosmo = Planck15.clone(Omega_cdm = 0.2685, h = 0.6711, Omega_b = 0.049)
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)



if rsdpos:
    hmesh = BigFileMesh('/global/cscratch1/sd/chmodi/m3127/H1mass/highres/%d-9100-fixed/fastpm_%0.4f/HImeshz-N%04d/'%(bs*10, aa, nc), 'ModelA').paint()
    if rsdpos: 
        with open('/global/cscratch1/sd/chmodi/m3127/H1mass/highres/2560-9100-fixed/fastpm_%0.4f/Header/attr-v2'%aa) as ff:
            for line in ff.readlines(): 
                if 'RSDFactor' in line: rsdfac = float(line.split()[-2])
else: 
    hmesh = BigFileMesh('/global/cscratch1/sd/chmodi/m3127/H1mass/highres/%d-9100-fixed/fastpm_%0.4f/HImesh-N%04d/'%(bs*10, aa, nc), 'ModelA').paint()
    rsdfac = 0

hmesh /= hmesh.cmean()
hmesh -= 1.
if rank == 0: print('RSD factor is : ', rsdfac)
##
##rankweight       = sum(masswt.compute())
##totweight        = comm.allreduce(rankweight)
##rankweight       = sum((masswt**2).compute())
##totweight2        = comm.allreduce(rankweight)
#noise = bs**3 / (hmesh.csum()**2 / (hmesh**2).csum())
noise = None 
if rank == 0 : print('Noise : ', noise)


#########################################
#dynamics
stages = numpy.linspace(0.1, aa, nsteps, endpoint=True)
if pmdisp: dynamic_model = NBodyModel(cosmo, truth_pm, B=B, steps=stages)
else: dynamic_model = ZAModel(cosmo, truth_pm, B=B, steps=stages)
if rank == 0: print(dynamic_model)

#noise
#Artifically low noise since the data is constructed from the model
truth_noise_model = map.NoiseModel(truth_pm, None, noisevar*(truth_pm.BoxSize/truth_pm.Nmesh).prod(), 1234)
truth_noise_model = None

#Create and save data if not found

dyn = BigFileCatalog(dfolder + 'fastpm_%0.4f/1'%aa)
s_truth = BigFileMesh(dfolder + 'linear', 'LinearDensityK').paint()
mock_model_setup = map.MockModel(dynamic_model, rsdpos=rsdpos, rsdfac=rsdfac)
fpos, linear, linearsq, shear = mock_model_setup.get_code().compute(['x', 'linear', 'linearsq', 'shear'], init={'parameters': s_truth})
grid = truth_pm.generate_uniform_particle_grid(shift=0.0, dtype='f4')
params, bmod = getbias(truth_pm, hmesh, [linear, linearsq, shear], fpos, grid)
title = ['%0.3f'%i for i in params]
kerror, perror = eval_bfit(hmesh, bmod, optfolder, noise=noise, title=title, fsize=15)
ipkerror = interp1d(kerror, perror, bounds_error=False, fill_value=(perror[0], perror[-1]))

mock_model = map.MockModel(dynamic_model, params=params, rsdpos=rsdpos, rsdfac=rsdfac)
data_p = mock_model.make_observable(s_truth)
data_p.mapp = hmesh.copy()
data_p.save(optfolder+'datap/')
if rank == 0: print('datap saved')

#data_n = truth_noise_model.add_noise(data_p)
#data_n.save(optfolder+'datan/')
#if rank == 0: print('datan saved')

fit_p = mock_model.make_observable(s_truth)
fit_p.save(optfolder+'fitp/')
if rank == 0: print('fitp saved')

##########

if rank == 0: print('data_p, data_n created')

#########################################
#Optimizer  
if cfg['init']['sinit'] is None:
    s_init = truth_pm.generate_whitenoise(777, mode='complex')\
        .apply(lambda k, v: v * (ipk(sum(ki **2 for ki in k) **0.5) / v.BoxSize.prod()) ** 0.5)\
        .c2r()*0.001
    sms = [4.0, 2.0, 1.0, 0.5, 0.0]
else: 
    s_init = BigFileMesh(cfg['init']['sinit'], 's').paint()
    sms = cfg['init']['sms']
    if sms is None: [4.0, 2.0, 1.0, 0.5, 0.0]

x0 = s_init
N0 = nc
C = x0.BoxSize[0] / x0.Nmesh[0]

for Ns in sms:
    if truth_pm.comm.rank == 0: print('\nDo for cell smoothing of %0.2f\n'%(Ns))
    #x0 = solve(N0, x0, 0.005, '%d-%0.2f'%(N0, Ns), Ns)
    sml = C * Ns
    rtol = 0.005
    run = '%d-%0.2f'%(N0, Ns)
    if Ns == sms[0]:
        if cfg['init']['sinit'] is not None: run += '-nit_%d-sm_%.2f'%(cfg['init']['nit'], cfg['init']['sml'])
    #obj = objectives.SmoothedFourierObjective(mock_model, truth_noise_model, data_p, prior_ps=ipk, error_ps=ipkerror, sml=sml)
    obj = objfunc(mock_model, truth_noise_model, data_p, prior_ps=ipk, error_ps=ipkerror, sml=sml, kmin=kmin, angle=angle)
    x0 = solve(N0, x0, rtol, run, Ns, prefix, mock_model, obj, data_p, truth_pm, optfolder, saveit=20, showit=5, title=None)    


