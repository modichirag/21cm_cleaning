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

#########################################

#Set parameters here
##
cfname = sys.argv[1]
with open(cfname, 'r') as ymlfile: cfg = yaml.load(ymlfile)
for i in cfg['basep'].keys(): locals()[i] = cfg['basep'][i]
zz = 1/aa-1

truth_pm = ParticleMesh(BoxSize=bs, Nmesh=(nc, nc, nc), dtype='f4')
comm = truth_pm.comm
rank = comm.rank

if numd <= 0: num = -1
else: num = int(bs**3 * numd)
if rank == 0: print('Number of objects : ', num)

objfunc = getattr(objectives, cfg['mods']['objective'])
map = getattr(lab, cfg['mods']['map'])

#
proj3058 = '/project/projectdirs/m3058/chmodi/cosmo4d/'
proj3035 = '/project/projectdirs/m3035/chmodi/cosmo4d/'
dfolder = proj3035 + 'data/z00/L%04d_N%04d_S%04d_%02dstep/'%(bs, int(nc), seed, 5)
hofolder = proj3058 + 'data/L%04d_N%04d_S%04d_%02dstep/'%(bs, int(nc*4), seed, 40)

ofolder = '/global/cscratch1/sd/chmodi/cosmo4d/output/version2/L%04d_N%04d_%02dstep-fof/lhd_S%04d/n%02d/biasmodel/'%(bs, nc, nsteps, seed, numd*1e4)
if pmdisp: ofolder += 'pm/'
else: ofolder += 'za/'
prefix = '_fourier'
if rsdpos: prefix += "_rsdpos"
if masswt: fname = 's999_mass%s'%prefix
else: fname = 's999_pos%s'%prefix
optfolder = ofolder + 'opt_%s/'%fname
if truth_pm.comm.rank == 0: print('Output Folder is %s'%optfolder)


for folder in [ofolder, optfolder]:
    try: os.makedirs(folder)
    except:pass


#########################################
#initiate & halos
klin, plin = numpy.loadtxt('../../data/ics_matterpow_0.dat', unpack = True)
ipk = interpolate(klin, plin)
cosmo = Planck15.clone(Omega_cdm = 0.2685, h = 0.6711, Omega_b = 0.049)

data = BigFileCatalog(hofolder + '/FOF/')
data = data.gslice(start = 0, stop = num)
#data['Mass'] = data['Length']*data.attrs['M0']*1e10
data['Mass'] = data['Length']*1e10

if masswt : 
    masswt = data['Mass'].copy()
else: masswt = data['Mass'].copy()*0 + 1.

hpos, hmass = data['PeakPosition'], masswt
rsdfac = 0
if rsdpos: 
    rsdfac = (aa**1*cosmo.efunc(zz) * 100)**-1
    if rank == 0:  print((rsdfac*data['PeakVelocity']*numpy.array([0, 0, 1]).reshape(1, -1)).std())
    hpos = data['PeakPosition'] + rsdfac*data['PeakVelocity']*numpy.array([0, 0, 1]).reshape(1, -1)
if rank == 0: print(rsdfac)
hlayout = truth_pm.decompose(hpos)
hmesh = truth_pm.paint(hpos, layout=hlayout, mass=hmass)
hmesh /= hmesh.cmean()
hmesh -= 1.

rankweight       = sum(masswt.compute())
totweight        = comm.allreduce(rankweight)
rankweight       = sum((masswt**2).compute())
totweight2        = comm.allreduce(rankweight)
noise = bs**3 / (totweight**2/totweight2)
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

s_truth = truth_pm.generate_whitenoise(seed, mode='complex')\
              .apply(lambda k, v: v * (ipk(sum(ki **2 for ki in k) **0.5) / v.BoxSize.prod()) ** 0.5)\
              .c2r()

s_truth = BigFileMesh(dfolder + 'mesh', 's').paint()
d_truth = BigFileMesh(dfolder + 'mesh', 'd').paint()

try: 
    params = numpy.loadtxt(optfolder + '/params.txt')
    kerror, perror = numpy.loadtxt(optfolder + '/error_ps.txt', unpack=True)
except Exception as e:
    mock_model_setup = map.MockModel(dynamic_model, rsdpos=rsdpos, rsdfac=rsdfac)
    fpos, linear, linearsq, shear = mock_model_setup.get_code().compute(['xp', 'linear', 'linearsq', 'shear'], init={'parameters': s_truth})
    grid = truth_pm.generate_uniform_particle_grid(shift=0.0, dtype='f4')
    params, bmod = getbias(truth_pm, hmesh, [linear, linearsq, shear], fpos, grid)
    title = ['%0.3f'%i for i in params]
    kerror, perror = eval_bfit(hmesh, bmod, optfolder, noise=noise, title=title, fsize=15)
    if rank == 0: 
        numpy.savetxt(optfolder + '/params.txt', params, header='b1, b2, bsq')
        numpy.savetxt(optfolder + '/error_ps.txt', numpy.array([kerror, perror]).T, header='kerror, perror')
##mock_model_setup = map.MockModel(dynamic_model, rsdpos=rsdpos, rsdfac=rsdfac)
##fpos, linear, linearsq, shear = mock_model_setup.get_code().compute(['xp', 'linear', 'linearsq', 'shear'], init={'parameters': s_truth})
##grid = truth_pm.generate_uniform_particle_grid(shift=0.0, dtype='f4')
##params, bmod = getbias(truth_pm, hmesh, [linear, linearsq, shear], fpos, grid)
##title = ['%0.3f'%i for i in params]
##kerror, perror = eval_bfit(hmesh, bmod, optfolder, noise=noise, title=title, fsize=15)

ipkerror = interp1d(kerror, perror, bounds_error=False, fill_value=(perror[0], perror[-1]))

mock_model = map.MockModel(dynamic_model, params=params, rsdpos=rsdpos, rsdfac=rsdfac)
data_p = mock_model.make_observable(s_truth)
data_p.mapp = hmesh.copy()
data_p.save(optfolder+'datap/')
if rank == 0: print('datap saved')


fit_p = mock_model.make_observable(s_truth)
fit_p.save(optfolder+'fitp/')
if rank == 0: print('fitp saved')


if rank == 0: print('data_p, data_n created')

################################################
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
    sml = C * Ns
    rtol = 0.005
    run = '%d-%0.2f'%(N0, Ns)
    if Ns == sms[0]:
        if cfg['init']['sinit'] is not None: run += '-nit_%d-sm_%.2f'%(cfg['init']['nit'], cfg['init']['sml'])
    obj = objfunc(mock_model, truth_noise_model, data_p, prior_ps=ipk, error_ps=ipkerror, sml=sml)
    x0 = solve(N0, x0, rtol, run, Ns, prefix, mock_model, obj, data_p, truth_pm, optfolder, saveit=20, showit=5, title=None)    



#########################################


