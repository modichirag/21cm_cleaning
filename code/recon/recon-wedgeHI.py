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
from cosmo4d.lab import report, dg, objectives, mapnoise
from abopt.algs.lbfgs import scalar as scalar_diag

from nbodykit.cosmology import Planck15, EHPower, Cosmology
from nbodykit.algorithms.fof import FOF
from nbodykit.lab import KDDensity, BigFileMesh, BigFileCatalog, ArrayCatalog, FieldMesh
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
with open(cfname, 'r') as ymlfile: cfg = yaml.load(ymlfile)
for i in cfg['basep'].keys(): locals()[i] = cfg['basep'][i]
zz = 1/aa-1

truth_pm = ParticleMesh(BoxSize=bs, Nmesh=(nc, nc, nc), dtype='f8')
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
initseed = 777
fname = 's%d_h1massA%s'%(initseed, prefix)
optfolder = ofolder + 'opt_%s/'%fname
optfolder = optfolder[:-1] + 'iter200/'
if truth_pm.comm.rank == 0: print('Output Folder is %s'%optfolder)


for folder in [ofolder, optfolder]:
    try: os.makedirs(folder)
    except:pass

####################################
#initiate

if rsdpos:
    if ray: hmesh = BigFileMesh('/global/cscratch1/sd/chmodi/m3127/H1mass/highres/%d-9100/fastpm_%0.4f/HImeshz-N%04d/'%(bs*10, aa, nc), 'ModelA').paint()
    else: hmesh = BigFileMesh('/global/cscratch1/sd/chmodi/m3127/H1mass/highres/%d-9100-fixed/fastpm_%0.4f/HImeshz-N%04d/'%(bs*10, aa, nc), 'ModelA').paint()
    if rsdpos: 
        if ray: pp = '/global/cscratch1/sd/chmodi/m3127/H1mass/highres/10240-9100/fastpm_%0.4f/Header/attr-v2'%aa
        else: pp = '/global/cscratch1/sd/chmodi/m3127/H1mass/highres/10240-9100-fixed/fastpm_%0.4f/Header/attr-v2'%aa
        with open(pp) as ff:
            for line in ff.readlines(): 
                if 'RSDFactor' in line: rsdfac = float(line.split()[-2])
else: 
    if ray: hmesh = BigFileMesh('/global/cscratch1/sd/chmodi/m3127/H1mass/highres/%d-9100/fastpm_%0.4f/HImesh-N%04d/'%(bs*10, aa, nc), 'ModelA').paint()
    else : hmesh = BigFileMesh('/global/cscratch1/sd/chmodi/m3127/H1mass/highres/%d-9100-fixed/fastpm_%0.4f/HImesh-N%04d/'%(bs*10, aa, nc), 'ModelA').paint()
    rsdfac = 0
rsdfac *= 100./aa ##Add hoc factor due to incorrect velocity dimensions in nbody.py


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
stages = numpy.linspace(0.01, aa, nsteps, endpoint=True)
if pmdisp: dynamic_model = NBodyModel(cosmo, truth_pm, B=B, steps=stages)
else: dynamic_model = ZAModel(cosmo, truth_pm, B=B, steps=stages)
if rank == 0: print(dynamic_model)

#noise
if stage2 is not None: truth_noise_model = mapnoise.ThermalNoise(truth_pm, seed=100, aa=aa, att=stage2,spread=spread, hex=hex, limk=2, Ns=Ndish)
else: truth_noise_model = mapnoise.ThermalNoise(truth_pm, seed=None, aa=aa, att=stage2,spread=spread, hex=hex, Ns=Ndish)
wedge_noise_model = mapnoise.WedgeNoiseModel(pm=truth_pm, power=1, seed=100, kmin=kmin, angle=angle)
#Create and save data if not found

s_truth = BigFileMesh(dfolder + 'linear', 'LinearDensityK').paint()
dyn = BigFileCatalog(dfolder + 'fastpm_%0.4f/1'%aa)
dlayout = truth_pm.decompose(dyn['Position'])
d_truth = truth_pm.paint(dyn['Position'], layout=dlayout)
hmesh = truth_pm.create(mode='real', value=hmesh[...])
s_truth = truth_pm.create(mode='real', value=s_truth[...])


try: data_p = map.Observable.load(optfolder+'/datap')
except: 
    data_p = map.Observable(hmesh, d_truth, s_truth)
    data_p.save(optfolder+'datap/')

try: 
    data_n = map.Observable.load(optfolder+'/datan')
except: 
    data_n = truth_noise_model.add_noise(data_p)
    data_n.save(optfolder+'datan/')

try: data_w = map.Observable.load(optfolder+'/dataw')
except: 
    data_w = wedge_noise_model.add_noise(data_n)
    data_w.save(optfolder+'dataw/')

if rank == 0: print('Data setup')

#Fit bias model and get noise here
try: 
    params = numpy.loadtxt(optfolder + '/params.txt')
    kerror, perror = numpy.loadtxt(optfolder + '/error_ps.txt', unpack=True)
    if stage2 is not None: 
        kerror, perror = numpy.loadtxt(optfolder + '/error_psn.txt', unpack=True)
        ivarmesh = BigFileMesh(optfolder + 'ivarmesh', 'ivar').paint()
    else: ivarmesh = None

except Exception as e:
    mock_model_setup = map.MockModel(dynamic_model, rsdpos=rsdpos, rsdfac=rsdfac)
    fpos, linear, linearsq, shear = mock_model_setup.get_code().compute(['xp', 'linear', 'linearsq', 'shear'], init={'parameters': s_truth})
    grid = truth_pm.generate_uniform_particle_grid(shift=0.0, dtype='f8')
    params, bmod = getbias(truth_pm, hmesh, [linear, linearsq, shear], fpos, grid)
    if rank ==0: numpy.savetxt(optfolder + '/params.txt', params, header='b1, b2, bsq')
    title = ['%0.3f'%i for i in params]
    kerror, perror = eval_bfit(hmesh, bmod, optfolder, noise=noise, title=title, fsize=15)
    if rank ==0: numpy.savetxt(optfolder + '/error_ps.txt', numpy.array([kerror, perror]).T, header='kerror, perror')

    if stage2: 
        ipkmodel = interp1d(kerror, perror, bounds_error=False, fill_value=(perror[0], perror[-1]))
        ivarmesh = truth_noise_model.get_ivarmesh(data_p, ipkmodel)
        FieldMesh(ivarmesh).save(optfolder+'ivarmesh', dataset='ivar', mode='real')
        kerror, perror = eval_bfit(data_n.mapp, bmod, optfolder, noise=noise, title=title, fsize=15, suff='-noise')        
        if rank ==0: numpy.savetxt(optfolder + '/error_psn.txt', numpy.array([kerror, perror]).T, header='kerror, perror')
    else: ivarmesh = None
    
ipkerror = interp1d(kerror, perror, bounds_error=False, fill_value=(perror[0], perror[-1]))
mock_model = map.MockModel(dynamic_model, params=params, rsdpos=rsdpos, rsdfac=rsdfac)

try: fitp_p = map.Observable.load(optfolder+'/fitp')
except:
    fit_p = mock_model.make_observable(s_truth)
    fit_p.save(optfolder+'fitp/')


if rank == 0: print('Setup done')

#########################################
#Optimizer  
#Initialize
smoothings = [0, 0.5, 1., 2., 4.] 
inpath = None
for ir, r in enumerate(smoothings):
    if os.path.isdir(optfolder + '/%d-%0.2f/best-fit'%(nc, r)): 
        inpath = optfolder + '/%d-%0.2f//best-fit'%(nc, r)
        sms = smoothings[:ir][::-1]
        lit = 100
        if r == 0:
            if rank == 0:print('\nAll done here already\nExiting\n')
            sys.exit()
    else:
        for iiter in range(100, -1, -20):
            path = optfolder + '/%d-%0.2f//%04d/fit_p/'%(nc, r, iiter)
            if os.path.isdir(path): 
                inpath = path
                sms = smoothings[:ir+1][::-1]
                lit = 100 - iiter
                break
    if inpath is not None:
        break

if inpath is not None:
    if rank == 0: print(inpath)
    s_init = BigFileMesh(inpath, 's').paint()
else:
    s_init = truth_pm.generate_whitenoise(initseed, mode='complex')\
        .apply(lambda k, v: v * (ipk(sum(ki **2 for ki in k) **0.5) / v.BoxSize.prod()) ** 0.5)\
        .c2r()*0.001
    sms = [4.0, 2.0, 1.0, 0.5, 0.0]
    lit = 100
##if cfg['init']['sinit'] is None:
##    s_init = truth_pm.generate_whitenoise(777, mode='complex')\
##        .apply(lambda k, v: v * (ipk(sum(ki **2 for ki in k) **0.5) / v.BoxSize.prod()) ** 0.5)\
##        .c2r()*0.001
##    sms = [4.0, 2.0, 1.0, 0.5, 0.0]
##else: 
##    s_init = BigFileMesh(cfg['init']['sinit'], 's').paint()
##    sms = cfg['init']['sms']
##    if sms is None: [4.0, 2.0, 1.0, 0.5, 0.0]
##
x0 = s_init
N0 = nc
C = x0.BoxSize[0] / x0.Nmesh[0]

for Ns in sms:
    if truth_pm.comm.rank == 0: print('\nDo for cell smoothing of %0.2f\n'%(Ns))
    #x0 = solve(N0, x0, 0.005, '%d-%0.2f'%(N0, Ns), Ns)
    sml = C * Ns
    rtol = 0.01
    maxiter = 200
    run = '%d-%0.2f'%(N0, Ns)
    if Ns == sms[0]:
#        if cfg['init']['sinit'] is not None: 
#            run += '-nit_%d-sm_%.2f'%(cfg['init']['nit'], cfg['init']['sml'])
#            maxiter -= int(cfg['init']['nit'])
        if inpath is not None:
            run += '-nit_%d-sm_%.2f'%(iiter, smoothings[ir])
            maxiter = lit
    if maxiter > 0:
        obj = objfunc(mock_model, truth_noise_model, data_n, prior_ps=ipk, error_ps=ipkerror, sml=sml, kmin=kmin, angle=angle, ivarmesh=ivarmesh)
        x0 = solve(N0, x0, rtol, run, Ns, prefix, mock_model, obj, data_p, truth_pm, optfolder, \
               saveit=20, showit=5, title=None, maxiter=maxiter)    

##
###################################################################################
#####Upsample
##
##try:
##    if cfg['upsample']:
##        if rank == 0: print('\nUpsampling\n')
##        new_pm = ParticleMesh(BoxSize=truth_pm.BoxSize, Nmesh=truth_pm.Nmesh*2, dtype='f4')
##        s_init = new_pm.upsample(x0, keep_mean=True)
##        if rsdpos:
##            hmesh = BigFileMesh('/global/cscratch1/sd/chmodi/m3127/H1mass/highres/%d-9100-fixed/fastpm_%0.4f/HImeshz-N%04d/'%(bs*10, aa, nc*2), 'ModelA').paint()
##        else: 
##            hmesh = BigFileMesh('/global/cscratch1/sd/chmodi/m3127/H1mass/highres/%d-9100-fixed/fastpm_%0.4f/HImesh-N%04d/'%(bs*10, aa, nc*2), 'ModelA').paint()
##        hmesh /= hmesh.cmean()
##        hmesh -= 1.
##
##        if pmdisp: dynamic_model = NBodyModel(cosmo, new_pm, B=B, steps=stages)
##        else: dynamic_model = ZAModel(cosmo, new_pm, B=B, steps=stages)
##
##        dnewfolder = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/%dstepT-B%d/%d-%d-9100-fixed/'%(nsteps, B, bs, nc*2)
##        s_truth = BigFileMesh(dnewfolder + 'linear', 'LinearDensityK').paint()
##        dyn = BigFileCatalog(dnewfolder + 'fastpm_%0.4f/1'%aa)
##        dlayout = new_pm.decompose(dyn['Position'])
##        d_truth = new_pm.paint(dyn['Position'], layout=dlayout)
##
##        mock_model = map.MockModel(dynamic_model, params=params, rsdpos=rsdpos, rsdfac=rsdfac)
##        truth_noise_model = None
##        data_p = map.Observable(hmesh, d_truth, s_truth)
##        data_p.save(optfolder+'datap_up/')
##        if rank == 0: print('datap saved')      
##
##        fit_p = mock_model.make_observable(s_truth)
##        fit_p.save(optfolder+'fitp_up/')
##        if rank == 0: print('fitp saved')
##
##        x0 = s_init
##        N0 = nc*2
##        C = x0.BoxSize[0] / x0.Nmesh[0]
##        rtol = 0.005
##        maxiter = 100
##
##        for Ns in [1, 0]:
##            sml = C * Ns
##            run = '%d-%0.2f'%(N0, Ns)
##            obj = objfunc(mock_model, truth_noise_model, data_p, prior_ps=ipk, error_ps=ipkerror, sml=sml, kmin=kmin, angle=angle)
##            x0 = solve(N0, x0, rtol, run, Ns, prefix, mock_model, obj, data_p, new_pm, optfolder + '/upsample/', \
##                       saveit=20, showit=5, title=None, maxiter=maxiter)    
##        print('\nFinished After Upsampling\n')
##
##except Exception as e:
##    print(e)
##    if rank == 0: print('\nFinished\n')
