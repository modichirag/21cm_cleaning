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
from nbodykit.lab import FFTPower, BigFileMesh, FieldMesh, BigFileCatalog, ArrayCatalog
import sys, os, json, yaml
from solve import solve
from getbiasparams import getbias, eval_bfit
sys.path.append('../')
sys.path.append('../utils/')
import HImodels


#initiatea
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
kmin, angle = cfg['mods']['kmin'], cfg['mods']['angle']
h1model = HImodels.ModelA(aa)
zz = 1/aa-1
if angle is None:
    angle = numpy.round(mapnoise.wedge(zz, att=cfg['mods']['wopt'], angle=True), 0)
if rank == 0: 
    print(angle)
try: spread
except : spread = 1.


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
if ray: dfolder = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/%dstepT-B%d/%d-%d-9100/'%(nsteps, B, bs, nc)
else: dfolder = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/%dstepT-B%d/%d-%d-9100-fixed/'%(nsteps, B, bs, nc)

ofolder = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/fastpm_%0.4f/wedge_kmin%.2f_%s/L%04d-N%04d/'%(aa, kmin, cfg['mods']['wopt'], bs, nc)
if ray: ofolder = ofolder[:-1]+'-R/'
if stage2 is not None:
    ofolder += 'thermal-%s/'%stage2
if hex: ofolder = ofolder[:-1] + '-hex/'
if spread != 1: ofolder = ofolder[:-1] + '-sp%.1f/'%spread
if pmdisp: 
    ofolder += 'T%02d-B%01d/'%(nsteps, B)
else: ofolder += 'ZA/'
if prefix is None:
    prefix = '_fourier'
    if rsdpos: prefix += "_rsdpos"

fname = 's999_h1massA%s'%prefix
optfolder = ofolder + 'opt_%s/'%fname
if truth_pm.comm.rank == 0: print('Output Folder is %s'%optfolder)


for folder in [ofolder, optfolder]:
    try: os.makedirs(folder)
    except:pass

####################################

new_pm = ParticleMesh(BoxSize=truth_pm.BoxSize, Nmesh=truth_pm.Nmesh*2, dtype='f4')
#####
#Data
if rsdpos:
    if ray: hmesh = BigFileMesh('/global/cscratch1/sd/chmodi/m3127/H1mass/highres/%d-9100/fastpm_%0.4f/HImeshz-N%04d/'%(bs*10, aa, nc*2), 'ModelA').paint()
    else: hmesh = BigFileMesh('/global/cscratch1/sd/chmodi/m3127/H1mass/highres/%d-9100-fixed/fastpm_%0.4f/HImeshz-N%04d/'%(bs*10, aa, nc*2), 'ModelA').paint()
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

hmesh /= hmesh.cmean()
hmesh -= 1.
rsdfac *= 100./aa ##Add hoc factor due to incorrect velocity dimensions in nbody.py
if rank == 0: print('RSD factor is : ', rsdfac)
noise = None 
if rank == 0 : print('Noise : ', noise)

if ray: dnewfolder = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/%dstepT-B%d/%d-%d-9100/'%(nsteps, B, bs, nc*2)
else: dnewfolder = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/%dstepT-B%d/%d-%d-9100-fixed/'%(nsteps, B, bs, nc*2)
#dnewfolder = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/%dstepT-B%d/%d-%d-9100-fixed/'%(nsteps, B, bs, nc*2)
s_truth = BigFileMesh(dnewfolder + 'linear', 'LinearDensityK').paint()
dyn = BigFileCatalog(dnewfolder + 'fastpm_%0.4f/1'%aa)
dlayout = new_pm.decompose(dyn['Position'])
d_truth = new_pm.paint(dyn['Position'], layout=dlayout)




#####
#Model
params = numpy.loadtxt(optfolder + '/params.txt')

stages = numpy.linspace(0.01, aa, nsteps, endpoint=True)
if pmdisp: dynamic_model = NBodyModel(cosmo, new_pm, B=B, steps=stages)
else: dynamic_model = ZAModel(cosmo, new_pm, B=B, steps=stages)
if rank == 0: print(dynamic_model)

#noise
if stage2 is not None: truth_noise_model = mapnoise.ThermalNoise(new_pm, seed=100, aa=aa, stage2=stage2,spread=spread, hex=hex)
else: truth_noise_model = mapnoise.ThermalNoise(new_pm, seed=None, aa=aa, stage2=stage2,spread=spread, hex=hex)
wedge_noise_model = mapnoise.WedgeNoiseModel(pm=new_pm, power=1, seed=100, kmin=kmin, angle=angle)
#Create and save data if not found

mock_model = map.MockModel(dynamic_model, params=params, rsdpos=rsdpos, rsdfac=rsdfac)
try: data_p = map.Observable.load(optfolder+'/datap_up')
except: 
    data_p = map.Observable(hmesh, d_truth, s_truth)
    data_p.save(optfolder+'datap_up/')

try: 
    data_n = map.Observable.load(optfolder+'/datan_up')
except: 
    data_n = truth_noise_model.add_noise(data_p)
    data_n.save(optfolder+'datan_up/')

try: data_w = map.Observable.load(optfolder+'/dataw_up')
except: 
    data_w = wedge_noise_model.add_noise(data_n)
    data_w.save(optfolder+'dataw_up/')


try: fit_p = map.Observable.load(optfolder+'/fitp_up')
except:
    fit_p = mock_model.make_observable(s_truth)
    fit_p.save(optfolder+'fitp_up/')


title = None
if stage2 is not None: 
    try:
        kerror, perror = numpy.loadtxt(optfolder + '/error_psnup.txt', unpack=True)
        ivarmesh = BigFileMesh(optfolder + 'ivarmesh_up', 'ivar').paint()
        ipkerror = interp1d(kerror, perror, bounds_error=False, fill_value=(perror[0], perror[-1]))
    except:
        #pkerror = FFTPower(data_n.mapp, second=-1* fit_p.mapp, mode='1d').power
        #kerror, perror = pkerror['k'], pkerror['power']
        kerror, perror = eval_bfit(data_n.mapp, fit_p.mapp, optfolder, noise=noise, title=title, fsize=15, suff='-noiseup')        
        ipkerror = interp1d(kerror, perror, bounds_error=False, fill_value=(perror[0], perror[-1]))
        if rank ==0: numpy.savetxt(optfolder + '/error_psnup.txt', numpy.array([kerror, perror]).T, header='kerror, perror')

        #pkerror = FFTPower(data_p.mapp, second=-1* fit_p.mapp, mode='1d').power
        #kerror, perror = pkerror['k'], pkerror['power']
        kerror, perror = eval_bfit(data_p.mapp, fit_p.mapp, optfolder, noise=noise, title=title, fsize=15, suff='-up')        
        ipkmodel = interp1d(kerror, perror, bounds_error=False, fill_value=(perror[0], perror[-1]))
        ivarmesh = truth_noise_model.get_ivarmesh(data_p, ipkmodel)
        FieldMesh(ivarmesh).save(optfolder+'ivarmesh_up', dataset='ivar', mode='real')
else: 
    ivarmesh = None
    try:
        kerror, perror = numpy.loadtxt(optfolder + '/error_psup.txt', unpack=True)
        ipkerror = interp1d(kerror, perror, bounds_error=False, fill_value=(perror[0], perror[-1]))
    except:
        pkerror = FFTPower(data_p.mapp, second=-1* fit_p.mapp, mode='1d').power
        kerror, perror = pkerror['k'], pkerror['power']
        ipkerror = interp1d(kerror, perror, bounds_error=False, fill_value=(perror[0], perror[-1]))
        if rank ==0: numpy.savetxt(optfolder + '/error_psup.txt', numpy.array([kerror, perror]).T, header='kerror, perror')

if rank == 0: print('Setup done')
#####################################
#Init and optimize

smoothings = cfg['basep']['sms'][::-1]
nsm = len(smoothings)

outfolder = optfolder + '/upsample%d/'%nsm
x0 =  optfolder + '/%d-%0.2f/best-fit/'%(nc, 0)
inpath = None
#if os.path.isdir(outfolder): 
try:
    for ir, r in enumerate(smoothings):
        if os.path.isdir(outfolder + '/%d-%0.2f/best-fit'%(nc*2, r)): 
            inpath = outfolder + '/%d-%0.2f//best-fit'%(nc*2, r)
            sms = smoothings[:ir][::-1]
            lit = maxiter0
            if r == 0:
                if rank == 0:print('\nAll done here already\nExiting\n')
                sys.exit()
        else:
            for iiter in range(100, -1, -20):
                path = outfolder + '/%d-%0.2f//%04d/fit_p/'%(nc*2, r, iiter)
                if os.path.isdir(path): 
                    inpath = path
                    sms = smoothings[:ir+1][::-1]
                    lit = maxiter0 - iiter
                    break
        if inpath is not None:
            if rank == 0: print('inpath: %s'%inpath)
            break
        s_init = BigFileMesh(inpath, 's').paint()

except Exception as e:
    if rank == 0: print(e)
    x0 =  optfolder + '/%d-%0.2f/best-fit/'%(nc, 0)
    s_init = BigFileMesh(x0, 's').paint()
    if rank == 0: print('Upsampling inint\n%s'%x0)
    s_init = new_pm.upsample(s_init, keep_mean=True)
    sms = smoothings[::-1]
    lit = maxiter0

N0 = nc*2
C = s_init.BoxSize[0] / s_init.Nmesh[0]
rtol = 0.005
maxiter = maxiter0
x0 = s_init

for Ns in sms:
    sml = C * Ns
    run = '%d-%0.2f'%(N0, Ns)
    maxiter = maxiter0
    if Ns == sms[0]:
        if inpath is not None:
            run += '-nit_%d-sm_%.2f'%(iiter, smoothings[ir])
            maxiter = lit
    if maxiter > 0:
        obj = objfunc(mock_model, truth_noise_model, data_p, prior_ps=ipk, error_ps=ipkerror, sml=sml, kmin=kmin, angle=angle, ivarmesh=ivarmesh)
        x0 = solve(N0, x0, rtol, run, Ns, prefix, mock_model, obj, data_p, new_pm, outfolder, \
               saveit=20, showit=5, title=None, maxiter=maxiter)    



##
##if cfg['init']['sinit'] is None:
##    s_init = BigFileMesh(x0, 's').paint()
##    if s_init.Nmesh[0] != nc*2: 
##        if rank == 0: print('Upsampling inint')
##        s_init = new_pm.upsample(s_init, keep_mean=True)
##else: 
##    if rank == 0: print('Initializing from previous iteration at :\n%s'%cfg['init']['sinit'])
##    s_init = BigFileMesh(cfg['init']['sinit'], 's').paint()
##
##sms = cfg['basep']['sms']
##nsm = len(sms)
##if cfg['init']['sinit'] is not None: sms =cfg['init']['sms'] 
##x0 = s_init
##N0 = nc*2
##C = x0.BoxSize[0] / x0.Nmesh[0]
##rtol = 0.005
##maxiter = 100
##
##for Ns in sms:
##    sml = C * Ns
##    run = '%d-%0.2f'%(N0, Ns)
##    outfolder = optfolder + '/upsample%d/'%nsm
##    if Ns == sms[0]:
##        if cfg['init']['sinit'] is not None: 
##            run += '-nit_%d-sm_%.2f'%(cfg['init']['nit'], cfg['init']['sml'])
##            maxiter -= int(cfg['init']['nit'])
##    if maxiter > 0:
##        obj = objfunc(mock_model, truth_noise_model, data_p, prior_ps=ipk, error_ps=ipkerror, sml=sml, kmin=kmin, angle=angle, ivarmesh=ivarmesh)
##        x0 = solve(N0, x0, rtol, run, Ns, prefix, mock_model, obj, data_p, new_pm, outfolder, \
##               saveit=20, showit=5, title=None, maxiter=maxiter)    
##
##
##
