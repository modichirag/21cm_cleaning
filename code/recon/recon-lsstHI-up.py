import warnings
from mpi4py import MPI
rank = MPI.COMM_WORLD.rank
#warnings.filterwarnings("ignore")
if rank!=0: warnings.filterwarnings("ignore")

import numpy
import numpy as  np
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

truth_pm = ParticleMesh(BoxSize=bs, Nmesh=(nc, nc, nc), dtype='f8')
comm = truth_pm.comm
rank = comm.rank
nc2  = nc*2

if numd <= 0: num = -1
else: num = int(bs**3 * numd)
if rank == 0: print('Number of objects : ', num)

objfunc = getattr(objectives, cfg['mods']['objective'])
map = getattr(lab, cfg['mods']['map'])

#
proj = '/project/projectdirs/m3058/chmodi/m3127/'
#proj = '/project/projectdirs/cosmosim/lbl/chmodi/cosmo4d/'
if ray: dfolder = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/%dstepT-B%d/%d-%d-9100/'%(nsteps, B, bs, nc)
else: dfolder = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/%dstepT-B%d/%d-%d-9100-fixed/'%(nsteps, B, bs, nc)

ofolder = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/reconlsstv2/fastpm_%0.4f/wedge_kmin%.2f_%s/L%04d-N%04d/'%(aa, kmin, cfg['mods']['wopt'], bs, nc)
if ray: ofolder = ofolder[:-1]+'-R/'
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
if lsstwt: prefix += '_ln%04d'%(lsstnumd*1e4)
if rsdpos: prefix += "_rsdpos"

initseed = 777
fname = 's%d_h1massD%s'%(initseed, "_"+prefix)
optfolder = ofolder + 'opt_%s/'%fname

#ofolder = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/reconlsst/fastpm_0.2000/wedge_kmin0.03_pess/L1024-N0256-R/'
if truth_pm.comm.rank == 0: print('Output Folder is %s'%optfolder)

###
for folder in [ofolder, optfolder]:
    try: os.makedirs(folder)
    except:pass

####################################

new_pm = ParticleMesh(BoxSize=truth_pm.BoxSize, Nmesh=truth_pm.Nmesh*2, dtype='f8')
lsstnum = int(lsstnumd * bs**3)
if rank == 0: print("Lsst number of obj : ", lsstnum)

#####
#Data
if rsdpos :
    pp = proj + '/HV10240-R/fastpm_%0.4f/Header/attr-v2'%aa
    with open(pp) as ff:
        for line in ff.readlines():
            if 'RSDFactor' in line: rsdfaccat = float(line.split()[-2])
else: rsdfaccat = 0.
rsdfac = rsdfaccat * 100./aa ##Add hoc factor due to incorrect velocity dimensions in nbody.py
if rank == 0: print('RSD factor for catalog is : ', rsdfaccat)
if rank == 0: print('RSD factor is : ', rsdfac)
noise = None 
if rank == 0 : print('Noise : ', noise)

stages = numpy.linspace(0.01, aa, nsteps, endpoint=True)
if pmdisp: dynamic_model = NBodyModel(cosmo, new_pm, B=B, steps=stages)
else: dynamic_model = ZAModel(cosmo, new_pm, B=B, steps=stages)
if rank == 0: print(dynamic_model)

#noise
if stage2 is not None: truth_noise_model = mapnoise.ThermalNoise(new_pm, seed=100, aa=aa, att=stage2,spread=spread, hex=hex, limk=2, Ns=Ndish, checkbase=True)
else: truth_noise_model = mapnoise.ThermalNoise(new_pm, seed=None, aa=aa, att=stage2,spread=spread, hex=hex, Ns=Ndish)
wedge_noise_model = mapnoise.WedgeNoiseModel(pm=new_pm, power=1, seed=100, kmin=kmin, angle=angle)
#Create and save data if not found



#################
###
######
###
###if rsdpos:
###    if ray: hmesh = BigFileMesh(proj + '/HV%d-R/fastpm_%0.4f/HImeshz-N%04d/'%(bs*10, aa, nc2), 'ModelD').paint()
###    else: hmesh = BigFileMesh(proj + '/HV%d-F/fastpm_%0.4f/HImeshz-N%04d/'%(bs*10, aa, nc2), 'ModelD').paint()
###else:
###    if ray: hmesh = BigFileMesh(proj + '/HV%d-R/fastpm_%0.4f/HImesh-N%04d/'%(bs*10, aa, nc2), 'ModelD').paint()
###    else: hmesh = BigFileMesh(proj + '/HV%d-F/fastpm_%0.4f/HImesh-N%04d/'%(bs*10, aa, nc2), 'ModelD').paint()
###
###hmesh /= hmesh.cmean()
###hmesh -= 1.
###
###if ray: dnewfolder = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/%dstepT-B%d/%d-%d-9100/'%(nsteps, B, bs, nc*2)
###else: dnewfolder = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/%dstepT-B%d/%d-%d-9100-fixed/'%(nsteps, B, bs, nc*2)
####dnewfolder = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/%dstepT-B%d/%d-%d-9100-fixed/'%(nsteps, B, bs, nc*2)
###s_truth = BigFileMesh(dnewfolder + 'linear', 'LinearDensityK').paint()
###dyn = BigFileCatalog(dnewfolder + 'fastpm_%0.4f/1'%aa)
###dlayout = new_pm.decompose(dyn['Position'])
###d_truth = new_pm.paint(dyn['Position'], layout=dlayout)
###
###
###

try:
    data_p = map.Observable.load(optfolder+'/datap_up')
    s_truth = data_p.s
    d_truth = data_p.d
    hmesh = data_p.mapp
    lmesh = data_p.mapp2
    
except Exception as e: 
    #data_p = map.Observable(hmesh, d_truth, s_truth)
    #data_p.save(optfolder+'datap_up/')
    print(e)
    cat = BigFileCatalog(proj + '/HV%d-R/fastpm_%0.4f/LL-M10p5/'%(bs*10, aa))
    cat  = cat.sort('Length', reverse=False)
    lsstcat = cat.gslice(start = cat.csize - lsstnum - 1, stop = cat.csize-1)
    if rank == 0: print("csize : ", lsstcat.csize)
    lsstmasswt = lsstcat['Mass'].copy().flatten()
    if not lsstmass: lsstmasswt = lsstmasswt*0 + 1.
    lsstposition = lsstcat['Position'] + lsstcat['Velocity']*np.array([0, 0, 1])*rsdfaccat
    llayout = new_pm.decompose(lsstposition)
    lmesh = new_pm.paint(lsstposition, mass=lsstmasswt, layout=llayout)
    lmesh /= lmesh.cmean()
    lmesh -= 1

    #llayout = new_pm.decompose(lsstcat['Position'])
    #lmesh = new_pm.paint(lsstcat['Position'] + lsstcat['Velocity']*np.array([0, 0, 1])*rsdfac, mass=lsstmasswt, layout=llayout)
    #lmesh /= lmesh.cmean()
    #lmesh -= 1

    if rsdpos:
        if ray: hmesh = BigFileMesh(proj + '/HV%d-R/fastpm_%0.4f/HImeshz-N%04d/'%(bs*10, aa, nc2), 'ModelD').paint()
        else: hmesh = BigFileMesh(proj + '/HV%d-F/fastpm_%0.4f/HImeshz-N%04d/'%(bs*10, aa, nc2), 'ModelD').paint()
    else:
        if ray: hmesh = BigFileMesh(proj + '/HV%d-R/fastpm_%0.4f/HImesh-N%04d/'%(bs*10, aa, nc2), 'ModelD').paint()
        else: hmesh = BigFileMesh(proj + '/HV%d-F/fastpm_%0.4f/HImesh-N%04d/'%(bs*10, aa, nc2), 'ModelD').paint()

    hmesh /= hmesh.cmean()
    hmesh -= 1.

    if ray: dnewfolder = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/%dstepT-B%d/%d-%d-9100/'%(nsteps, B, bs, nc*2)
    else: dnewfolder = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/%dstepT-B%d/%d-%d-9100-fixed/'%(nsteps, B, bs, nc*2)
    s_truth = BigFileMesh(dnewfolder + 'linear', 'LinearDensityK').paint()
    s_truth = new_pm.create(mode='real', value=s_truth[...])
    dyn = BigFileCatalog(dnewfolder + 'fastpm_%0.4f/1'%aa)
    dlayout = new_pm.decompose(dyn['Position'])
    d_truth = new_pm.paint(dyn['Position'], layout=dlayout)
    hmesh = new_pm.create(mode='real', value=hmesh[...])
    
    data_p = map.Observable(hmesh, d_truth, s_truth, lmesh)
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

##
#Model
title = None
paramsfile = '/paramsup%s.txt'
try:
    params = numpy.loadtxt(optfolder + paramsfile%'')
    params_lsst = numpy.loadtxt(optfolder + paramsfile%'_lsst')
    #mock_model = map.MockModel(dynamic_model, params=params, rsdpos=rsdpos, rsdfac=rsdfac)
    mock_model = map.MockModel(dynamic_model, params=params, params2 = params_lsst, rsdpos=rsdpos, rsdfac=rsdfac)
    fit_p = map.Observable.load(optfolder+'/fitp_up')
    #
    ivarmesh = BigFileMesh(optfolder + 'ivarmesh_up', 'ivar').paint()
    kerror, perror = numpy.loadtxt(optfolder + '/error_psnup.txt', unpack=True)
    ipkerror = interp1d(kerror, perror, bounds_error=False, fill_value=(perror[0], perror[-1]))
    kerror, perror = numpy.loadtxt(optfolder + '/error_psup.txt', unpack=True)
    ipkmodel = interp1d(kerror, perror, bounds_error=False, fill_value=(perror[0], perror[-1]))
    kerror_lsst, perror_lsst = numpy.loadtxt(optfolder + '/error_psup_lsst.txt', unpack=True)
    ipkerror_lsst = interp1d(kerror_lsst, perror_lsst, bounds_error=False, fill_value=(perror_lsst[0], perror_lsst[-1]))

except Exception as e:

    print('Exception occured : ', e)
   
    mock_model_setup = map.MockModel(dynamic_model, rsdpos=rsdpos, rsdfac=rsdfac)
    fpos, linear, linearsq, shear = mock_model_setup.get_code().compute(['xp', 'linear', 'linearsq', 'shear'], init={'parameters': s_truth})
    grid = new_pm.generate_uniform_particle_grid(shift=0.0, dtype='f8')
    #For LSST
    params_lsst, bmod = getbias(new_pm, lmesh, [linear, linearsq, shear], fpos, grid, fitb2=True)
    if rank ==0: numpy.savetxt(optfolder + paramsfile%'_lsst', params_lsst, header='b1, b2, bsq')
    #For HI
    params, bmod = getbias(new_pm, hmesh, [linear, linearsq, shear], fpos, grid, fitb2=True)
    #params, bmod = getbias(new_pm, hmesh, [linear, linearsq, shear], fpos, grid)
    if rank ==0: numpy.savetxt(optfolder + paramsfile%'', params, header='b1, b2, bsq')
    #Create model and save
    mock_model = map.MockModel(dynamic_model, params=params, params2 = params_lsst, rsdpos=rsdpos, rsdfac=rsdfac)
    fit_p = mock_model.make_observable(s_truth)
    fit_p.save(optfolder+'fitp_up/')
    #Quantify error
    kerror, perror = eval_bfit(data_n.mapp, fit_p.mapp, optfolder, noise=noise, title=title, fsize=15, suff='-noiseup')        
    ipkerror = interp1d(kerror, perror, bounds_error=False, fill_value=(perror[0], perror[-1]))
    if rank ==0: numpy.savetxt(optfolder + '/error_psnup.txt', numpy.array([kerror, perror]).T, header='kerror, perror')
    kerror, perror = eval_bfit(data_p.mapp, fit_p.mapp, optfolder, noise=noise, title=title, fsize=15, suff='-up')        
    if rank ==0: numpy.savetxt(optfolder + '/error_psup.txt', numpy.array([kerror, perror]).T, header='kerror, perror')
    ipkmodel = interp1d(kerror, perror, bounds_error=False, fill_value=(perror[0], perror[-1]))
    ivarmesh = truth_noise_model.get_ivarmesh(data_p, ipkmodel)
    FieldMesh(ivarmesh).save(optfolder+'ivarmesh_up', dataset='ivar', mode='real')        
    kerror_lsst, perror_lsst = eval_bfit(data_p.mapp2, fit_p.mapp2, optfolder, noise=noise, title=title, fsize=15, suff="-up_lsst")
    if rank ==0: numpy.savetxt(optfolder + '/error_psup_lsst.txt', numpy.array([kerror_lsst, perror_lsst]).T, header='kerror, perror')
    ipkerror_lsst = interp1d(kerror_lsst, perror_lsst, bounds_error=False, fill_value=(perror_lsst[0], perror_lsst[-1]))


##if stage2 is not None: 
##    try:
##        ivarmesh = BigFileMesh(optfolder + 'ivarmesh_up', 'ivar').paint()
##        kerror, perror = numpy.loadtxt(optfolder + '/error_psnup.txt', unpack=True)
##        ipkerror = interp1d(kerror, perror, bounds_error=False, fill_value=(perror[0], perror[-1]))
##        kerror, perror = numpy.loadtxt(optfolder + '/error_psup.txt', unpack=True)
##        ipkmodel = interp1d(kerror, perror, bounds_error=False, fill_value=(perror[0], perror[-1]))
##        kerror_lsst, perror_lsst = numpy.loadtxt(optfolder + '/error_psup_lsst.txt', unpack=True)
##        ipkerror_lsst = interp1d(kerror_lsst, perror_lsst, bounds_error=False, fill_value=(perror_lsst[0], perror_lsst[-1]))
##    except:
##        kerror, perror = eval_bfit(data_n.mapp, fit_p.mapp, optfolder, noise=noise, title=title, fsize=15, suff='-noiseup')        
##        ipkerror = interp1d(kerror, perror, bounds_error=False, fill_value=(perror[0], perror[-1]))
##        if rank ==0: numpy.savetxt(optfolder + '/error_psnup.txt', numpy.array([kerror, perror]).T, header='kerror, perror')
##        kerror, perror = eval_bfit(data_p.mapp, fit_p.mapp, optfolder, noise=noise, title=title, fsize=15, suff='-up')        
##        if rank ==0: numpy.savetxt(optfolder + '/error_psup.txt', numpy.array([kerror, perror]).T, header='kerror, perror')
##        ipkmodel = interp1d(kerror, perror, bounds_error=False, fill_value=(perror[0], perror[-1]))
##        ivarmesh = truth_noise_model.get_ivarmesh(data_p, ipkmodel)
##        FieldMesh(ivarmesh).save(optfolder+'ivarmesh_up', dataset='ivar', mode='real')        
##        kerror_lsst, perror_lsst = eval_bfit(data_p.mapp2, fit_p.mapp2, optfolder, noise=noise, title=title, fsize=15, suff="-up_lsst")
##        if rank ==0: numpy.savetxt(optfolder + '/error_psup_lsst.txt', numpy.array([kerror_lsst, perror_lsst]).T, header='kerror, perror')
##        ipkerror_lsst = interp1d(kerror_lsst, perror_lsst, bounds_error=False, fill_value=(perror_lsst[0], perror_lsst[-1]))
##         
##else: 
##    ivarmesh = None
##    try:
##        kerror, perror = numpy.loadtxt(optfolder + '/error_psup.txt', unpack=True)
##        ipkmodel = interp1d(kerror, perror, bounds_error=False, fill_value=(perror[0], perror[-1]))
##        kerror, perror = numpy.loadtxt(optfolder + '/error_psnup.txt', unpack=True)
##        ipkerror = interp1d(kerror, perror, bounds_error=False, fill_value=(perror[0], perror[-1]))
##    except:
##        pkerror = FFTPower(data_p.mapp, second=-1* fit_p.mapp, mode='1d').power
##        kerror, perror = pkerror['k'], pkerror['power']
##        ipkmodel = interp1d(kerror, perror, bounds_error=False, fill_value=(perror[0], perror[-1]))
##        if rank ==0: numpy.savetxt(optfolder + '/error_psup.txt', numpy.array([kerror, perror]).T, header='kerror, perror')
##
    
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
            for iiter in range(100, -1, -10):
                path = outfolder + '/%d-%0.2f//%04d/fit_p/'%(nc*2, r, iiter)
                if os.path.isdir(path): 
                    inpath = path
                    sms = smoothings[:ir+1][::-1]
                    lit = maxiter0 - iiter
                    break
    if rank == 0: print('inpath: %s'%inpath)
    s_init = BigFileMesh(inpath, 's').paint()

except Exception as e:
    print('Exception :', e)
    try:
        x0 =  optfolder + '/%d-%0.2f/best-fit/'%(nc, 0)
        s_init = BigFileMesh(x0, 's').paint()
    except :
        x0 =  optfolder + '/%d-%0.2f-nit_0-sm_%0.2f/best-fit/'%(nc, 0, 0)
        s_init = BigFileMesh(x0, 's').paint()

    if rank == 0: print('Upsampling inint\n%s'%x0)
    s_init = new_pm.upsample(s_init, keep_mean=True)
    sms = smoothings[::-1]
    lit = maxiter0

if rank == 0 : print(inpath, lit)


N0 = nc*2
C = s_init.BoxSize[0] / s_init.Nmesh[0]
rtol = 0.005
maxiter = maxiter0
x0 = s_init
##Photoz smoothing
zz = 1/aa-1
sigz = lambda z : 120*((1+z)/5)**-0.5
if photosigma is None: photosigma = sigz(zz)

for Ns in sms:
    sml = C * Ns
    run = '%d-%0.2f'%(N0, Ns)
    maxiter = maxiter0
    if Ns == sms[0]:
        if inpath is not None:
            run += '-nit_%d-sm_%.2f'%(iiter, smoothings[ir])
            maxiter = lit
            
    if maxiter > 0:

#        obj = objfunc(mock_model, truth_noise_model, data_n, prior_ps=ipk, error_ps=ipkerror, sml=sml, kmin=kmin, angle=angle, ivarmesh=ivarmesh,
#                      shotnoise=1/lsstnumd, photosigma=photosigma, error_ps_lsst=ipkerror_lsst, lsstwt=lsstwt, h1wt=h1wt)
#
#        prior, chi2h1, chi2lsst, chi2 = obj.get_code().compute(['prior', 'chi2HI', 'chi2lsst', 'chi2'], init={'parameters': data_p.s})
#        if new_pm.comm.rank == 0:
#            print('\nprior, chi2h1, chi2lsst, chi2 at data.s \n',  "%.3e"%prior, "%.3e"%chi2h1, "%.3e"%chi2lsst, "%.3e"%chi2) # for 2d chi2 is close to total pixels.
#
#        prior, chi2h1, chi2lsst, chi2 = obj.get_code().compute(['prior', 'chi2HI', 'chi2lsst', 'chi2'], init={'parameters': x0})
#        if new_pm.comm.rank == 0:
#            print('\nprior, chi2h1, chi2lsst, chi2 at x0 \n',   "%.3e"%prior, "%.3e"%chi2h1, "%.3e"%chi2lsst, "%.3e"%chi2) # for 2d chi2 is close to total pixels.
#
#        x0 = solve(N0, x0, rtol, run, Ns, prefix, mock_model, obj, data_p, new_pm, optfolder, \
#                   saveit=10, showit=10, title=None, maxiter=maxiter, map2=True)    
#

        obj = objfunc(mock_model, truth_noise_model, data_n, prior_ps=ipk, error_ps=ipkerror, sml=sml, kmin=kmin, angle=angle, ivarmesh=ivarmesh,
                      shotnoise=1/lsstnumd, photosigma=photosigma, error_ps_lsst=ipkerror_lsst, lsstwt=lsstwt, h1wt=h1wt)
        
        prior, chi2h1, chi2lsst, chi2 = obj.get_code().compute(['prior', 'chi2HI', 'chi2lsst', 'chi2'], init={'parameters': data_p.s})
        if new_pm.comm.rank == 0:
            print('\nprior, chi2h1, chi2lsst, chi2 at data.s \n',  "%.3e"%prior, "%.3e"%chi2h1, "%.3e"%chi2lsst, "%.3e"%chi2) 

        prior, chi2h1, chi2lsst, chi2 = obj.get_code().compute(['prior', 'chi2HI', 'chi2lsst', 'chi2'], init={'parameters': x0})
        if new_pm.comm.rank == 0:
            print('\nprior, chi2h1, chi2lsst, chi2 at x0 \n',   "%.3e"%prior, "%.3e"%chi2h1, "%.3e"%chi2lsst, "%.3e"%chi2)  
            
        x0 = solve(N0, x0, rtol, run, Ns, prefix, mock_model, obj, data_p, new_pm, outfolder, \
                   saveit=50, showit=25, title=None, maxiter=maxiter, map2=True)    

