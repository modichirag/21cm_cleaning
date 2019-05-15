import numpy
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from cosmo4d.lab import (UseComplexSpaceOptimizer,
                        ZAModel,
                        LBFGS, ParticleMesh)

from cosmo4d.lab import mapfinal as map
from cosmo4d.lab import dg

from abopt.algs.lbfgs import scalar as scalar_diag

from nbodykit.cosmology import Planck15, EHPower, Cosmology
from nbodykit.algorithms.fof import FOF
from nbodykit.lab import KDDensity, BigFileMesh, BigFileCatalog, ArrayCatalog
import os, json

#########################################

#Set parameters here
bs, nc = 256., 128
truth_pm = ParticleMesh(BoxSize=bs, Nmesh=(nc, nc, nc), dtype='f4')
rank = truth_pm.comm.rank
nsteps = 5
aa = 0.2000
B = 1

noisevar = 0.01
smooth = None


#
proj = '/project/projectdirs/cosmosim/lbl/chmodi/cosmo4d/'
dfolder = '/global/cscratch1/sd/chmodi/m3127/cm_lowres/%dstepT-B%d/%d-%d-9100-fixed/'%(nsteps, B, bs, nc)

ofolder = '/global/cscratch1/sd/chmodi/m3127/21cm_cleaning/recon/'
prefix = 'za'
fname = 's999_%s'%prefix
basefolder = ofolder + 'opt_%s/'%fname
optfolder = basefolder 
if truth_pm.comm.rank == 0:
    print('Output Folder is %s'%optfolder)


for folder in [ofolder, optfolder]:
    try:
        os.makedirs(folder)
    except:
        pass


#initiate

klin, plin = numpy.loadtxt('../../data/pklin_1.0000.txt', unpack = True)
pk = interpolate(klin, plin)
#cosmo = Planck15.clone(Omega_cdm = 0.2685, h = 0.6711, Omega_b = 0.049)
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)
#########################################
#dynamics
stages = numpy.linspace(0.1, aa, nsteps, endpoint=True)
dynamic_model = ZAModel(cosmo, truth_pm, B=B, steps=stages)

#noise
#Artifically low noise since the data is constructed from the model
truth_noise_model = map.NoiseModel(truth_pm, None, noisevar*(truth_pm.BoxSize/truth_pm.Nmesh).prod(), 1234)
mock_model = map.MockModel(dynamic_model)

#Create and save data if not found

dyn = BigFileCatalog(dfolder + 'fastpm_%0.4f/1'%aa)
s_truth = BigFileMesh(dfolder + 'linear', 'LinearDensityK').paint()
data_p = mock_model.make_observable(s_truth)
layout = truth_pm.decompose(dyn['Position'])
#data_p.mapp = truth_pm.paint(dyn['Position'], layout=layout)
data_p.save(optfolder+'datap/')


data_n = truth_noise_model.add_noise(data_p)
data_n.save(optfolder+'datan/')

fit_p = mock_model.make_observable(s_truth)
fit_p.save(optfolder+'fitp/')

s_init = truth_pm.generate_whitenoise(777, mode='complex')\
        .apply(lambda k, v: v * (pk(sum(ki **2 for ki in k) **0.5) / v.BoxSize.prod()) ** 0.5)\
        .c2r()*0.001
##
#s_init = BigFileMesh(finfolder, 's').paint()


if rank == 0: print('data_p, data_n created')

#########################################
#optimizer

def solve(Nmesh, x0, rtol, run, Nsm):
    
    pm = truth_pm.resize(Nmesh=(Nmesh, Nmesh, Nmesh))
    atol = pm.Nmesh.prod() * rtol
    x0 = pm.upsample(x0, keep_mean=True)
    #data = data_n.downsample(pm)
    #IDEAL no noise limit
    data = data_p.downsample(pm)
 
    # smooth the data. This breaks the noise model but we don't need it
    # for lower resolution anyways.
    sml = pm.BoxSize[0] / Nmesh * Nsm
 
    dynamic_model = ZAModel(cosmo, truth_pm, B=B, steps=stages)
    mock_model = map.MockModel(dynamic_model)
    
    # an approximate noise model, due to smoothing this is correct only at large scale.
    noise_model = truth_noise_model.downsample(pm)
 
    obj = map.SmoothedObjective(mock_model, noise_model, data, prior_ps=pk, sml=sml)#, noised=noised)
 
    prior, chi2 = obj.get_code().compute(['prior', 'chi2'], init={'parameters': data.s})
    if pm.comm.rank == 0:
        print(prior, chi2) # for 2d chi2 is close to total pixels.
 
    fit_p = mock_model.make_observable(data.s)
    r = obj.evaluate(fit_p, data)
 
    try:
        os.makedirs(optfolder + '%s' % run)
    except:
        pass
    try:
        os.makedirs(optfolder + '%s/2pt' % run)
    except:
        pass
    obj.save_report(r, optfolder + "%s/truth.png" % run)
    dg.save_2ptreport(r, optfolder + "%s/2pt/truth.png" % run, pm)
 

    optimizer = LBFGS(m=10, diag_update=scalar_diag)
 
    prob = obj.get_problem(atol=atol, precond=UseComplexSpaceOptimizer)
 
    def monitor(state):
        if pm.comm.rank == 0:
            print(state)
        if state.nit % 5 == 0:
            fit_p = mock_model.make_observable(state['x'])
            if state.nit % 20 == 0:
                fit_p.save(optfolder + '%s/%04d/fit_p' % (run, state['nit']))
            r = obj.evaluate(fit_p, data)
            #obj.save_report(r, optfolder + "%s/%s%02d-%04d.png"% (run, prefix, int(Nsm*10), state['nit']))
            dg.save_report(r, optfolder + "%s/%s_N%02d-%04d.png"% (run, prefix, int(Nsm*10),  state['nit']), pm)
            dg.save_2ptreport(r, optfolder + "%s/2pt/%s_N%02d-%04d.png"% (run, prefix, int(Nsm*10), state['nit']), pm)
            if pm.comm.rank == 0:
                print('saved')
 
    state = optimizer.minimize(prob, x0=x0, monitor=monitor)
    fit_p = mock_model.make_observable(state['x'])
    fit_p.save(optfolder + '%s/best-fit' % run)
    r = obj.evaluate(fit_p, data)
    obj.save_report(r, optfolder + "%s/%s%02d-best-fit.png" % (run, prefix, int(Nsm*10)))
    dg.save_2ptreport(r, optfolder + "%s/2pt/%s_N%02d-best-fit.png" % (run, prefix, int(Nsm*10)), pm)
    return state.x



#Optimizer


    
def gaussian_smoothing(sm):
    def kernel(k, v):
        return numpy.exp(- 0.5 * sm ** 2 * sum(ki ** 2 for ki in k)) * v
    return kernel
  

x0 = s_init
N0 = nc
C = x0.BoxSize[0] / x0.Nmesh[0]


for Ns in [4.0, 2.0, 1.0, 0.5, 0.]:
    x0 = solve(N0, x0, 0.005, '%d-%0.2f'%(N0, Ns), Ns)
    if truth_pm.comm.rank == 0:
        print('Do for cell smoothing of %0.2f'%(Ns))
    
