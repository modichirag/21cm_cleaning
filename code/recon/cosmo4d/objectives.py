
import numpy
#from . import base
from cosmo4d import base
from cosmo4d.engine import Literal
from cosmo4d.iotools import save_map, load_map
from nbodykit.lab import FieldMesh
import re, json, warnings



class Objective(base.Objective):
    def __init__(self, mock_model, noise_model, data, prior_ps, M0=0):
        self.prior_ps = prior_ps
        self.mock_model = mock_model
        self.noise_model = noise_model
        self.data = data
        self.pm = mock_model.pm
        self.engine = mock_model.engine
        self.M0 = M0

    def get_code(self):
        pass

##    def evaluate(self, model, data):
##      pass


###########################



def fingauss(pm, R):
    kny = numpy.pi*pm.Nmesh[0]/pm.BoxSize[0]
    def tf(k):
        k2 = sum(((2*kny/numpy.pi)*numpy.sin(ki*numpy.pi/(2*kny)))**2  for ki in k)
        wts = numpy.exp(-0.5*k2* R**2)
        return wts
    return tf            



class SmoothedObjective(Objective):
    """ The smoothed objecte smoothes the residual before computing chi2.
        It breaks the noise model at small scale, but the advantage is that
        the gradient in small scale is stronglly suppressed and we effectively
        only fit the large scale. Since we know usually the large scale converges
        very slowly this helps to stablize the solution.
    """
    def __init__(self, mock_model, noise_model, data, prior_ps, sml):
        Objective.__init__(self, mock_model, noise_model, data, prior_ps)
        self.sml = sml

    def get_code(self):
        import numpy
        pm = self.mock_model.pm

        code = self.mock_model.get_code()

        data = self.data.mapp

        code.add(x1='model', x2=Literal(data * -1), y='residual')
        if self.noise_model is not None: 
            code.multiply(x1='residual', x2=Literal(self.noise_model.ivar2d ** 0.5), y='residual')
        code.r2c(real='residual', complex='C')
        smooth_window = lambda k: numpy.exp(- self.sml ** 2 * sum(ki ** 2 for ki in k))
        code.transfer(complex='C', tf=smooth_window)
        code.c2r(real='residual', complex='C')
        code.to_scalar(x='residual', y='chi2')
        code.create_whitenoise(dlinear_k='dlinear_k', powerspectrum=self.prior_ps, whitenoise='pvar')
        code.to_scalar(x='pvar', y='prior')
        # the whitenoise is not properly normalized as d_k / P**0.5
        code.multiply(x1='prior', x2=Literal(pm.Nmesh.prod()**-1.), y='prior')
        code.add(x1='prior', x2='chi2', y='objective')
        return code




class SmoothedFourierObjective(Objective):
    """ The smoothed objecte smoothes the residual before computing chi2.
        It breaks the noise model at small scale, but the advantage is that
        the gradient in small scale is stronglly suppressed and we effectively
        only fit the large scale. Since we know usually the large scale converges
        very slowly this helps to stablize the solution.
    """
    def __init__(self, mock_model, noise_model, data, prior_ps, error_ps, sml, ivarmesh=None):
        Objective.__init__(self, mock_model, noise_model, data, prior_ps)
        self.sml = sml
        self.error_ps = error_ps
        self.ivarmesh = ivarmesh
        if self.ivarmesh is not None: self.ivarmeshc = self.ivarmesh.r2c()

    def get_code(self):
        import numpy
        pm = self.mock_model.pm

        code = self.mock_model.get_code()

        data = self.data.mapp

        code.add(x1='model', x2=Literal(data * -1), y='residual')
        code.r2c(real='residual', complex='C')
        smooth_window = lambda k: numpy.exp(- self.sml ** 2 * sum(ki ** 2 for ki in k))
        code.transfer(complex='C', tf=smooth_window)
        if self.ivarmesh is None: code.create_whitenoise(dlinear_k='C', powerspectrum=self.error_ps, whitenoise='perror')
        else: 
            if pm.comm.rank == 0: print('Using ivarmesh')
            code.multiply(x1='C', x2=Literal(self.ivarmeshc**0.5), y='perrorc')
            code.c2r(complex='perrorc', real='perror')

        code.to_scalar(x='perror', y='chi2')
        #code.c2r(real='residual', complex='C')
        #code.to_scalar(x='residual', y='chi2')
        code.create_whitenoise(dlinear_k='dlinear_k', powerspectrum=self.prior_ps, whitenoise='pvar')
        code.to_scalar(x='pvar', y='prior')
        # the whitenoise is not properly normalized as d_k / P**0.5
        code.multiply(x1='prior', x2=Literal(pm.Nmesh.prod()**-1.), y='prior')
        code.add(x1='prior', x2='chi2', y='objective')
        return code





class SmoothedFourierWedgeObjective(Objective):
    """ The smoothed objecte smoothes the residual before computing chi2.
        It breaks the noise model at small scale, but the advantage is that
        the gradient in small scale is stronglly suppressed and we effectively
        only fit the large scale. Since we know usually the large scale converges
        very slowly this helps to stablize the solution.
    """
    def __init__(self, mock_model, noise_model, data, prior_ps, error_ps, sml, kmin, angle, ivarmesh=None):
        Objective.__init__(self, mock_model, noise_model, data, prior_ps)
        self.sml = sml
        self.error_ps = error_ps
        self.kmin = kmin
        self.angle = angle
        self.ivarmesh = ivarmesh
        if ivarmesh is not None: self.ivarmeshc = self.ivarmesh.r2c()

    def get_code(self):
        import numpy
        pm = self.mock_model.pm
        code = self.mock_model.get_code()
        data = self.data.mapp

        code.add(x1='model', x2=Literal(data * -1), y='residual')
        code.r2c(real='residual', complex='C')
        smooth_window = lambda k: numpy.exp(- self.sml ** 2 * sum(ki ** 2 for ki in k))
        code.transfer(complex='C', tf=smooth_window)
        #code.create_whitenoise(dlinear_k='C', powerspectrum=self.error_ps, whitenoise='perror')
        if self.ivarmesh is None: code.create_whitenoise(dlinear_k='C', powerspectrum=self.error_ps, whitenoise='perror')
        else: 
            if pm.comm.rank == 0: print('Using ivarmesh')
            code.multiply(x1='C', x2=Literal(self.ivarmeshc**0.5), y='perrorc')
            code.c2r(complex='perrorc', real='perror')

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

        code.r2c(real='perror', complex='perrorc')
        code.transfer(complex='perrorc', tf=tf)
        code.c2r(complex='perrorc', real='perror')

        code.to_scalar(x='perror', y='chi2')
        #code.c2r(real='residual', complex='C')
        #code.to_scalar(x='residual', y='chi2')
        code.create_whitenoise(dlinear_k='dlinear_k', powerspectrum=self.prior_ps, whitenoise='pvar')
        code.to_scalar(x='pvar', y='prior')
        # the whitenoise is not properly normalized as d_k / P**0.5
        code.multiply(x1='prior', x2=Literal(pm.Nmesh.prod()**-1.), y='prior')
        code.add(x1='prior', x2='chi2', y='objective')
        return code










class SmoothedFourierWedgeCrossObjective(Objective):
    """ The smoothed objecte smoothes the residual before computing chi2.
        It breaks the noise model at small scale, but the advantage is that
        the gradient in small scale is stronglly suppressed and we effectively
        only fit the large scale. Since we know usually the large scale converges
        very slowly this helps to stablize the solution.
    """
    def __init__(self, mock_model, noise_model, data, prior_ps, error_ps, sml, kmin, angle, ivarmesh=None):
        Objective.__init__(self, mock_model, noise_model, data, prior_ps)
        self.sml = sml
        self.error_ps = error_ps
        self.kmin = kmin
        self.angle = angle
        self.ivarmesh = ivarmesh
        if ivarmesh is not None: self.ivarmeshc = self.ivarmesh.r2c()

    def get_code(self):
        import numpy
        pm = self.mock_model.pm
        code = self.mock_model.get_code()
        data = self.data.mapp

        code.add(x1='model', x2=Literal(data * -1), y='residual')
        code.r2c(real='residual', complex='C')
        smooth_window = lambda k: numpy.exp(- self.sml ** 2 * sum(ki ** 2 for ki in k))
        code.transfer(complex='C', tf=smooth_window)
        #code.create_whitenoise(dlinear_k='C', powerspectrum=self.error_ps, whitenoise='perror')
        if self.ivarmesh is None: code.create_whitenoise(dlinear_k='C', powerspectrum=self.error_ps, whitenoise='perror')
        else: 
            if pm.comm.rank == 0: print('Using ivarmesh')
            code.multiply(x1='C', x2=Literal(self.ivarmeshc**0.5), y='perrorc')
            code.c2r(complex='perrorc', real='perror')

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

        code.r2c(real='perror', complex='perrorc')
        code.transfer(complex='perrorc', tf=tf)
        code.c2r(complex='perrorc', real='perror')

        code.to_scalar(x='perror', y='chi2')
        #code.c2r(real='residual', complex='C')
        #code.to_scalar(x='residual', y='chi2')
        code.create_whitenoise(dlinear_k='dlinear_k', powerspectrum=self.prior_ps, whitenoise='pvar')
        code.to_scalar(x='pvar', y='prior')
        # the whitenoise is not properly normalized as d_k / P**0.5
        code.multiply(x1='prior', x2=Literal(pm.Nmesh.prod()**-1.), y='prior')
        code.add(x1='prior', x2='chi2', y='objective')
        return code
    




class SmoothedFourierWedgeLSSTObjective(Objective):
    """ The smoothed objecte smoothes the residual before computing chi2.
        It breaks the noise model at small scale, but the advantage is that
        the gradient in small scale is stronglly suppressed and we effectively
        only fit the large scale. Since we know usually the large scale converges
        very slowly this helps to stablize the solution.
/    """
    def __init__(self, mock_model, noise_model, data, prior_ps, error_ps, sml, kmin, angle,
                 datalsst=None, ivarmesh=None, shotnoise=0, photosigma=0, error_ps_lsst=None, lsstwt=1., h1wt=1.):
        
        Objective.__init__(self, mock_model, noise_model, data, prior_ps)
        self.sml = sml
        self.error_ps = error_ps
        self.kmin = kmin
        self.angle = angle
        self.ivarmesh = ivarmesh
        if ivarmesh is not None: self.ivarmeshc = self.ivarmesh.r2c().real
        
        #self.ivarmesh = None
        #lsst params
        self.photosigma = photosigma
        if error_ps_lsst is None: self.error_ps_lsst = lambda x: error_ps(x) + shotnoise
        else: self.error_ps_lsst = lambda x: error_ps_lsst(x) + shotnoise
        self.datalsst = datalsst
        self.lsstwt = lsstwt
        self.h1wt = h1wt
        if not self.lsstwt : print("No lsst ") 
        
    def get_code(self):
        import numpy
        pm = self.mock_model.pm
        code = self.mock_model.get_code()
        data = self.data.mapp

        code.add(x1='model', x2=Literal(data * -1), y='residual')
        code.r2c(real='residual', complex='C')
        smooth_window = lambda k: numpy.exp(- self.sml ** 2 * sum(ki ** 2 for ki in k))
        code.transfer(complex='C', tf=smooth_window)
        #code.create_whitenoise(dlinear_k='C', powerspectrum=self.error_ps, whitenoise='perror')
        if self.ivarmesh is None:
            code.create_whitenoise(dlinear_k='C', powerspectrum=self.error_ps, whitenoise='perror')
        else: 
            if pm.comm.rank == 0: print('Using ivarmesh')
            code.multiply(x1='C', x2=Literal(self.ivarmeshc**0.5), y='perrorc')
            code.c2r(complex='perrorc', real='perror')


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

        code.r2c(real='perror', complex='perrorc')
        code.transfer(complex='perrorc', tf=tf)
        code.c2r(complex='perrorc', real='perror')
        code.to_scalar(x='perror', y='chi2HI')
        code.multiply(x1='chi2HI', x2=Literal(self.h1wt), y='chi2HI')

        #add LSST data
        if self.datalsst is None: code.add(x1='model2', x2=Literal(self.data.mapp2 * -1), y='residuallsst')
        else: code.add(x1='model2', x2=Literal(self.datalsst.mapp * -1), y='residuallsst')

        ##
        def phototf(k): #Photoz smoothing
            kmesh = sum(ki ** 2 for ki in k)**0.5
            kmesh[kmesh == 0] = 1
            mumesh = k[2]/(kmesh + 1e-10)
            weights = numpy.exp(-0.5 * kmesh**2 * mumesh**2 * self.photosigma**2)
            return weights
        
        if pm.comm.rank == 0: print('\nsmoothing for photoz with smoothing = ', self.photosigma)

        code.r2c(real='residuallsst', complex='residuallsstc')
        code.transfer(complex='residuallsstc', tf=phototf)
        smooth_window = lambda k: numpy.exp(- self.sml ** 2 * sum(ki ** 2 for ki in k))
        code.transfer(complex='residuallsstc', tf=smooth_window)
        code.create_whitenoise(dlinear_k='residuallsstc', powerspectrum=self.error_ps_lsst, whitenoise='perrorlsst')
        #code.multiply(x1='residuallsstc', x2=Literal(self.ivarmeshc**0.5), y='perrorclsst')
        #code.c2r(complex='perrorclsst', real='perrorlsst')


        code.to_scalar(x='perrorlsst', y='chi2lsst')
        code.multiply(x1='chi2lsst', x2=Literal(self.lsstwt), y='chi2lsst')
        
        code.create_whitenoise(dlinear_k='dlinear_k', powerspectrum=self.prior_ps, whitenoise='pvar')
        code.to_scalar(x='pvar', y='prior')
        # the whitenoise is not properly normalized as d_k / P**0.5
        #In v2 I think I am disabling this normalization since it is waaaay downweighted
        code.multiply(x1='prior', x2=Literal(pm.Nmesh.prod()**-1.), y='prior')

        code.add(x1='chi2HI', x2='chi2lsst', y='chi2')
        code.add(x1='prior', x2='chi2', y='objective')
        #code.add(x1='prior', x2='chi2HI', y='objective')
        return code



    
class SmoothedFourierWedgeLSSTObjectiveZANoise(Objective):
    """ The smoothed objecte smoothes the residual before computing chi2.
        It breaks the noise model at small scale, but the advantage is that
        the gradient in small scale is stronglly suppressed and we effectively
        only fit the large scale. Since we know usually the large scale converges
        very slowly this helps to stablize the solution.
/    """
    def __init__(self, mock_model, noise_model, data, prior_ps, error_ps, sml, kmin, angle,
                 datalsst=None, ivarmesh=None, shotnoise=0, photosigma=0, error_ps_lsst=None, lsstwt=1., h1wt=1., knl=1.):
        
        Objective.__init__(self, mock_model, noise_model, data, prior_ps)
        self.sml = sml
        self.error_ps = error_ps
        self.kmin = kmin
        self.angle = angle
        self.ivarmesh = ivarmesh
        self.knl = knl
        if ivarmesh is not None: self.ivarmeshc = self.ivarmesh.r2c().real
        
        #self.ivarmesh = None
        #lsst params
        self.photosigma = photosigma
        if error_ps_lsst is None: self.error_ps_lsst = lambda x: error_ps(x) + shotnoise
        else: self.error_ps_lsst = lambda x: error_ps_lsst(x) + shotnoise
        self.datalsst = datalsst
        self.lsstwt = lsstwt
        self.h1wt = h1wt
        if not self.lsstwt : print("No lsst ")
        
        
    def get_code(self):
        import numpy
        pm = self.mock_model.pm
        code = self.mock_model.get_code()
        data = self.data.mapp

        code.add(x1='model', x2=Literal(data * -1), y='residual')
        code.r2c(real='residual', complex='C')
        smooth_window = lambda k: numpy.exp(- self.sml ** 2 * sum(ki ** 2 for ki in k))
        smooth_windowZA = lambda k: numpy.exp(- self.knl ** -2 * sum(ki ** 2 for ki in k))
        code.transfer(complex='C', tf=smooth_window)
        code.transfer(complex='C', tf=smooth_windowZA)
        #code.create_whitenoise(dlinear_k='C', powerspectrum=self.error_ps, whitenoise='perror')
        if self.ivarmesh is None:
            code.create_whitenoise(dlinear_k='C', powerspectrum=self.error_ps, whitenoise='perror')
        else: 
            if pm.comm.rank == 0: print('Using ivarmesh')
            code.multiply(x1='C', x2=Literal(self.ivarmeshc**0.5), y='perrorc')
            code.c2r(complex='perrorc', real='perror')


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

        code.r2c(real='perror', complex='perrorc')
        code.transfer(complex='perrorc', tf=tf)
        code.c2r(complex='perrorc', real='perror')
        code.to_scalar(x='perror', y='chi2HI')
        code.multiply(x1='chi2HI', x2=Literal(self.h1wt), y='chi2HI')

        #add LSST data
        if self.datalsst is None: code.add(x1='model2', x2=Literal(self.data.mapp2 * -1), y='residuallsst')
        else: code.add(x1='model2', x2=Literal(self.datalsst.mapp * -1), y='residuallsst')

        ##
        def phototf(k): #Photoz smoothing
            kmesh = sum(ki ** 2 for ki in k)**0.5
            kmesh[kmesh == 0] = 1
            mumesh = k[2]/(kmesh + 1e-10)
            weights = numpy.exp(-0.5 * kmesh**2 * mumesh**2 * self.photosigma**2)
            return weights
        
        if pm.comm.rank == 0: print('\nsmoothing for photoz with smoothing = ', self.photosigma)

        code.r2c(real='residuallsst', complex='residuallsstc')
        code.transfer(complex='residuallsstc', tf=phototf)
        smooth_window = lambda k: numpy.exp(- self.sml ** 2 * sum(ki ** 2 for ki in k))
        smooth_windowZA = lambda k: numpy.exp(- self.knl ** -2 * sum(ki ** 2 for ki in k))
        code.transfer(complex='residuallsstc', tf=smooth_window)
        code.transfer(complex='residuallsstc', tf=smooth_windowZA)
        code.create_whitenoise(dlinear_k='residuallsstc', powerspectrum=self.error_ps_lsst, whitenoise='perrorlsst')
        #code.multiply(x1='residuallsstc', x2=Literal(self.ivarmeshc**0.5), y='perrorclsst')
        #code.c2r(complex='perrorclsst', real='perrorlsst')
        code.to_scalar(x='perrorlsst', y='chi2lsst')
        code.multiply(x1='chi2lsst', x2=Literal(self.lsstwt), y='chi2lsst')
        
        code.create_whitenoise(dlinear_k='dlinear_k', powerspectrum=self.prior_ps, whitenoise='pvar')
        code.to_scalar(x='pvar', y='prior')
        # the whitenoise is not properly normalized as d_k / P**0.5
        #In v2 I think I am disabling this normalization since it is waaaay downweighted
        code.multiply(x1='prior', x2=Literal(pm.Nmesh.prod()**-1.), y='prior')

        code.add(x1='chi2HI', x2='chi2lsst', y='chi2')
        code.add(x1='prior', x2='chi2', y='objective')
        #code.add(x1='prior', x2='chi2HI', y='objective')
        return code



    

class SmoothedFourierWedgeLSSTObjectiveLowpass(Objective):
    """ The smoothed objecte smoothes the residual before computing chi2.
        It breaks the noise model at small scale, but the advantage is that
        the gradient in small scale is stronglly suppressed and we effectively
        only fit the large scale. Since we know usually the large scale converges
        very slowly this helps to stablize the solution.
/    """
    def __init__(self, mock_model, noise_model, data, prior_ps, error_ps, sml, kmin, angle,
                 datalsst=None, ivarmesh=None, shotnoise=0, photosigma=0, error_ps_lsst=None, lsstwt=1., h1wt=1., smlmax=None):
        
        Objective.__init__(self, mock_model, noise_model, data, prior_ps)
        self.sml = sml
        self.smlmax = smlmax
        self.error_ps = error_ps
        self.kmin = kmin
        self.angle = angle
        self.ivarmesh = ivarmesh
        if ivarmesh is not None: self.ivarmeshc = self.ivarmesh.r2c().real
        
        #self.ivarmesh = None
        #lsst params
        self.photosigma = photosigma
        if error_ps_lsst is None: self.error_ps_lsst = lambda x: error_ps(x) + shotnoise
        else: self.error_ps_lsst = lambda x: error_ps_lsst(x) + shotnoise
        self.datalsst = datalsst
        self.lsstwt = lsstwt
        self.h1wt = h1wt
        if not self.lsstwt : print("No lsst ") 
        
    def get_code(self):
        import numpy
        pm = self.mock_model.pm
        code = self.mock_model.get_code()
        data = self.data.mapp

        code.add(x1='model', x2=Literal(data * -1), y='residual')
        code.r2c(real='residual', complex='C')
        smooth_window = lambda k: numpy.exp(- self.sml ** 2 * sum(ki ** 2 for ki in k))
        code.transfer(complex='C', tf=smooth_window)
        #
        #code.create_whitenoise(dlinear_k='C', powerspectrum=self.error_ps, whitenoise='perror')
        if self.ivarmesh is None:
            code.create_whitenoise(dlinear_k='C', powerspectrum=self.error_ps, whitenoise='perror')
        else: 
            if pm.comm.rank == 0: print('Using ivarmesh')
            code.multiply(x1='C', x2=Literal(self.ivarmeshc**0.5), y='perrorc')
            code.c2r(complex='perrorc', real='perror')


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

        code.r2c(real='perror', complex='perrorc')
        code.transfer(complex='perrorc', tf=tf)
        code.c2r(complex='perrorc', real='perror')
        code.multiply(x1='perror', x2=Literal(self.h1wt), y='perror')
        code.to_scalar(x='perror', y='chi2HI')
        #code.multiply(x1='chi2HI', x2=Literal(self.h1wt), y='chi2HI')

        #add LSST data
        if self.datalsst is None: code.add(x1='model2', x2=Literal(self.data.mapp2 * -1), y='residuallsst')
        else: code.add(x1='model2', x2=Literal(self.datalsst.mapp * -1), y='residuallsst')

        ##
        def phototf(k): #Photoz smoothing
            kmesh = sum(ki ** 2 for ki in k)**0.5
            kmesh[kmesh == 0] = 1
            mumesh = k[2]/(kmesh + 1e-10)
            weights = numpy.exp(-0.5 * kmesh**2 * mumesh**2 * self.photosigma**2)
            return weights
        
        if pm.comm.rank == 0: print('\nsmoothing for photoz with smoothing = ', self.photosigma)

        code.r2c(real='residuallsst', complex='residuallsstc')
        code.transfer(complex='residuallsstc', tf=phototf)
        smooth_window = lambda k: numpy.exp(- self.sml ** 2 * sum(ki ** 2 for ki in k))
        code.transfer(complex='residuallsstc', tf=smooth_window)
        code.create_whitenoise(dlinear_k='residuallsstc', powerspectrum=self.error_ps_lsst, whitenoise='perrorlsst')
        #code.multiply(x1='residuallsstc', x2=Literal(self.ivarmeshc**0.5), y='perrorclsst')
        #code.c2r(complex='perrorclsst', real='perrorlsst')


        code.multiply(x1='perrorlsst', x2=Literal(self.lsstwt), y='perrorlsst')
        code.to_scalar(x='perrorlsst', y='chi2lsst')
        #code.multiply(x1='chi2lsst', x2=Literal(self.lsstwt), y='chi2lsst')
        
        code.create_whitenoise(dlinear_k='dlinear_k', powerspectrum=self.prior_ps, whitenoise='pvar')
        code.multiply(x1='pvar', x2=Literal(pm.Nmesh.prod()**-1.), y='pvar')
        code.to_scalar(x='pvar', y='prior')
        # the whitenoise is not properly normalized as d_k / P**0.5
        #In v2 I think I am disabling this normalization since it is waaaay downweighted
        #code.multiply(x1='prior', x2=Literal(pm.Nmesh.prod()**-1.), y='prior')

        code.add(x1='chi2HI', x2='chi2lsst', y='chi2')
        #code.add(x1='prior', x2='chi2', y='objective')
        #code.add(x1='prior', x2='chi2HI', y='objective')

        code.add(x1='perror', x2='perrorlsst', y='perrorall')
        code.add(x1='pvar', x2='perrorall', y='loss')

        #
        low_pass_window = lambda k: 1 - numpy.exp(- (self.sml + self.smlmax) ** 2 * sum(ki ** 2 for ki in k))
        lowpass = False
        if self.smlmax is not None:
            if self.sml < self.smlmax:
                lowpass = True
                if pm.comm.rank == 0: print('\nLow pass filtering\n')
                
        if lowpass :
            code.r2c(real='loss', complex='lossC')
            code.transfer(complex='lossC', tf=low_pass_window)
            code.c2r(complex='lossC', real='loss')

        code.to_scalar(x='loss', y='objective')
        
        return code


