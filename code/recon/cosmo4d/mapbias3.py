import numpy
from . import base
from .engine import Literal
from .iotools import save_map, load_map
from nbodykit.lab import FieldMesh


class Observable(base.Observable):
    def __init__(self, mapp, d, s, mapp2=None):
        self.mapp = mapp
        self.s = s
        self.d = d
        self.mapp2 = mapp2

    def save(self, path):
        if self.mapp2 is not None: save_map(self.mapp2, path, 'mapp2')
        save_map(self.mapp, path, 'mapp')
        save_map(self.s, path, 's')
        save_map(self.d, path, 'd')

    @classmethod
    def load(kls, path):
        try:
            return Observable(load_map(path, 'mapp'),
                              load_map(path, 'd'),
                              load_map(path, 's'),
                              load_map(path, 'mapp2'))
    
        except Exception as e:
            #print("Exception in load Observable : ", e)
            return Observable(load_map(path, 'mapp'),
                          load_map(path, 'd'),
                          load_map(path, 's'))
    
    def downsample(self, pm):
        return Observable(
                pm.downsample(self.mapp, resampler='nearest', keep_mean=True),
                pm.downsample(self.d, resampler='nearest', keep_mean=True),
                pm.downsample(self.s, resampler='nearest', keep_mean=True),
        )




class MockModel(base.MockModel):
    def __init__(self, dynamic_model, params=[1., 0., 0.], params2 = [1.0, 0., 0.], rsdpos=False, rsdfac=0, smoothing=4.):
        self.dynamic_model = dynamic_model
        self.pm = dynamic_model.pm
        self.engine = dynamic_model.engine
        self.b1, self.b2, self.bs2 = params
        self.b12, self.b22, self.bs22 = params2
        self.nc = self.pm.Nmesh[0]
        self.ncsize = self.nc**3
        self.rsdpos = rsdpos
        self.rsdfac = rsdfac
        if self.pm.comm.rank == 0: print('\nBias params for HI : ', self.b1, self.b2, self.bs2)
        if self.pm.comm.rank == 0: print('\nBias params for LSST : ', self.b12, self.b22, self.bs22)
        self.sml = smoothing
        
    def get_code(self):
        code = base.MockModel.get_code(self)

        code.c2r(real='linear', complex='dlinear_k')
        #
        code.r2c(real='linear', complex='C')
        smooth_window = lambda k: numpy.exp(- self.sml ** 2 * sum(ki ** 2 for ki in k))
        code.transfer(complex='C', tf=smooth_window)
        code.c2r(real='linearsm', complex='C')
        #gen. lag. mesh
        code.generate_shear(source='linearsm', shear='shear')
        code.multiply(x1='linearsm', x2='linearsm', y='linearsq')
        code.multiply(x1='linearsq', x2=Literal(-1.), y='neglsq')
        code.add(x1='shear', x2='neglsq', y='shear')

        #subtract means
        code.sum(x='linearsq', y='sum')
        code.multiply(x1='sum', x2=Literal(-1./self.ncsize), y='negmean')
        code.add(x1='linearsq', x2='negmean', y='linearsq')

        code.sum(x='shear', y='sum')
        code.multiply(x1='sum', x2=Literal(-1./self.ncsize), y='negmean')
        code.add(x1='shear', x2='negmean', y='shear')

        #readout at lag. pos.
        qq = self.engine.q
        qlayout = self.engine.pm.decompose(qq)
        code.readout(x=Literal(qq), mesh='linear', value='d0', layout=Literal(qlayout), resampler='nearest')
        code.readout(x=Literal(qq), mesh='linearsq', value='d2', layout=Literal(qlayout), resampler='nearest')
        code.readout(x=Literal(qq), mesh='shear', value='ds2', layout=Literal(qlayout), resampler='nearest')
        code.reshape_scalar(x='d0', y='d0')
        code.reshape_scalar(x='d2', y='d2')
        code.reshape_scalar(x='ds2', y='ds2')

        #paint at shifted pos.
        code.assign(x='x', y='xp')
        if self.rsdpos:
            los = numpy.array([0, 0, self.rsdfac]).reshape(1, -1)
            code.multiply(x1='v', x2=Literal(los), y='xv')
            code.add(x1='xp', x2='xv', y='xp')
        code.decompose(x='xp', layout='xplayout')
        code.paint(x='xp', mesh='ed0', layout='xplayout', mass='d0')
        code.paint(x='xp', mesh='ed2', layout='xplayout', mass='d2')
        code.paint(x='xp', mesh='es2', layout='xplayout', mass='ds2')
        
        #multiply by bias
        code.multiply(x1=Literal(self.b1), x2='ed0', y='b1ed0')
        code.multiply(x1=Literal(self.b2), x2='ed2', y='b1ed2')
        code.multiply(x1=Literal(self.bs2), x2='es2', y='b1es2')

        #add to get model
        code.add(x1='b1ed0', x2='b1ed2', y='model')
        code.add(x1='model', x2='b1es2', y='model')

        #Repeat for the other observable
        code.multiply(x1=Literal(self.b12), x2='ed0', y='b1ed02')
        code.multiply(x1=Literal(self.b22), x2='ed2', y='b1ed22')
        code.multiply(x1=Literal(self.bs22), x2='es2', y='b1es22')

        #add to get model
        code.add(x1='b1ed02', x2='b1ed22', y='model2')
        code.add(x1='model2', x2='b1es22', y='model2')
        #code.assign(x='model', y='model2')

        return code

    def make_observable(self, initial):
        code = self.get_code()
        model2, model, final = code.compute(['model2', 'model', 'final'], init={'parameters':initial})
        return Observable(mapp=model, s=initial, d=final, mapp2=model2)



class DataModel(base.MockModel):
    def __init__(self, dynamic_model):
        self.dynamic_model = dynamic_model
        self.pm = dynamic_model.pm
        self.engine = dynamic_model.engine

    def get_code(self):
        code = base.MockModel.get_code(self)
        code.assign(x='final', y='model')
        return code

    def make_observable(self, initial):
        code = self.get_code()
        model, final = code.compute(['model', 'final'], init={'parameters':initial})
        return Observable(mapp=model, s=initial, d=final)




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
