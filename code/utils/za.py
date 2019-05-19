import numpy
import numpy as np


def laplace(k, v):
    kk = sum(ki ** 2 for ki in k)
    mask = (kk == 0).nonzero()
    kk[mask] = 1
    b = v / kk
    b[mask] = 0
    return b



def gradient(dir, order=1):
    if order == 0:
        def kernel(k, v):
            # clear the nyquist to ensure field is real                         
            mask = v.i[dir] != v.Nmesh[dir] // 2
            return v * (1j * k[dir]) * mask
    if order == 1:
        def kernel(k, v):
            cellsize = (v.BoxSize[dir] / v.Nmesh[dir])
            w = k[dir] * cellsize

            a = 1 / (6.0 * cellsize) * (8 * numpy.sin(w) - numpy.sin(2 * w))
            # a is already zero at the nyquist to ensure field is real          
            return v * (1j * a)
    return kernel



def doza(dlin_k, q, resampler='cic', z=0, displacement=False, dgrow=1):
    """ Run first order LPT on linear density field, returns displacements of particles            
        reading out at q. The result has the same dtype as q.                                      
    """
    basepm = dlin_k.pm

    ndim = len(basepm.Nmesh)
    delta_k = basepm.create('complex')
    
    source = numpy.zeros((len(q), ndim), dtype=q.dtype)
    for d in range(len(basepm.Nmesh)):
        disp = dlin_k.apply(laplace) \
                    .apply(gradient(d), out=Ellipsis) \
                    .c2r(out=Ellipsis)
        source[..., d] = disp.readout(q, resampler=resampler)*dgrow
        
    if displacement: return source
    pos = q + source
    #pos[pos < 0] += bs
    #pos[pos > bs] -= bs
    return pos

