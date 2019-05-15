import numpy as np
from nbodykit.utils import DistributedArray
from nbodykit.lab import BigFileCatalog, MultipleSpeciesCatalog

class ModelA():
    
    def __init__(self, aa):

        self.aa = aa
        self.zz = 1/aa-1

        self.alp = (1+2*self.zz)/(2+2*self.zz)
        #self.mcut = 1e9*( 1.8 + 15*(3*self.aa)**8 )
        self.mcut = 3e9*( 1 + 10*(3*self.aa)**8)
        ###self.normhalo = 3e5*(1+(3.5/self.zz)**6) 
        ###self.normhalo = 3e7 *(4+(3.5/self.zz)**6)
        self.normhalo = 8e5*(1+(3.5/self.zz)**6) 
        self.normsat = self.normhalo*(1.75 + 0.25*self.zz)


    def assignHI(self, halocat, cencat, satcat):
        mHIhalo = self.assignhalo(halocat['Mass'].compute())
        mHIsat = self.assignsat(satcat['Mass'].compute())
        mHIcen = self.assigncen(mHIhalo, mHIsat, satcat['GlobalID'].compute(), 
                                cencat.csize, cencat.comm)
        
        return mHIhalo, mHIcen, mHIsat
        
        
    def assignhalo(self, mhalo):
        xx  = mhalo/self.mcut+1e-10
        mHI = xx**self.alp * np.exp(-1/xx)
        mHI*= self.normhalo
        return mHI

    def assignsat(self, msat):
        xx  = msat/self.mcut+1e-10
        mHI = xx**self.alp * np.exp(-1/xx)
        mHI*= self.normsat
        return mHI
        

    def getinsat(self, mHIsat, satid, totalsize, localsize, comm):
       
        #print(comm.rank, np.all(np.diff(satid) >=0))
        #diff = np.diff(satid)
        #if comm.rank == 260: 
        #    print(satid[:-1][diff <0], satid[1:][diff < 0])

        da = DistributedArray(satid, comm)
        
        mHI = da.bincount(mHIsat, shared_edges=False)
        
        zerosize = totalsize - mHI.cshape[0]
        zeros = DistributedArray.cempty(cshape=(zerosize, ), dtype=mHI.local.dtype, comm=comm)
        zeros.local[...] = 0
        mHItotal = DistributedArray.concat(mHI, zeros, localsize=localsize)
        return mHItotal
        
    def assigncen(self, mHIhalo, mHIsat, satid, censize, comm):
        #Assumes every halo has a central...which it does...usually
        mHItotal = self.getinsat(mHIsat, satid, censize, mHIhalo.size, comm)
        return mHIhalo - mHItotal.local
        
      
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos


    def createmesh(self, bs, nc, halocat, cencat, satcat, mode='galaxies', position='RSDpos', weight='HImass'):
        '''use this to create mesh of HI
        '''
        comm = halocat.comm
        if mode == 'halos': catalogs = [halocat]
        elif mode == 'galaxies': catalogs = [cencat, satcat]
        elif mode == 'all': catalogs = [halocat, cencat, satcat]
        else: print('Mode not recognized')
        
        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
        totweight        = comm.allreduce(rankweight)

        for cat in catalogs: cat[weight] /= totweight/float(nc)**3            
        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)
        mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                                 position=position,weight=weight)

        return mesh
        
    


class ModelA2(ModelA):
    '''Same as model A with a different RSD for satellites
    '''
    def __init__(self, aa):

        super().__init__(aa)
        
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity_HI']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos

        

        
    
class ModelB():
    
    def __init__(self, aa, h=0.6776):

        self.aa = aa
        self.zz = 1/aa-1
        self.h = h
        #self.mcut = 1e9*( 1.8 + 15*(3*self.aa)**8 )
        self.mcut = 3e9*( 1 + 10*(3*self.aa)**8) 
        self.normhalo = 1
        #self.slope, self.intercept = np.polyfit([8.1, 11], [0.2, -1.], deg=1)


    def assignHI(self, halocat, cencat, satcat):
        mHIsat = self.assignsat(satcat['Mass'].compute())
        mHIcen = self.assigncen(cencat['Mass'].compute())
        
        mHIhalo = self.assignhalo(mHIcen, mHIsat, satcat['GlobalID'].compute(), 
                                halocat.csize, halocat.comm)
        return mHIhalo, mHIcen, mHIsat

    def assignhalo(self, mHIcen, mHIsat, satid, hsize, comm):
        #Assumes every halo has a central...which it does...usually
        mHItotal = self.getinsat(mHIsat, satid, hsize, mHIcen.size, comm)
        return mHIcen + mHItotal.local


    def getinsat(self, mHIsat, satid, totalsize, localsize, comm):
        da = DistributedArray(satid, comm)
        mHI = da.bincount(mHIsat, shared_edges=False)
        zerosize = totalsize - mHI.cshape[0]
        zeros = DistributedArray.cempty(cshape=(zerosize, ), dtype=mHI.local.dtype, comm=comm)
        zeros.local[...] = 0
        mHItotal = DistributedArray.concat(mHI, zeros, localsize=localsize)
        return mHItotal

##    def _assign(self, mstellar):
##        xx = np.log10(mstellar)
##        yy = self.slope* xx + self.intercept
##        mh1 = mstellar * 10**yy
##        return mh1
##        

##    def _assign(self, mstellar):
##       xx = np.log10(mstellar)
##       mm = 9e8
##       aa = 0.0
##       bb = 0.5
##       yy = np.log10(((10**xx/mm)**aa +(10**xx/mm)**bb)**-1) 
##       mh1 = mstellar * 10**yy
##       return mh1

    def _assign(self, mstellar):
        '''Takes in M_stellar and gives M_HI in M_solar
        '''
        mm = 3e8 #5e7
        f = 0.18 #0.35
        alpha = 0.4 #0.35
        mfrac = f*(mm/(mstellar + mm))**alpha
        mh1 = mstellar * mfrac
        return mh1
        
        

    def assignsat(self, msat, scatter=None):
        mstellar = self.moster(msat, scatter=scatter)/self.h
        mh1 = self._assign(mstellar)
        mh1 = mh1*self.h #* np.exp(-self.mcut/msat)
        return mh1


    def assigncen(self, mcen, scatter=None):
        mstellar = self.moster(mcen, scatter=scatter)/self.h
        mh1 = self._assign(mstellar)
        mh1 = mh1*self.h #* np.exp(-self.mcut/mcen)
        return mh1


    def moster(self, Mhalo, scatter=None):
        """ 
        moster(Minf,z): 
        Returns the stellar mass (M*/h) given Minf and z from Table 1 and                                                                  
        Eq. (2,11-14) of Moster++13 [1205.5807]. 
        This version now works in terms of Msun/h units,
        convert to Msun units in the function
        To get "true" stellar mass, add 0.15 dex of lognormal scatter.                    
        To get "observed" stellar mass, add between 0.1-0.45 dex extra scatter.         

        """
        z = self.zz
        Minf = Mhalo/self.h
        zzp1  = z/(1+z)
        M1    = 10.0**(11.590+1.195*zzp1)
        mM    = 0.0351 - 0.0247*zzp1
        beta  = 1.376  - 0.826*zzp1
        gamma = 0.608  + 0.329*zzp1
        Mstar = 2*mM/( (Minf/M1)**(-beta) + (Minf/M1)**gamma )
        Mstar*= Minf
        if scatter is not None: 
            Mstar = 10**(np.log10(Mstar) + np.random.normal(0, scatter, Mstar.size))
        return Mstar*self.h
        #                                                                                                                                          

    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos


    def createmesh(self, bs, nc, halocat, cencat, satcat, mode='galaxies', position='RSDpos', weight='HImass'):
        '''use this to create mesh of HI
        '''
        comm = halocat.comm
        if mode == 'halos': catalogs = [halocat]
        elif mode == 'galaxies': catalogs = [cencat, satcat]
        elif mode == 'all': catalogs = [halocat, cencat, satcat]
        else: print('Mode not recognized')
        
        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
        totweight        = comm.allreduce(rankweight)

        for cat in catalogs: cat[weight] /= totweight/float(nc)**3            
        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)
        mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                                 position=position,weight=weight)

        return mesh



##
##
##
##    
##class ModelB():
##    '''Assign differently in satellites here
##    '''
##    def __init__(self, aa, h=0.6776):
##
##        self.aa = aa
##        self.zz = 1/aa-1
##        self.h = h
##        self.mcut = 1e9*( 1.8 + 15*(3*self.aa)**8 )
##        self.normhalo = 1
##        self.slope, self.intercept = np.polyfit([8.1, 11], [0.2, -1.], deg=1)
##
##
##    def assignHI(self, halocat, cencat, satcat):
##        mHIsat = self.assignsat(satcat['Mass'].compute(), satcat['HaloMass'].compute())
##        mHIcen = self.assigncen(cencat['Mass'].compute())
##        
##        mHIhalo = self.assignhalo(mHIcen, mHIsat, satcat['GlobalID'].compute(), 
##                                halocat.csize, halocat.comm)
##        return mHIhalo, mHIcen, mHIsat
##
##    def assignhalo(self, mHIcen, mHIsat, satid, hsize, comm):
##        #Assumes every halo has a central...which it does...usually
##        mHItotal = self.getinsat(mHIsat, satid, hsize, mHIcen.size, comm)
##        return mHIcen + mHItotal.local
##
##
##    def getinsat(self, mHIsat, satid, totalsize, localsize, comm):
##        da = DistributedArray(satid, comm)
##        mHI = da.bincount(mHIsat, shared_edges=False)
##        zerosize = totalsize - mHI.cshape[0]
##        zeros = DistributedArray.cempty(cshape=(zerosize, ), dtype=mHI.local.dtype, comm=comm)
##        zeros.local[...] = 0
##        mHItotal = DistributedArray.concat(mHI, zeros, localsize=localsize)
##        return mHItotal
##
##    def assignsat(self, msat, mhalo, mmax=15.5, scatter=None):
##        mstellar = self.moster(msat, scatter=scatter)/self.h
##
##        xx = np.log10(mstellar)
##        yy = self.slope* xx + self.intercept
##        mh1 = mstellar * 10**yy
##        mh1 =  mh1*self.h * np.exp(-self.mcut/msat)
##        
##        #
##        mcutoff = np.log10(mstellar) + 3
##        mscale = (1/(mmax-mcutoff))
##        poor = mscale*(np.log10(mhalo)-mcutoff)
##        poor = np.array(list(map(lambda i: max(0.01, i), poor)))
##        poor = np.array(list(map(lambda i: min(1, i), poor))).flatten()
##        mask = np.random.uniform(size=poor.size) > poor
##        mask = mask *bool(0)
##        #print(mask.sum()/mask.size)
##        mh1 = mh1 * mask
##        return mh1
##
##
##    def assigncen(self, mcen, scatter=None):
##        mstellar = self.moster(mcen, scatter=scatter)/self.h
##        xx = np.log10(mstellar)
##        yy = self.slope* xx + self.intercept
##        mh1 = mstellar * 10**yy
##        return mh1*self.h * np.exp(-self.mcut/mcen)
##
##
##    def moster(self, Mhalo, scatter=None):
##        """ 
##        moster(Minf,z): 
##        Returns the stellar mass (M*/h) given Minf and z from Table 1 and                                                                  
##        Eq. (2,11-14) of Moster++13 [1205.5807]. 
##        This version now works in terms of Msun/h units,
##        convert to Msun units in the function
##        To get "true" stellar mass, add 0.15 dex of lognormal scatter.                    
##        To get "observed" stellar mass, add between 0.1-0.45 dex extra scatter.         
##
##        """
##        z = self.zz
##        Minf = Mhalo/self.h
##        zzp1  = z/(1+z)
##        M1    = 10.0**(11.590+1.195*zzp1)
##        mM    = 0.0351 - 0.0247*zzp1
##        beta  = 1.376  - 0.826*zzp1
##        gamma = 0.608  + 0.329*zzp1
##        Mstar = 2*mM/( (Minf/M1)**(-beta) + (Minf/M1)**gamma )
##        Mstar*= Minf
##        if scatter is not None: 
##            Mstar = 10**(np.log10(Mstar) + np.random.normal(0, scatter, Mstar.size))
##        return Mstar*self.h
##        #                                                                                                                                          
##
##    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
##        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
##        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
##        srsdpos = satcat['Position']+satcat['Velocity']*los * rsdfac
##        return hrsdpos, crsdpos, srsdpos
##
##
##    def createmesh(self, bs, nc, halocat, cencat, satcat, mode='galaxies', position='RSDpos', weight='HImass'):
##        '''use this to create mesh of HI
##        '''
##        comm = halocat.comm
##        if mode == 'halos': catalogs = [halocat]
##        elif mode == 'galaxies': catalogs = [cencat, satcat]
##        elif mode == 'all': catalogs = [halocat, cencat, satcat]
##        else: print('Mode not recognized')
##        
##        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
##        totweight        = comm.allreduce(rankweight)
##
##        for cat in catalogs: cat[weight] /= totweight/float(nc)**3            
##        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)
##        mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
##                                 position=position,weight=weight)
##
##        return mesh
##





class ModelC(ModelA):
    '''Vanilla model with no centrals and satellites, only halo
    Halos have the COM velocity but do not have any dispersion over it
    '''
    def __init__(self, aa):

        super().__init__(aa)
        self.normsat = 0

        #self.alp = 1.0
        #self.mcut = 1e9
        #self.normhalo = 2e5*(1+3/self.zz**2) 
        #self.normhalo = 1.1e5*(1+4/self.zz) 
        self.alp = 0.9
        self.mcut = 1e10
        self.normhalo = 3.5e6*(1+1/self.zz) 


    def derivate(self, param, delta):
        if param == 'alpha':
            self.alp = (1+delta)*self.alp
        elif param == 'mcut':
            self.mcut = 10**( (1+delta)*np.log10(self.mcut))
        elif param == 'norm':
            self.mcut = 10**( (1+delta)*np.log10(self.normhalo))
        else:
            print('Parameter to vary not recongnized. Should be "alpha", "mcut" or "norm"')
            


    def assignHI(self, halocat, cencat, satcat):
        mHIhalo = self.assignhalo(halocat['Mass'].compute())
        mHIsat = self.assignsat(satcat['Mass'].compute())
        mHIcen = self.assigncen(cencat['Mass'].compute())
        
        return mHIhalo, mHIcen, mHIsat
        
    def assignsat(self, msat):
        return msat*0
        
    def assigncen(self, mcen):
        return mcen*0
        

    def createmesh(self, bs, nc, halocat, cencat, satcat, mode='halos', position='RSDpos', weight='HImass'):
        '''use this to create mesh of HI
        '''
        comm = halocat.comm
        if mode == 'halos': catalogs = [halocat]
        else: print('Mode not recognized')
        
        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
        totweight        = comm.allreduce(rankweight)

        for cat in catalogs: cat[weight] /= totweight/float(nc)**3            
        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)
        mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                                 position=position,weight=weight)

        return mesh




class ModelD(ModelC):
    '''Vanilla model with no centrals and satellites, only halo
    Halos have the COM velocity and a dispersion from VN18 added over it
    '''
    def __init__(self, aa):

        super().__init__(aa)
        self.vdisp = self._setupvdisp()


    def _setupvdisp(self):
        vzdisp0 = np.array([31, 34, 39, 44, 51, 54])
        vzdispal = np.array([0.35, 0.37, 0.38, 0.39, 0.39, 0.40])
        vdispz = np.arange(0, 6)
        vdisp0fit = np.polyfit(vdispz, vzdisp0, 1)
        vdispalfit = np.polyfit(vdispz, vzdispal, 1)
        vdisp0 = self.zz * vdisp0fit[0] + vdisp0fit[1]
        vdispal = self.zz * vdispalfit[0] + vdispalfit[1]
        return lambda M: vdisp0*(M/1e10)**vdispal
        
      
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        dispersion = np.random.normal(0, self.vdisp(halocat['Mass'].compute())).reshape(-1, 1)
        hvel = halocat['Velocity']*los + dispersion*los
        hrsdpos = halocat['Position']+ hvel*rsdfac
        
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos


