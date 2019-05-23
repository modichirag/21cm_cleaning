#!/usr/bin/env python3
#
# Plots the scales probed by PUMA, including the wedge.
#
import numpy as np
import matplotlib.pyplot as plt
from   astropy.cosmology import FlatLambdaCDM



cc = FlatLambdaCDM(H0=67.7,Om0=0.309167)


def wedge_R(zz,bfac=3.0,D=6.0):
    """The foreground wedge, for a dish of D meters."""
    # First compute the beam, assuming using bfac*fov.
    etaA = 0.7                  # Aperture efficiency.
    lam21= 0.21 * (1+zz)
    thta = 1.22 * lam21/(2*D) / np.sqrt(etaA)
    beam = bfac * thta
    # Now compute R.
    Hz   = cc.H(zz).value
    chi  = cc.comoving_distance(zz).value
    R    = np.sin(beam)*chi*Hz/(1+zz)/(2.997925e5)
    return(R)
    #








def thermal_n(kperp,zz,D=6.0,Ns=256,hex=True):
    """The thermal noise for PUMA -- note noise rescaling from 5->5/4 yr."""
    # Some constants.
    etaA = 0.7                          # Aperture efficiency.
    Aeff = etaA*np.pi*(D/2)**2          # m^2
    lam21= 0.21*(1+zz)                  # m
    nuobs= 1420/(1+zz)                  # MHz
    # The cosmology-dependent factors.
    hub  = cc.H(0).value / 100.0
    Ez   = cc.H(zz).value / cc.H(0).value
    chi  = cc.comoving_distance(zz).value * hub         # Mpc/h.
    OmHI = 4e-4*(1+zz)**0.6 / Ez**2
    Tbar = 0.188*hub*(1+zz)**2*Ez*OmHI  # K
    # Eq. (3.3) of Chen++19
    d2V  = chi**2*2997.925/Ez*(1+zz)**2
    # Eq. (3.5) of Chen++19
    if hex:    # Hexagonal array of Ns^2 elements.
        n0,c1,c2,c3,c4,c5 = (Ns/D)**2,0.5698,-0.5274,0.8358,1.6635,7.3177
        uu   = kperp*chi/(2*np.pi)
        xx   = uu*lam21/Ns/D                # Dimensionless.
        nbase= n0*(c1+c2*xx)/(1+c3*xx**c4)*np.exp(-xx**c5) * lam21**2 + 1e-30
        nbase[uu<   D/lam21    ]=1e-30
        nbase[uu>Ns*D/lam21*1.3]=1e-30
    else:      # Square array of Ns^2 elements.
        n0,c1,c2,c3,c4,c5 = (Ns/D)**2,0.4847,-0.33,1.3157,1.5974,6.8390
        uu   = kperp*chi/(2*np.pi)
        xx   = uu*lam21/Ns/D                # Dimensionless.
        nbase= n0*(c1+c2*xx)/(1+c3*xx**c4)*np.exp(-xx**c5) * lam21**2 + 1e-30
        nbase[uu<   D/lam21    ]=1e-30
        nbase[uu>Ns*D/lam21*1.4]=1e-30
    # Eq. (3.2) of Chen++19, updated to match PUMA specs:
    npol = 2
    fsky = 0.5
    tobs = 5.*365.25*24.*3600.          # sec.
    tobs/= 4.0                          # Scale to 1/2-filled array.
    # the signal entering OMT is given by eta_dish*T_s + (1-eta_dish)*T_g
    # and after hitting both with eta_omt and adding amplifier noise you get:
    # T_ampl + eta_omt.eta_dish.T_s + eta_omt(1-eta_dish)T_g
    # so normalizing to have 1 in front of Ts we get
    # T_ampl/(eta_omt*eta_dish) + T_g (1-eta_dish)/(eta_dish) + T_sky
    # Putting in T_ampl=50K T_g=30K eta_omt=eta_dish=0.9 gives:
    Tamp = 50.0/0.9**2                  # K
    Tgnd = 30.0/0.9*(1-0.9)             # K
    Tsky = 2.7 + 25*(400./nuobs)**2.75  # K
    Tsys = Tamp + Tsky + Tgnd
    Omp  = (lam21/D)**2/etaA
    # Return Pth in "cosmological units", with the Tbar divided out.
    Pth  = (Tsys/Tbar)**2*(lam21**2/Aeff)**2 *\
           4*np.pi*fsky/Omp/(npol*1420e6*tobs*nbase) * d2V
    return(Pth)
    #








def make_scales_plot(kmin=0.1):
    """Does the work of making the wedge/scales figure."""
    # Bias and shot-noise values from Table 1, model A, Modi+19.
    zlist = [2.00,4.00,6.00]
    blist = [1.91,2.58,3.72]
    nlist = [53.3,10.1,9.24]	# P_{SN} in (Mpc/h)^3.
    # Read the linear theory P(k,z=0).
    pklin = np.loadtxt("../../data/pklin_1.0000.txt")
    # Now make the figure.
    cmap   = plt.get_cmap('magma')
    fig,ax = plt.subplots(1,4,figsize=(9,3.),\
                          gridspec_kw={'width_ratios':[5,5,5,1]})
    for ii in range(ax.size-1):
        zz = zlist[ii]
        aa = 1.0/(1.0+zz)
        bb = blist[ii]
        sn = nlist[ii]
        # Compute the linear growth factor and growth rate -- this
        # isn't part of the cosmology package so approximate it.
        ff   = (0.3/(0.3+0.7/(1+zz)**3))**0.55
        aval = np.logspace(np.log10(1/(1+zz)),0.0,100)
        Dz   = np.exp(-np.trapz(cc.Om(1/aval-1)**0.55,x=np.log(aval)))
        # Scale the linear theory P(k) and compute Kaiser.
        pk = pklin.copy()
        pk[:,1] *= Dz**2
        xx = np.linspace(0.0,0.8,100)
        yy = np.linspace(0.0,0.8,100)
        X,Y= np.meshgrid(xx,yy,indexing='ij')
        kk = np.sqrt(X**2+Y**2)
        mu = Y/(kk+1e-10)
        P  = np.interp(kk,pk[:,0],pk[:,1])
        P *= (bb+ff*mu**2)**2
        SNR= P/(P+sn+thermal_n(X,zz))
        im = ax[ii].imshow(SNR.T,origin='lower',vmin=0,vmax=1,\
                           extent=[xx[0],xx[-1],yy[0],yy[-1]],cmap=cmap)
        # Put on the wedge.
        if False:
            ax[ii].plot(xx,np.clip(wedge_R(zz,1.0)*xx,kmin,1e9),\
                        '--',color='grey')
            ax[ii].plot(xx,np.clip(wedge_R(zz,3.0)*xx,kmin,1e9),\
                        ':' ,color='grey')
        else:
            ax[ii].plot(xx,wedge_R(zz,1.0)*xx,'--',color='grey')
            ax[ii].plot(xx,wedge_R(zz,3.0)*xx,':' ,color='grey')
        # Tidy up the plot.
        ax[ii].set_xlim(xx[0],xx[-1])
        ax[ii].set_ylim(yy[0],yy[-1])
        ax[ii].set_xscale('linear')
        ax[ii].set_yscale('linear')
        ax[ii].set_aspect('equal')
        ax[ii].text(0.05,0.95*yy[-1],"$z={:.1f}$".format(zz),va='center',color='w')
        #
    plt.colorbar(im,cax=ax[-1])
    ax[-1].set_aspect(15)
    # Put on some more labels.
    ax[0].set_xlabel(r'$k_\perp\quad [h\ {\rm Mpc}^{-1}]$')
    ax[1].set_xlabel(r'$k_\perp\quad [h\ {\rm Mpc}^{-1}]$')
    ax[2].set_xlabel(r'$k_\perp\quad [h\ {\rm Mpc}^{-1}]$')
    ax[0].set_ylabel(r'$k_\parallel\quad [h\ {\rm Mpc}^{-1}]$')
    # and finish up.
    plt.tight_layout()
    plt.savefig('puma_scales.pdf')
    #











if __name__=="__main__":
    make_scales_plot()
    #
