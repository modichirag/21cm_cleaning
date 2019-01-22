#include	<cmath>
#include	<cstdlib>
#include	<iostream>
#include	<iomanip>
#include	<fstream>
#include	<sstream>
#include	<vector>
#include	<string>
#include	<algorithm>
#include	<exception>
#include	"omp.h"
#include	"fftw3.h"



// OpenMP code to do a matched filter of the density field against a
// template to point sources.
//
// Author:	Martin White	(UCB)
// Written:	13-Sep-2018
// Modified:	13-Sep-2018	(Copied from void21/matched)



// This definition should match that used in the void finder.
const	long	Ng=1024L;			// Should be even
const	float	Lbox=256.;			// In Mpc/h.
const	float	FluxMin=0;
const	float	FluxMax=1e30;
const	float	kzMin=0.00*Lbox;		// Modes removed due to fg
const	float	muMin=0.00;			// Modes removed due to fg
const	float	kFid=0.2;			// Noise norm scale, h/Mpc.
const	float	nP=100.;			// Noise normalization.
const	int	ZAxis=2;


struct	Source {
  float	pos[3];
};



void	myexit(int flag)
{
  exit(flag);
}

void	myexception(std::exception& e)
{
  std::cerr<<"Standard exception: "<<e.what()<<std::endl;
  myexit(1);
}

float	periodic(const float x)
{
  if (x>=1.0) return(x-1);
  if (x< 0.0) return(x+1);
  return(x);
}


class Cosmology {
private:
  const double OmM,hh,ns,f;
  double wm,wb,s_ak,zobs;
  double T_AK(const double x) const {
  // The "T" function of Adachi & Kasai (2012), used in chi(z) computation.
    const double b1=2.64086441;
    const double b2=0.883044401;
    const double b3=0.0531249537;
    const double c1=1.39186078;
    const double c2=0.512094674;
    const double c3=0.0394382061;
    const double x3=x*x*x;
    double tmp = 2+x3*(b1+x3*(b2+x3*b3));
    tmp /= 1+x3*(c1+x3*(c2+x3*c3));
    tmp *= sqrt(x);
    return(tmp);
  }
public:
  double chi;
  Cosmology() : OmM(0.3106),hh(0.68),ns(0.97),f(1.0) {
    wm  = OmM * hh*hh;
    wb  = 0.022;
    s_ak= pow((1-OmM)/OmM,0.333333);
    zobs= 0;
    chi = 0;
  }
  void set_zobs(const double zz) {
    zobs = zz;
    chi  = T_AK(s_ak)-T_AK(s_ak/(1+zz));
    chi *= 2997.925/sqrt(s_ak*OmM);
  }
  double tk_eh(const double k) const {
  // Transfer function from E&H, k in h/Mpc.
    double theta= 2.728/2.7;
    double s    = 44.5*log(9.83/wm)/sqrt( 1.+10.*exp(0.75*log(wb)) ) * hh;
    double a    = 1.-0.328*log(431.*wm)* wb/wm
                + 0.380*log(22.3*wm)*(wb/wm)*(wb/wm);
    double gamma = a+(1.-a)/( 1.+exp(4.*log(0.43*k*s)) ); gamma *= OmM*hh;
    double q     = k*theta*theta/gamma;
    double L0    = log( 2.*exp(1.)+1.8*q );
    double C0    = 14.2 + 731./(1.+62.5*q);
    return( L0/(L0+C0*q*q) );
  }
  double pofk(const double kR, const double kZ) const {
  // The (unnormalized) linear theory power spectrum.
    double kk = sqrt(kR*kR+kZ*kZ) + 1e-30;
    double mu = kZ/kk;
#ifdef	RED
    double RR = 1+f*mu*mu;
#else
    const double RR = 1;
#endif
    double pk = RR*RR * pow(kk/0.2,ns)*tk_eh(kk)*tk_eh(kk);
    return(pk);
  }
};





class AbstractTelescope {
public:
  virtual double num_baselines(const double kR) const = 0;
  // Want this to be a function of kR, not kX and kY separately,
  // to fix rotational symmetry.
  virtual void print_name() const = 0;
  // Print what kind of telescope for the logs.
};




class PerfectTelescope: public AbstractTelescope {
public:
  PerfectTelescope() {}		// Don't need to do anything
  double num_baselines(const double kR) const {
    return(1.0);
  }
  void print_name() const {
    std::cout<<"# Perfect telescope."<<std::endl;
  }
};




class CylinderTelescope: public AbstractTelescope {
private:
  const int		Ncyl,Nfeed,Nbin;
  double		kRmin,kRmax;
  std::vector<double>	nbase;
public:
  CylinderTelescope(const int ncyl, const int nfeed, const float width,
    const float length, const float wavelength, const float chi) : Ncyl(ncyl),
    Nfeed(nfeed),Nbin(64) {
    // Work out the spacings in wavelengths.
    const double dkx = 2*M_PI/wavelength * width;
    const double dky = 2*M_PI/wavelength * length/nfeed;
    kRmin=1e30;  kRmax=-1e30;
    for (int i=0; i<ncyl*nfeed; ++i) {
      int icyl1  = i/nfeed;
      int ifeed1 = i%nfeed;
      for (int j=0; j<ncyl*nfeed; ++j) {
        int icyl2  = j/nfeed;
        int ifeed2 = j%nfeed;
        double kx  = (icyl1 -icyl2 )*dkx;
        double ky  = (ifeed1-ifeed2)*dky;
        double kR  = sqrt( kx*kx+ky*ky );
        if (kR<kRmin) kRmin=kR;
        if (kR>kRmax) kRmax=kR;
      }
    }
    kRmin *= 0.999/chi;
    kRmax *= 1.001/chi;
    // These are now in "ell" units, convert to physical k using chi.
    nbase.resize(Nbin);
    for (int i=0; i<ncyl*nfeed; ++i) {
      int icyl1  = i/nfeed;
      int ifeed1 = i%nfeed;
      for (int j=0; j<ncyl*nfeed; ++j) {
        int icyl2  = j/nfeed;
        int ifeed2 = j%nfeed;
        double kx  = (icyl1 -icyl2 )*dkx;
        double ky  = (ifeed1-ifeed2)*dky;
        double kR  = sqrt( kx*kx+ky*ky )/chi;
        int ibin   = Nbin * kR/kRmax;
        nbase[ibin]+=1.0;
      }
    }
    // and finally normalize the counts at kFid to 1.
    int fbin = Nbin*kFid/kRmax;
    if (fbin<0 || fbin>=Nbin) {
      std::cerr<<"Fiducial k, "<<kFid<<", out of kR range."<<std::endl;
      myexit(1);
    }
    const double norm=1.0/(nbase[fbin]+1e-30);
    for (int ibin=0; ibin<Nbin; ++ibin) nbase[ibin] *= norm;
  }
  double num_baselines(const double kR) const {
    // The # baselines is independent of kZ, the frequency direction.
    if (kR>kRmin && kR<kRmax) {
      int ibin=kR/kRmax*Nbin;
      return(nbase[ibin]);
    }
    else
      return(0);
  }
  void print_name() const {
    std::cout<<"# Cylinder telescope ("<<Ncyl<<","<<Nfeed<<")"<<std::endl;
  }
  void print_feed_hist() const {
  // A feature useful for debugging.
    for (int ibin=0; ibin<Nbin; ++ibin) {
      double kR=kRmax*(ibin+0.5)/Nbin;
      std::cout<<std::setw(10)<<ibin
               <<std::fixed<<std::setw(15)<<std::setprecision(4)<<kR
               <<std::fixed<<std::setw(15)<<std::setprecision(4)<<nbase[ibin]
               <<std::endl;
    }
  }
};



class DishTelescope: public AbstractTelescope {
private:
  const int		Nside,Nbin;
  const double		Rdish;
  double		kRmin,kRmax;
  std::vector<double>	nbase;
public:
  DishTelescope(const int nside, const float radius,
    const float wavelength, const float chi) : Nside(nside),Rdish(radius),
    Nbin(64) {
    // Work out the spacings in wavelengths.
    const double dkx = 2*M_PI/wavelength * (2*Rdish);
    const double dky = 2*M_PI/wavelength * (2*Rdish);
    kRmin=1e30; kRmax=-1e30;
    for (int i=0; i<Nside*Nside; ++i) {
      int ix1 = i/Nside;
      int iy1 = i%Nside;
      for (int j=0; j<Nside*Nside; ++j) {
        int ix2    = j/Nside;
        int iy2    = j%Nside;
        double kx  = (ix1-ix2)*dkx;
        double ky  = (iy1-iy2)*dky;
        double kR  = sqrt( kx*kx+ky*ky );
        if (kR<kRmin) kRmin=kR;
        if (kR>kRmax) kRmax=kR;
      }
    }
    // These are now in "ell" units, convert to physical k using chi.
    kRmin *= 0.999/chi;
    kRmax *= 1.001/chi;
    nbase.resize(Nbin);
    for (int i=0; i<Nside*Nside; ++i) {
      int ix1 = i/Nside;
      int iy1 = i%Nside;
      for (int j=0; j<Nside*Nside; ++j) {
        int ix2    = j/Nside;
        int iy2    = j%Nside;
        double kx  = (ix1-iy2)*dkx;
        double ky  = (iy1-iy2)*dky;
        double kR  = sqrt( kx*kx+ky*ky )/chi;
        int ibin   = Nbin * kR/kRmax;
        nbase[ibin]+=1.0;
      }
    }
    // and finally normalize the counts at kFid to 1.
    int fbin = Nbin*kFid/kRmax;
    if (fbin<0 || fbin>=Nbin) {
      std::cerr<<"Fiducial k, "<<kFid<<", out of kR range."<<std::endl;
      myexit(1);
    }
    const double norm=1.0/(nbase[fbin]+1e-30);
    for (int ibin=0; ibin<Nbin; ++ibin) nbase[ibin] *= norm;
  }
  double num_baselines(const double kR) const {
    // The # baselines is independent of kZ, the frequency direction.
    if (kR>kRmin && kR<kRmax) {
      int ibin=kR/kRmax*Nbin;
      return(nbase[ibin]);
    }
    else
      return(0);
  }
  void print_name() const {
    std::cout<<"# Dish telescope ("<<Nside<<","<<Rdish<<")"<<std::endl;
  }
  void print_feed_hist() const {
  // A feature useful for debugging.
    for (int ibin=0; ibin<Nbin; ++ibin) {
      double kR=kRmax*(ibin+0.5)/Nbin;
      std::cout<<std::setw(10)<<ibin
               <<std::fixed<<std::setw(15)<<std::setprecision(4)<<kR
               <<std::fixed<<std::setw(15)<<std::setprecision(4)<<nbase[ibin]
               <<std::endl;
    }
  }
};



double	noise_weight(const double kR, const double kZ,
                     const Cosmology& C, const AbstractTelescope& T) {
// Returns N^{-1}(kR,kZ), with k's in box units.
  double signal = C.pofk(kR/Lbox,kZ/Lbox);
  double mu     = fabs(kZ)/sqrt(kR*kR+kZ*kZ);
  double nb     = T.num_baselines(kR/Lbox) + 1e-30;
  double noise  = C.pofk(kFid,0)/nP/nb + 1e30*(kZ<kzMin) + 1e30*(mu<muMin);
  return(1.0/(signal+noise+1e-30));
}







void	match(std::vector<double>& di, std::vector<Source>& src,
              const Cosmology& C, const AbstractTelescope& T) {
// Does the matched filter between the density field and the profile.
// Prints two histograms: the S/N distribution overall and at known
// positions, src.
// Also writes a file containing the locations where A>threshold.
  const double Ngrid=0.9999*Ng;
  const long Ng2=Ng+2;
  fftw_complex *dk = (fftw_complex *)&di[0];
  // Generate the FFTW plan files and do the FTs.
  fftw_init_threads();
  fftw_plan_with_nthreads(omp_get_max_threads());
  fftw_plan planA = fftw_plan_dft_r2c_3d(Ng,Ng,Ng,&di[0],&dk[0],FFTW_ESTIMATE);
  fftw_plan planB = fftw_plan_dft_c2r_3d(Ng,Ng,Ng,&dk[0],&di[0],FFTW_ESTIMATE);
  fftw_execute(planA);
  // Multiply dk by template (1), times the (assumed diagonal) weights.
  const double tPi2 = 2*M_PI * 2*M_PI;
  const double Ng3inv=1.0/Ng/Ng/Ng;
  const double nfid = C.pofk(kFid,0)/nP;
  double varL=0,varN=0,fish=0;
#pragma omp parallel for shared(dk) reduction(+:varL,varN,fish)
  for (int ix=0; ix<Ng; ++ix) {
    int iix = (ix<=Ng/2)?ix:ix-Ng;
    for (int iy=0; iy<Ng; ++iy) {
      int iiy = (iy<=Ng/2)?iy:iy-Ng;
      for (int iz=0; iz<Ng/2+1; ++iz) {
        int iiz = (iz<=Ng/2)?iz:iz-Ng;
        long   ip    = Ng*(Ng/2+1)*ix+(Ng/2+1)*iy+iz;
        double k2    = tPi2*(iix*iix+iiy*iiy+iiz*iiz);
        double kZ    = 2*M_PI*abs(iiz);			// Note abs.
        double kR    = 2*M_PI*sqrt(iix*iix+iiy*iiy*1.0);
        double nwt   = noise_weight(kR,kZ,C,T);
        double prodr = dk[ip][0]*1+dk[ip][1]*0;
        double prodi = dk[ip][1]*1-dk[ip][0]*0;
        // Do the matched filter.
        dk[ip][0] = prodr * nwt*Ng3inv;
        dk[ip][1] = prodi * nwt*Ng3inv;
        // Now propagate the separate signal and noise variances -- don't
        // need to model regions where noise is very large.  We can also
        // neglect the Fisher prefactor on the filter since it cancels
        // between L and N.
        double vsqr   = 1;
        double signal = C.pofk(kR/Lbox,kZ/Lbox);
        double noise  = nfid/(T.num_baselines(kR/Lbox) + 1e-30);
        varL += signal*vsqr*nwt*nwt*(ip!=0);
        varN += noise *vsqr*nwt*nwt*(ip!=0);
        fish +=        vsqr*    nwt*(ip!=0);
      }
    }
  }
  dk[0][0] = dk[0][1] = 0;
  fftw_execute(planB);
  fftw_destroy_plan(planB);
  fftw_destroy_plan(planA);
  fftw_cleanup_threads();
  // Print out the S/N weights.
  std::cout<<"# sigma_N/sigma_L="<<std::scientific<<std::setprecision(4)
           <<sqrt(varN/varL)<<std::endl;
  // and normalize the filter -- not reqd but do it anyway.
  double fishinv = 1.0/fish;
#pragma omp parallel for
  for (long ip=0; ip<di.size(); ++ip) di[ip] *= fishinv;
  // Now we can accumulate statistics.
  double avg=0,std=0;
#pragma omp parallel for reduction(+:avg,std)
  for (int ix=0; ix<Ng; ++ix)
    for (int iy=0; iy<Ng; ++iy)
      for (int iz=0; iz<Ng; ++iz) {
        long ip = Ng2*Ng*ix + Ng2*iy + iz;
        avg += di[ip];
        std += di[ip]*di[ip];
      }
  avg /= Ng*Ng*Ng;
  std  = sqrt(std/Ng/Ng/Ng-avg*avg);
  std::cout<<"# Amplitude is "<<std::scientific<<std::setprecision(4)
           <<avg<<"+/-"<<std<<std::endl;
  // Want to set limits based on a round number for sigma, so that we
  // can co-add different simulations on the same scale.
  int scale = int(2*log10(std));
  std  = pow(10.0,scale/2.0);
  //double dmin =-10.0*std;
  //double dmax = 10.0*std;
  double dmin =-25.0*std;
  double dmax = 25.0*std;
  std::cout<<"# Clipping to ["<<dmin<<","<<dmax<<"]"<<std::endl;
  // Make and print histograms.
  const int Nbin=2048;
  std::vector<long> allh(Nbin),srch(Nbin);
  for (int ix=0; ix<Ng; ++ix)
    for (int iy=0; iy<Ng; ++iy)
      for (int iz=0; iz<Ng; ++iz) {
        long ip = Ng2*Ng*ix + Ng2*iy + iz;
        int ibin= Nbin*(di[ip]-dmin)/(dmax-dmin);
        if (ibin<  0  ) ibin=0;
        if (ibin>=Nbin) ibin=Nbin-1;
        allh[ibin]++;
      }
  for (long j=0; j<src.size(); ++j) {
    int ix=int(Ngrid*src[j].pos[0]);
    int iy=int(Ngrid*src[j].pos[1]);
    int iz=int(Ngrid*src[j].pos[2]);
    long ip = Ng2*Ng*ix + Ng2*iy + iz;
    int ibin= Nbin*(di[ip]-dmin)/(dmax-dmin);
    if (ibin<  0  ) ibin=0;
    if (ibin>=Nbin) ibin=Nbin-1;
    srch[ibin]++;
  }
  for (int ibin=0; ibin<Nbin; ++ibin) {
    float rho = dmin + (ibin+0.5)*(dmax-dmin)/Nbin;
    std::cout<<std::fixed<<std::setw(15)<<std::setprecision(5)<<(rho-avg)/std
             <<std::setw(15)<<allh[ ibin]
             <<std::setw(15)<<srch[ibin]
             <<std::endl;
  }
#ifdef	DUMPA
  // Finally let's write the positions where local maxima in A are greater
  // than threshold.
  std::ofstream fs("Apeak_list.txt");
  const double Amin=0.5;
  for (int ix=0; ix<Ng; ++ix) {
    int ixp = (ix+1   )%Ng;
    int ixm = (ix-1+Ng)%Ng;
    for (int iy=0; iy<Ng; ++iy) {
      int iyp = (iy+1   )%Ng;
      int iym = (iy-1+Ng)%Ng;
      for (int iz=0; iz<Ng; ++iz) {
        int izp = (iz+1   )%Ng;
        int izm = (iz-1+Ng)%Ng;
        long ip = Ng2*Ng*ix + Ng2*iy + iz;
        // Check for above threshold and local maximum.
        if (di[ip]>Amin &&
            di[ip]>di[Ng2*Ng*ixp+Ng2*iy +iz ] &&
            di[ip]>di[Ng2*Ng*ix +Ng2*iyp+iz ] &&
            di[ip]>di[Ng2*Ng*ix +Ng2*iy +izp] &&
            di[ip]>di[Ng2*Ng*ixm+Ng2*iy +iz ] &&
            di[ip]>di[Ng2*Ng*ix +Ng2*iym+iz ] &&
            di[ip]>di[Ng2*Ng*ix +Ng2*iy +izm])
          fs<<std::setw(5)<<ix<<std::setw(5)<<iy<<std::setw(5)<<iz
            <<std::setw(15)<<std::setprecision(5)<<di[ip]<<std::endl;
      }
    }
  }
  fs.close();
#endif
}








std::vector<double>	read_prop(const char fbase[], const int Nfile)
// Reads the objects and generates an Ng^3 density grid using CIC.
// The grid is in "FFTW-format".
{
  const long Ng2=Ng+2;
  std::vector<double> D;
  try{D.resize(Ng2*Ng*Ng);} catch(std::exception& e) {myexception(e);}
  long ntotal=0;
  double mtotal=0;
  for (int ifile=0; ifile<Nfile; ++ifile) {
    std::ostringstream ss;
    if (Nfile==1)
      ss<<fbase<<".prop";
    else
      ss<<fbase<<".prop."<<std::setw(2)<<std::setfill('0')<<ifile;
    std::ifstream fm(ss.str().c_str(),std::ios::binary);
    if (!fm) {
      std::cerr<<"Unable to open "<<ss.str()<<" for reading."<<std::endl;
      myexit(1);
    }
    std::ifstream fr(ss.str().c_str(),std::ios::binary);
    if (!fr) {
      std::cerr<<"Unable to open "<<ss.str()<<" for reading."<<std::endl;
      myexit(1);
    }
#ifdef	RED
    std::ifstream fv(ss.str().c_str(),std::ios::binary);
    if (!fv) {
      std::cerr<<"Unable to open "<<ss.str()<<" for reading."<<std::endl;
      myexit(1);
    }
#endif
    int nobj=0;
    fm.read((char *)&nobj,sizeof(int));
    if (fm.fail()){std::cerr<<"Unable to read "<<ss.str()<<std::endl;myexit(1);}
    long seeksize;
    seeksize = sizeof(int) + 1L*nobj*sizeof(float);
    fr.seekg(seeksize);
    if (fr.fail()){std::cerr<<"Unable to seek "<<ss.str()<<std::endl;myexit(1);}
#ifdef	RED
    seeksize = sizeof(int) + 4L*nobj*sizeof(float);
    fv.seekg(seeksize);
    if (fv.fail()){std::cerr<<"Unable to seek "<<ss.str()<<std::endl;myexit(1);}
#endif
    const int BlkSiz=16777216;
    std::vector<float> bum,buf;
    try{
      bum.resize(1*BlkSiz);
      buf.resize(3*BlkSiz);
    } catch(std::exception& e){myexception(e);}
#ifdef	RED
    std::vector<float> buv;
    try{buv.resize(3*BlkSiz);} catch(std::exception& e){myexception(e);}
#endif
    long ngot=0;
    while (ngot<nobj) {
      int nget=(nobj-ngot>BlkSiz)?BlkSiz:nobj-ngot;
      fm.read((char *)&bum[0],1*nget*sizeof(float));
      if (fm.fail()) {
        std::cerr<<"Unable to read weights from "<<ss.str()<<std::endl;
        myexit(1);
      }
      fr.read((char *)&buf[0],3*nget*sizeof(float));
      if (fr.fail()) {
        std::cerr<<"Unable to read positions from "<<ss.str()<<std::endl;
        myexit(1);
      }
#ifdef	RED
      fv.read((char *)&buv[0],3*nget*sizeof(float));
      if (fv.fail()) {
        std::cerr<<"Unable to read velocities from "<<ss.str()<<std::endl;
        myexit(1);
      }
      for (int j=0; j<nget; ++j)
        buf[3*j+ZAxis] = buf[3*j+ZAxis]+buv[3*j+ZAxis];
#endif
#pragma omp parallel for shared(buf)
      for (int j=0; j<3*nget; ++j) buf[j] = periodic(buf[j]);
      float dx,dy,dz,Ngrid=0.9999*Ng;
      for (int j=0; j<nget; ++j) {
        float mm=bum[j];
        int ix=int(Ngrid*buf[3*j+0]); dx=Ngrid*buf[3*j+0]-ix;
        int iix=(ix+1)%Ng;
        int iy=int(Ngrid*buf[3*j+1]); dy=Ngrid*buf[3*j+1]-iy;
        int iiy=(iy+1)%Ng;
        int iz=int(Ngrid*buf[3*j+2]); dz=Ngrid*buf[3*j+2]-iz;
        int iiz=(iz+1)%Ng;
        D[Ng*Ng2* ix+Ng2* iy+ iz] += (1.0-dx)*(1.0-dy)*(1.0-dz)*mm;
        D[Ng*Ng2*iix+Ng2* iy+ iz] += (  dx  )*(1.0-dy)*(1.0-dz)*mm;
        D[Ng*Ng2* ix+Ng2*iiy+ iz] += (1.0-dx)*(  dy  )*(1.0-dz)*mm;
        D[Ng*Ng2* ix+Ng2* iy+iiz] += (1.0-dx)*(1.0-dy)*(  dz  )*mm;
        D[Ng*Ng2*iix+Ng2*iiy+ iz] += (  dx  )*(  dy  )*(1.0-dz)*mm;
        D[Ng*Ng2*iix+Ng2* iy+iiz] += (  dx  )*(1.0-dy)*(  dz  )*mm;
        D[Ng*Ng2* ix+Ng2*iiy+iiz] += (1.0-dx)*(  dy  )*(  dz  )*mm;
        D[Ng*Ng2*iix+Ng2*iiy+iiz] += (  dx  )*(  dy  )*(  dz  )*mm;
        mtotal+=mm;
        ntotal++;
      }
      ngot += nget;
    }
#ifdef	RED
    fv.close();
#endif
    fr.close();
    fm.close();
  }
  std::cout<<"# Read "<<ntotal<<" objects with weight "
           <<std::scientific<<mtotal<<" from "<<fbase<<std::endl;
  std::cout.flush();
  // Now normalize D to mean density.
  double mdens = ((mtotal/Ng)/Ng)/Ng;
#pragma omp parallel for shared(D,mdens)
  for (long j=0; j<D.size(); ++j) D[j]/=mdens;
  return(D);
}










std::vector<double>	read_part(const char fbase[], const int Nfile)
// Reads the particles and generates an Ng^3 density grid using CIC.
// The grid is in "FFTW-format".
{
  const long Ng2=Ng+2;
  std::vector<double> D;
  try{D.resize(Ng2*Ng*Ng);} catch(std::exception& e) {myexception(e);}
  long ntotal=0;
  for (int ifile=0; ifile<Nfile; ++ifile) {
    std::ostringstream ss;
    if (Nfile==1)
      ss<<fbase<<".bin";
    else
      ss<<fbase<<".bin."<<std::setw(2)<<std::setfill('0')<<ifile;
    std::ifstream fr(ss.str().c_str(),std::ios::binary);
    if (!fr) {
      std::cerr<<"Unable to open "<<ss.str()<<" for reading."<<std::endl;
      myexit(1);
    }
#ifdef	RED
    std::ifstream fv(ss.str().c_str(),std::ios::binary);
    if (!fv) {
      std::cerr<<"Unable to open "<<ss.str()<<" for reading."<<std::endl;
      myexit(1);
    }
#endif
    struct FileHeader {
      int   npart,nsph,nstar;
      float aa,softlen;
    } header;
    int eflag,hsize;
    fr.read((char *)&eflag,sizeof(int));
    if (fr.fail()){std::cerr<<"Unable to read "<<ss.str()<<std::endl;myexit(1);}
    fr.read((char *)&hsize,sizeof(int));
    if (fr.fail()){std::cerr<<"Unable to read "<<ss.str()<<std::endl;myexit(1);}
    if (hsize!=sizeof(header)) {
      std::cerr<<"Header size mismatch in "<<ss.str()<<std::endl;
      myexit(1);
    }
    fr.read((char *)&header,sizeof(header));
#ifdef	RED
    long seeksize = 2*sizeof(int) + sizeof(header);
    seeksize += 3L*header.npart*sizeof(float);
    fv.seekg(seeksize);
    if (fv.fail()){std::cerr<<"Unable to seek "<<ss.str()<<std::endl;myexit(1);}
#endif
    const int BlkSiz=16777216;
    std::vector<float> buf;
    try{buf.resize(3*BlkSiz);} catch(std::exception& e){myexception(e);}
#ifdef	RED
    std::vector<float> buv;
    try{buv.resize(3*BlkSiz);} catch(std::exception& e){myexception(e);}
#endif
    long ngot=0;
    while (ngot<header.npart) {
      int nget=(header.npart-ngot>BlkSiz)?BlkSiz:header.npart-ngot;
      fr.read((char *)&buf[0],3*nget*sizeof(float));
      if (fr.fail()) {
        std::cerr<<"Unable to read positions from "<<ss.str()<<std::endl;
        myexit(1);
      }
#ifdef	RED
      fv.read((char *)&buv[0],3*nget*sizeof(float));
      if (fv.fail()) {
        std::cerr<<"Unable to read velocities from "<<ss.str()<<std::endl;
        myexit(1);
      }
      for (int j=0; j<nget; ++j)
        buf[3*j+ZAxis] = buf[3*j+ZAxis]+buv[3*j+ZAxis];
#endif
#pragma omp parallel for shared(buf)
      for (int j=0; j<3*nget; ++j) buf[j] = periodic(buf[j]);
      float dx,dy,dz,Ngrid=0.9999*Ng;
      for (int j=0; j<nget; ++j) {
        int ix=int(Ngrid*buf[3*j+0]); dx=Ngrid*buf[3*j+0]-ix;
        int iix=(ix+1)%Ng;
        int iy=int(Ngrid*buf[3*j+1]); dy=Ngrid*buf[3*j+1]-iy;
        int iiy=(iy+1)%Ng;
        int iz=int(Ngrid*buf[3*j+2]); dz=Ngrid*buf[3*j+2]-iz;
        int iiz=(iz+1)%Ng;
        D[Ng*Ng2* ix+Ng2* iy+ iz] += (1.0-dx)*(1.0-dy)*(1.0-dz);
        D[Ng*Ng2*iix+Ng2* iy+ iz] += (  dx  )*(1.0-dy)*(1.0-dz);
        D[Ng*Ng2* ix+Ng2*iiy+ iz] += (1.0-dx)*(  dy  )*(1.0-dz);
        D[Ng*Ng2* ix+Ng2* iy+iiz] += (1.0-dx)*(1.0-dy)*(  dz  );
        D[Ng*Ng2*iix+Ng2*iiy+ iz] += (  dx  )*(  dy  )*(1.0-dz);
        D[Ng*Ng2*iix+Ng2* iy+iiz] += (  dx  )*(1.0-dy)*(  dz  );
        D[Ng*Ng2* ix+Ng2*iiy+iiz] += (1.0-dx)*(  dy  )*(  dz  );
        D[Ng*Ng2*iix+Ng2*iiy+iiz] += (  dx  )*(  dy  )*(  dz  );
        ntotal++;
      }
      ngot += nget;
    }
#ifdef	RED
    fv.close();
#endif
    fr.close();
  }
  std::cout<<"# Read "<<ntotal<<" particles from "<<fbase<<std::endl;
  std::cout.flush();
  // Now normalize D to mean density.
  double mdens = ((double(ntotal)/double(Ng))/Ng)/Ng;
#pragma omp parallel for shared(D,mdens)
  for (long j=0; j<D.size(); ++j) D[j]/=mdens;
  return(D);
}





std::vector<Source>	read_src(const char fname[])
{
  std::vector<Source> s;  s.reserve(10000);
  std::ifstream fs(fname);
  if (!fs) {
    std::cerr<<"Unable to open "<<fname<<" for reading."<<std::endl;
    myexit(1);
  }
  std::string buf;
  do {
    getline(fs,buf);
  } while (buf[0]=='#' && fs.eof()==0);
  try {
    while (fs.eof()==0) {
      struct Source ss;  float vel[3],flx;
      std::istringstream(buf) >> ss.pos[0] >> ss.pos[1] >> ss.pos[2]
                              >> vel[0]    >> vel[1]    >> vel[2] >> flx;
      if (flx>FluxMin && flx<FluxMax) s.push_back(ss);
      getline(fs,buf);
    }
  } catch(std::exception& e) {myexception(e);}
  fs.close();
  std::cout<<"# Read "<<s.size()<<" sources in range from "<<fname<<std::endl;
  return(s);
}





int	main(int argc, char **argv)
{
  if (argc!=5) {
    std::cout<<"Usage: match <fbase> <Nfile> <SrcFile> <zz>"<<std::endl;
    myexit(1);
  }
  double zz = atof(argv[4]);
  std::cout<<"# Matched filter running on "
           <<omp_get_max_threads()<<" threads."<<std::endl;
  std::cout<<"# Working with Lbox="<<Lbox<<"Mpc/h."<<std::endl;
  std::cout<<"# Computing FTs on a "<<Ng<<"^3 grid."<<std::endl;
#ifdef	RED
  std::cout<<"# Using redshift space with ZAxis="<<ZAxis<<std::endl;
#else
  std::cout<<"# Using real space."<<std::endl;
#endif
  std::cout<<"# Assuming observation redshift is "<<zz<<std::endl;
  std::cout<<"# Setting noise with effective nP("<<kFid<<")="<<nP<<std::endl;
  std::cout<<"# Using kzMin="<<kzMin<<", muMin="<<muMin<<std::endl;
  std::cout.flush();

  // Make cosmology and instrument.
  Cosmology  C;  C.set_zobs(zz);
  std::cout<<"# Comoving distance to z="<<zz
           <<" is "<<C.chi<<"Mpc/h."<<std::endl;
  const DishTelescope T(128,6.0,0.21*(1+zz),C.chi);
  //T.print_feed_hist();
  T.print_name();

  // Read the object data, making a density grid.
  std::vector<double> H = read_prop(argv[1],atoi(argv[2]));

  // Read the source data.
  std::cout<<"# Keeping sources with "<<FluxMin<<"<F<"<<FluxMax<<std::endl;
  std::vector<Source> S = read_src(argv[3]);

  // Print the distribution of S/N.
  match(H,S,C,T);

  return(0);
}
