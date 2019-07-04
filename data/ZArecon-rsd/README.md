the cross and auto spectra with the true fields is for reconstructed field in recon.... as well as noise+wedge field in dataw.... file. 

1d power spectra files have columns:

k, xm.power, xs.power, xd.power, pm1.power, pm2.power, ps1.power, ps2.power, pd1.power, pd2.power

where m=data i.e HI, s=initial filed, d=final field
1 is reconstruction/noisy data, while 2 is truth

2d power spectra is in Nmu8 subfolder and each of the columns above is now a new file with that suffix,
and columns in files are different (8) mu-bins.

Files are generated in code/recon/saverep.py which uses code/recon/cosmo4d/report.py to estimate power spectra, in case you need to check the order of the output. 
