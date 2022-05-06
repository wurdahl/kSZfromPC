import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import pywigxjpf as pywig
from joblib import Parallel, delayed
import sys, argparse, multiprocessing
from common import *
from scipy.signal import savgol_filter

print("  Creating pywig tables...")
NSIDE_MAX = 4096
pywig.wig_table_init(3*NSIDE_MAX, 3)
pywig.wig_temp_init(3*NSIDE_MAX)


####################
## Reconstruction ##
####################

L_RECONST_MAX = 50

def GammakSZ(l1, l2, l, Cltd) :
  pref = np.sqrt((2.0*l1+1)*(2.0*l2+1)*(2.0*l+1)/4.0/np.pi)
  wig = pywig.wig3jj(2*l1, 2*l2, 2*l, 0, 0, 0) # 2*[j1,j2,j3,m1,m2,m3]
  return pref*wig*Cltd[l2]

def getNinv(l, ls, Cltd, ClTT, Cldd) :
  Ninv = 0.0
  if l < L_RECONST_MAX+1 :
    print("Working on l =", l)
    for l1 in ls: # TODO: don't include monopole, dipole contributions?
      for l2 in ls:
        Ninv += GammakSZ(l1, l2, l, Cltd)**2 / ClTT[l1] / Cldd[l2]
    Ninv /= (2.0*l + 1.0)
  else :
    Ninv = 1.0e50 # N = 1.0e-50
  return Ninv

if True :

  NSIDE_WORKING = NSIDE
  #OUTPUT_DIR = MAPS_OUTPUT_DIR+"ksz_reconstruction/"
  OUTPUT_DIR = "."
  #mkdir_p(OUTPUT_DIR)

  #Overdensity maps
  #tau_map = -1.0*getMap("N_uncertain", NSIDE=NSIDE_WORKING) #
  tau_map = -1.0*hp.read_map("./MAPS/overdensity_NS_512_R_2048_P_2048_DV_64.fits")
  rho_map = -1.0*tau_map#getMap("N_uncertain", NSIDE=NSIDE_WORKING) # _uncertain
  
  #Velocity Maps
  #vrad_map = getMap("vrad", NSIDE=NSIDE_WORKING) # Directly averaged velocity
  vrad_map = hp.read_map("./MAPS/velocityField_NS_512_R_2048_P_2048_DV_64.fits")
  #kSZ Map
  #ksz_map = getMap("../ksz", NSIDE=NSIDE_WORKING) # websky kSZ
  ksz_map = hp.read_map("./MAPS/kSZ_NS_512_R_2048_P_2048_DV_64.fits")
  # ksz_map = getMap("ksz", NSIDE=NSIDE_WORKING) # kSZ for this bin only
  # ksz_map = getMap("../ksz_halos", NSIDE=NSIDE_WORKING) # kSZ from halo catalogue only

 
 
  #Read in CMB Map
  CMB_map = 0.0*hp.sphtfunc.synfast(cls=np.ones(6144),nside=512);#hp.alm2map(CMB_alms, NSIDE_WORKING)
  CMB_alms = hp.map2alm(CMB_map) #hp.fitsfunc.read_alm('lensed_alm.fits').astype(np.complex)
  #Higher redshift kSZ that we aven't modelled- Later
  #patchy_ksz_map = getMap("../ksz_patchy", NSIDE=NSIDE_WORKING) # websky patchy kSZ
  #Residual tSZ - contaminant
  #tsz_map = T_CMB*y_to_tSZ*getMap("../tsz", NSIDE=NSIDE_WORKING) # websky tSZ
  Obs_T_map = ksz_map + CMB_map# + patchy_ksz_map + tsz_map

  #CMB power spectra
  CMB_PS = hp.anafast(CMB_map)
  ksz_PS = hp.anafast(ksz_map)
  # patchy_ksz_PS = hp.anafast(patchy_ksz_map)
  # tsz_PS = hp.anafast(tsz_map)
  # Obs_T_PS = hp.anafast(Obs_T_map)
  psplot(CMB_PS, label="CMB")
  psplot(ksz_PS, label="kSZ")
  # psplot(patchy_ksz_PS, label="Patchy kSZ")
  # psplot(tsz_PS, label="tSZ")
  # psplot(Obs_T_PS, label="Total")
  plt.legend()
  plt.savefig(OUTPUT_DIR+"CMB_PS.png")
  plt.close()


  # try reconstructing...
  print("Generating power spectra.")
  ClTT = hp.anafast(Obs_T_map)
  ClTT_filtered = np.concatenate(([1.0], savgol_filter(ClTT[1:], 51, 3)))
  # Cldd = hp.anafast(rho_map)
  # Cltd = hp.anafast(rho_map, map2=tau_map)
  ls = np.arange(ClTT.size)

  print("Generating alms.")
  dTlm = hp.map2alm(Obs_T_map)
  dlm = hp.map2alm(rho_map)

  print("Generating rescaled alms.")
  dTlm_resc = hp.almxfl(dTlm, 1.0/ClTT)
  dT_resc = hp.alm2map(dTlm_resc, NSIDE)
  dlm = -1.0*dlm # dlm_resc = hp.almxfl(dlm, 1.0) #Cltd/Cldd
  d_resc = hp.alm2map(dlm, NSIDE)


  # Compute noise (expensive, need to optimize?)
  # print("Computing noise.")
  # ncores = multiprocessing.cpu_count()
  # Ninv = [ getNinv(l, ls, Cltd, ClTT, Cldd) for l in ls ]
  # Ninv = Parallel(n_jobs=ncores)(delayed(getNinv)(l, ls, Cltd, ClTT, Cldd) for l in ls)
  # N = 1.0/np.array(Ninv)
  # N = np.zeros_like(ls, dtype=np.int)
  # N[:100] = 1.0

  unnorm_veff_reconstlm = hp.map2alm(dT_resc*d_resc)
  unnorm_veff_reconst_ps = hp.alm2cl(unnorm_veff_reconstlm)
  unnorm_veff_reconst = hp.alm2map(unnorm_veff_reconstlm, NSIDE)

  dT_resc_ps = hp.anafast(dT_resc)
  d_resc_ps = hp.anafast(d_resc)
  psplot(dT_resc_ps)
  psplot(d_resc_ps)
  plt.savefig(OUTPUT_DIR+'interm_ps.png')
  plt.close()

  # # Plot reconstructed velocity maps
  # hp.mollview(veff_reconst)
  # plt.title('Reconstructed velocity map')
  # plt.savefig(OUTPUT_DIR+'veff_reconst_map.png')
  # plt.close()
  # hp.mollview(vrad_map)
  # plt.title('Simulated velocity map')
  # plt.savefig(OUTPUT_DIR+'veff_map.png')
  # plt.close()

    
  print("Plot stuff")
  # Plot velocity power spectra
  vrad_PS = hp.anafast(vrad_map)
  psplot(vrad_PS, label="True velocity", norm=True)
  psplot(unnorm_veff_reconst_ps, label="Unnorm. reconstructed velocity", norm=True)
  plt.legend()
  plt.savefig(OUTPUT_DIR+"velocity_PS.png")
  plt.close()

  # Correlation between velocity maps
  plt.semilogx( hp.anafast(vrad_map, unnorm_veff_reconst)/np.sqrt(vrad_PS*unnorm_veff_reconst_ps), label="vrad_map")
  plt.legend()
  plt.savefig(OUTPUT_DIR+"corr_coeff.png")
  plt.close()