#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# In[18]:


def Correlation(map1, map2):
    return hp.anafast(map1,map2)/np.sqrt(hp.anafast(map1)*hp.anafast(map2))


# In[16]:


fileNames = os.listdir('./MAPS/kSZ/')
nside=2048


# In[3]:


kSZ = np.load('./MAPS/kSZ/'+fileNames[0])

for i in range(1,3):
    kSZ = np.vstack((kSZ,np.load('./MAPS/kSZ/'+fileNames[i])))


# In[4]:


overdensity = np.load('./MAPS/overdensity/'+fileNames[0])

for i in range(1,3):
    overdensity = np.vstack((overdensity,np.load('./MAPS/overdensity/'+fileNames[i])))


# In[5]:


velocityField =  np.load('./MAPS/velocityField/'+fileNames[0])

for i in range(1,3):
    velocityField = np.vstack((velocityField,np.load('./MAPS/velocityField/'+fileNames[i])))


# In[6]:


lensedkSZ =  np.load('./MAPS/lensedkSZ/'+fileNames[0])

for i in range(1,3):
    lensedkSZ = np.vstack((lensedkSZ,np.load('./MAPS/lensedkSZ/'+fileNames[i])))


# In[7]:


lensedOverdensity =  np.load('./MAPS/lensedOverdensity/'+fileNames[0])

for i in range(1,3):
    lensedOverdensity = np.vstack((lensedOverdensity,np.load('./MAPS/lensedOverdensity/'+fileNames[i])))


# # Quadratic Estimator

# In[8]:


import camb
#import pywigxjpf as pywig
from joblib import Parallel, delayed
import sys, argparse, multiprocessing
#from common import *
from scipy.signal import savgol_filter


# In[9]:


#Make Fake CMB
h=0.69
pars = camb.CAMBparams()
pars.set_cosmology(H0=100.0*h, ombh2=0.048*h**2, omch2=0.262*h**2, mnu=0.06, omk=0)
pars.InitPower.set_params(As=2e-9, ns=0.96, r=0)
pars.set_for_lmax(6144, lens_potential_accuracy=0)
results = camb.get_results(pars)
powers =results.get_cmb_power_spectra(pars, CMB_unit='K')
l=np.arange(0,len(powers['total'][:,0]))
cambFactor = l*(l+1)/(2*np.pi)
CMB_camb = powers['total'][:,0]/cambFactor
CMB_camb[0]=0.0

#rho map is just the overdensity
def quadEst(ksz_map, rho_map):
    
    ksz_PS = hp.anafast(ksz_map)

    CMB_map = hp.sphtfunc.synfast(CMB_camb,nside=nside);
    
    Obs_T_map = ksz_map + CMB_map

    ClTT = hp.anafast(Obs_T_map)
    ClTT_filtered = np.concatenate(([1.0], savgol_filter(ClTT[1:], 51, 3)))
    
    dTlm = hp.map2alm(Obs_T_map)
    dlm = hp.map2alm(rho_map)

    dTlm_resc = hp.almxfl(dTlm, 1.0/ClTT)
    dT_resc = hp.alm2map(dTlm_resc, nside)
    dlm = -1.0*dlm # dlm_resc = hp.almxfl(dlm, 1.0) #Cltd/Cldd
    d_resc = hp.alm2map(dlm, nside)

    unnorm_veff_reconstlm = hp.map2alm(dT_resc*d_resc)
    unnorm_veff_reconst_ps = hp.alm2cl(unnorm_veff_reconstlm)
    unnorm_veff_reconst = hp.alm2map(unnorm_veff_reconstlm, nside)

    return unnorm_veff_reconst


# In[19]:


totalCorrelation = Correlation(quadEst(kSZ[0],overdensity[0]),velocityField[0])

for i in range(1,3):
    totalCorrelation = totalCorrelation + Correlation(quadEst(kSZ[i],overdensity[i]),velocityField[i])


# In[22]:


totalCorrelationLensed = Correlation(quadEst(lensedkSZ[0],lensedOverdensity[0]),velocityField[0])

for i in range(1,3):
    totalCorrelationLensed = totalCorrelationLensed + Correlation(quadEst(lensedkSZ[i],lensedOverdensity[i]),velocityField[i])


# In[25]:


fig = plt.figure(dpi=800)
plt.semilogx(totalCorrelation/4)
plt.semilogx(totalCorrelationLensed/4)
plt.legend(["Not Lensed","Lensed"])
plt.title("Velocity and Velocity Reconstruction Correlation")
fig.savefig("CorrelationComparison.png")


# In[ ]:




