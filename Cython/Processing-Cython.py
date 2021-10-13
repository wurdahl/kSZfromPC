#!/usr/bin/env python
# coding: utf-8

# In[1]:


import struct
import numpy as np
import os
from joblib import Parallel, delayed


# In[2]:


#import matplotlib.pyplot as plt
import healpy as hp
nside = 1024
npix = hp.nside2npix(nside)

#%matplotlib inline


# In[3]:


rangeOfInterest=512#only look at particles within this
radialDivs = 32
ROIs = np.linspace(0,rangeOfInterest, radialDivs+1)

boxSize=1024 # side length of box
particleSize=768 #the total number of particles is n**3

run_Ident = "_NS_"+str(nside)+"_R_"+str(rangeOfInterest)+"_P_"+str(particleSize)+"_DV_"+str(radialDivs)


# In[4]:
#this can convert cartesian data into radial data
#velocity still needs to be divided by angular diamter distance squared
import ProcessingFunctions

# In[5]:


# code for determinig angular diameter distance from comving distance

from classy import Class
from scipy.interpolate import interp1d

LambdaCDM = Class()

LambdaCDM.set({ 'omega_b':0.022032 , 'omega_cdm' :0.12038 , 'h':0.67556 , 'A_s':2.215e-9 , 'n_s' :0.9619 , 'tau_reio' :0.0925})

LambdaCDM.set({ 'output' :'tCl,pCl,lCl,mPk','lensing' :'yes' ,'P_k_max_1/Mpc': 3.0})

LambdaCDM.compute()

comovDist = LambdaCDM.get_background()['comov. dist.']
angDiaDist = LambdaCDM.get_background()['ang.diam.dist.']
redshifts = LambdaCDM.get_background()['z']

getAngDiaDist = interp1d(comovDist,angDiaDist)
getRedshift = interp1d(comovDist, redshifts)

def getScalingFactor(comov_dist):
    return 1/(1+getRedshift(comov_dist))

del comovDist
del angDiaDist
del redshifts
del LambdaCDM


# Reading lp-cola unformatted files
def unf_read_file(file, p_list=[], np=6):
    with open(file, mode='rb') as f:
        tot_n = 0
        cnt = 0
        while (True):
            cnt += 1
            r = f.read(4)
            if not r: break
            a1 = struct.unpack('i', r)
            r = f.read(a1[0])
            n = struct.unpack('i', r)
            r = f.read(8)
            a, b = struct.unpack('2i', r)
            r = f.read(b)
            p_list.extend(struct.unpack(str(n[0] * np) + 'f', r))
            r = f.read(4)
            tot_n += n[0]
    f.close()
    return tot_n


# In[8]:

direc = "/home/urdahl/home/dockerFirst/Data/"
inputFiles = os.listdir(direc)
print("Found "+str(len(inputFiles))+" Files")

#import time

def readSetToBins(startFile, stopFile, index):
    tempArray = []
    numcount = np.zeros((radialDivs,npix))
    totalVelThread = np.zeros((radialDivs,npix))
    for i in range(startFile,stopFile):
        path = direc + inputFiles[i]
        if os.path.isfile(path):
            #begin = time.perf_counter()
            unf_read_file(path, p_list=tempArray)
            print("Reading file "+path)
            reshaped = np.reshape(tempArray,(-1,6))

            tempArray = []
            #fileTime = time.perf_counter()
            #print("file done at "+str(fileTime-begin))
            offset = np.append((boxSize/2)*np.ones((np.shape(reshaped)[0],3)),np.zeros((np.shape(reshaped)[0],3)),axis=1)

            reshaped = np.subtract(reshaped,offset)
            del offset
            #timeOffset = time.perf_counter()
            #print("offset done at "+str(timeOffset-begin))
            sphereConversion = ProcessingFunctions.convertToSpherical(reshaped)
            del reshaped
            #reshaped = []

            sphereConversion[:,3] = sphereConversion[:,3]/np.power(getAngDiaDist(sphereConversion[:,0]),2)
            #timeConversion = time.perf_counter()
            #print("Sphere Conversion done at " +str(timeConversion-begin))
            #bin points into the correct bins (radial, theta, phi bins)

            for j in range(0,radialDivs):
                #select all the point in each radial range
                #beginSelect = time.perf_counter()
                radialRange = sphereConversion[(sphereConversion[:,0]>ROIs[j]) & (sphereConversion[:,0]<ROIs[j+1])]
                #endSelect = time.perf_counter()
                #print("it took "+str(endSelect-beginSelect)+"to select range")
                #determine how many points are in each bin
                pixIndicies = hp.ang2pix(nside,radialRange[:,1],radialRange[:,2])

                #do the math for the SZ effect

                #sum all velocities in each bin together
                #beginVel = time.perf_counter()
                if(len(pixIndicies)>0):
                    numcount[j] = np.add(numcount[j], ProcessingFunctions.binParticles(pixIndicies, npix))
                    totalVelThread[j] = np.add(totalVelThread[j],ProcessingFunctions.binVelocities(pixIndicies,radialRange[:,3],npix))
                #endVel = time.perf_counter()
                #print("it took "+str(endVel-beginVel)+"to bin Vel")

            #binTime = time.perf_counter()
            #print("Binning done at "+str(binTime-begin))
            print(str(i)+" done")

    return [numcount, totalVelThread]


# In[9]:

num_Files = len(inputFiles)
numProcess = int(num_Files/4)
ranges = np.linspace(0,num_Files,numProcess+1).astype(int)

returnValues = Parallel(n_jobs=-1)(delayed(readSetToBins)(ranges[i],ranges[i+1], i) for i in range(0,numProcess))

#big pause here for some reason, the program says its done with the last file but then it take 20 seconds to move on

print("Read all Files")


# In[10]:


#structure of returnValues is weird
#the first length is 32 corresponding to each process
#the second length is 2 corresponding to the two arrays that were returned from each process: numcount, totalVelThread
#the third length is number of radial divs
#the fourth length corresponds to the number of pixels in each projection

#combine the returns from each processor by summing the first axis

#this can be a very memory intensive part:

numpyReturn = np.array(returnValues)
del returnValues

outputCount = np.sum(numpyReturn[0:numProcess,0],axis=0)
outputkSZ = np.sum(numpyReturn[0:numProcess,1],axis=0)

del numpyReturn

# In[11]:


#combine the different radial divs for viewing
numcount = np.sum(outputCount,axis=0)
print(sum(numcount))

n_bar = np.average(numcount)
overdensity = (numcount-n_bar)/n_bar

#hp.mollview(numcount,xsize=3200, max=1000)


# In[12]:


#hp.mollview(overdensity,xsize=3200,max=5)
hp.fitsfunc.write_map("MAPS/overdensity"+run_Ident+".fits", overdensity, overwrite=True)


# In[13]:


almostkSZ = np.sum(outputkSZ,axis=0)
#hp.mollview(almostkSZ,xsize=3200,min=-30,max=30)


# In[14]:


#plt.hist(almostkSZ,bins=np.linspace(-30,30));


# In[15]:


OmegaB = 0.048
OmegaM = 0.31
fb = OmegaB/OmegaM

h = 0.69
H = (3.2407789/h)*10**-18
G = 6.674*10**-8
unitLength = 3.085678*10**24 #cm/h - This is one megaparsec
#unitMass = 1.989*10**43 #g
unitMass = (3.0*OmegaM*H*H*(boxSize*unitLength)**3)/(8.0*3.14*G*(particleSize**3))
unitVelocity = 10**5 #cm/s

c=2.99792458*10**10

YHe=0.2477
mH=1.6735575*(10**-24) #g
mHe=6.464731*(10**-24) #g
mu = (1-YHe)/mH+YHe/mHe

sigmaT = 6.6524*(10**-25) #cm^2

#convert l-picola units to SI units
correctUnits = almostkSZ*(unitMass*unitVelocity/unitLength**2)
#summation is now in (g/s)

almosterkSZ = -(sigmaT*fb*mu)*(correctUnits/(c*hp.nside2resol(nside)**2))
#hp.mollview(almosterkSZ,xsize=3200,min=-2*10**-6,max=2*10**-6)
hp.fitsfunc.write_map("MAPS/kSZ"+run_Ident+".fits", almosterkSZ, overwrite=True)


# In[16]:


#plt.hist(almosterkSZ,bins=np.linspace(-2*10**-6,2*10**-6));


# In[24]:


convergenceFactors = np.zeros((radialDivs,radialDivs));

dr = rangeOfInterest/radialDivs

#for each layer of the kSZ determine the accumalted convergence
for kSZLayer in range(0,radialDivs):
    #add up the convergence contribution of every intervening layer from 0 up to the current kSZ layer
    for lensingLayer in range(0,kSZLayer):
        kSZDist = (kSZLayer+0.5)*(dr)
        lensingLayerDist = (lensingLayer+0.5)*(dr)

        convergenceFactors[kSZLayer,lensingLayer] = (1/(lensingLayerDist*getScalingFactor(lensingLayerDist)))*(kSZDist-lensingLayerDist)/kSZDist


# In[33]:


#Now we determine the convergence maps

def getConvergenceForPixel(pixelIndex):

    dr = rangeOfInterest/radialDivs

    convergencePixels = np.zeros(radialDivs)

    #for each layer of the kSZ determine the accumalted convergence
    for kSZLayer in range(0,radialDivs):
        #add up the convergence contribution of every intervening layer from 0 up to the current kSZ layer
        for lensingLayer in range(0,kSZLayer):
            kSZDist = (kSZLayer+0.5)*(dr)
            lensingLayerDist = (lensingLayer+0.5)*(dr)

            convergencePixels[kSZLayer] = convergencePixels[kSZLayer] + outputCount[lensingLayer,pixelIndex]*(1/(lensingLayerDist*getScalingFactor(lensingLayerDist)))*(kSZDist-lensingLayerDist)/kSZDist

    return convergencePixels

def getConvergenceForPixelMat(pixelIndex):
    return np.dot(convergenceFactors,outputCount[:,pixelIndex])

def getConvergenceForRange(start, finish):
    convergences = np.zeros((finish-start,radialDivs))
    for pixel in range(start, finish):
        convergences[pixel-start] = getConvergenceForPixelMat(pixel)
    return convergences


# In[34]:


npix = hp.nside2npix(nside)
numProcess = 64
pixelSteps = np.linspace(0,npix,numProcess+1).astype(int)
print("Starting to Calculate Convergences")
convergenceReturn = Parallel(n_jobs=-1)(delayed(getConvergenceForRange)(pixelSteps[i],pixelSteps[i+1]) for i in range(0,numProcess))
print("Calculated Convergences")

#to each pixel:\n#convergenceReturn = Parallel(n_jobs=npix)(delayed(getConvergenceForPixel)(pixel) for pixel in range(0,npix))')


# In[35]:


#to not get a ragged array, npix must be divisible by numProcess
convergenceMaps = np.transpose(np.array(convergenceReturn).reshape((npix,radialDivs)))


# In[36]:


H0Squared = 1

prefactors = (3/2)*H0Squared*OmegaM

convergenceMaps = prefactors*convergenceMaps


# In[37]:


#hp.mollview(np.sum(convergenceMaps,axis=0))
hp.fitsfunc.write_map("MAPS/convergence"+run_Ident+".fits", np.sum(convergenceMaps,axis=0), overwrite=True)


# In[ ]: