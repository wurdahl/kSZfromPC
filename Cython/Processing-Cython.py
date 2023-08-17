#!/usr/bin/env python
# coding: utf-8
import sys
print(sys.version)

import struct
import numpy as np
import os
from joblib import Parallel, delayed

#import matplotlib.pyplot as plt
import healpy as hp
#The third command line arguement sets the nside
nside = int(sys.argv[3])
npix = hp.nside2npix(nside)

#Hubble parameter
h = 0.69

#The fourth arguement is the range of interest
rangeOfInterest = int(sys.argv[4])/h #only look at particles within this
#The fifth command line arguemnt sets the number of radial divisions to use
#This is the number of radial blocks the data will be sorted into
radialDivs = int(sys.argv[5])

ROIs = np.linspace(0,rangeOfInterest, radialDivs+1)

boxSize= 4096/h # side length of box
particleSize = 2048 #the total number of particles is n**3

#The second command line arguement is the random seed used for the l-picola simulation
SimSeed = sys.argv[2]

run_Ident = "_NS_"+str(nside)+"_R_"+str(int(rangeOfInterest))+"_P_"+str(particleSize)+"_DV_"+str(radialDivs)+"_Sd_"+SimSeed

#the name of the directory in AllData where the data you want to analyze
dataDirec = sys.argv[1]+"/"

direc = "./"+dataDirec

#this can convert cartesian data into radial data
#velocity still needs to be divided by angular diamter distance squared
import ProcessingFunctions

# code for determinig angular diameter distance from comving distance

from classy import Class
from scipy.interpolate import interp1d

LambdaCDM = Class()

LambdaCDM.set({ 'omega_b':0.022032 , 'omega_cdm' :0.125 , 'h': h, 'A_s':2.215e-9 , 'n_s' :0.96 , 'tau_reio' :0.0925})

LambdaCDM.set({ 'output' :'tCl,pCl,lCl,mPk','lensing' :'yes' ,'P_k_max_1/Mpc': 3.0})

LambdaCDM.compute()

comovDist = LambdaCDM.get_background()['comov. dist.']
angDiaDist = LambdaCDM.get_background()['ang.diam.dist.']
redshifts = LambdaCDM.get_background()['z']

getAngDiaDist = interp1d(comovDist,angDiaDist)
getRedshift = interp1d(comovDist, redshifts)

def getScalingFactor(comov_dist):
    #add the window function which is the normalized dN/Dx which is from the data, then convert that to a dN/dz to feed into ccl
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



inputFiles = os.listdir(direc)
print("Found "+str(len(inputFiles))+" Files")

#import time

def readSetToBins(fileName, index):
    tempArray = []
    path = direc + fileName

    if os.path.isfile(path):
            
        unf_read_file(path, p_list=tempArray)
        print("Reading file "+path)
        reshaped = np.reshape(tempArray,(-1,6))
        
        #divide the position particles by h to convert them into MPc instead of MPc/h
        reshaped = reshaped/np.array([h,h,h,1,1,1])
    
        tempArray = []
            
        numParticles = reshaped.shape[0]
            
        #contains which radial bin each particle is in
        partRadial =  np.zeros(numParticles)

        #move the box so it is centered around the origin
        offset = np.append((boxSize/2)*np.ones((np.shape(reshaped)[0],3)),np.zeros((np.shape(reshaped)[0],3)),axis=1)
            
        reshaped = np.subtract(reshaped,offset)
        del offset
            
        sphereConversion = ProcessingFunctions.convertToSpherical(reshaped)
        del reshaped
        

        #extract the radial velocity of the particles
        partVelocity = sphereConversion[:,3]

        #make a velocity over angular diameter distance conversion for use later in making kSZ
        partVelocityAdjusted = sphereConversion[:,3]/np.power(getAngDiaDist(sphereConversion[:,0]),2)
            
        #get the pixel index for each particle              
        partIndex = hp.ang2pix(nside,sphereConversion[:,1],sphereConversion[:,2])

        for j in range(0,radialDivs):
            #set the values of the array to the correspodning radial bin
            #this sorts each particle into the correct radial bin
            partRadial[(sphereConversion[:,0]>ROIs[j]) & (sphereConversion[:,0]<ROIs[j+1])] = j
        
        #If the particle is outside the circle, then it is set to negative one, which will be used later to exclude it
        partRadial[(sphereConversion[:,0]>ROIs[j+1])] = -1

	#also is the particle is within the minimum radius, then it is also set to -1 so that it will be excluded
        partRadial[(sphereConversion[:,0]<ROIs[0])] = -1
                                          
    #print(str(index)+" done")
   
    return [partIndex,partRadial, partVelocity, partVelocityAdjusted]


num_Files = len(inputFiles)
numProcess = num_Files

if(num_Files==0):
    print("no files found, likely ran out of memory in l-picola")
    exit()

returnValues = Parallel(n_jobs=6)(delayed(readSetToBins)(inputFiles[i], i) for i in range(0,numProcess))

#big pause here for some reason, the program says its done with the last file but then it take 20 seconds to move on

print("Read all Files")


#combine the returns from each processor by summing the first axis

#this can be a very memory intensive part:

#make the longest list possible that can hold all of the particles
outputIndicies = np.full((particleSize**3+6000),0)
outputRadial = np.full((particleSize**3+6000),-1)
outputVelocity = np.full((particleSize**3+6000),0.0,dtype=np.double)
outputVelocityAdjusted = np.full((particleSize**3+6000),0.0,dtype=np.double)

#unroll the array into single long arrays
#this is where each loop of the array should start writing to the array
beginSec = 0

for i in range(0,numProcess):
    #this is the number of particles in each returned list
    listParticles = returnValues[i][0].shape[0]
    #print("Particles in file: " + str(listParticles))
    #the first return of each process is particleIndex
    
    outputIndicies[beginSec:beginSec+listParticles] = returnValues[i][0]
    
    #the second return is the radial index
    outputRadial[beginSec:beginSec+listParticles] = returnValues[i][1]

    #the third return is the particle's radial Velocity
    outputVelocity[beginSec:beginSec+listParticles] = returnValues[i][2]

    #the fourth return is the particle's radial velocity divided by the angular diameter distance squared
    outputVelocityAdjusted[beginSec:beginSec+listParticles] = returnValues[i][3]
    
    beginSec = beginSec + listParticles
del returnValues

print("Made numpy arrays and deleted return array")
print("Mean distance:" + str(np.mean(outputRadial)))
allMaps = ProcessingFunctions.binParticles(outputIndicies, outputRadial.astype(np.int64), outputVelocity, outputVelocityAdjusted, npix, radialDivs)

del outputIndicies
del outputRadial
del outputVelocity
del outputVelocityAdjusted

print("Binnned particles into maps")

outputCount = allMaps[0]
velocityField = allMaps[1]
adjustedVelocityField = allMaps[2]

del allMaps

#check for NANs
if(np.any(np.isnan(outputCount))):
    print("NAN is outputCount, program failure")
    exit()


#divide velocity field by output count in otder to get average velocity in each cell
velocityField = np.divide(velocityField,outputCount)
velocityField[velocityField==np.inf] = 0
velocityField[np.isnan(velocityField)] = 0

adjustedVelocityField = np.divide(adjustedVelocityField,outputCount)
adjustedVelocityField[adjustedVelocityField==np.inf] = 0
adjustedVelocityField[np.isnan(adjustedVelocityField)] = 0


#combine the different radial divs for viewing
numcount = np.sum(outputCount,axis=0)
print(sum(numcount))

dNdx = np.sum(outputCount,axis=1)/np.sum(outputCount)

n_bar = np.average(numcount)
overdensity = (numcount-n_bar)/n_bar

if(np.any(np.isnan(overdensity))):
    print("overdensity is nan, program failure")
    exit()

#we want the middle of the ROIs to represent where the box is
def moving_average(a, n=2) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

#convert the l-picola coordinates into actual MPc and get the value in between the layer boundaries
comovValues = moving_average(ROIs)

#convert the comoving coordinates into redshift values
redshiftValues = getRedshift(comovValues)

#hp.fitsfunc.write_map("./MAPS/overdensity"+run_Ident+".fits", overdensity, overwrite=True)
np.save("./MAPS/overdensity/"+run_Ident, overdensity)
#Save full density data:
#np.save("./MAPS/density"+run_Ident,outputCount)

#Save the eight Super Bins of overdenstiy starting from half way to the edge
start = radialDivs//2
for i in range(0,8):
    layerCount = np.sum(outputCount[start+start//8*(i):start+start//8*(i+1)],axis=0)
    layerAvg = np.average(layerCount)
    np.save("./MAPS/overdensity/"+run_Ident+"_SB_"+str(i), (layerCount-layerAvg)/layerAvg)

velocityFieldMap = np.sum(velocityField,axis=0)

#hp.fitsfunc.write_map("./MAPS/velocityField"+run_Ident+".fits", velocityFieldMap, overwrite=True)
np.save("./MAPS/velocityField/"+run_Ident, velocityFieldMap)

#Save the eight Super Bins of velocityField starting from half way to the edge
start = radialDivs//2
for i in range(0,8):
    np.save("./MAPS/velocityField/"+run_Ident+"_SB_"+str(i), np.sum(velocityField[start+start//8*(i):start+start//8*(i+1)],axis=0))   


# In[15]:

OmegaB = 0.048
OmegaM = 0.31
fb = OmegaB/OmegaM

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
correctUnits = adjustedVelocityField*(unitMass*unitVelocity/unitLength**2)
#summation is now in (g/s)

almosterkSZ = -(sigmaT*fb*mu)*(outputCount*correctUnits/(c*hp.nside2resol(nside)**2))
#hp.mollview(almosterkSZ,xsize=3200,min=-2*10**-6,max=2*10**-6)
#hp.fitsfunc.write_map("./MAPS/kSZ"+run_Ident+".fits", np.sum(almosterkSZ[1:],axis=0), overwrite=True)
#hp.fitsfunc.write_map("./MAPS/kSZ"+run_Ident+".fits", np.sum(almosterkSZ[1:],axis=0))
np.save("./MAPS/kSZ/"+run_Ident, np.sum(almosterkSZ[1:],axis=0))


convergenceFactors = np.zeros((radialDivs,radialDivs));

dr = rangeOfInterest/radialDivs

#for each layer of the kSZ determine the accumalted convergence
for lensedLayer in range(0,radialDivs):
    #add up the convergence contribution of every intervening layer from 0 up to the current kSZ layer
    for lensingLayer in range(0,lensedLayer):
        lensedDist = (lensedLayer+0.5)*(dr)
        lensingLayerDist = (lensingLayer+0.5)*(dr)
        
        convergenceFactors[lensedLayer,lensingLayer] = (1/(lensingLayerDist*getScalingFactor(lensingLayerDist)))*(lensedDist-lensingLayerDist)/lensedDist

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
convergenceReturn = Parallel(n_jobs=32)(delayed(getConvergenceForRange)(pixelSteps[i],pixelSteps[i+1]) for i in range(0,numProcess))
print("Calculated Convergences")

#to each pixel:\n#convergenceReturn = Parallel(n_jobs=npix)(delayed(getConvergenceForPixel)(pixel) for pixel in range(0,npix))')


# In[35]:


#to not get a ragged array, npix must be divisible by numProcess
convergenceMaps = np.transpose(np.array(convergenceReturn).reshape((npix,radialDivs)))


# In[36]:

H = 72.1*10**3
c=2.998*10**8

H0Squared = (H/c)**2

prefactors = (3/2)*H0Squared*OmegaM*((boxSize)**3)/(particleSize**3*hp.pixelfunc.nside2pixarea(nside))

convergenceMaps = prefactors*convergenceMaps

#hp.mollview(np.sum(convergenceMaps,axis=0))
#hp.fitsfunc.write_map("./MAPS/convergence"+run_Ident+".fits", convergenceMaps[-1], overwrite=True)

#hp.fitsfunc.write_map("./MAPS/midConvergence"+run_Ident+".fits", convergenceMaps[radialDivs//2], overwrite=True)

#np.save("./MAPS/convergence"+run_Ident,convergenceMaps)

kalms = hp.sphtfunc.map2alm(convergenceMaps[-1])

lmax=hp.Alm.getlmax(len(kalms))
ls, ms = hp.Alm.getlm(lmax)
lFactor = -ls*(ls+1)
lFactor[0] = 1

baseAngle = hp.pixelfunc.pix2ang(nside, np.arange(0,npix))

lensedkSZ = np.zeros(npix)
lensedOverdensity = np.zeros((radialDivs,npix))


for i in range(1,radialDivs):
    kalms=hp.sphtfunc.map2alm(convergenceMaps[i])
    
    lensPotential = -2.0*kalms/(lFactor)
    lensPotential[0]=0+0j
    
    divLensPot = hp.alm2map_der1(lensPotential,nside)
    
    #deflectedTheta = baseAngle[0]+divLensPot[1]
    deflectedTheta = baseAngle[0]
    #deflectedPhi = baseAngle[1]+divLensPot[2]
    deflectedPhi = baseAnlgle[1]+0.001

    if(i%50==0):
        print("add layer "+ str(i))
 
    lensedkSZ = lensedkSZ + hp.pixelfunc.get_interp_val(almosterkSZ[i],deflectedTheta,deflectedPhi)

    lensedOverdensity[i] = hp.pixelfunc.get_interp_val(outputCount[i],deflectedTheta, deflectedPhi)

print("The averages for lensedOverdensity are "+str(np.mean(lensedOverdensity,axis=1)))
print("The average for lensedkSZ is "+str(np.mean(lensedkSZ)))

#hp.fitsfunc.write_map("./MAPS/lensedkSZ"+run_Ident+".fits", lensedkSZ, overwrite=True)
np.save("./MAPS/lensedkSZ/"+run_Ident, lensedkSZ)

lensedOverdensityOverall = np.sum(lensedOverdensity,axis=0)
lensedOverdensityOverall = lensedOverdensityOverall/np.mean(lensedOverdensityOverall)-1

np.save("./MAPS/lensedOverdensity/"+run_Ident, lensedOverdensityOverall)

#Save the eight Super Bins of lensed overdensity starting from half way to the edge
start = radialDivs//2
for i in range(0,8):
    superBinLensedOverdensity = np.sum(lensedOverdensity[start+start//8*(i):start+start//8*(i+1)],axis=0)
    superBinLensedOverdensity = superBinLensedOverdensity/np.mean(superBinLensedOverdensity)-1
    np.save("./MAPS/lensedOverdensity/"+run_Ident+"_SB_"+str(i), superBinLensedOverdensity )   

