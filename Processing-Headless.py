#!/usr/bin/env python
# coding: utf-8

# In[1]:


import struct
import numpy as np
import os
import threading


# In[2]:


import matplotlib.pyplot as plt
import healpy as hp
nside = 128
npix = hp.nside2npix(nside)

#get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


def getRadialUnitVecs(spherePos):
    unitVecs = np.zeros((len(spherePos),3))
    
    #negative because you want it pointing towards the origin    
    unitVecs[:,0] = -np.multiply(np.cos(spherePos[:,2]),np.sin(spherePos[:,1]))
    unitVecs[:,1] = -np.multiply(np.sin(spherePos[:,2]),np.sin(spherePos[:,1]))
    unitVecs[:,2] = -np.cos(spherePos[:,1])
    
    return unitVecs


# In[4]:


def convertToSpherical(xyz):
    #variables
    sphericalConversion = np.zeros((len(xyz),4))
    vel = np.zeros((len(xyz),3))
    
    # col 0: r, col 1: theta, col 2: phi
    
    xySquared = xyz[:,0]**2 + xyz[:,1]**2
    # r
    sphericalConversion[:,0] = np.sqrt(xySquared+xyz[:,2]**2)
    # theta - many options for this calc
    sphericalConversion[:,1] = np.arctan2(np.sqrt(xySquared), xyz[:,2])
    # phi
    sphericalConversion[:,2] = np.arctan2(xyz[:,1],xyz[:,0])
    
    #convert velocity to radial velocity over radius
    
    vel[:,0] = xyz[:,3]/sphericalConversion[:,0]
    vel[:,1] = xyz[:,4]/sphericalConversion[:,0]
    vel[:,2] = xyz[:,5]/sphericalConversion[:,0]
    
    unitVectors = getRadialUnitVecs(sphericalConversion)
    
    #get the radial 
    for i in range(len(xyz)):
        sphericalConversion[i,3] = np.dot(vel[i],unitVectors[i])
    
    return sphericalConversion


# In[5]:


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


# In[6]:


def readSetToBins(startFile, stopFile, index):
    tempArray = []
    numcount = []
    totalVelThread = np.zeros(npix)
    for i in range(startFile,stopFile):
        path = "Data/cone_test_lightcone."+str(i)
        if os.path.isfile(path):
            unf_read_file(path, p_list=tempArray)

            reshaped = np.reshape(tempArray,(-1,6))
            tempArray = []
                                 
            offset = np.append(64*np.ones((np.shape(reshaped)[0],3)),np.zeros((np.shape(reshaped)[0],3)),axis=1)
            
            reshaped = np.subtract(reshaped,offset)
            
            sphereConversion = convertToSpherical(reshaped)
            
            reshaped = []
            
            #only take points within a certain radius
            sphereConversion = sphereConversion[sphereConversion[:,0]<30]

            #determine how many points are in each bin
            pixIndicies = hp.ang2pix(nside,sphereConversion[:,1],sphereConversion[:,2])

            if len(numcount)==0:
                numcount = np.bincount(pixIndicies, minlength=npix)
            else:
                numcount = np.add(numcount, np.bincount(pixIndicies, minlength=npix) )
                
            #do the math for the SZ effect
            
            #sum all velocities in each bin together
            if(len(pixIndicies)>0):
                for j in range(np.amin(pixIndicies),np.amax(pixIndicies)+1):
                    velInBin = sphereConversion[pixIndicies==j][:,3]
                    totalVelThread[j] = np.sum(velInBin,axis=0)
            
            if i%10==0:
                print(i)
        
        #set global variable to output for return
        outputCount[index] = numcount
        outputSZ[index] = totalVelThread


# In[7]:


numThreads = 32
ranges = np.linspace(0,32,numThreads+1).astype(int)
outputCount = [None]*numThreads
outputSZ = [None]*numThreads

threads = [None]*numThreads

for i in range(0,numThreads):
    print("Reading files from "+str(ranges[i])+" to " +str(ranges[i+1]))
    threads[i] = threading.Thread(target=readSetToBins, args=(ranges[i],ranges[i+1], i))


for i in range(0,numThreads):
    threads[i].start()

for i in range(0,numThreads):
    threads[i].join()


# In[8]:


numcount = np.sum(outputCount,axis=0)
print(sum(numcount))

n_bar = np.average(numcount)
overdensity = (numcount-n_bar)/n_bar

hp.fitsfunc.write_map("DensityCount.fits", numcount, overwrite=True)


# In[9]:


hp.fitsfunc.write_map("overdensity.fits", overdensity, overwrite=True)


# In[14]:


almostSZ = np.sum(outputSZ,axis=0)
hp.fitsfunc.write_map("kSZ.fits", almostSZ, overwrite=True)


# In[ ]:




