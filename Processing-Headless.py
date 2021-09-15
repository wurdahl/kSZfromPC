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
nside = 64
npix = hp.nside2npix(nside)


# In[3]:


def convertToSpherical(xyz):
    # col 0: r, col 1: theta, col 2: phi
    sphericalPositons = np.zeros((len(xyz),3))
    xySquared = xyz[:,0]**2 + xyz[:,1]**2
    # r
    sphericalPositons[:,0] = np.sqrt(xySquared+xyz[:,2]**2)
    # theta - many options for this calc
    sphericalPositons[:,1] = np.arctan2(np.sqrt(xySquared), xyz[:,2])
    # phi
    sphericalPositons[:,2] = np.arctan2(xyz[:,1],xyz[:,0])
    return sphericalPositons


# In[4]:


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


# In[5]:


def readSetToBins(startFile, stopFile, index):
    tempArray = []
    numcount = []
    for i in range(startFile,stopFile):
        path = "Data/cone_test_lightcone."+str(i)
        if os.path.isfile(path):
            unf_read_file(path, p_list=tempArray)

            positionFormatted = np.reshape(tempArray,(-1,6))[:,0:3]
            tempArray = []
            
            positionFormatted = np.subtract(positionFormatted,64*np.ones(np.shape(positionFormatted)))
            
            sphereCoord = convertToSpherical(positionFormatted)
            
            positionFormatted = []
            
            #only take points within a certain radius
            sphereCoord = sphereCoord[sphereCoord[:,0]<200]

            pixIndicies = hp.ang2pix(nside,sphereCoord[:,1],sphereCoord[:,2])

            if len(numcount)==0:
                numcount = np.bincount(pixIndicies, minlength=npix)
            else:
                numcount = np.add(numcount, np.bincount(pixIndicies, minlength=npix) )
            if i%10==0:
                print(i)
        
        outputs[index] = numcount


# In[6]:


numThreads = 16
ranges = np.linspace(0,32,numThreads+1).astype(int)
outputs = [None]*numThreads

threads = [None]*numThreads

for i in range(0,numThreads):
    print("Reading files from "+str(ranges[i])+" to " +str(ranges[i+1]))
    threads[i] = threading.Thread(target=readSetToBins, args=(ranges[i],ranges[i+1], i))


for i in range(0,numThreads):
    threads[i].start()

for i in range(0,numThreads):
    threads[i].join()


# In[9]:


numcount = np.cumsum(outputs,axis=0)
print(sum(numcount[-1]))

print("NSIDE = "+str(nside))

hp.fitsfunc.write_map("Map.fits", numcount[-1], overwrite=True)


# In[ ]:





# In[ ]:




