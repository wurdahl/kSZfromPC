import ProcessingFunctions

import numpy as np
import cython

import time

data = np.random.rand(1000,3)

begin = time.perf_counter()
#output = ProcessingFunctions.getRadialUnitVecs(data)
end  = time.perf_counter()

print("Cython done in "+str(end-begin)+" seconds")

def getRadialUnitVecs(spherePos):
    unitVecs = np.zeros((len(spherePos),3))
    
    #negative because you want it pointing towards the origin    
    unitVecs[:,0] = np.multiply(np.cos(spherePos[:,2]),np.sin(spherePos[:,1]))
    unitVecs[:,1] = np.multiply(np.sin(spherePos[:,2]),np.sin(spherePos[:,1]))
    unitVecs[:,2] = np.cos(spherePos[:,1])
    
    return unitVecs

begin2 = time.perf_counter()
output2 = getRadialUnitVecs(data)
end2 = time.perf_counter()

print("numpy done in "+str(end2-begin2)+" seconds")


#testing spherical conversion

data = np.random.rand(1000,6)

begin3 = time.perf_counter()
output3 = ProcessingFunctions.convertToSpherical(data)
end3 = time.perf_counter()

print("cython done in "+str(end3-begin3)+ " seconds")

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
    
    vel[:,0] = xyz[:,3]
    vel[:,1] = xyz[:,4]
    vel[:,2] = xyz[:,5]
    
    unitVectors = getRadialUnitVecs(sphericalConversion)
    
    #get the radial 
    for i in range(len(xyz)):
        sphericalConversion[i,3] = np.dot(vel[i],unitVectors[i])
    
    return sphericalConversion

begin4 = time.perf_counter()
output4 = convertToSpherical(data)
end4 = time.perf_counter()

print("numpy done in "+str(end4-begin4)+" seconds")

npix = 12*(2048**2)
data = np.random.randint(npix,size=npix)
begin5 = time.perf_counter()
output5 = ProcessingFunctions.binParticles(data,npix)
end5 = time.perf_counter()
print("cython binning done in "+ str(end5-begin5))

begin6 = time.perf_counter()
output6 = np.bincount(data,minlength = npix)
end6 = time.perf_counter()
print("numpy binning done in "+str(end6-begin6))




