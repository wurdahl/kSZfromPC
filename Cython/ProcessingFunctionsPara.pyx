# distutils: language=c++

import numpy as np
cimport numpy as np

import healpy as hp
import os
import struct

from libc.math cimport sin
from libc.math cimport cos
from libc.math cimport pow
from libc.math cimport sqrt
from libc.math cimport atan2

from libcpp cimport vector

#getRadialUnitVecs
cdef getRadialUnitVecs(double[:,:] spherePos): 

    cdef int rows = spherePos.shape[0]
    unitVecs = np.zeros((rows,3))
    cdef double[:,:] unitVecs_view = unitVecs

    cdef int i

    for i in range(0,rows):
        unitVecs_view[i,0] = cos(spherePos[i,2])*sin(spherePos[i,1])
        unitVecs_view[i,1] = sin(spherePos[i,2])*sin(spherePos[i,1])
        unitVecs_view[i,2] = cos(spherePos[i,1])

    return unitVecs

#convertToSpherical

cpdef convertToSpherical(double[:,:] xyz):
    cdef int rows = xyz.shape[0]
    #variables
    sphericalConversion = np.zeros((rows,4))
    cdef double[:,:] sphericalConversion_view = sphericalConversion    
    
    # col 0: r, col 1: theta, col 2: phi
    cdef int i
    cdef double xySquared     

    for i in range(0,rows):

        xySquared = pow(xyz[i,0],2) + pow(xyz[i,1],2)

        # r
        sphericalConversion_view[i,0] = sqrt(xySquared+pow(xyz[i,2],2))
        
        # theta - many options for this calc
        sphericalConversion_view[i,1] = atan2(sqrt(xySquared), xyz[i,2])
    
        # phi
        sphericalConversion_view[i,2] = atan2(xyz[i,1],xyz[i,0])
    
    #unitVectors = getRadialUnitVecs(sphericalConversion)
    
    cdef double[:,:] unitVectors = getRadialUnitVecs(sphericalConversion)

    #get the radial 
    for i in range(rows):
        sphericalConversion_view[i,3] = unitVectors[i,0]*xyz[i,3]+unitVectors[i,1]*xyz[i,4]+unitVectors[i,2]*xyz[i,5]

    return sphericalConversion


#bin Particles

cpdef binParticles(long[:] pixIndicies, long npix):
       
    numcount = np.zeros(npix, dtype=np.int64)
    cdef long[:] numcount_view = numcount

    cdef int i

    for i in range(0,pixIndicies.shape[0]):
        numcount_view[pixIndicies[i]] +=1

    return numcount
    
#bin Velocities

cpdef binVelocities(long[:] pixIndicies, double[:] velocity, long npix):

    #velMap = np.zeros(npix)
    #cdef double[:] velMap_view = velMap

    cdef double[:] velMap_view = np.zeros(npix)

    cdef int i

    for i in range(0,pixIndicies.shape[0]):
        velMap_view[pixIndicies[i]] +=velocity[i]

    return velMap_view

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

#readSetToBins

cpdef readSetToBins(inputFiles, direc,ROIs, long nside,long npix, long boxSize, long radialDivs, getAngDiaDist):
    tempArray = []
    numcount = np.zeros((radialDivs,npix))
    totalVelThread = np.zeros((radialDivs,npix))
    for i in range(len(inputFiles)):
        path = direc + inputFiles[i]
        if os.path.isfile(path):
            
            unf_read_file(path, p_list=tempArray)
            print("Reading file "+path)
            reshaped = np.reshape(tempArray,(-1,6))

            tempArray = []
            
            
            offset = np.append((boxSize/2)*np.ones((np.shape(reshaped)[0],3)),np.zeros((np.shape(reshaped)[0],3)),axis=1)

            reshaped = np.subtract(reshaped,offset)
            del offset
            
            sphereConversion = convertToSpherical(reshaped)
            del reshaped

            sphereConversion[:,3] = sphereConversion[:,3]/np.power(getAngDiaDist(sphereConversion[:,0]),2)
            
            
            for j in range(0,radialDivs):
                #select all the point in each radial range
                
                radialRange = sphereConversion[(sphereConversion[:,0]>ROIs[j]) & (sphereConversion[:,0]<ROIs[j+1])]
                
                #determine how many points are in each bin
                pixIndicies = hp.ang2pix(nside,radialRange[:,1],radialRange[:,2])

                #do the math for the SZ effect

                #sum all velocities in each bin together
                
                if(len(pixIndicies)>0):
                    numcount[j] = np.add(numcount[j], binParticles(pixIndicies, npix))
                    totalVelThread[j] = np.add(totalVelThread[j],binVelocities(pixIndicies,radialRange[:,3],npix))
                
            print(str(i)+" done")

    return np.array([numcount, totalVelThread])


from cython.parallel import prange
cpdef readFilesMP(inputFiles, direc,ROIs, long nside, long npix, long boxSize, long radialDivs, getAngDiaDist):
    cdef long num_Files = len(inputFiles)
    cdef double[:,:,:] returnArray = np.zeros((2,radialDivs, npix))

    cdef int i

    cdef double[:,:,:] temp = np.zeros((2,radialDivs,npix))
    
    for i in prange(num_Files, nogil=True):
        with gil:
            infiles = inputFiles[i]
        temp = readSetToBins(infiles,direc, ROIs, nside, npix, boxSize, radialDivs, getAngDiaDist)

        #returnArray[0,:,:] = np.sum(returnArray[0,:,:],temp[0])
        #returnArray[1,:,:] = np.sum(returnArray[1,:,:],temp[1])
        #cdef int a,b,c
        with gil:
            for a in range(2):
                for b in range(radialDivs):
                    for c in range(npix):
                        returnArray[a,b,c] += temp[a,b,c]

    return returnArray




