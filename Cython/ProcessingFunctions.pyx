# distutils: language=c++

import numpy as np
cimport numpy as np

from libc.math cimport sin
from libc.math cimport cos
from libc.math cimport pow
from libc.math cimport sqrt
from libc.math cimport atan2
from libc.math cimport isnan

from cython.parallel import prange

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

cpdef binParticles(long[:] partIndicies, long[:] partRadial, double[:] partVelocity, long npix, long radialBins):
       
    densityFeild = np.zeros((radialBins, npix), dtype=np.int64)
    velocityFeild = np.zeros((radialBins,npix),dtype=np.double)
    
    cdef long[:,:] densityFeild_view = densityFeild
    cdef double[:,:] velocityFeild_view = velocityFeild

    cdef Py_ssize_t i

    for i in prange(partIndicies.shape[0],nogil=True):
        if(partRadial[i]!=-1):
            densityFeild_view[partRadial[i], partIndicies[i]] +=1
            velocityFeild_view[partRadial[i], partIndicies[i]] += partVelocity[i]

    return densityFeild, velocityFeild
    
#readSetToBins

#getConvergenceForPixel

#getConvergenceForRange
