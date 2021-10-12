# distutils: language=c++

import numpy as np
cimport numpy as np

from libc.math cimport sin
from libc.math cimport cos
from libc.math cimport pow
from libc.math cimport sqrt
from libc.math cimport atan2

from libcpp cimport vector

#getRadialUnitVecs
cdef getRadialUnitVecs(spherePosNp):
    cdef double[:,:] spherePos = spherePosNp 

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

cpdef convertToSpherical(xyz):
    cdef double[:,:] xyz_view = xyz
    cdef int rows = xyz_view.shape[0]
    #variables
    sphericalConversion = np.zeros((rows,4))
    cdef double[:,:] sphericalConversion_view = sphericalConversion    
    
    # col 0: r, col 1: theta, col 2: phi
    cdef int i
    cdef double xySquared     

    for i in range(0,rows):

        xySquared = pow(xyz_view[i,0],2) + pow(xyz_view[i,1],2)

        # r
        sphericalConversion_view[i,0] = sqrt(xySquared+pow(xyz_view[i,2],2))
        
        # theta - many options for this calc
        sphericalConversion_view[i,1] = atan2(sqrt(xySquared), xyz_view[i,2])
    
        # phi
        sphericalConversion_view[i,2] = atan2(xyz_view[i,1],xyz_view[i,0])
    
    unitVectors = getRadialUnitVecs(sphericalConversion)
    
    cdef double[:,:] unitVectors_view = unitVectors

    #get the radial 
    for i in range(rows):
        sphericalConversion_view[i,3] = unitVectors_view[i,0]*xyz_view[i,3]+unitVectors_view[i,1]*xyz_view[i,4]+unitVectors_view[i,2]*xyz_view[i,5]
    
    return sphericalConversion


#bin Particles

cpdef binParticles(np.ndarray pixIndicies, long npix):
    cdef long[:] pixIndicies_view = pixIndicies     

    numcount = np.zeros(npix, dtype=np.int64)
    cdef long[:] numcount_view = numcount

    cdef int i

    for i in range(0,pixIndicies_view.shape[0]):
        numcount_view[pixIndicies_view[i]] +=1

    return numcount
    
#bin Velocities

cpdef binVelocities(np.ndarray pixIndicies, np.ndarray velocity, long npix):
    cdef long[:] pixIndicies_view = pixIndicies

    velMap = np.zeros(npix)
    cdef double[:] velMap_view = velMap

    cdef int i

    for i in range(0,pixIndicies_view.shape[0]):
        velMap_view[pixIndicies_view[i]] +=velocity[i]

    return velMap

#readSetToBins

#getConvergenceForPixel

#getConvergenceForRange
