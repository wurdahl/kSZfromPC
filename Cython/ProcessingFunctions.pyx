# distutils: language=c++

import numpy as np
cimport numpy as np

#getRadialUnitVecs
def getRadialUnitVecs(spherePos):
    unitVecs = np.zeros((len(spherePos),3))

    #negative because you want it pointing towards the origin
    unitVecs[:,0] = np.multiply(np.cos(spherePos[:,2]),np.sin(spherePos[:,1]))
    unitVecs[:,1] = np.multiply(np.sin(spherePos[:,2]),np.sin(spherePos[:,1]))
    unitVecs[:,2] = np.cos(spherePos[:,1])

    return unitVecs

#convertToSpherical

#unf_read_file

#readSetToBins

#getConvergenceForPixel

#getConvergenceForRange
