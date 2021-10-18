from setuptools import setup
from Cython.Build import cythonize
import numpy
import healpy
import os
import struct

setup(
    ext_modules=cythonize("ProcessingFunctions.pyx"),
    include_dirs=[numpy.get_include(),healpy.get_include(),os.get_include(), struct.get_include()]
)
