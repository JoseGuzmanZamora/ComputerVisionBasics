from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy 

setup(
    ext_modules=cythonize("cython_logo.pyx"),  
    include_dirs=[numpy.get_include()]
)  
