# build with:
#   python setup.py build_ext --inplace

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("bilerp_cython", ["bilerp_cython.pyx"],
              include_dirs = [numpy.get_include()]
              #libraries = [...],
              #library_dirs = [...]
          )
    ]

setup(
    name = 'bilerp (cython version)',
    ext_modules = cythonize(extensions)
)

# setup(
#     ext_modules=[
#         
# )

# import pyximport
# pyximport.install(setup_args={"script_args":["--compiler=g++"],
#                               "include_dirs":numpy.get_include()},
#                   reload_support=True)

# from bilerp_cython import bilerp_cython as fptc
# fptc(2,1,4,4)
