# build with:
#   python setup.py build_ext --inplace

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("find_pixel_trace_cython", ["find_pixel_trace_cython.pyx"],
              include_dirs = [numpy.get_include()]
              #libraries = [...], library_dirs = [...]
          )
    ]

setup(
    name = 'find_pixel_trace (cython version)',
    ext_modules = cythonize(extensions)
)
