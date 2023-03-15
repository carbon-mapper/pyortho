#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# dot extension module
_sunpos = Extension("_sunpos",
                    sources=["sunpos.i","SunPos.c"],
                    headers=['SunPos.h'],
                    include_dirs = [numpy_include]
                   )

# dot setup
setup(  name        = "SUNPOS",
        description = "Piecewise-Linear Continuum-Removal function",
        author      = "B. Bue",
        version     = "1.0",
        ext_modules = [_sunpos]
        )
