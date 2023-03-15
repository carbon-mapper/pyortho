#!/bin/bash
python setup.py build_ext --inplace
if [ ! -e ../bilerp_cython.so ]; then
    ln -s $(pwd)/bilerp_cython.so ../bilerp_cython.so
fi
