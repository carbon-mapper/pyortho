#!/bin/bash
# build script to compile pyortho cython dependencies (find_pixel_trace, sunpos)

# save working directory
cwd=$(pwd)

# if undefined, get PYORT_ROOT from first input argument or default to $cwd
if [ -z $PYORT_ROOT ]; then
    if [ ! -z $1 ]; then
	PYORT_ROOT=$1
    else
	echo "PYORT_ROOT undefined, using current working directory ($cwd)"
	PYORT_ROOT=$cwd
    fi
fi

if [ ! -d $PYORT_ROOT ]; then
    echo "PYORT_ROOT=$PYORT_ROOT directory not found"
    echo "unable to build find_pixel_trace_cython and sunpos libraries"
    exit 1;
fi

PYEXT_ROOT=$PYORT_ROOT/external

FPT_PATH=find_pixel_trace_cython
FPT_ROOT=$PYORT_ROOT/$FPT_PATH
FPT_LIB=$FPT_ROOT/find_pixel_trace_cython.so

SUNPOS_PATH=sunpos-1.1
SUNPOS_ROOT=$PYEXT_ROOT/$SUNPOS_PATH
SUNPOS_LIB=$SUNPOS_ROOT/_sunpos.so

BUILD_LOG=$PYORT_ROOT/build_link_cython.log
rm -f $BUILD_LOG

if [ ! -d $FPT_ROOT ]; then
    echo "$FPT_PATH path not found (PYORT_ROOT=$PYORT_ROOT)" >& $BUILD_LOG
    echo "unable to build find_pixel_trace_cython library" >& $BUILD_LOG
else
    echo "Compiling $FPT_LIB"
    # build find_pixel_trace_cython
    cd $FPT_ROOT
    python setup.py build_ext --inplace >& $BUILD_LOG
    #ln -sf ./find_pixel_trace_cython/find_pixel_trace_cython.so .
fi

if [ ! -d $SUNPOS_ROOT ]; then
    echo "$SUNPOS_PATH path not found (PYEXT_ROOT=$PYEXT_ROOT)" >& $BUILD_LOG
    echo "unable to build sunpos library" >& $BUILD_LOG
else
    echo "Compiling $SUNPOS_LIB"
    # build sunpos
    cd $SUNPOS_ROOT
    python setup_swig.py build >& $BUILD_LOG

    # link to library in most recent build dir
    builddir=$(ls -t build/|head -n 1)
    ln -sf ./build/$builddir/_sunpos.so $SUNPOS_ROOT
fi

echo "Compliation complete. Compile log written to $BUILD_LOG"

# restore working directory
cd $cwd
if [ ! -f $FPT_LIB ] || [ ! -f $SUNPOS_LIB ]; then
   exit 1;
fi



