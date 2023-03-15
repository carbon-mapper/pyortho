#!/bin/bash
  
REPO_BASE_DIR=$(pwd)

echo "Installing pyortho..."

cd $REPO_BASE_DIR
./build_link_cython.sh

cd $REPO_BASE_DIR/external/sunpos-1.1/
python setup_swig.py build_ext --inplace
cp _sunpos.so $REPO_BASE_DIR/
cp sunpos.py $REPO_BASE_DIR/
