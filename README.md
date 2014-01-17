libcudann
=========

A fork of libcudann -- artificial neural network implemented in CUDA.

Original code by Luca Donati: http://sourceforge.net/projects/libcudann/

The library supports standard multilayer perceptron architecture with
an arbitrary number of layers. Activation function can be either
sigmoid or hyperbolic tangent. Network is trained through
batch backpropagation algorithm.

CPU version of the algorithm is implemented for reference. Common
speedups are between 100-200 times (depends on hardware).

Compilation and installation:

    cd src
    make
    sudo make install

Before running make edit Makefile and set variables CUDA_LIB_DIR and
CUDA_INCLUDE_DIR to point to correct paths.

Compilation and installation of Python wrapper:

    cd python
    python setup.py build
    sudo python setup.py install

Example of C++ library usage can be found in src/main.cpp

Example of Python wrapper usage can be found in python/cudann_test.py
