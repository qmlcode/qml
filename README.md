# QML: Quantum Machine Learning

A toolkit for representation learning of molecules and solids.

Current author list:
* Anders S. Christensen (University of Basel)
* Felix Faber (University of Basel)

## 1) Installation

By default FML compiles with GCC's gfortran and your system's standard BLAS+LAPACK libraries (-lblas -llapack). I recommend you switch to MKL for the math libraries, but the difference between ifort and gfortran shouldn't be to significant. BLAS and LAPACK implementations are the only libraries required for FML. Additionally you need a functional Python 2.x interpreter and NumPy (with F2PY) installed. Most Linux systems can install BLAS and LAPACK like this:

    sudo apt-get install libblas-dev liblapack-dev

Ok, on to the installation instructions:

1.1) First you clone this repository: 

    git clone https://github.com/qmlcode/qml.git

1.2) Then you simply compile by typing make in the fml folder:

    make

Note: If you access to the Intel compilers, you can replace the default `Makefile` with a different `Makefile.*` from the `makefiles/` folder. E.g. `Makefile.intel` will compile the F2PY interface with `ifort` and link to MKL. The default `Makefile` is identical to `Makefile.gnu`.

1.3) To make everything accessible to your Python export the fml root-folder to your PYTHONPATH.

    export PYTHONPATH=/path/to/installation/qml:$PYTHONPATH

## 2) How to use:


## 2.1) Generating representations using the `Compound` class:

## 2.2) Generating representations via the `qml.representations` module:


    from qml.representations import *





## 2.3) Benchmarks for QM7:

## 3.1) Calculate kernels using the `Compound` class:

## 3.2) Calculate kernels using the `qml.kernels` module:

## 3.3) Benchmarks for QM7:

## 4.1) 
