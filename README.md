# QML: Quantum Machine Learning

A toolkit for representation learning of molecules and solids.

Current author list:
* Anders S. Christensen (University of Basel)
* Felix Faber (University of Basel)
* O. Anatole von Lilienfeld (University of Basel)

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

1.4) TODO: create `setup.py` install script

## 2) Representations of compounds:

### 2.1) Supported representations:

Currently QML supports the following representations for molecules:

* Molecular coulomb matrix (sorted by row-norm, or unsorted)
* Atomic coulomb matrix (sorted by distance to query atom, or row-norm)
* ARAD

Currently QML supports the following representations for solids:

* ARAD

### 2.2) Generating representations using the `Compound` class:
The following example demonstrates how to generate a representation via the `qml.Compound` class.

```python
from qml import Compound

# Read in an xyz or cif file.
water = Compound(xyz="water.xyz")

# Generate a molecular coulomb matrices sorted by row norm.
water.generate_coulomb_matrix(size=5, sort="row-norm")

print water.coulomb_matrix
```


## 2.3) Generating representations via the `qml.representations` module:

```python
from qml.representations import *

# Dummy atomtypes and coordinates
atomtypes = ["O", "H", H"]
coordinates = np.array([1.464, 0.707, 1.056],
                       [0.878, 1.218, 0.498],
                       [2.319, 1.126, 0.952])

# Generate a molecular coulomb matrices sorted by row norm.
cm1 = generate_coulomb_matrix(atomtypes, coordinates,
                                size=5, sort="row-norm")
print cm1

# Generate all atomic coulomb matrices sorted by distance to
# query atom.
cm2 = generate_atomic_coulomb_matrix(atomtypes, coordinates,
                                size=5, sort="distance")
print cm2
```

## 2.4) Benchmarks for QM7:
Following benchmarks were executed on a single core on an Intel Core i5-6260U @ 1.80 GHz CPU.

Generate ~7K molecular coulomb matrices = 0.06s 
Generate ~100K atomic coulomb matrices = 0.22s


 
## 3.1) Calculate kernels using the `Compound` class:


## 3.2) Calculate kernels using the `qml.kernels` module directly


```python
from qml.kernels import laplacian_kernel
```

## 3.3) Benchmarks for QM7:

## 4.1) 
