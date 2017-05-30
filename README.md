# QML: A Python Toolking for Quantum Machine Learning

A Python2/3 toolkit for representation learning of molecules and solids. 

Current author list:
* Anders S. Christensen (University of Basel)
* Felix Faber (University of Basel)
* Bing Huang (University of Basel)
* O. Anatole von Lilienfeld (University of Basel)

## 1) Installation

Installing prerequisite modules (for most Linux systems):

```bash
sudo apt-get install python-pip gfortran libblas-dev liblapack-dev

```
These should already be installed on most systems. Alternatively the Intel compilers and MKL math-libraries are supported as well (see section 1.3).


### 1.1) Installing via `pip`:

The easiest way to install is using the official, built-in Python package manager, `pip`:

```bash
pip install git+https://github.com/qmlcode/qml --user --upgrade
```

Additionally you can use `pip2 install ...` or `pip3 install ...` to get the Python2 or Python3 versions explicitly. QML supports both flavors.

To uninstall simply use `pip` again. 

```bash
pip uninstall qml
```

### 1.2) Installing via `setup.py` with Intel compiler:

If you have Intel compilers installed, you can compile QML with ifort/MKL using the following options:

```bash
pip install git+https://github.com/qmlcode/qml.git --user --upgrade --global-option="build" --global-option="--compiler=intelem" --global-option="--fcompiler=intelem"
```

### 1.3) Note on Apple/Mac support:

Install QML requires a Fortran compiler. On Darwin you can install it using `brew`:

```bash
brew install gcc
```

Note: the Clang Fortran compiler in brew does currently not support OpenMP, so this disables parallelism in QML.


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

... to be updated


## 3.2) Calculate kernels using the `qml.kernels` module directly


```python
from qml.kernels import laplacian_kernel
```
... to be updated
## 3.3) Benchmarks for QM7:
... to be updated

## 4.1) 
... to be updated
