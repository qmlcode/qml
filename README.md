# QML: A Python Toolkit for Quantum Machine Learning
[![Build Status](https://travis-ci.org/qmlcode/qml.svg?branch=master)](https://travis-ci.org/qmlcode/qml)

QML is a Python2/3-compatible toolkit for representation learning of properties of molecules and solids. QML is not a high-level framework where you can do `model.train()`, but supplies the building blocks to carry out efficient and accurate machine learning on chemical compounds. As such, the goal is to provide usable and efficient implementations of concepts such as representations and kernels, which can then be implemented in other machine learning codes.

#### Current list of contributors:
* Anders S. Christensen (University of Basel)
* Felix Faber (University of Basel)
* Bing Huang (University of Basel)
* Lars A. Bratholm (University of Copenhagen)
* O. Anatole von Lilienfeld (University of Basel)

## 1) Citing QML:

Until the preprint is available from arXiv, please cite this GitHub repository as:

    AS Christensen, F Faber, B Huang, LA Bratholm, OA von Lilienfeld (2017) "QML: A Python Toolkit for Quantum Machine Learning" https://github.com/qmlcode/qml

## 2) Installation

Installing prerequisite modules (for most Linux systems):

```bash
sudo apt-get install python-pip gfortran libblas-dev liblapack-dev git

```
These should already be installed on most systems. The Intel compilers and MKL math-libraries are supported as well (see section 1.3).


### 2.1) Installing via `pip`:

The easiest way to install the stable version from the Python Package Index (PyPI) is using the official, built-in Python package manager, `pip`:

```bash
pip install qml --user -U
```

Alternatively, you can install the most recent stable development snapshot directly from GitHub:

```bash
pip install git+https://github.com/qmlcode/qml --user -U
```

Use `pip2 install ...` or `pip3 install ...` to get the Python2 or Python3 versions explicitly. QML supports both flavors.

To uninstall simply use `pip` again. 

```bash
pip uninstall qml
```

### 2.2) Installing via `setup.py` with Intel compiler:

If you have Intel compilers installed (2016 and newer), you can compile QML with Ifort/MKL using the following options:

```bash
pip install git+https://github.com/qmlcode/qml.git --user --upgrade --global-option="build" --global-option="--compiler=intelem" --global-option="--fcompiler=intelem"
```

### 2.3) Note on Apple/Mac support:

Installing QML requires a Fortran compiler. On Darwin you can install it using `brew`:

```bash
brew install gcc
```

Note the Clang Fortran compiler from brew unfortunately does not support OpenMP. Therefore parallelism via OpenMP is disabled as default for Darwin systems.

## 3) Get help:

Documentation in found in the file [`DOCUMENTATION.md`](../blob/master/DOCUMENTATION.md).


## 4) License:

QML is freely available under the terms of the MIT license.


