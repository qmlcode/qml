# MIT License
#
# Copyright (c) 2017 Anders Steen Christensen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function

import os
import numpy as np

import qml
import qml.data

from qml.kernels import laplacian_kernel
from qml.math import cho_solve

from qml.representations import get_slatm_mbtypes

from qml.kernels import get_local_kernels_gaussian
from qml.kernels import get_local_kernels_laplacian

from qml.kernels.wrappers import get_atomic_kernels_gaussian, get_atomic_kernels_laplacian

def get_energies(filename):
    """ Returns a dictionary with heats of formation for each xyz-file.
    """

    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    energies = dict()

    for line in lines:
        tokens = line.split()

        xyz_name = tokens[0]
        hof = float(tokens[1])

        energies[xyz_name] = hof

    return energies

def test_krr_gaussian_local_cmat():

    test_dir = os.path.dirname(os.path.realpath(__file__))

    # Parse file containing PBE0/def2-TZVP heats of formation and xyz filenames
    data = get_energies(test_dir + "/data/hof_qm7.txt")

    # Generate a list of qml.data.Compound() objects"
    mols = []


    for xyz_file in sorted(data.keys())[:1000]:

        # Initialize the qml.data.Compound() objects
        mol = qml.data.Compound(xyz=test_dir + "/qm7/" + xyz_file)

        # Associate a property (heat of formation) with the object
        mol.properties = data[xyz_file]

        # This is a Molecular Coulomb matrix sorted by row norm
        mol.generate_atomic_coulomb_matrix(size=23, sorting="row-norm")

        mols.append(mol)

    # Shuffle molecules
    np.random.seed(666)
    np.random.shuffle(mols)

    # Make training and test sets
    n_test  = 100
    n_train = 200

    training = mols[:n_train]
    test  = mols[-n_test:]

    X = np.concatenate([mol.representation for mol in training])
    Xs = np.concatenate([mol.representation for mol in test])

    N = np.array([mol.natoms for mol in training])
    Ns = np.array([mol.natoms for mol in test])

    # List of properties
    Y = np.array([mol.properties for mol in training])
    Ys = np.array([mol.properties for mol in test])

    # Set hyper-parameters
    sigma = 724.0
    llambda = 10**(-6.5)

    K = get_local_kernels_gaussian(X, X, N, N, [sigma])[0]
    assert np.allclose(K, K.T), "Error in local Gaussian kernel symmetry"

    K_test = np.loadtxt(test_dir + "/data/K_local_gaussian.txt")
    assert np.allclose(K, K_test), "Error in local Gaussian kernel (vs. reference)"

    K_test = get_atomic_kernels_gaussian(training, training, [sigma])[0]
    assert np.allclose(K, K_test), "Error in local Gaussian kernel (vs. wrapper)"

    # Solve alpha
    K[np.diag_indices_from(K)] += llambda
    alpha = cho_solve(K,Y)

    # Calculate prediction kernel
    Ks = get_local_kernels_gaussian(Xs, X, Ns, N, [sigma])[0]

    Ks_test = np.loadtxt(test_dir + "/data/Ks_local_gaussian.txt")
    # Somtimes a few coulomb matrices differ because of parallel sorting and numerical error
    # Allow up to 5 molecules to differ from the supplied reference.
    differences_count = len(set(np.where(Ks - Ks_test > 1e-7)[0]))
    assert differences_count < 5, "Error in local Laplacian kernel (vs. reference)"
    # assert np.allclose(Ks, Ks_test), "Error in local Gaussian kernel (vs. reference)"

    Ks_test = get_atomic_kernels_gaussian(test, training, [sigma])[0]
    assert np.allclose(Ks, Ks_test), "Error in local Gaussian kernel (vs. wrapper)"

    Yss = np.dot(Ks, alpha)

    mae = np.mean(np.abs(Ys - Yss))
    print(mae)
    assert abs(19.0 - mae) < 1.0, "Error in local Gaussian kernel-ridge regression"

def test_krr_laplacian_local_cmat():

    test_dir = os.path.dirname(os.path.realpath(__file__))

    # Parse file containing PBE0/def2-TZVP heats of formation and xyz filenames
    data = get_energies(test_dir + "/data/hof_qm7.txt")

    # Generate a list of qml.data.Compound() objects"
    mols = []


    for xyz_file in sorted(data.keys())[:1000]:

        # Initialize the qml.data.Compound() objects
        mol = qml.data.Compound(xyz=test_dir + "/qm7/" + xyz_file)

        # Associate a property (heat of formation) with the object
        mol.properties = data[xyz_file]

        # This is a Molecular Coulomb matrix sorted by row norm
        mol.generate_atomic_coulomb_matrix(size=23, sorting="row-norm")

        mols.append(mol)

    # Shuffle molecules
    np.random.seed(666)
    np.random.shuffle(mols)

    # Make training and test sets
    n_test  = 100
    n_train = 200

    training = mols[:n_train]
    test  = mols[-n_test:]

    X = np.concatenate([mol.representation for mol in training])
    Xs = np.concatenate([mol.representation for mol in test])

    N = np.array([mol.natoms for mol in training])
    Ns = np.array([mol.natoms for mol in test])

    # List of properties
    Y = np.array([mol.properties for mol in training])
    Ys = np.array([mol.properties for mol in test])

    # Set hyper-parameters
    sigma = 10**(3.6)
    llambda = 10**(-12.0)

    K = get_local_kernels_laplacian(X, X, N, N, [sigma])[0]
    assert np.allclose(K, K.T), "Error in local Laplacian kernel symmetry"

    K_test = np.loadtxt(test_dir + "/data/K_local_laplacian.txt")
    assert np.allclose(K, K_test), "Error in local Laplacian kernel (vs. reference)"

    K_test = get_atomic_kernels_laplacian(training, training, [sigma])[0]
    assert np.allclose(K, K_test), "Error in local Laplacian kernel (vs. wrapper)"

    # Solve alpha
    K[np.diag_indices_from(K)] += llambda
    alpha = cho_solve(K,Y)

    # Calculate prediction kernel
    Ks = get_local_kernels_laplacian(Xs, X, Ns, N, [sigma])[0]


    Ks_test = np.loadtxt(test_dir + "/data/Ks_local_laplacian.txt")

    # Somtimes a few coulomb matrices differ because of parallel sorting and numerical error
    # Allow up to 5 molecules to differ from the supplied reference.
    differences_count = len(set(np.where(Ks - Ks_test > 1e-7)[0]))
    assert differences_count < 5, "Error in local Laplacian kernel (vs. reference)"

    Ks_test = get_atomic_kernels_laplacian(test, training, [sigma])[0]
    assert np.allclose(Ks, Ks_test), "Error in local Laplacian kernel (vs. wrapper)"

    Yss = np.dot(Ks, alpha)

    mae = np.mean(np.abs(Ys - Yss))
    assert abs(8.7 - mae) < 1.0, "Error in local Laplacian kernel-ridge regression"

if __name__ == "__main__":

    test_krr_gaussian_local_cmat()
    test_krr_laplacian_local_cmat()
