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

import sys

import os
import numpy as np

np.set_printoptions(linewidth=666)

import qml

from qml.math import cho_solve

from qml.representations import generate_fchl_acsf

from qml.kernels import get_local_kernel
from qml.kernels import get_local_symmetric_kernel

from time import time

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

def test_energy():

    test_dir = os.path.dirname(os.path.realpath(__file__))

    # Parse file containing PBE0/def2-TZVP heats of formation and xyz filenames
    data = get_energies(test_dir + "/data/hof_qm7.txt")

    # Generate a list of qml.data.Compound() objects
    mols = []

    Qall = []
    for xyz_file in sorted(data.keys())[:1000]:

        # Initialize the qml.data.Compound() objects
        mol = qml.Compound(xyz=test_dir + "/qm7/" + xyz_file)

        # Associate a property (heat of formation) with the object
        mol.properties = data[xyz_file]

        mol.representation = generate_fchl_acsf(mol.nuclear_charges, mol.coordinates, gradients=False, pad=27)

        Qall.append(mol.nuclear_charges)

        mols.append(mol)

    # Shuffle molecules
    np.random.seed(666)
    np.random.shuffle(mols)

    # Make training and test sets
    n_test  = 99
    n_train = 101

    training = mols[:n_train]
    test  = mols[-n_test:]
    training_indexes = list(range(n_train))
    test_indexes = list(range(n_train, n_train+n_test))

    # List of representations
    X  = np.array([mol.representation for mol in training])
    Xs = np.array([mol.representation for mol in test])
    Xall = np.array([mol.representation for mol in training + test])

    Q  = np.array([mol.nuclear_charges for mol in training])
    Qs = np.array([mol.nuclear_charges for mol in test])
    Qall = np.array([mol.nuclear_charges for mol in training + test])

    # List of properties
    Y = np.array([mol.properties for mol in training])
    Ys = np.array([mol.properties for mol in test])

    # Set hyper-parameters
    sigma = 3.0
    llambda = 1e-10

    K = get_local_symmetric_kernel(X, Q, sigma)

    # Solve alpha
    alpha = cho_solve(K,Y, l2reg=llambda)

    # Calculate test kernel
    Ks = get_local_kernel(X, Xs, Q, Qs, sigma)

    # Calculate test prediction kernel
    Ks = get_local_kernel(X, Xs, Q, Qs, sigma)
    Yss = np.dot(Ks, alpha)

    mae = np.mean(np.abs(Ys - Yss))
    assert mae < 4.0, "ERROR: Too high MAE!"
    

if __name__ == "__main__":

    test_energy()
