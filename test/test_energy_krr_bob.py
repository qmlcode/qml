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

from qml.ml.kernels import laplacian_kernel
from qml.ml.math import cho_solve
from qml.ml.representations import get_slatm_mbtypes


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

def test_krr_bob():

    test_dir = os.path.dirname(os.path.realpath(__file__))

    # Parse file containing PBE0/def2-TZVP heats of formation and xyz filenames
    data = get_energies(test_dir + "/data/hof_qm7.txt")

    # Generate a list of qml.data.Compound() objects
    mols = []

    for xyz_file in sorted(data.keys())[:1000]:

        # Initialize the qml.data.Compound() objects
        mol = qml.data.Compound(xyz=test_dir + "/qm7/" + xyz_file)

        # Associate a property (heat of formation) with the object
        mol.properties = data[xyz_file]

        # This is a Molecular Coulomb matrix sorted by row norm
        mol.generate_bob()

        mols.append(mol)

    # Shuffle molecules
    np.random.seed(666)
    np.random.shuffle(mols)

    # Make training and test sets
    n_test  = 300
    n_train = 700

    training = mols[:n_train]
    test  = mols[-n_test:]

    # List of representations
    X  = np.array([mol.representation for mol in training])
    Xs = np.array([mol.representation for mol in test])

    # List of properties
    Y = np.array([mol.properties for mol in training])
    Ys = np.array([mol.properties for mol in test])

    # Set hyper-parameters
    sigma = 26214.40
    llambda = 1e-10

    # Generate training Kernel
    K = laplacian_kernel(X, X, sigma)

    # Solve alpha
    K[np.diag_indices_from(K)] += llambda
    alpha = cho_solve(K,Y)

    # Calculate prediction kernel
    Ks = laplacian_kernel(X, Xs, sigma)
    Yss = np.dot(Ks.transpose(), alpha)

    mae = np.mean(np.abs(Ys - Yss))
    print(mae)
    assert mae < 2.6, "ERROR: Too high MAE!"

if __name__ == "__main__":

    test_krr_bob()
