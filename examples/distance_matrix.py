#!/usr/bin/env python2
#
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

import sys
import os

import numpy as np

import qml
from qml.distance import *


if __name__ == "__main__":

    # Generate a list of qml.Compound() objects
    mols = []

    for xyz_file in sorted(os.listdir("qm7/")):

        # Initialize the qml.Compound() objects
        mol = qml.Compound(xyz="qm7/" + xyz_file)

            # This is a Molecular Coulomb matrix sorted by row norm
        mol.generate_coulomb_matrix(size=23, sorting="unsorted")

        mols.append(mol)


    # Shuffle molecules
    np.random.seed(666)
    np.random.shuffle(mols)

    # Make training and test sets
    n_test  = 5
    n_train = 10

    training = mols[:n_train]
    test  = mols[-n_test:]

    # List of representations
    X  = np.array([mol.coulomb_matrix for mol in training])
    Xs = np.array([mol.coulomb_matrix for mol in test])


    D = manhattan_distance(X, Xs)
    print "Manhattan Distances:"
    print D

    D = l2_distance(X, Xs)
    print "L2 Distances:"
    print D

    D = p_distance(X, Xs, p=3)
    print "p-norm = 3 Distances:"
    print D

    D = p_distance(X, Xs, p=2.5)
    print "p-norm = 3 Distances:"
    print D
