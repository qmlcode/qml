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

import numpy as np
import qml
from qml.representations import *

def test_coulomb_matrix(mol):

    # Generate coulomb matrix representation, sorted by row-norm, using the Compound class
    mol.generate_coulomb_matrix(size = 5, sorting = "row-norm")

    # Generate coulomb matrix representation, sorted by row-norm,  using the python interface
    cm = generate_coulomb_matrix(mol.nuclear_charges,
                mol.coordinates, size = 5, sorting = "row-norm")

    assert np.allclose(mol.representation, cm), "Error in coulomb matrix representation"

def test_atomic_coulomb_matrix(mol):

    # Generate atomic coulomb matrix representation, sorted by distance, using the Compound class
    mol.generate_atomic_coulomb_matrix(size = 5, sorting = "distance")

    # Generate atomic coulomb matrix representation, sorted by distance, using the python interface
    acm = generate_atomic_coulomb_matrix(size = 5, sorting = "distance")

    assert np.allclose(mol.representation, acm), "Error in atomic coulomb matrix representation"

def test_eigenvalue_coulomb_matrix(mol):

    # Generate coulomb matrix eigenvalue representation using the Compound class
    mol.generate_eigenvalue_coulomb_matrix(size = 5)

    # Generate coulomb matrix eigenvalue representation using the python interface
    ecm = generate_eigenvalue_coulomb_matrix(mol.nuclear_charges,
                mol.coordinates, size = 5)

    assert np.allclose(mol.representation, ecm), "Error in coulomb matrix eigenvalue representation"

def test_bob(mol):

    # Generate bag of bonds representation using the Compound class
    mol.generate_bob(asize = {"C": 2, "H": 5, "O": 10}, size = 5)

    # Generate bag of bonds representation using the python interface
    bob = generate_bob(mol.nuclear_charges,
            mol.coordinates, mol.atomtypes, asize = {"C": 2, "H": 5, "O": 10}, size = 5)

    assert np.allclose(mol.representation, bob), "Error in bag of bonds representation"

def test_representations():

    # Generate a compound
    mol = qml.Compound(xyz = "qm7/0001.xyz")

    test_coulomb_matrix(mol)
    test_atomic_coulomb_matrix(mol)
    test_eigenvalue_coulomb_matrix(mol)
    test_bob(mol)
