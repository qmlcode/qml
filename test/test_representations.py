# MIT License
#
# Copyright (c) 2017-2018 Anders Steen Christensen, Lars Andersen Bratholm
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

from collections import defaultdict
import numpy as np
import os

import qml

from qml.representations import *

def get_asize(mols, pad):

    asize = defaultdict()

    for mol in mols:
        for key, value in mol.natypes.items():
            try:
                asize[key] = max(asize[key], value + pad)
            except KeyError:
                asize[key] = value + pad
    return asize

def test_representations():
    files = ["qm7/0101.xyz",
             "qm7/0102.xyz",
             "qm7/0103.xyz",
             "qm7/0104.xyz",
             "qm7/0105.xyz",
             "qm7/0106.xyz",
             "qm7/0107.xyz",
             "qm7/0108.xyz",
             "qm7/0109.xyz",
             "qm7/0110.xyz"]

    path = test_dir = os.path.dirname(os.path.realpath(__file__))

    mols = []
    for xyz_file in files:
        mol = qml.data.Compound(xyz=path + "/" + xyz_file)
        mols.append(mol)

    size = max(mol.nuclear_charges.size for mol in mols) + 1

    asize = get_asize(mols,1)

    coulomb_matrix(mols, size, path)
    atomic_coulomb_matrix(mols, size, path)
    eigenvalue_coulomb_matrix(mols, size, path)
    bob(mols, size, asize, path)

def coulomb_matrix(mols, size, path):

    # Generate coulomb matrix representation, sorted by row-norm
    for i, mol in enumerate(mols): 
        mol.generate_coulomb_matrix(size = size, sorting = "row-norm")

    X_test = np.asarray([mol.representation for mol in mols])
    X_ref = np.loadtxt(path + "/data/coulomb_matrix_representation_row-norm_sorted.txt")
    assert np.allclose(X_test, X_ref), "Error in coulomb matrix representation"

    # Generate coulomb matrix representation, unsorted, using the Compound class
    for i, mol in enumerate(mols): 
        mol.generate_coulomb_matrix(size = size, sorting = "unsorted")

    X_test = np.asarray([mol.representation for mol in mols])
    X_ref = np.loadtxt(path + "/data/coulomb_matrix_representation_unsorted.txt")
    assert np.allclose(X_test, X_ref), "Error in coulomb matrix representation"

def atomic_coulomb_matrix(mols, size, path):

    # Generate coulomb matrix representation, sorted by distance
    for i, mol in enumerate(mols): 
        mol.generate_atomic_coulomb_matrix(size = size, sorting = "distance")

    X_test = np.concatenate([mol.representation for mol in mols])
    X_ref = np.loadtxt(path + "/data/atomic_coulomb_matrix_representation_distance_sorted.txt")
    assert np.allclose(X_test, X_ref), "Error in atomic coulomb matrix representation"
    # Compare to old implementation (before 'indices' keyword)
    X_ref = np.loadtxt(path + "/data/atomic_coulomb_matrix_representation_distance_sorted_no_indices.txt")
    assert np.allclose(X_test, X_ref), "Error in atomic coulomb matrix representation"


    # Generate coulomb matrix representation, sorted by row-norm
    for i, mol in enumerate(mols): 
        mol.generate_atomic_coulomb_matrix(size = size, sorting = "row-norm")

    X_test = np.concatenate([mol.representation for mol in mols])
    X_ref = np.loadtxt(path + "/data/atomic_coulomb_matrix_representation_row-norm_sorted.txt")
    assert np.allclose(X_test, X_ref), "Error in atomic coulomb matrix representation"

    # Generate coulomb matrix representation, sorted by distance, with soft cutoffs
    for i, mol in enumerate(mols): 
        mol.generate_atomic_coulomb_matrix(size = size, sorting = "distance",
                central_cutoff = 4.0, central_decay = 0.5,
                interaction_cutoff = 5.0, interaction_decay = 1.0)

    X_test = np.concatenate([mol.representation for mol in mols])
    X_ref = np.loadtxt(path + "/data/atomic_coulomb_matrix_representation_distance_sorted_with_cutoff.txt")
    assert np.allclose(X_test, X_ref), "Error in atomic coulomb matrix representation"

    # Generate coulomb matrix representation, sorted by row-norm, with soft cutoffs
    for i, mol in enumerate(mols): 
        mol.generate_atomic_coulomb_matrix(size = size, sorting = "row-norm",
                central_cutoff = 4.0, central_decay = 0.5,
                interaction_cutoff = 5.0, interaction_decay = 1.0)

    X_test = np.concatenate([mol.representation for mol in mols])
    X_ref = np.loadtxt(path + "/data/atomic_coulomb_matrix_representation_row-norm_sorted_with_cutoff.txt")
    assert np.allclose(X_test, X_ref), "Error in atomic coulomb matrix representation"

    # Generate only two atoms in the coulomb matrix representation, sorted by distance
    for i, mol in enumerate(mols): 
        mol.generate_atomic_coulomb_matrix(size = size, sorting = "distance")
        representation_subset = mol.representation[1:3]
        mol.generate_atomic_coulomb_matrix(size = size, sorting = "distance", indices = [1,2])
        for i in range(2):
            for j in range(153):
                diff = representation_subset[i,j] - mol.representation[i,j]
                if abs(diff) > 1e-9:
                    print (i,j,diff, representation_subset[i,j],mol.representation[i,j])
        assert np.allclose(representation_subset, mol.representation), \
                "Error in atomic coulomb matrix representation"

    # Generate only two atoms in the coulomb matrix representation, sorted by row-norm
    for i, mol in enumerate(mols): 
        mol.generate_atomic_coulomb_matrix(size = size, sorting = "row-norm")
        representation_subset = mol.representation[1:3]
        mol.generate_atomic_coulomb_matrix(size = size, sorting = "row-norm", indices = [1,2])
        for i in range(2):
            for j in range(153):
                diff = representation_subset[i,j] - mol.representation[i,j]
                if abs(diff) > 1e-9:
                    print (i,j,diff, representation_subset[i,j],mol.representation[i,j])
        assert np.allclose(representation_subset, mol.representation), \
                "Error in atomic coulomb matrix representation"

def eigenvalue_coulomb_matrix(mols, size, path):

    # Generate coulomb matrix representation, sorted by row-norm
    for i, mol in enumerate(mols): 
        mol.generate_eigenvalue_coulomb_matrix(size = size)

    X_test = np.asarray([mol.representation for mol in mols])
    X_ref = np.loadtxt(path + "/data/eigenvalue_coulomb_matrix_representation.txt")
    assert np.allclose(X_test, X_ref), "Error in eigenvalue coulomb matrix representation"

def bob(mols, size, asize, path):

    for i, mol in enumerate(mols): 
        mol.generate_bob(size=size, asize=asize)

    X_test = np.asarray([mol.representation for mol in mols])
    X_ref = np.loadtxt(path + "/data/bob_representation.txt")
    assert np.allclose(X_test, X_ref), "Error in bag of bonds representation"

def print_mol(mol):
    n = len(mol.representation.shape)
    if n == 1:
        for item in mol.representation:
            print("{:.9e}".format(item), end='  ')
        print()
    elif n == 2:
        for atom in mol.representation:
            for item in atom:
                print("{:.9e}".format(item), end='  ')
            print()

if __name__ == "__main__":
    test_representations()

