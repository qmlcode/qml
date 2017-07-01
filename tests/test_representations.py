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

import itertools
import contextlib
import numpy as np
import os
import qml
from qml.representations import *
from qml.data import NUCLEAR_CHARGE


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
        mol = qml.Compound(xyz=path + "/" + xyz_file)
        mols.append(mol)
    
    size = max(mol.nuclear_charges.size for mol in mols) + 1

    coulomb_matrix_test(mols, size, path)
    atomic_coulomb_matrix_test(mols, size, path)

def coulomb_matrix_test(mols, size, path):

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

def atomic_coulomb_matrix_test(mols, size, path):

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


    #for mol in mols:
    #    print_mol(mol)
    quit()

def test_eigenvalue_coulomb_matrix():

    size = mol.nuclear_charges.size + 1 # +1 to check that dummy atoms are handled correctly

    # Generate coulomb matrix eigenvalue representation using the Compound class
    mol.generate_eigenvalue_coulomb_matrix(size = size)

    # Generate coulomb matrix eigenvalue representation using the python interface
    ecm = generate_eigenvalue_coulomb_matrix(mol.nuclear_charges,
                mol.coordinates, size = size)


    assert np.allclose(mol.representation, ecm), "Error in coulomb matrix eigenvalue representation"

    # Compare with python implementation
    ecm2 = eigenvalue_coulomb_matrix(mol.nuclear_charges, mol.coordinates, size = size)

    assert np.allclose(ecm2, ecm), "Error in coulomb matrix eigenvalue representation"

def test_bob():

    asize = dict([(key, value+1) for key,value in mol.natypes.items()])

    # Generate bag of bonds representation using the Compound class
    mol.generate_bob(asize)

    # Generate bag of bonds representation using the python interface
    bob = generate_bob(mol.nuclear_charges,
            mol.coordinates, mol.atomtypes, asize = asize)

    assert np.allclose(mol.representation, bob), "Error in bag of bonds representation"

    # Compare with python implementation
    bob = bob_reference(mol.nuclear_charges, mol.coordinates, mol.atomtypes, size = (mol.natoms), asize = asize)

    assert np.allclose(mol.representation, bob), "Error in bag of bonds representation"

def test_local_bob():

    asize = dict([(key, value+1) for key,value in mol.natypes.items()])

    print (asize)
    print (mol.atomtypes)
    # Generate atomic coulomb matrix representation, sorted by row-norm, using the Compound class
    bob = generate_local_bob(mol.nuclear_charges,
            mol.coordinates, mol.atomtypes, asize = asize,
            interaction_cutoff = 5.0, interaction_decay = 1.0,
            central_cutoff = 3.0, central_decay = 0.8, variant="sncf1", localization=2)
    bob2 = local_bob_reference(mol.nuclear_charges,
            mol.coordinates, mol.atomtypes, asize = asize,
            interaction_cutoff = 5.0, interaction_decay = 1.0,
            central_cutoff = 3.0, central_decay = 0.8, variant="sncf1", localization=2)

    for k in range(mol.atomtypes.size):
        for i in range(bob[k].size):
            diff = abs(bob[k][i] - bob2[k][i])
            if diff > 1e-6:
                print(k,i,bob[k][i],bob2[k][i])


    ## Generate atomic coulomb matrix representation, sorted by distance,
    ## with cutoffs, using the Compound class
    #mol.generate_atomic_coulomb_matrix(size = size, sorting = "distance",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0)

    #acm = atomic_coulomb_matrix(mol.nuclear_charges, mol.coordinates, size, sorting = "distance",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0)

    #assert np.allclose(mol.representation, acm), "Error in atomic coulomb matrix representation"

    ## Generate atomic coulomb matrix representation, sorted by row-norm,
    ## with cutoffs, using the Compound class
    #mol.generate_atomic_coulomb_matrix(size = size, sorting = "row-norm",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0)

    #acm = atomic_coulomb_matrix(mol.nuclear_charges, mol.coordinates, size, sorting = "row-norm",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0)

    #assert np.allclose(mol.representation, acm), "Error in atomic coulomb matrix representation"


    ## Generate the sncf1 variant of the atomic coulomb matrix representation, sorted by distance,
    ## with cutoffs using the python interface
    #acm = generate_atomic_coulomb_matrix(mol.nuclear_charges,
    #        mol.coordinates, size = size, sorting = "distance",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0,
    #        variant = "sncf1", localization = 2.0)

    #acm2 = atomic_coulomb_matrix(mol.nuclear_charges,
    #        mol.coordinates, size = size, sorting = "distance",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0,
    #        variant = "sncf1", localization = 2.0)

    #assert np.allclose(acm2, acm), "Error in atomic coulomb matrix representation"

    ## Generate the sncf1 variant of the atomic coulomb matrix representation, sorted by row-norm,
    ## with cutoffs using the python interface
    #acm = generate_atomic_coulomb_matrix(mol.nuclear_charges,
    #        mol.coordinates, size = size, sorting = "row-norm",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0,
    #        variant = "sncf1", localization = 2.0)

    #acm2 = atomic_coulomb_matrix(mol.nuclear_charges,
    #        mol.coordinates, size = size, sorting = "row-norm",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0,
    #        variant = "sncf1", localization = 2.0)

    #assert np.allclose(acm2, acm), "Error in atomic coulomb matrix representation"

    ## Generate the sncf2 variant of the atomic coulomb matrix representation, sorted by distance,
    ## with cutoffs using the python interface
    #acm = generate_atomic_coulomb_matrix(mol.nuclear_charges,
    #        mol.coordinates, size = size, sorting = "distance",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0,
    #        variant = "sncf2", localization = 2.0)

    #acm2 = atomic_coulomb_matrix(mol.nuclear_charges,
    #        mol.coordinates, size = size, sorting = "distance",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0,
    #        variant = "sncf2", localization = 2.0)

    #assert np.allclose(acm2, acm), "Error in atomic coulomb matrix representation"

    ## Generate the sncf2 variant of the atomic coulomb matrix representation, sorted by distance,
    ## with cutoffs using the python interface
    #acm = generate_atomic_coulomb_matrix(mol.nuclear_charges,
    #        mol.coordinates, size = size, sorting = "row-norm",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0,
    #        variant = "sncf2", localization = 2.0)

    #acm2 = atomic_coulomb_matrix(mol.nuclear_charges,
    #        mol.coordinates, size = size, sorting = "row-norm",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0,
    #        variant = "sncf2", localization = 2.0)

    #assert np.allclose(acm2, acm), "Error in atomic coulomb matrix representation"
    pass

def print_mol(mol):
    n = len(mol.representation.shape)
    if n == 1:
        for item in np.concatenate(mol.representation):
            print("{:.9e}".format(item), end='  ')
        print()
    elif n == 2:
        for atom in mol.representation:
            for item in atom:
                print("{:.9e}".format(item), end='  ')
            print()

if __name__ == "__main__":
    #test_coulomb_matrix()
    #test_atomic_coulomb_matrix()
    #test_eigenvalue_coulomb_matrix()
    #test_bob()
    #test_local_bob()
    test_representations()

