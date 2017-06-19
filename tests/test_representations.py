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
import os
import qml
from qml.representations import *
from qml.data import NUCLEAR_CHARGE


test_dir = os.path.dirname(os.path.realpath(__file__))

# Generate a compound
mol = qml.Compound(xyz = "%s/qm7/0100.xyz" % test_dir)
#mol = qml.Compound(xyz = "%s/qm7/0002.xyz" % test_dir)

def test_coulomb_matrix():

    size = mol.nuclear_charges.size + 1 # +1 to check that dummy atoms are handled correctly

    # Generate coulomb matrix representation, sorted by row-norm, using the Compound class
    mol.generate_coulomb_matrix(size = size, sorting = "row-norm")

    # Generate coulomb matrix representation, sorted by row-norm,  using the python interface
    cm = generate_coulomb_matrix(mol.nuclear_charges,
        mol.coordinates, size = size, sorting = "row-norm")

    assert np.allclose(mol.representation, cm), "Error in coulomb matrix representation"

    # compare unsorted coulomb matrix implementations
    mol.generate_coulomb_matrix(size = size, sorting = "unsorted")

    cm_matrix, cm_feature_vector = unsorted_coulomb_matrix(mol.nuclear_charges, mol.coordinates, size)

    assert np.allclose(cm_feature_vector, mol.representation), "Error in coulomb matrix representation"

    # compare sorted coulomb matrix implementation
    cm_feature_vector = unsorted_to_sorted(cm_feature_vector)

    assert np.allclose(cm, cm_feature_vector), "Error in coulomb matrix representation"

def unsorted_to_sorted(cm):

    size = (-1 + int(np.sqrt(1 + 8*cm.size))) // 2
    norms = np.zeros(size)
    for i in range(size):
        for j in range(size):
            if j < i:
                idx = (i * (i+1)) // 2 + j
            else:
                idx = (j * (j+1)) // 2 + i
            norms[i] += cm[idx]**2

    sortargs = np.argsort(norms)[::-1]
    
    sorted_cm = np.zeros(cm.size)

    for i in range(size):
        si = sortargs[i]
        for j in range(i+1):
            sj = sortargs[j]
            if j < i:
                idx = (i*(i+1))//2 + j
            else:
                idx = (j*(j+1))//2 + i
            if sj < si:
                idy = (si*(si+1))//2 + sj
            else:
                idy = (sj*(sj+1))//2 + si

            sorted_cm[idx] = cm[idy]

    return sorted_cm

def unsorted_coulomb_matrix(nuclear_charges, coordinates, size):

    natoms = nuclear_charges.size
    cm = np.zeros((size, size))
    feature_vector = np.zeros((size*(size+1))//2)
    for i in range(natoms):
        cm[i,i] = 0.5 * nuclear_charges[i] ** 2.4
        idx = (i*(i+1))//2
        feature_vector[idx+i] = cm[i,i]
        for j in range(i):
            d = np.sqrt(np.sum((coordinates[i] - coordinates[j])**2))
            cm[i,j] = nuclear_charges[i] * nuclear_charges[j] / d
            cm[j,i] = nuclear_charges[i] * nuclear_charges[j] / d
            feature_vector[idx+j] = cm[i,j]

    return cm, feature_vector

def test_atomic_coulomb_matrix():

    size = mol.nuclear_charges.size + 1 # +1 to check that dummy atoms are handled correctly

    # Generate atomic coulomb matrix representation, sorted by distance, using the Compound class
    mol.generate_atomic_coulomb_matrix(size = size, sorting = "distance")

    # Generate atomic coulomb matrix representation, sorted by distance, using the python interface
    acm = generate_atomic_coulomb_matrix(mol.nuclear_charges,
            mol.coordinates, size = size, sorting = "distance")

    assert np.allclose(mol.representation, acm), "Error in atomic coulomb matrix representation"


    # Generate atomic coulomb matrix representation, sorted by distance, with reference implementation
    acm = atomic_coulomb_matrix(mol.nuclear_charges, mol.coordinates, size, "distance")

    assert np.allclose(mol.representation, acm), "Error in atomic coulomb matrix representation"
    
    # Generate atomic coulomb matrix representation, sorted by row-norm, using the Compound class
    mol.generate_atomic_coulomb_matrix(size = size, sorting = "row-norm")

    acm = atomic_coulomb_matrix(mol.nuclear_charges, mol.coordinates, size, sorting = "row-norm")

    assert np.allclose(mol.representation, acm), "Error in atomic coulomb matrix representation"

def atomic_coulomb_matrix(nuclear_charges, coordinates, size, sorting = "distance"):

    cm_matrix, cm_feature_vector = unsorted_coulomb_matrix(nuclear_charges, coordinates, size)
    natoms = nuclear_charges.size

    if sorting == "row-norm":
        sorting = np.zeros((natoms, natoms))
        for k in range(natoms):
            norms = np.zeros(natoms)
            norms[k] = np.inf
            for i in range(natoms):
                if i == k:
                    continue
                for j in range(natoms):
                    # uncomment below, to ignore the central atom in the sorting
                    #if j == k:
                    #    continue
                    if j < i:
                        idx = (i * (i+1)) // 2 + j
                    else:
                        idx = (j * (j+1)) // 2 + i
                    norms[i] += cm_feature_vector[idx]**2

            sorting[k] = np.argsort(norms)[::-1]

    else: # sort by distances
        sorting = np.zeros((natoms, natoms))
        for i in range(natoms):
            distances = np.sum((coordinates - coordinates[i].T)**2, axis = 1)
            sorting[i] = np.argsort(distances)

    sorted_cm = np.zeros((natoms, cm_feature_vector.size))

    sortargs = np.arange(size)

    for k in range(natoms):
        sortargs[:natoms] = sorting[k]
        for i in range(size):
            si = sortargs[i]
            for j in range(i+1):
                sj = sortargs[j]
                if j < i:
                    idx = (i*(i+1))//2 + j
                else:
                    idx = (j*(j+1))//2 + i
                if sj < si:
                    idy = (si*(si+1))//2 + sj
                else:
                    idy = (sj*(sj+1))//2 + si

                sorted_cm[k, idx] = cm_feature_vector[idy]

    return sorted_cm

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

def eigenvalue_coulomb_matrix(nuclear_charges, coordinates, size):
    coulomb_matrix, cm_feature_vector = unsorted_coulomb_matrix(nuclear_charges, 
             coordinates, size = size)
    eigenvalues = np.linalg.eigh(coulomb_matrix)[0]
    return sorted(eigenvalues, reverse = True)

def test_bob():

    asize = dict([(key, value) for key,value in mol.natypes.items()])

    # Generate bag of bonds representation using the Compound class
    mol.generate_bob(asize)

    # Generate bag of bonds representation using the python interface
    bob = generate_bob(mol.nuclear_charges,
            mol.coordinates, mol.atomtypes, asize = asize)

    assert np.allclose(mol.representation, bob), "Error in bag of bonds representation"

    # Compare with python implementation
    bob = bob_reference(mol.nuclear_charges, mol.coordinates, mol.atomtypes, size = (mol.natoms), asize = asize)

    print(bob)
    print(mol.representation)

    assert np.allclose(mol.representation, bob), "Error in bag of bonds representation"

def bob_reference(nuclear_charges, coordinates, atomtypes, size = 23, asize = {"O":3, "C":7, "N":3, "H":16, "S":1}):

    coulomb_matrix = generate_coulomb_matrix(nuclear_charges,
            coordinates, size = size, sorting = "unsorted")

    coulomb_matrix = vector_to_matrix(coulomb_matrix)

    atoms = sorted(asize, key=asize.get)
    nmax = [asize[key] for key in atoms]

    descriptor = []
    positions = dict([(element, np.where(atomtypes == element)[0]) for element in atoms])
    print(atomtypes)
    for i, (element1, size1) in enumerate(zip(atoms,nmax)):
        pos1 = positions[element1]
        feature_vector = np.zeros(size1)
        feature_vector[:pos1.size] = np.diag(coulomb_matrix)[pos1]
        descriptor.append(feature_vector)
        for j, (element2, size2) in enumerate(zip(atoms,nmax)):
            if i > j:
                continue
            if i == j:
                size = (size1*(size1-1))//2
                feature_vector = np.zeros(size)
                sub_matrix = coulomb_matrix[np.ix_(pos1,pos1)]
                feature_vector[:(pos1.size*(pos1.size-1))//2] = sub_matrix[np.triu_indices(pos1.size, 1)]
                feature_vector[::-1].sort()
                descriptor.append(feature_vector)
            else:
                pos2 = positions[element2]
                feature_vector = np.zeros(size1*size2)
                feature_vector[:pos1.size*pos2.size] = coulomb_matrix[np.ix_(pos1,pos2)].ravel()
                feature_vector[::-1].sort()
                descriptor.append(feature_vector)

    return np.concatenate(descriptor)

def vector_to_matrix(vec):
    size = (-1 + int(np.sqrt(1 + 8 * vec.size))) // 2
    mat = np.zeros((size,size))

    count = 0
    for i in range(size):
        for j in range(i+1):
            mat[i,j] = vec[count]
            mat[j,i] = mat[i,j]
            count += 1

    return mat


if __name__ == "__main__":
    test_coulomb_matrix()
    test_atomic_coulomb_matrix()
    test_eigenvalue_coulomb_matrix()
    test_bob()

