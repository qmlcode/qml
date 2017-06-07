# MIT License
#
# Copyright (c) 2016 Anders Steen Christensen and Lars A. Bratholm
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
from __future__ import division

import numpy as np

from .frepresentations import fgenerate_coulomb_matrix
from .frepresentations import fgenerate_unsorted_coulomb_matrix
from .frepresentations import fgenerate_local_coulomb_matrix
from .frepresentations import fgenerate_atomic_coulomb_matrix


def vector_to_matrix(v):
    if not (np.sqrt(8*v.shape[0]+1) == int(np.sqrt(8*v.shape[0]+1))):
        print("ERROR: Can not make a square matrix.")
        exit(1)

    n = v.shape[0]
    l = (-1 + int(np.sqrt(8*n+1)))//2
    M = np.empty((l,l))

    index = 0
    for i in range(l):
        for j in range(l):
            if j > i:
                continue

            M[i,j] = v[index]
            M[j,i] = M[i,j]

            index += 1
    return M


def generate_coulomb_matrix(coordinates, nuclear_charges, size=23, sorting="row-norm"):

    if (sorting == "row-norm"):
        return fgenerate_coulomb_matrix(nuclear_charges, \
            coordinates, len(nuclear_charges), size)

    elif (sorting == "unsorted"):
        return fgenerate_unsorted_coulomb_matrix(nuclear_charges, \
            coordinates, len(nuclear_charges), size)

    else:
        print("ERROR: Unknown sorting scheme requested")


def generate_atomic_coulomb_matrix(coordinates, nuclear_charges, size=23, sorting ="row-norm"):

    if (sorting == "row-norm"):
        return fgenerate_local_coulomb_matrix(nuclear_charges,\
            coordinates, len(nuclear_charges), size)

    elif (sorting == "distance"):
        return fgenerate_atomic_coulomb_matrix(nuclear_charges, \
            coordinates, len(nuclear_charges), size)

    else:
        print("ERROR: Unknown sorting scheme requested")


def generate_bob(coordinates, nuclear_charges, atomtypes, size=23, asize={"O":3, "C":7, "N":3, "H":16, "S":1}):

    natoms = len(nuclear_charges)

    coulomb_matrix = fgenerate_unsorted_coulomb_matrix(nuclear_charges,
            coordinates, natoms, size)

    coulomb_matrix = vector_to_matrix(coulomb_matrix)
    descriptor = []
    atomtypes = np.asarray(atomtypes)
    for atom1, size1 in asize.items():
        pos1 = np.where(atomtypes == atom1)[0]
        feature_vector = np.zeros(size1)
        feature_vector[:pos1.size] = np.diag(coulomb_matrix)[pos1]
        feature_vector.sort()
        descriptor.append(feature_vector[:])
        for atom2, size2 in asize.items():
            if atom1 > atom2:
                continue
            if atom1 == atom2:
                size = size1*(size1-1)//2
                feature_vector = np.zeros(size)
                sub_matrix = coulomb_matrix[np.ix_(pos1,pos1)]
                feature_vector[:pos1.size*(pos1.size-1)//2] = sub_matrix[np.triu_indices(pos1.size, 1)]
                feature_vector.sort()
                descriptor.append(feature_vector[:])
            else:
                pos2 = np.where(atomtypes == atom2)[0]
                feature_vector = np.zeros(size1*size2)
                feature_vector[:pos1.size*pos2.size] = coulomb_matrix[np.ix_(pos1,pos2)].ravel()
                feature_vector.sort()
                descriptor.append(feature_vector[:])

    return np.concatenate(descriptor)


def generate_eigenvalue_coulomb_matrix(coordinates, nuclear_charges, size=23):
    coulomb_matrix = fgenerate_coulomb_matrix(nuclear_charges, \
             coordinates, len(nuclear_charges), size)
    descriptor = np.linalg.eigh(vector_to_matrix(coulomb_matrix))[0]
    return descriptor
