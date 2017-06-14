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

import numpy as np

from .frepresentations import fgenerate_coulomb_matrix
from .frepresentations import fgenerate_unsorted_coulomb_matrix
from .frepresentations import fgenerate_local_coulomb_matrix
from .frepresentations import fgenerate_atomic_coulomb_matrix
from .frepresentations import fgenerate_eigenvalue_coulomb_matrix
from .frepresentations import fgenerate_bob

from .data import NUCLEAR_CHARGE

def generate_coulomb_matrix(nuclear_charges, coordinates, size = 23, sorting = "row-norm"):
    """ Generates a sorted molecular coulomb matrix, sorted either by ``"row-norm"`` or ``"unsorted"``.
    ``size=`` denotes the max number of atoms in the molecule (thus the size of the resulting square matrix.
    The resulting matrix is the upper triangle put into the form of a 1D-vector.

    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param coordinates: Input coordinates.
    :type coordinates: numpy array
    :param size: Max number of atoms in representation.
    :type size: integer
    :param sorting: Matrix sorting scheme, "row-norm" or "unsorted".
    :type sorting: string
    :return: 1D Coulomb matrix representation
    :rtype: numpy array
    """

    if (sorting == "row-norm"):
        return fgenerate_coulomb_matrix(nuclear_charges, \
            coordinates, size)

    elif (sorting == "unsorted"):
        return fgenerate_unsorted_coulomb_matrix(nuclear_charges, \
            coordinates, size)

    else:
        print("ERROR: Unknown sorting scheme requested")
        raise SystemExit

def generate_atomic_coulomb_matrix(nuclear_charges, coordinates, size = 23, sorting = "distance"):
    """ Generates a list of sorted Coulomb matrices, sorted either by ``"row-norm"`` or ``"distance"``, the latter refers to sorting by distance to each query atom.
    ``size=`` denotes the max number of atoms in the molecule (thus the size of the resulting square matrix.
    The resulting matrix is the upper triangle put into the form of a 1D-vector.

    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param coordinates: Input coordinates.
    :type coordinates: numpy array
    :param size: Max number of atoms in representation.
    :type size: integer
    :param sorting: Matrix sorting scheme, "row-norm" or "distance".
    :type sorting: string
    :return: List of 1D Coulomb matrix representations.
    :rtype: numpy array
    """

    if (sorting == "row-norm"):
        return fgenerate_local_coulomb_matrix(nuclear_charges,
            coordinates, nuclear_charges.size, size)

    elif (sorting == "distance"):
        return fgenerate_atomic_coulomb_matrix(nuclear_charges,
            coordinates, nuclear_charges.size, size)

    else:
        print("ERROR: Unknown sorting scheme requested")
        raise SystemExit

def generate_eigenvalue_coulomb_matrix(nuclear_charges, coordinates, size = 23):
    """ Generates an eigenvalue molecular coulomb matrix, sorted by descending values.
    ``size=`` denotes the max number of atoms in the molecule (thus the size of the resulting square matrix.
    The resulting matrix is in the form of a 1D-vector.

    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param coordinates: Input coordinates.
    :type coordinates: numpy array
    :param size: Max number of atoms in representation.
    :type size: integer
    :return: 1D Coulomb matrix representation
    :rtype: numpy array
    """
    return fgenerate_eigenvalue_coulomb_matrix(nuclear_charges,
        coordinates, size)

def generate_bob(nuclear_charges, coordinates, atomtypes, asize = {"O":3, "C":7, "N":3, "H":16, "S":1}):
    """ Generates a bag-of-bonds (BOB) representation of the molecule.
    ``asize=`` is the maximum number of atoms of each type (necessary to generate bags of minimal sizes).
    The resulting matrix is the BOB representation put into the form of a 1D-vector.
    Reference: 10.1021/acs.jpclett.5b00831

    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param coordinates: Input coordinates.
    :type coordinates: numpy array
    :param size: Max number of atoms in representation.
    :type size: integer
    :param asize: Max number of each element type.
    :type asize: dict
    :return: 1D BOB representation.
    :rtype: numpy array
    """

    n = 0
    ids = np.zeros(len(asize), dtype=int)
    for i, (key, value) in enumerate(asize.items()):
        n += value * (1+value)
        ids[i] = NUCLEAR_CHARGE[key]
        for j in range(i):
            v = asize.values()[j]
            n += 2 * value * v
    n /= 2

    return fgenerate_bob(nuclear_charges, coordinates, nuclear_charges, ids, asize.values(), n)
