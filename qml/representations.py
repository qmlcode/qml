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
    """ Creates a Coulomb Matrix representation of a molecule.
        A matrix :math:`M` is constructed with elements

        .. math::

            M_{ij} =
              \\begin{cases}
                 \\tfrac{1}{2} Z_{i}^{2.4} & \\text{if } i = j \\\\
                 \\frac{Z_{i}Z_{j}}{\\| {\\bf R}_{i} - {\\bf R}_{j}\\|}       & \\text{if } i \\neq j
              \\end{cases},

        where :math:`i` and :math:`j` are atom indices, :math:`Z` is nuclear charge and
        :math:`\\bf R` is the coordinate in euclidean space.
        if ``sorting = 'row-norm'``, the atom indices are reordered such that

            :math:`\\sum_j M_{1j}^2 \\geq \\sum_j M_{2j}^2 \\geq ... \\geq \\sum_j M_{nj}^2`

        The upper triangular of M, including the diagonal, is concatenated to a 1D
        vector representation.
        The representation is calculated using an OpenMP parallel Fortran routine.

        :param nuclear_charges: Nuclear charges of the atoms in the molecule
        :type nuclear_charges: numpy array
        :param coordinates: 3D Coordinates of the atoms in the molecule
        :type coordinates: numpy array
        :param size: The size of the largest molecule supported by the representation
        :type size: integer
        :param sorting: How the atom indices are sorted ('row-norm', 'unsorted')
        :type sorting: string

        :return: 1D representation - shape (size(size+1)/2,)
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
    return fgenerate_eigenvalue_coulomb_matrix(nuclear_charges,
        coordinates, size)

def generate_bob(nuclear_charges, coordinates, atomtypes, asize = {"O":3, "C":7, "N":3, "H":16, "S":1}):

    n = 0
    atoms = sorted(asize, key=asize.get)
    nmax = [asize[key] for key in atoms]
    print(atoms,nmax)
    ids = np.zeros(len(nmax), dtype=int)
    for i, (key, value) in enumerate(zip(atoms,nmax)):
        n += value * (1+value)
        ids[i] = NUCLEAR_CHARGE[key]
        for j in range(i):
            v = nmax[j]
            n += 2 * value * v
    n /= 2

    return fgenerate_bob(nuclear_charges, coordinates, nuclear_charges, ids, nmax, n)
