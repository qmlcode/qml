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
    ids = np.zeros(len(asize), dtype=int)
    for i, (key, value) in enumerate(asize.items()):
        n += value * (1+value)
        ids[i] = NUCLEAR_CHARGE[key]
        for j in range(i):
            v = list(asize.values())[j]
            n += 2 * value * v
    n /= 2

    return fgenerate_bob(nuclear_charges, coordinates, nuclear_charges, ids, list(asize.values()), n)
