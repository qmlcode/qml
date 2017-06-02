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


if __name__ == "__main__":

    # Generate a compound
    mol = qml.Compound(xyz="qm7/0001.xyz")

    # Generate a representation using the Compound class
    mol.generate_coulomb_matrix(size=5, sorting="row-norm")
    print(mol.coulomb_matrix)

    # Generate a representation using the python interface
    cm2 = generate_coulomb_matrix(mol.coordinates,
                mol.nuclear_charges, size=5, sorting="row-norm")
    print(cm2)

    # Generate an atomic coulomb matrix, sorted by distance
    mol.generate_atomic_coulomb_matrix(size=5, sorting="distance")
    print(mol.atomic_coulomb_matrix)

    bob = generate_bob(mol.coordinates,
            mol.nuclear_charges, mol.atomtypes, asize={"C": 2, "H": 5, "O": 10}, size=5)
    print(bob)

    mol.generate_bob(asize={"C": 2, "H": 5, "O": 10}, size=5)
    print(mol.bob)

    # Generate a representation using the python interface
    cm3 = generate_eigenvalue_coulomb_matrix(mol.coordinates,
                mol.nuclear_charges, size=5)
    print(cm3)

    mol.generate_eigenvalue_coulomb_matrix(size=5)
    print(mol.eigenvalue_coulomb_matrix)
