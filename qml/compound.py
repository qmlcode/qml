# MIT License
#
# Copyright (c) 2016 Anders Steen Christensen, Felix Faber
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
import collections

from .data import NUCLEAR_CHARGE

from .representations import generate_coulomb_matrix
from .representations import generate_atomic_coulomb_matrix
from .representations import generate_bob
from .representations import generate_eigenvalue_coulomb_matrix
from .representations import generate_slatm_representation

from .arad import generate_arad_representation

class Compound(object):

    def __init__(self, xyz = None):

        empty_array = np.asarray([], dtype = float)

        self.molid = float("nan")
        self.name = None

        # Information about the compound
        self.natoms = float("nan")
        self.natypes = {}
        self.atomtypes = empty_array
        self.atomtype_indices = collections.defaultdict(list)
        self.nuclear_charges = empty_array
        self.coordinates = empty_array
        self.active_atoms = empty_array
        self.unit_cell = empty_array

        # Container for misc properties
        self.energy = float("nan")
        self.properties = empty_array
        self.properties2 = empty_array

        # Representations:
        self.representation = empty_array

        if xyz is not None:
            self.read_xyz(xyz)

    def generate_coulomb_matrix(self, size = 23, sorting = "row-norm"):

        self.representation = generate_coulomb_matrix(self.nuclear_charges, 
            self.coordinates, size = size, sorting = sorting)

    def generate_eigenvalue_coulomb_matrix(self, size = 23):

        self.representation = generate_eigenvalue_coulomb_matrix(
                self.nuclear_charges, self.coordinates, size = size)

    def generate_atomic_coulomb_matrix(self, size = 23, sorting = "row-norm"):

        self.representation = generate_atomic_coulomb_matrix(
            self.nuclear_charges, self.coordinates, size = size, sorting = sorting)

    def generate_bob(self, asize = {"O":3, "C":7, "N":3, "H":16, "S":1}):

        self.representation = generate_bob(self.nuclear_charges, self.coordinates, 
                self.atomtypes, asize = asize)

    def generate_arad_representation(self, size = 23):

        self.representation = generate_arad_representation(self.coordinates,
                self.nuclear_charges, size=size)

        assert (self.representation).shape[0] == size, "ERROR: Check ARAD descriptor size!"
        assert (self.representation).shape[2] == size, "ERROR: Check ARAD descriptor size!"

    def generate_slatm_representation(self, mbtypes,
        local=False, sigmas=[0.05,0.05], dgrids=[0.03,0.03], rcut=4.8, 
        alchemy=False, rpower=6, iprt=False):

        slatm = generate_slatm_representation(self.coordinates, self.nuclear_charges,
                mbtypes, local=local, sigmas=sigmas, dgrids=dgrids, rcut=rcut,
                alchemy=alchemy, rpower=rpower, iprt=iprt)
        if local: slatm = np.asarray(slatm)
        self.representation = slatm
        print(self.representation)
        print(self.representation.shape)


    def read_xyz(self, filename):
    
        f = open(filename, "r")
        lines = f.readlines()
        f.close()
    
        self.natoms = int(lines[0])
        self.atomtypes = np.empty(self.natoms, dtype=str)
        self.nuclear_charges = np.empty(self.natoms, dtype=int)
        self.coordinates = np.empty((self.natoms, 3), dtype=float)
    
        self.name = filename
    
        for i, line in enumerate(lines[2:]):
            tokens = line.split()
    
            if len(tokens) < 4:
                break
    
            self.atomtypes[i] = tokens[0]
            self.atomtype_indices[tokens[0]].append(i)
            self.nuclear_charges[i] = NUCLEAR_CHARGE[tokens[0]]
    
            self.coordinates[i] = np.asarray(tokens[1:4], dtype=float)
    
        self.natypes = dict([(key, len(value)) for key,value in self.atomtype_indices.items()])
