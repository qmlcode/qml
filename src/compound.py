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

from .data import NUCLEAR_CHARGE

from .representations import fgenerate_coulomb_matrix
from .representations import fgenerate_unsorted_coulomb_matrix
from .representations import fgenerate_local_coulomb_matrix
from .representations import fgenerate_atomic_coulomb_matrix

from .arad import ARAD

class Compound:

    def __init__(self, xyz=None):

        self.molid = float("nan")
        self.name = None

        # Information about the compound
        self.natoms = float("nan")
        self.atomtypes = None
        self.nuclear_charges = None
        self.coordinates = None
        self.active_atoms = None
        self.unit_cell = None

        # Container for misc properties
        self.energy = float("nan")
        self.properties = []
        self.properties2 = []

        # Representations:
        self.coulomb_matrix = None
        self.atomic_coulomb_matrix = None
        self.arad_representation = None
        self.aras_representation = None

        if xyz is not None:
            self.read_xyz(xyz)

    def generate_coulomb_matrix(self, size=23, sorting="row-norm"):

        if (sorting == "row-norm"):
            self.coulomb_matrix = fgenerate_coulomb_matrix(self.nuclear_charges, \
                self.coordinates, self.natoms, size)

        elif (sorting == "unsorted"):
            self.coulomb_matrix = fgenerate_unsorted_coulomb_matrix(self.nuclear_charges, \
                self.coordinates, self.natoms, size)

        else:
            print("ERROR: Unknown sorting scheme requested")


    def generate_atomic_coulomb_matrix(self,size=23, sorting ="row-norm"):

        if (sorting == "row-norm"):
            self.local_coulomb_matrix = fgenerate_local_coulomb_matrix( \
                self.nuclear_charges, self.coordinates, self.natoms, size)

        elif (sorting == "distance"):
            self.atomic_coulomb_matrix = fgenerate_atomic_coulomb_matrix( \
                self.nuclear_charges, self.coordinates, self.natoms, size)

        else:
            print("ERROR: Unknown sorting scheme requested")


    def generate_arad_representation(self, size=23):
        arad = ARAD(maxMolSize=size,maxAts=size)
        self.arad_representation = arad.describe(np.array(self.coordinates), \
                np.array(self.nuclear_charges))

        assert (self.arad_representation).shape[0] == size, "ERROR: Check ARAD descriptor size!"
        assert (self.arad_representation).shape[2] == size, "ERROR: Check ARAD descriptor size!"


    def read_xyz(self, filename):

        f = open(filename, "r")
        lines = f.readlines()
        f.close()

        self.natoms = int(lines[0])
        self.atomtypes = []
        self.nuclear_charges = []
        self.coordinates = []

        self.name = filename

        for line in lines[2:]:
            tokens = line.split()

            if len(tokens) != 4:
                break

            self.atomtypes.append(tokens[0])
            self.nuclear_charges.append(NUCLEAR_CHARGE[tokens[0]])

            x = float(tokens[1])
            y = float(tokens[2])
            z = float(tokens[3])

            self.coordinates.append(np.array([x, y, z]))

        self.coordinates = np.array(self.coordinates)
