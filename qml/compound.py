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
from .representations import generate_slatm

from .arad import generate_arad_representation

class Compound(object):
    """ The ``Compound`` class is used to store data from  

        :param xyz: Option to initialize the ``Compound`` with data from an XYZ file.
        :type xyz: string
    """

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
        self.unit_cell = None

        # Container for misc properties
        self.energy = float("nan")
        self.properties = empty_array
        self.properties2 = empty_array

        # Representations:
        self.representation = empty_array

        if xyz is not None:
            self.read_xyz(xyz)

    def generate_coulomb_matrix(self, size = 23, sorting = "row-norm"):
        """ Generates a sorted molecular coulomb and stores it in the ``representation`` variable.
    Sorting either by ``"row-norm"`` or ``"unsorted"``.
    ``size=`` denotes the max number of atoms in the molecule (thus the size of the resulting square matrix.
    The resulting matrix is the upper triangle put into the form of a 1D-vector.

    :param size: Max number of atoms in representation.
    :type size: integer
    :param sorting: Matrix sorting scheme, "row-norm" or "unsorted".
    :type sorting: string
    """
        self.representation = generate_coulomb_matrix(self.nuclear_charges, 
            self.coordinates, size = size, sorting = sorting)

    def generate_eigenvalue_coulomb_matrix(self, size = 23):
        """ Generates the eigenvalue-Coulomb matrix representation and stores it in the ``representation`` variable.
    ``size=`` denotes the max number of atoms in the molecule (thus the size of the resulting square matrix.
    The resulting matrix is in the form of a 1D-vector.

    :param size: Max number of atoms in representation.
    :type size: integer
    """
        self.representation = generate_eigenvalue_coulomb_matrix(
                self.nuclear_charges, self.coordinates, size = size)

    def generate_atomic_coulomb_matrix(self, size = 23, sorting = "row-norm"):
        """ Generates a list of sorted Coulomb matrices and stores it in the ``representation`` variable.
    Sorting either by ``"row-norm"`` or ``"distance"``, the latter refers to sorting by distance to each query atom.
    ``size=`` denotes the max number of atoms in the molecule (thus the size of the resulting square matrix.
    The resulting matrix is the upper triangle put into the form of a 1D-vector.

    :param size: Max number of atoms in representation.
    :type size: integer
    :param sorting: Matrix sorting scheme, "row-norm" or "distance".
    :type sorting: string
    """
        self.representation = generate_atomic_coulomb_matrix(
            self.nuclear_charges, self.coordinates, size = size, sorting = sorting)

    def generate_bob(self, asize = {"O":3, "C":7, "N":3, "H":16, "S":1}):
        """ Generates a bag-of-bonds (BOB) representation of the molecule and stores it in the ``representation`` variable. 
    ``asize=`` is the maximum number of atoms of each type (necessary to generate bags of minimal sizes).
    The resulting matrix is the BOB representation put into the form of a 1D-vector.

    :param size: Max number of atoms in representation.
    :type size: integer
    :param asize: Max number of each element type.
    :type asize: dict
    """
        self.representation = generate_bob(self.nuclear_charges, self.coordinates, 
                self.atomtypes, asize = asize)

    def generate_arad_representation(self, size = 23):
        """Generates the representation for the ARAD-kernel. Note that this representation is incompatible with generic ``qml.kernel.*`` kernels.
    :param size: Max number of atoms in representation.
    :type size: integer
    """
        self.representation = generate_arad_representation(self.coordinates,
                self.nuclear_charges, size=size)

        assert (self.representation).shape[0] == size, "ERROR: Check ARAD descriptor size!"
        assert (self.representation).shape[2] == size, "ERROR: Check ARAD descriptor size!"

    def generate_slatm(self, mbtypes,
        local=False, sigmas=[0.05,0.05], dgrids=[0.03,0.03], rcut=4.8, pbc='000',
        alchemy=False, rpower=6):
        """Generate Spectrum of London and Axillrod-Teller-Muto potential (SLATM) representation.
    Both global (``local=False``) and local (``local=True``) SLATM are available.

    A version that works for periodic boundary conditions will be released soon.

    NOTE: You will need to run the ``get_slatm_mbtypes()`` function to get the ``mbtypes`` input (or generate it manually).

    :param mbtypes: Many-body types for the whole dataset, including 1-, 2- and 3-body types. Could be obtained by calling ``get_slatm_mbtypes()``.
    :type mbtypes: list
    :param local: Generate a local representation. Defaulted to False (i.e., global representation); otherwise, atomic version.
    :type local: bool
    :param sigmas: Controlling the width of Gaussian smearing function for 2- and 3-body parts, defaulted to [0.05,0.05], usually these do not need to be adjusted.
    :type sigmas: list
    :param dgrids: The interval between two sampled internuclear distances and angles, defaulted to [0.03,0.03], no need for change, compromised for speed and accuracy.
    :type dgrids: list
    :param rcut: Cut-off radius, defaulted to 4.8 Angstrom.
    :type rcut: float
    :param alchemy: Swith to use the alchemy version of SLATM. (default=False)
    :type alchemy: bool
    :param pbc: defaulted to '000', meaning it's a molecule; the three digits in the string corresponds to x,y,z direction
    :type pbc: string
    :param rpower: The power of R in 2-body potential, defaulted to London potential (=6).
    :type rpower: float
    :return: 1D SLATM representation
    :rtype: numpy array
    """

        slatm = generate_slatm(self.coordinates, self.nuclear_charges, mbtypes, local=local, 
                sigmas=sigmas, dgrids=dgrids, rcut=rcut, unit_cell=self.unit_cell, 
                alchemy=alchemy, rpower=rpower)
        if local: slatm = np.asarray(slatm)
        self.representation = slatm


    def read_xyz(self, filename):
        """(Re-)initializes the Compound-object with data from an xyz-file.

    :param filename: Input xyz-filename.
    :type filename: string
    """
    
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
