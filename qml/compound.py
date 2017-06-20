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

from .arad import ARAD

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

            :param size: The size of the largest molecule supported by the representation
            :type size: integer
            :param sorting: How the atom indices are sorted ('row-norm', 'unsorted')
            :type sorting: string

            :return: 1D representation - shape (size(size+1)/2,)
            :rtype: numpy array
        """

        self.representation = generate_coulomb_matrix(self.nuclear_charges, 
            self.coordinates, size = size, sorting = sorting)

    def generate_eigenvalue_coulomb_matrix(self, size = 23):
        """ Creates an eigenvalue Coulomb Matrix representation of a molecule.
            A matrix :math:`M` is constructed with elements

            .. math::

                M_{ij} =
                  \\begin{cases}
                     \\tfrac{1}{2} Z_{i}^{2.4} & \\text{if } i = j \\\\
                     \\frac{Z_{i}Z_{j}}{\\| {\\bf R}_{i} - {\\bf R}_{j}\\|}       & \\text{if } i \\neq j
                  \\end{cases},

            where :math:`i` and :math:`j` are atom indices, :math:`Z` is nuclear charge and
            :math:`\\bf R` is the coordinate in euclidean space.
            The molecular representation of the molecule is then the sorted eigenvalues of M.
            The representation is calculated using an OpenMP parallel Fortran routine.

            :param size: The size of the largest molecule supported by the representation
            :type size: integer

            :return: 1D representation - shape (size, )
            :rtype: numpy array
        """

        self.representation = generate_eigenvalue_coulomb_matrix(
                self.nuclear_charges, self.coordinates, size = size)

    def generate_atomic_coulomb_matrix(self, size = 23, sorting = "row-norm", 
            central_cutoff = 1e6, central_decay = -1, interaction_cutoff = 1e6, interaction_decay = -1):
    """ Creates a Coulomb Matrix representation of the local environment of a central atom.
        For each central atom :math:`k`, a matrix :math:`M` is constructed with elements

        .. math::

            M_{ij}(k) =
              \\begin{cases}
                 \\tfrac{1}{2} Z_{i}^{2.4} \\cdot f_{ik}^2 & \\text{if } i = j \\\\
                 \\frac{Z_{i}Z_{j}}{\\| {\\bf R}_{i} - {\\bf R}_{j}\\|} \\cdot f_{ik}f_{jk}f_{ij} & \\text{if } i \\neq j
              \\end{cases},

        where :math:`i`, :math:`j` and :math:`k` are atom indices, :math:`Z` is nuclear charge and
        :math:`\\bf R` is the coordinate in euclidean space.

        :math:`f_{ij}` is a function that masks long range effects:

        .. math::

            f_{ij} =
              \\begin{cases}
                 1 & \\text{if } \\|{\\bf R}_{i} - {\\bf R}_{j} \\| \\leq r - \Delta r \\\\
                 \\tfrac{1}{2} \\big(1 + \\cos\\big(\\pi \\tfrac{\\|{\\bf R}_{i} - {\\bf R}_{j} \\|
                    - r + \Delta r}{\Delta r} \\big)\\big)     
                    & \\text{if } r - \Delta r < \\|{\\bf R}_{i} - {\\bf R}_{j} \\| \\leq r - \Delta r \\\\
                 0 & \\text{if } \\|{\\bf R}_{i} - {\\bf R}_{j} \\| > r
              \\end{cases},

        where the parameters ``central_cutoff`` and ``central_decay`` corresponds to the variables
        :math:`r` and :math:`\Delta r` respectively for interactions involving the central atom,
        and ``interaction_cutoff`` and ``interaction_decay`` corresponds to the variables
        :math:`r` and :math:`\Delta r` respectively for interactions not involving the central atom.

        if ``sorting = 'row-norm'``, the atom indices are ordered such that

            :math:`\\sum_j M_{1j}(k)^2 \\geq \\sum_j M_{2j}(k)^2 \\geq ... \\geq \\sum_j M_{nj}(k)^2`

        if ``sorting = 'distance'``, the atom indices are ordered such that

        .. math::

            \\|{\\bf R}_{1} - {\\bf R}_{k}\\| \\leq \\|{\\bf R}_{2} - {\\bf R}_{k}\\|
                \\leq ... \\leq \\|{\\bf R}_{n} - {\\bf R}_{k}\\|

        The upper triangular of M, including the diagonal, is concatenated to a 1D
        vector representation.
        The representation is calculated using an OpenMP parallel Fortran routine.

        :param size: The size of the largest molecule supported by the representation
        :type size: integer
        :param sorting: How the atom indices are sorted ('row-norm', 'distance')
        :type sorting: string
        :param central_cutoff: The distance from the central atom, where the coulomb interaction
            element will be zero
        :type central_cutoff: float
        :param central_decay: The distance over which the the coulomb interaction decays from full to none
        :type central_decay: float
        :param interaction_cutoff: The distance between two non-central atom, where the coulomb interaction
            element will be zero
        :type interaction_cutoff: float
        :param interaction_decay: The distance over which the the coulomb interaction decays from full to none
        :type interaction_decay: float


        :return: nD representation - shape (:math:`N_{atoms}`, size(size+1)/2)
        :rtype: numpy array
    """

        self.representation = generate_atomic_coulomb_matrix(
            self.nuclear_charges, self.coordinates, size = size,
            sorting = sorting, central_cutoff = central_cutoff, central_decay = central_decay,
            interaction_cutoff = interaction_cutoff, interaction_decay = interaction_decay)

    def generate_bob(self, asize = {"O":3, "C":7, "N":3, "H":16, "S":1}):
        """ Creates a Bag of Bonds (BOB) representation of a molecule.
            The representation expands on the coulomb matrix representation.
            For each element a bag (vector) is constructed for self interactions
            (e.g. ('C', 'H', 'O')).
            For each element pair a bag is constructed for interatomic interactions
            (e.g. ('CC', 'CH', 'CO', 'HH', 'HO', 'OO')), sorted by value.
            The self interaction of element :math:`I` is given by

                :math:`\\tfrac{1}{2} Z_{I}^{2.4}`,

            with :math:`Z_{i}` being the nuclear charge of element :math:`i`
            The interaction between atom :math:`i` of element :math:`I` and 
            atom :math:`j` of element :math:`J` is given by

                :math:`\\frac{Z_{I}Z_{J}}{\\| {\\bf R}_{i} - {\\bf R}_{j}\\|}`

            with :math:`R_{i}` being the euclidean coordinate of atom :math:`i`.
            The sorted bags are concatenated to an 1D vector representation.
            The representation is calculated using an OpenMP parallel Fortran routine.

            :param asize: The maximum number of atoms of each element type supported by the representation
            :type size: dictionary

            :return: 1D representation
            :rtype: numpy array
        """

        self.representation = generate_bob(self.nuclear_charges, self.coordinates, 
                self.atomtypes, asize = asize)

    def generate_arad_representation(self, size = 23):

        arad = ARAD(maxMolSize = size, maxAts = size)
        self.representation = arad.describe(self.coordinates,
                self.nuclear_charges)

        assert (self.representation).shape[0] == size, "ERROR: Check ARAD descriptor size!"
        assert (self.representation).shape[2] == size, "ERROR: Check ARAD descriptor size!"

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
