# MIT License
#
# Copyright (c) 2017-2018 Anders Steen Christensen, Lars Andersen Bratholm, Bing Huang
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

import copy
import itertools

import numpy as np
from sklearn.base import BaseEstimator

from .data import Data
from ..utils import is_positive_integer_or_zero_array, get_unique
from ..ml.representations.frepresentations import fgenerate_coulomb_matrix
from ..ml.representations.frepresentations import fgenerate_unsorted_coulomb_matrix
from ..ml.representations.frepresentations import fgenerate_local_coulomb_matrix
from ..ml.representations.frepresentations import fgenerate_atomic_coulomb_matrix
from ..ml.representations.representations import get_slatm_mbtypes
from ..ml.representations.representations import generate_slatm

class _BaseRepresentation(BaseEstimator):
    """
    Base class for representations
    """

    def fit(self, X, y=None):
        """
        The fit routine is needed for scikit-learn pipelines.
        Must return self.
        """
        raise NotImplementedError

    def transform(self, X):
        """
        The transform routine is needed for scikit-learn pipelines.
        Needs to store the representations in self.data.representations
        and return self.data.
        """
        raise NotImplementedError


    def _preprocess_input(self, X):
        """
        Convenience function that processes X in a way such that
        X can both be a data object or an array of indices.

        :param X: Data object or array of indices
        :type X: Data object or array
        :return: array of indices
        :rtype: array
        """

        if isinstance(X, Data):
            self._set_data(X)
            # Part of the sklearn CV hack.
            if not hasattr(self.data, 'indices'):
                self.data.indices = np.arange(len(self.data))
        elif self.data and is_positive_integer_or_zero_array(X) \
                and max(X) <= self.data.natoms.size:
            # This forces a copy to be made, which is helpful when
            # using scikit-learn.
            self._set_data(self.data)
            self.data.indices = np.asarray(X, dtype=int).ravel()
        else:
            print("Expected X to be array of indices or Data object. Got %s" % str(X))
            raise SystemExit

        self._set_representation_type()

    def _set_data(self, data):
        if data and data.natoms is None:
            print("Error: Empty Data object passed to %s representation" % self.__class__.__name__)
            raise SystemExit
        # Shallow copy should be fine
        self.data = copy.copy(data)

    def _set_representation_type(self):
        self.data.representation_type = self._representation_type

    def _check_elements(self, nuclear_charges):
        """
        Check that the elements in the given nuclear_charges was
        included in the fit.
        """

        elements_transform = get_unique(nuclear_charges)
        if not np.isin(elements_transform, self.elements).all():
            print("Warning: Trying to transform molecules with elements",
                  "not included during fit in the %s method" % self.__class__.__name__)

class _MolecularRepresentation(_BaseRepresentation):
    """
    Base class for molecular representations
    """

    _representation_type = "molecular"

class _AtomicRepresentation(_BaseRepresentation):
    """
    Base class for molecular representations
    """

    _representation_type = "atomic"

class CoulombMatrix(_MolecularRepresentation):
    """
    Coulomb Matrix representation as described in 10.1103/PhysRevLett.108.058301

    """

    _representation_short_name = "cm"

    def __init__(self, data=None, size=23, sorting="row-norm"):
        """
        Coulomb Matrix representation of a molecule.
        Sorting of the elements can either be done by ``sorting="row-norm"`` or ``sorting="unsorted"``.
        A matrix :math:`M` is constructed with elements

        .. math::

            M_{ij} =
              \\begin{cases}
                 \\tfrac{1}{2} Z_{i}^{2.4} & \\text{if } i = j \\\\
                 \\frac{Z_{i}Z_{j}}{\\| {\\bf R}_{i} - {\\bf R}_{j}\\|}       & \\text{if } i \\neq j
              \\end{cases},

        where :math:`i` and :math:`j` are atom indices, :math:`Z` is nuclear charge and
        :math:`\\bf R` is the coordinate in euclidean space.
        If ``sorting = 'row-norm'``, the atom indices are reordered such that

            :math:`\\sum_j M_{1j}^2 \\geq \\sum_j M_{2j}^2 \\geq ... \\geq \\sum_j M_{nj}^2`

        The upper triangular of M, including the diagonal, is concatenated to a 1D
        vector representation.

        If ``sorting = 'unsorted``, the elements are sorted in the same order as the input coordinates
        and nuclear charges.

        The representation is calculated using an OpenMP parallel Fortran routine.

        :param size: The size of the largest molecule supported by the representation
        :type size: integer
        :param sorting: How the atom indices are sorted ('row-norm', 'unsorted')
        :type sorting: string
        :param data: Optional Data object containing all molecules used in training \
                and/or prediction
        :type data: Data object
        """

        self.size = size
        self.sorting = sorting
        self.data = data

    def fit(self, X, y=None):
        """
        :param X: Data object or indices to use from the \
                Data object passed at initialization
        :type X: Data object or array of indices
        :param y: Dummy argument for scikit-learn
        :type y: NoneType
        :return: self
        :rtype: object
        """
        self._preprocess_input(X)

        natoms = self.data.natoms[self.data.indices]

        if self.size < max(natoms):
            print("Warning: Maximum size of system increased from %d to %d"
                    % (self.size, max(natoms)))
            self.size = max(natoms)

        return self

    def transform(self, X):
        """
        :param X: Data object or indices to use from the \
                Data object passed at initialization
        :type X: Data object or array of indices
        :return: Data object
        :rtype: Data object
        """

        self._preprocess_input(X)

        nuclear_charges = self.data.nuclear_charges[self.data.indices]
        coordinates = self.data.coordinates[self.data.indices]
        natoms = self.data.natoms[self.data.indices]

        if max(natoms) > self.size:
            print("Error: Representation can't be generated for molecule of size \
                %d when variable 'size' is %d" % (max(natoms), self.size))
            raise SystemExit

        representations = []
        if (self.sorting == "row-norm"):
            for charge, xyz in zip(nuclear_charges, coordinates):
                representations.append(
                        fgenerate_coulomb_matrix(charge, xyz, self.size))

        elif (self.sorting == "unsorted"):
            for charge, xyz in zip(nuclear_charges, coordinates):
                representations.append(
                        fgenerate_unsorted_coulomb_matrix(charge, xyz, self.size))
        else:
            print("ERROR: Unknown sorting scheme requested")
            raise SystemExit

        self.data.representations = np.asarray(representations)

        return self.data

    # TODO Make it possible to pass data in other ways as well
    # e.g. dictionary
    def generate(self, X):
        """
        :param X: Data object or indices to use from the \
                Data object passed at initialization
        :type X: Data object or array of indices
        :return: Representations of shape (n_samples, representation_size)
        :rtype: array
        """

        return self.fit(X).transform(X).representations

# TODO add reference
# TODO add support for atomic properties
#        The representation can be calculated for a subset by either specifying
#        ``indices = [0,1,...]``, where :math:`[0,1,...]` are the requested atom indices,
#        or by specifying ``indices = 'C'`` to only calculate central carbon atoms.
class AtomicCoulombMatrix(_AtomicRepresentation):
    """
    Atomic Coulomb Matrix representation as described in 

    """

    _representation_short_name = "acm"

    def __init__(self, data=None, size=23, sorting="distance", central_cutoff=10.0,
            central_decay=-1, interaction_cutoff=10.0, interaction_decay=-1):
        """
        Creates a Coulomb Matrix representation of the local environment of a central atom.
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

        :param data: Optional Data object containing all molecules used in training \
                and/or prediction
        :type data: Data object
        :param size: The maximum number of atoms within the cutoff radius supported by the representation
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
        """

        self.data = data
        self.size = size
        self.sorting = sorting
        self.central_cutoff = central_cutoff
        self.central_decay = central_decay
        self.interaction_cutoff = interaction_cutoff
        self.interaction_decay = interaction_decay

    def fit(self, X, y=None):
        """
        :param X: Data object or indices to use from the \
                Data object passed at initialization
        :type X: Data object or array of indices
        :param y: Dummy argument for scikit-learn
        :type y: NoneType
        :return: self
        :rtype: object
        """
        self._preprocess_input(X)

        ## Giving indices doesn't make sense when predicting energies
        #if self.data.property_type == 'energies' and not is_none(self.indices):
        #    print("Error: Specifying variable 'indices' in representation %s \
        #            is not compatiple with 'property_type' specified in the \
        #            Data object" % self.__class__.__name__)

        return self

    def transform(self, X):
        """
        :param X: Data object or indices to use from the \
                Data object passed at initialization
        :type X: Data object or array of indices
        :return: Data object
        :rtype: Data object
        """

        self._preprocess_input(X)

        nuclear_charges = self.data.nuclear_charges[self.data.indices]
        coordinates = self.data.coordinates[self.data.indices]
        natoms = self.data.natoms[self.data.indices]

        representations = []
        if (self.sorting == "row-norm"):
            for charge, xyz, n in zip(nuclear_charges, coordinates, natoms):
                representations.append(fgenerate_local_coulomb_matrix(np.arange(n)+1, n, charge,
                    xyz, n, self.size, self.central_cutoff,
                    self.central_decay, self.interaction_cutoff, self.interaction_decay))

        elif (self.sorting == "distance"):
            for charge, xyz, n in zip(nuclear_charges, coordinates, natoms):
                representations.append(fgenerate_atomic_coulomb_matrix(np.arange(1,n+1), n, charge,
                    xyz, n, self.size, self.central_cutoff,
                    self.central_decay, self.interaction_cutoff, self.interaction_decay))

        else:
            print("ERROR: Unknown sorting scheme requested")
            raise SystemExit

        self.data.representations = representations

        return self.data

    # TODO Make it possible to pass data in other ways as well
    # e.g. dictionary
    def generate(self, X):
        """
        :param X: Data object or indices to use from the \
                Data object passed at initialization
        :type X: Data object or array of indices
        :return: Representations of shape (n_samples, representation_size)
        :rtype: array
        """

        return self.fit(X).transform(X).representations

class _SLATM(object):
    """
    Routines for global and local SLATM
    """

    def _fit(self, X, y=None):
        """
        :param X: Data object or indices to use from the \
                Data object passed at initialization
        :type X: Data object or array of indices
        :param y: Dummy argument for scikit-learn
        :type y: NoneType
        :return: self
        :rtype: object
        """
        self._preprocess_input(X)

        nuclear_charges = self.data.nuclear_charges[self.data.indices]

        self.elements = get_unique(nuclear_charges)
        self.mbtypes = get_slatm_mbtypes(nuclear_charges)

        return self

    def _transform(self, X, local):
        """
        :param X: Data object or indices to use from the \
                Data object passed at initialization
        :type X: Data object or array of indices
        :param local: Use local or global version
        :type local: bool
        :return: Data object
        :rtype: Data object
        """

        self._preprocess_input(X)

        nuclear_charges = self.data.nuclear_charges[self.data.indices]
        coordinates = self.data.coordinates[self.data.indices]
        natoms = self.data.natoms[self.data.indices]

        # Check that the molecules being transformed doesn't contain elements
        # not used in the fit.
        self._check_elements(nuclear_charges)

        representations = []
        for charge, xyz in zip(nuclear_charges, coordinates):
            representations.append(
                    np.asarray(
                        generate_slatm(xyz, charge, self.mbtypes, local=local,
                        sigmas=[self.sigma2, self.sigma3],
                        dgrids=[self.dgrid2, self.dgrid3], rcut=self.rcut,
                        alchemy=self.alchemy, rpower=-self.rpower)))

        self.data.representations = np.asarray(representations)

        return self.data

class GlobalSLATM(_SLATM, _MolecularRepresentation):
    """
    Global version of SLATM.
    """

    _representation_short_name = "slatm"

    def __init__(self, data=None, sigma2=0.05, sigma3=0.05, dgrid2=0.03,
            dgrid3=0.03, rcut=4.8, alchemy=False, rpower=-6):
        """
        Generate Spectrum of London and Axillrod-Teller-Muto potential (SLATM) representation.

        :param sigma2: Width of Gaussian smearing function for 2-body part.
        :type sigma2: float
        :param sigma3: Width of Gaussian smearing function for 3-body part.
        :type sigma3: float
        :param dgrid2: The interval between two sampled internuclear distances.
        :type dgrid2: float
        :param dgrid3: The interval between two sampled internuclear angles.
        :type dgrid3: float
        :param rcut: Cut-off radius.
        :type rcut: float
        :param alchemy: Use the alchemy version of SLATM.
        :type alchemy: bool
        :param rpower: The scaling power of R in 2-body potential.
        :type rpower: float
        :param data: Optional Data object containing all molecules used in training \
                and/or prediction
        :type data: Data object
        """

        self.data = data
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.dgrid2 = dgrid2
        self.dgrid3 = dgrid3
        self.rcut = rcut
        self.alchemy = alchemy
        self.rpower = rpower

        # Will be changed during fit
        self.elements = None
        self.mbtypes = None

    def fit(self, X, y=None):
        """
        :param X: Data object or indices to use from the \
                Data object passed at initialization
        :type X: Data object or array of indices
        :param y: Dummy argument for scikit-learn
        :type y: NoneType
        :return: self
        :rtype: object
        """

        return self._fit(X, y)

    def transform(self, X):
        """
        :param X: Data object or indices to use from the \
                Data object passed at initialization
        :type X: Data object or array of indices
        :return: Data object
        :rtype: Data object
        """

        return self._transform(X, False)

    # TODO Make it possible to pass data in other ways as well
    # e.g. dictionary
    def generate(self, X):
        """
        :param X: Data object or indices to use from the \
                Data object passed at initialization
        :type X: Data object or array of indices
        :return: Representations of shape (n_samples, representation_size)
        :rtype: array
        """

        return self.fit(X).transform(X).representations

class AtomicSLATM(_SLATM, _AtomicRepresentation):
    """
    Local version of SLATM.
    """

    _representation_short_name = "aslatm"

    def __init__(self, data=None, sigma2=0.05, sigma3=0.05, dgrid2=0.03,
            dgrid3=0.03, rcut=4.8, alchemy=False, rpower=-6):
        """
        Generate Spectrum of London and Axillrod-Teller-Muto potential (SLATM) representation.

        :param sigma2: Width of Gaussian smearing function for 2-body part.
        :type sigma2: float
        :param sigma3: Width of Gaussian smearing function for 3-body part.
        :type sigma3: float
        :param dgrid2: The interval between two sampled internuclear distances.
        :type dgrid2: float
        :param dgrid3: The interval between two sampled internuclear angles.
        :type dgrid3: float
        :param rcut: Cut-off radius.
        :type rcut: float
        :param alchemy: Use the alchemy version of SLATM.
        :type alchemy: bool
        :param rpower: The scaling power of R in 2-body potential.
        :type rpower: float
        :param data: Optional Data object containing all molecules used in training \
                and/or prediction
        :type data: Data object
        """

        self.data = data
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.dgrid2 = dgrid2
        self.dgrid3 = dgrid3
        self.rcut = rcut
        self.alchemy = alchemy
        self.rpower = rpower

        # Will be changed during fit
        self.elements = None
        self.mbtypes = None

    def fit(self, X, y=None):
        """
        :param X: Data object or indices to use from the \
                Data object passed at initialization
        :type X: Data object or array of indices
        :param y: Dummy argument for scikit-learn
        :type y: NoneType
        :return: self
        :rtype: object
        """

        return self._fit(X, y)

    def transform(self, X):
        """
        :param X: Data object or indices to use from the \
                Data object passed at initialization
        :type X: Data object or array of indices
        :return: Data object
        :rtype: Data object
        """

        return self._transform(X, True)

    # TODO Make it possible to pass data in other ways as well
    # e.g. dictionary
    def generate(self, X):
        """
        :param X: Data object or indices to use from the \
                Data object passed at initialization
        :type X: Data object or array of indices
        :return: Representations of shape (n_samples, representation_size)
        :rtype: array
        """

        return self.fit(X).transform(X).representations
