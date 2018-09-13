# MIT License
#
# Copyright (c) 2018 Lars Andersen Bratholm
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
from ..utils import is_positive_integer_or_zero_array, get_unique, get_pairs, is_string
from ..representations.frepresentations import fgenerate_coulomb_matrix
from ..representations.frepresentations import fgenerate_unsorted_coulomb_matrix
from ..representations.frepresentations import fgenerate_local_coulomb_matrix
from ..representations.frepresentations import fgenerate_atomic_coulomb_matrix
from ..representations.representations import get_slatm_mbtypes
from ..representations.representations import generate_slatm
from ..representations.facsf import fgenerate_acsf
from ..fchl.fchl_representations import generate_representation

class _BaseRepresentation(BaseEstimator):
    """
    Base class for representations
    """

    # Variables that has to be set in child methods
    _representation_short_name = None
    _representation_type = None
    alchemy = None


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

    def _extract_data(self, X):
        """
        Convenience function that processes X in a way such that
        X can both be a data object or an array of indices.

        :param X: Data object or array of indices
        :type X: Data object or array
        :return: Data object
        :rtype: Data object
        """

        if isinstance(X, Data):
            self._check_data(X)
            data = copy.copy(X)
            if not hasattr(data, "_indices"):
                data._indices = np.arange(len(data))

        elif self.data and is_positive_integer_or_zero_array(X) \
                and max(X) <= self.data.natoms.size:
            # A copy here might avoid some unintended behaviour
            # if multiple models is used sequentially.
            data = copy.copy(self.data)
            data._indices = np.asarray(X, dtype=int).ravel()
        else:
            print("Expected X to be array of indices or Data object. Got %s" % str(X))
            raise SystemExit

        # Store representation type / name for later use
        data._representation_type = self._representation_type
        data._representation_short_name = self._representation_short_name
        data._representation_alchemy = self.alchemy

        return data

    def _check_data(self, X):
        if X.natoms is None:
            print("Error: Empty Data object passed to %s" % self.__class__.__name__)
            raise SystemExit

    def _set_data(self, data):
        if data:
            self._check_data(data)

        self.data = data

    def _check_elements(self, nuclear_charges):
        """
        Check that the elements in the given nuclear_charges was
        included in the fit.
        """

        elements_transform = get_unique(nuclear_charges)
        if not np.isin(elements_transform, self.elements).all():
            print("Warning: Trying to transform molecules with elements",
                  "not included during fit in the %s method." % self.__class__.__name__,
                  "%s used in training but trying to transform %s" % (str(self.elements), str(element_transform)))

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

        return self.fit(X).transform(X)._representations

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
    alchemy = None

class CoulombMatrix(_MolecularRepresentation):
    """
    Coulomb Matrix representation as described in 10.1103/PhysRevLett.108.058301

    """

    _representation_short_name = "cm"

    def __init__(self, data=None, size='auto', sorting="row-norm"):
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

        :param size: The size of the largest molecule supported by the representation.
                     `size='auto'` will try to determine this automatically.
        :type size: integer
        :param sorting: How the atom indices are sorted ('row-norm', 'unsorted')
        :type sorting: string
        :param data: Optional Data object containing all molecules used in training \
                and/or prediction
        :type data: Data object
        """

        self.size = size
        self.sorting = sorting
        self._set_data(data)

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

        data = self._extract_data(X)

        natoms = data.natoms#[data._indices]

        if self.size == 'auto':
            self.size = max(natoms)

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

        data = self._extract_data(X)

        nuclear_charges = data.nuclear_charges[data._indices]
        coordinates = data.coordinates[data._indices]
        natoms = data.natoms[data._indices]

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

        data._representations = np.asarray(representations)

        return data


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
    alchemy = True

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
        :param size: The maximum number of atoms within the cutoff radius supported by the representation.
                     `size='auto'` will try to determine this automatically.
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
        :param interaction_decay: The distance over which the coulomb interaction decays from full to none
        :type interaction_decay: float
        """

        self._set_data(data)
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
        data = self._extract_data(X)

        natoms = data.natoms#[data._indices]

        if self.size == 'auto':
            self.size = min(max(natoms), 2 * self.central_cutoff**3)

        return self

    def transform(self, X):
        """
        :param X: Data object or indices to use from the \
                Data object passed at initialization
        :type X: Data object or array of indices
        :return: Data object
        :rtype: Data object
        """

        data = self._extract_data(X)

        nuclear_charges = data.nuclear_charges[data._indices]
        coordinates = data.coordinates[data._indices]
        natoms = data.natoms[data._indices]

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

        data._representations = representations

        return data


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
        data = self._extract_data(X)

        if is_string(self.elements) and self.elements == 'auto' \
                and is_string(self.element_pairs) and self.element_pairs == 'auto':
            nuclear_charges = data.nuclear_charges#[data._indices]
            self.elements = get_unique(nuclear_charges)
            self.element_pairs = get_slatm_mbtypes(nuclear_charges)
        elif not is_string(self.elements) and is_string(self.element_pairs) \
                and self.element_pairs == 'auto':
            self.element_pairs = get_slatm_mbtypes(self.elements * 3)
        elif is_string(self.elements) and self.elements == 'auto' and \
                not is_string(self.element_pairs):
            self.elements = get_unique(self.element_pairs)

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

        data = self._extract_data(X)

        nuclear_charges = data.nuclear_charges[data._indices]
        coordinates = data.coordinates[data._indices]
        natoms = data.natoms[data._indices]

        # Check that the molecules being transformed doesn't contain elements
        # not used in the fit.
        self._check_elements(nuclear_charges)

        representations = []
        for charge, xyz in zip(nuclear_charges, coordinates):
            representations.append(
                    np.asarray(
                        generate_slatm(xyz, charge, self.element_pairs, local=local,
                        sigmas=[self.sigma2, self.sigma3],
                        dgrids=[self.dgrid2, self.dgrid3], rcut=self.rcut,
                        alchemy=self.alchemy, rpower=-self.rpower)))

        data._representations = np.asarray(representations)

        return data

# TODO add reference
class GlobalSLATM(_SLATM, _MolecularRepresentation):
    """
    Global version of SLATM.
    """

    _representation_short_name = "slatm"

    def __init__(self, data=None, sigma2=0.05, sigma3=0.05, dgrid2=0.03,
            dgrid3=0.03, rcut=4.8, alchemy=False, rpower=-6, elements='auto',
            element_pairs='auto'):
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
        :param elements: Atomnumber of elements that the representation should support.
                         `elements='auto'` will try to determine this automatically.
        :type elements: list
        :param element_pairs: Atomnumbers of element pairs that the representation should support.
                         `element_pairs='auto'` will try to determine this automatically.
        :type element_pairs: list
        :param data: Optional Data object containing all molecules used in training \
                and/or prediction
        :type data: Data object
        """

        self._set_data(data)
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.dgrid2 = dgrid2
        self.dgrid3 = dgrid3
        self.rcut = rcut
        self.alchemy = alchemy
        self.rpower = rpower
        # Will be changed during fit
        self.elements = elements
        self.element_pairs = element_pairs

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


# TODO add reference
class AtomicSLATM(_SLATM, _AtomicRepresentation):
    """
    Local version of SLATM.
    """

    _representation_short_name = "aslatm"

    def __init__(self, data=None, sigma2=0.05, sigma3=0.05, dgrid2=0.03,
            dgrid3=0.03, rcut=4.8, alchemy=False, rpower=-6, elements='auto',
            element_pairs='auto'):
        """
        Generate Spectrum of London and Axillrod-Teller-Muto potential (SLATM) representation.

        :param data: Optional Data object containing all molecules used in training \
                and/or prediction
        :type data: Data object
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
        :param elements: Atomnumber of elements that the representation should support.
                         `elements='auto'` will try to determine this automatically.
        :type elements: list
        :param element_pairs: Atomnumbers of element pairs that the representation should support.
                         `element_pairs='auto'` will try to determine this automatically.
        :type element_pairs: list
        """

        self._set_data(data)
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.dgrid2 = dgrid2
        self.dgrid3 = dgrid3
        self.rcut = rcut
        self.alchemy = alchemy
        self.rpower = rpower
        # Will be changed during fit
        self.elements = elements
        self.element_pairs = element_pairs

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

class AtomCenteredSymmetryFunctions(_AtomicRepresentation):
    """
    The variant of Atom-Centered Symmetry Functions used in 10.1039/C7SC04934J
    """

    _representation_short_name = "acsf"
    alchemy = False

    def __init__(self, data=None, nbasis=3, precision=2, cutoff=5.0, elements='auto'):
        """
        :param data: Optional Data object containing all molecules used in training \
                and/or prediction
        :type data: Data object
        :param nbasis: Number of basis functions to use
        :type nbasis: integer
        :param cutoff: Cutoff radius
        :type cutoff: float
        :param precision: Precision in the basis functions. A value of 2 corresponds to the \
                        basis functions intersecting at half maximum. Higher values makes \
                        the basis functions narrower.
        :type precision: float
        :param elements: Atomnumber of elements that the representation should support.
                         `elements='auto'` will try to determine this automatically.
        :type elements: list
        """

        self._set_data(data)
        self.nbasis = nbasis
        self.precision = precision
        self.cutoff = cutoff
        # Will be changed during fit
        self.elements = elements

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
        data = self._extract_data(X)

        if is_string(self.elements) and self.elements == 'auto':
            nuclear_charges = data.nuclear_charges#[data._indices]
            self.elements = get_unique(nuclear_charges)

        return self

    def transform(self, X):
        """
        :param X: Data object or indices to use from the \
                Data object passed at initialization
        :type X: Data object or array of indices
        :return: Data object
        :rtype: Data object
        """

        data = self._extract_data(X)

        nuclear_charges = data.nuclear_charges[data._indices]
        coordinates = data.coordinates[data._indices]
        natoms = data.natoms[data._indices]

        # Check that the molecules being transformed doesn't contain elements
        # not used in the fit.
        self._check_elements(nuclear_charges)

        # Calculate parameters needed for fortran input.
        # This is a heuristic that cuts down the number of hyper-parameters
        # and might be subject to future change.
        min_distance = 0.8
        Rs = np.linspace(min_distance, self.cutoff, self.nbasis)
        eta = 4 * np.log(self.precision) * ((self.nbasis-1)/(self.cutoff-min_distance))**2
        Ts = np.linspace(0, np.pi, self.nbasis)
        zeta = - np.log(self.precision) / np.log(np.cos(np.pi / (4 * (self.nbasis - 1)))**2)
        n_elements = len(self.elements)
        size = n_elements * self.nbasis + (n_elements * (n_elements + 1)) // 2 * self.nbasis ** 2

        representations = []
        for charge, xyz, n in zip(nuclear_charges, coordinates, natoms):
            representations.append(
                        fgenerate_acsf(xyz, charge, self.elements, Rs, Rs, Ts,
                            eta, eta, zeta, self.cutoff, self.cutoff, n, size))

        data._representations = np.asarray(representations)

        return data

class FCHLRepresentation(_BaseRepresentation):
    """
    The representation from 10.1063/1.5020710
    """

    _representation_short_name = "fchl"

    def __init__(self, data=None, size='auto', cutoff=10.0):
        """
        :param data: Optional Data object containing all molecules used in training \
                and/or prediction
        :type data: Data object
        :param size: Max number of atoms in representation. `max_size='auto'` Will try to determine this
                         automatically.
        :type size: integer
        :param cutoff: Spatial cut-off distance - must be the same as used in the kernel function call.
        :type cutoff: float
        """

        self._set_data(data)
        self.size = size
        self.cutoff = cutoff

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
        data = self._extract_data(X)

        natoms = data.natoms#[data._indices]

        if self.size == 'auto':
            self.size = max(natoms)

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

        data = self._extract_data(X)

        # Store cutoff to make sure that the kernel cutoff is less
        data._representation_cutoff = self.cutoff

        nuclear_charges = data.nuclear_charges[data._indices]
        coordinates = data.coordinates[data._indices]
        natoms = data.natoms[data._indices]

        if max(natoms) > self.size:
            print("Error: Representation can't be generated for molecule of size \
                %d when variable 'size' is %d" % (max(natoms), self.size))
            raise SystemExit

        representations = []
        for charge, xyz in zip(nuclear_charges, coordinates):
            representations.append(
                    generate_representation(xyz, charge, max_size=self.size, cut_distance=self.cutoff))

        data._representations = np.asarray(representations)

        return data
