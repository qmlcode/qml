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

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from .data import Data
from ..utils import is_numeric_array, get_unique, is_positive_integer_or_zero_array

class AtomScaler(BaseEstimator):
    """
    Subtracts any constant offset or linear dependency on the number of atoms of each element from the property
    """

    def __init__(self, data=None, elements='auto'):
        """
        :param data: Data object (optional)
        :type data: Data object
        :param elements: Elements to support. If `elements='auto'` try to determine this automatically.
        :type elements: array
        :param normalize: Normalize the transformed data such that the standard deviation is 1
        :type normalize: bool
        """
        # Shallow copy should be fine
        self._set_data(data)
        self.elements = elements

        # Initialize model
        self.model = LinearRegression()

    def _preprocess_input(self, X):
        """
        Convenience function that processes X in a way such that
        X can both be a data object, or an array of indices. And y
        can be either values to transform or None.

        :param X: Data object, floating values or integer array of indices
        :type X: Data object or array
        :param y: Values or None
        :type y: array or None
        :return: Nuclear charges and values to transform
        :rtype: tuple
        """

        if isinstance(X, Data):

            self._check_data(X)

            data = copy.copy(X)

            # Part of the sklearn CV hack.
            if not hasattr(data, '_indices'):
                data._indices = np.arange(len(data))

            if hasattr(data, '_has_transformed_labels'):
                print("Error: Target data has already been transformed by %s" % self.__class__.__name__)
                raise SystemExit

            transformed_labels = np.zeros(len(data), dtype=bool)
            transformed_labels[data._indices] = True
            data._has_transformed_labels = transformed_labels

        elif self.data and is_positive_integer_or_zero_array(X) \
                and max(X) <= self.data.natoms.size:
            # A copy here might avoid some unintended behaviour
            # if multiple models is used sequentially.
            data = copy.copy(self.data)
            data._indices = np.asarray(X, dtype=int).ravel()


            if hasattr(data, '_has_transformed_labels'):
                if any(data._has_transformed_labels[data._indices] == True):
                    print("Error: Target data has already been transformed by %s" % self.__class__.__name__)
                    raise SystemExit
                data._has_transformed_labels[data._indices] = True
            else:
                transformed_labels = np.zeros(len(data.energies), dtype=bool)
                transformed_labels[data._indices] = True
                data._has_transformed_labels = transformed_labels

        else:
            print("Expected X to be array of indices or Data object. Got %s" % str(X))
            raise SystemExit

        return data

    def _check_data(self, X):
        if X.natoms is None:
            print("Error: Empty Data object passed to the %s transformer" % self.__class__.__name__)
            raise SystemExit

        if X.energies is None:
            print("Error: Expected Data object to have non-empty attribute 'energies'" % self.__class__.__name__)
            raise SystemExit


    def _set_data(self, data):
        if data:
            self._check_data(data)
        self.data = data

    def fit_transform(self, X, y=None):
        """
        Fit and transform the data with a linear model.
        Supports three different types of input.
        1) X is a list of nuclear charges and y is values to transform.
        2) X is an array of indices of which to transform.
        3) X is a data object

        :param X: List with nuclear charges or Data object.
        :type X: list
        :param y: Values to transform
        :type y: array or None
        :return: Array of transformed values or Data object, depending on input
        :rtype: array or Data object
        """

        if not isinstance(X, Data) and y is not None:
            data = None
            nuclear_charges = X
        else:
            data = self._preprocess_input(X)
            nuclear_charges = data.nuclear_charges[data._indices]
            y = data.energies[data._indices]

        if self.elements == 'auto':
            self.elements = get_unique(nuclear_charges)
        else:
            self._check_elements(nuclear_charges)


        features = self._featurizer(nuclear_charges)

        delta_y = y - self.model.fit(features, y).predict(features)

        if data:
            # Force copy
            data.energies = data.energies.copy()
            data.energies[data._indices] = delta_y
            return data
        else:
            return delta_y

    def _check_elements(self, nuclear_charges):
        """
        Check that the elements in the given nuclear_charges was
        included in the fit.
        """

        elements_transform = get_unique(nuclear_charges)
        if not np.isin(elements_transform, self.elements).all():
            print("Warning: Trying to transform molecules with elements",
                  "not included during fit in the %s method." % self.__class__.__name__,
                  "%s used in training but trying to transform %s" % (str(self.elements), str(elements_transform)))

    def _featurizer(self, X):
        """
        Get the counts of each element as features.
        """

        n = len(X)
        m = len(self.elements)
        element_to_index = {v:i for i, v in enumerate(self.elements)}
        features = np.zeros((n,m), dtype=int)

        for i, x in enumerate(X):
            count_dict = {k:v for k,v in zip(*np.unique(x, return_counts=True))}
            for key, value in count_dict.items():
                if key not in element_to_index:
                    continue
                j = element_to_index[key]
                features[i, j] = value

        return features

    def transform(self, X, y=None):
        """
        Transform the data with the fitted linear model.
        Supports three different types of input.
        1) X is a list of nuclear charges and y is values to transform.
        2) X is an array of indices of which to transform.
        3) X is a data object

        :param X: List with nuclear charges or Data object.
        :type X: list
        :param y: Values to transform
        :type y: array or None
        :return: Array of transformed values or Data object, depending on input
        :rtype: array or Data object
        """

        if not isinstance(X, Data) and y is not None:
            data = None
            nuclear_charges = X
        else:
            data = self._preprocess_input(X)
            nuclear_charges = data.nuclear_charges[data._indices]
            y = data.energies[data._indices]

        self._check_elements(nuclear_charges)

        features = self._featurizer(nuclear_charges)

        delta_y = y - self.model.predict(features)

        if data:
            # Force copy
            data.energies = data.energies.copy()
            data.energies[data._indices] = delta_y
            return data
        else:
            return delta_y

