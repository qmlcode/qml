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

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from .data import Data

from ..utils import is_numeric_array, get_unique

class AtomScaler(BaseEstimator):
    """
    Subtracts any constant offset or linear dependency on the number of atoms of each element from the property
    """

    def __init__(self, data=None, elements='auto', normalize=True):
        """
        :param data: Data object (optional)
        :type data: Data object
        :param elements: Elements to support. If `elements='auto'` try to determine this automatically.
        :type elements: array
        :param normalize: Normalize the transformed data such that the standard deviation is 1
        :type normalize: bool
        """
        # Shallow copy should be fine
        self._set_data(data, make_copy=False)
        self.elements = elements
        self.normalize = normalize

        # Initialize model
        self.model = LinearRegression()

        # Constant unless normalize==True
        self.std = 1

    def _preprocess_input(self, X, y):
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
            if X.energies is None:
                print("Error: Expected Data object to have non-empty attribute 'energies'" % self.__class__.__name__)
                raise SystemExit

            # NOTE: If memory of the data object is ever an issue,
            # not making a shallow copy here should be fine
            self._set_data(X, make_copy=True)
            # Part of the sklearn CV hack.
            if not hasattr(self.data, 'indices'):
                self.data.indices = np.arange(len(self.data))

            return X.nuclear_charges, X.energies

        elif self.data and is_positive_integer_or_zero_array(X) \
                and max(X) <= self.data.natoms.size:
            # A copy here might avoid some unintended behaviour
            # if multiple models is used sequentially.
            self._set_data(self.data, make_copy = True)
            self.data.indices = np.asarray(X, dtype=int).ravel()

            if X.energies is None:
                print("Error: Expected Data object to have non-empty attribute 'energies'" % self.__class__.__name__)
                raise SystemExit

            return self.data.nuclear_charges[self.data.indices], self.data.energies[self.data.indices]

        elif len(X) > 0 and is_numeric_array(y):
            if is_numeric_array(X[0]):
                return X, y
            else:
                print("Error: Expected numeric array. Got %s" % str(X))
                raise SystemExit
        else:
            print("Expected X to be array of indices or Data object. Got %s" % str(X))
            raise SystemExit

    def _postprocess_output(self, y):
        """
        """

        if self.normalize:
            self.std = np.std(y)

        y /= self.std


    def _set_data(self, data, make_copy=False):
        if data and data.natoms is None:
            print("Error: Empty Data object passed to the %s transformer" % self.__class__.__name__)
            raise SystemExit
        if make_copy:
            # Shallow copy should be fine
            self.data = copy.copy(data)
        else:
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

        nuclear_charges, y = self._preprocess_input(X, y)

        if self.elements == 'auto':
            self.elements = get_unique(nuclear_charges)
        else:
            self_check_elements(nuclear_charges)



        X = self._featurizer(nuclear_charges)

        delta_y = y - self.model.fit(X, y).predict(X)

        output = self._postprocess_output(delta_y)

        return output

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
                j = element_to_index[key]
                features[i, j] = value

        return features

    def transform(self, X, y):
        X = self._featurizer(X)
        return y - self.model.predict(X)
