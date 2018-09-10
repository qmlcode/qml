# MIT License
#
# Copyright (c) 2018 Lars A. Bratholm
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

from __future__ import division, absolute_import, print_function

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error

from ..utils import is_numeric_array
from .data import Data
from ..math import cho_solve

class _BaseModel(BaseEstimator):
    """
    Base class for all regression models
    """

    _estimator_type = "regressor"

    def fit(self, X):
        raise NotImplementedError

    def predict(self, X):
        return NotImplementedError

    def score(self, X, y=None):
        """
        Make predictions on `X` and return a score

        :param X: Data object
        :type X: object
        :param y: Energies
        :type y: array
        :return: score
        :rtype: float
        """

        # Make predictions
        y_pred = self.predict(X)

        # Get the true values
        if is_numeric_array(y):
            pass

        elif isinstance(X, Data):
            try:
                y = X.energies[X._indices]
            except:
                print("No kernel energies found in data object in module %s" % self.__class__.__name__)
                raise SystemExit

        else:
            print("Expected variable 'X' to be Data object. Got %s" % str(X))
            raise SystemExit

        # Return the score
        if self.scoring == 'mae':
            return mean_absolute_error(y, y_pred)
        elif self.scoring == 'neg_mae':
            return - mean_absolute_error(y, y_pred)
        elif self.scoring == 'rmsd':
            return np.sqrt(mean_squared_error(y, y_pred))
        elif self.scoring == 'neg_rmsd':
            return - np.sqrt(mean_squared_error(y, y_pred))
        elif self.scoring == 'neg_log_mae':
            return - np.log(mean_absolute_error(y, y_pred))

class KernelRidgeRegression(_BaseModel):
    """
    Standard Kernel Ridge Regression using a cholesky solver
    """

    def __init__(self, l2_reg=1e-10, scoring='neg_mae'):
        """
        :param llambda: l2 regularization
        :type llambda: float
        :param scoring: Metric used for scoring ('mae', 'neg_mae', 'rmsd', 'neg_rmsd', 'neg_log_mae')
        :type scoring: string
        """
        self.l2_reg = l2_reg
        self.scoring = scoring

        self.alpha = None

    def fit(self, X, y=None):
        """
        Fit the Kernel Ridge Regression model using a cholesky solver

        :param X: Data object or kernel
        :type X: object or array
        :param y: Energies
        :type y: array
        """

        if isinstance(X, Data):
            try:
                K, y = X._kernel, X.energies[X._indices]
            except:
                print("No kernel matrix and/or energies found in data object in module %s" % self.__class__.__name__)
                raise SystemExit
        elif is_numeric_array(X) and X.ndim == 2 and X.shape[0] == X.shape[1] and y is not None:
            K = X
        else:
            print("Expected variable 'X' to be kernel matrix or Data object. Got %s" % str(X))
            raise SystemExit


        K[np.diag_indices_from(K)] += self.l2_reg

        self.alpha = cho_solve(K, y)

    def predict(self, X):
        """
        Fit the Kernel Ridge Regression model using a cholesky solver

        :param X: Data object
        :type X: object
        :param y: Energies
        :type y: array
        """

        # Check if model has been fit
        if self.alpha is None:
            print("Error: The %s model has not been trained yet" % self.__class__.__name__)
            raise SystemExit

        if isinstance(X, Data):
            try:
                K = X._kernel
            except:
                print("No kernel matrix found in data object in module %s" % self.__class__.__name__)
                raise SystemExit
        elif is_numeric_array(X) and X.ndim == 2 and X.shape[1] == self.alpha.size:
            K = X
        elif is_numeric_array(X) and X.ndim == 2 and X.shape[0] == self.alpha.size:
            K = X.T
        else:
            print("Expected variable 'X' to be kernel matrix or Data object. Got %s" % str(X))
            raise SystemExit

        return np.dot(K, self.alpha)
