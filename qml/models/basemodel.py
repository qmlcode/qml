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

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error

from ..utils import is_numeric_array
from ..data import Data

class _BaseModel(BaseEstimator):
    """
    Base class for all regression models
    """

    _estimator_type = "regressor"

    def fit(self, X):
        raise NotImplementedError

    def predict(self, X):
        return NotImplementedError

    def score(self, X, y=None, multioutput = False):
        """
        Make predictions on `X` and return a score

        :param X: Data object
        :type X: object
        :param y: Energies
        :type y: array
        :param multioutput: Return the score for each sample or an averaged score.
        :type multioutput: bool
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
                y = X.energies[X.indices]
            except:
                print("No kernel energies found in data object in module %s" % self.__class__.__name__)
                raise SystemExit

        else:
            print("Expected variable 'X' to be Data object. Got %s" % str(X))
            raise SystemExit

        # Translate bool to string for sklearn
        if multioutput:
            multioutput = 'uniform_average'
        else:
            multioutput = 'raw_values'

        # Return the score
        if self.scoring == 'mae':
            return mean_absolute_error(y, y_pred, multioutput=multioutput)
        elif self.scoring == 'neg_mae':
            return - mean_absolute_error(y, y_pred, multioutput=multioutput)
        elif self.scoring == 'rmsd':
            return np.sqrt(mean_squared_error(y, y_pred, multioutput=multioutput))
        elif self.scoring == 'neg_rmsd':
            return - np.sqrt(mean_squared_error(y, y_pred, multioutput=multioutput))
        elif self.scoring == 'neg_log_mae':
            return - np.log(mean_absolute_error(y, y_pred, multioutput=multioutput))


