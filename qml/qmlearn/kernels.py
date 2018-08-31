# MIT License
#
# Copyright (c) 2016 Anders Steen Christensen, Felix A. Faber, Lars A. Bratholm
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

from .data import Data

from ..ml.kernels.fkernels import fgaussian_kernel
from ..ml.kernels.fkernels import fgaussian_kernel_symmetric
from ..ml.kernels.fkernels import fget_vector_kernels_gaussian
from ..ml.kernels.fkernels import fget_vector_kernels_gaussian_symmetric
from ..ml.kernels.fkernels import flaplacian_kernel
from ..ml.kernels.fkernels import flaplacian_kernel_symmetric
from ..ml.kernels.fkernels import fget_vector_kernels_laplacian
from ..ml.kernels.fkernels import fget_vector_kernels_laplacian_symmetric

class _BaseKernel(BaseEstimator):
    """
    Base class for kernels
    """

    def fit(self, X):
        """
        The fit routine is needed for scikit-learn pipelines.
        This is actually never used but has to be here.
        """
        raise NotImplementedError

    def transform(self, X, y = None):
        """
        The transform routine is needed for scikit-learn pipelines.
        """
        raise NotImplementedError

    def generate(self, **params):
        raise NotImplementedError

    def _check_data_object(self, X):
        # Check that we have a data object
        if not isinstance(X, Data):
            print("Error: Expected Data object as input in %s" % self.__class__.__name__)
            raise SystemExit

    def _set_representations(self, rep):
        self.representations = rep

    def _transform(self, X):

        self._check_data_object(X)

        # Kernel between representation stored in fit and representation
        # given in data object. The order matters
        kernel = self.generate(X.representations, self.representations, X.representation_type)

        X.kernel = kernel

        return X

    def _fit_transform(self, X, y=None):

        self._check_data_object(X)

        # Store representation for future transform calls
        self._set_representations(X.representations)

        kernel = self.generate(X.representations, representation_type=X.representation_type)

        X.kernel = kernel

        return X


class GaussianKernel(_BaseKernel):
    """
    Gaussian kernel
    """

    def __init__(self, sigma='auto', alchemy=True):
        """
        :param sigma: Scale parameter of the gaussian kernel. `sigma='auto'` will try to guess
                      a good value (very approximate).
        :type sigma: float
        :param alchemy: Determines if contributions from atoms of differing elements should be
                        included in the kernel.
        :type alchemy: bool
        """
        self.sigma = sigma
        self.alchemy = alchemy

        self.representations = None

    def transform(self, X):
        return self._transform(X)

    def fit_transform(self, X, y=None):
        return self._fit_transform(X)

    def _quick_estimate_sigma(self, X, representation_type, sigma_init=100, count=1):
        """
        Use 50 random points for atomic, 200 for molecular, to get an approximate guess for sigma
        """

        if count > 10:
            print("Error. Could not automatically determine parameter `sigma` in the kernel %s"
                    % self.__class__.__name__)
            raise SystemExit

        if representation_type == 'molecular':
            n = 200
        else:
            n = 50

        if len(X) < n:
            n = len(X)

        self.sigma = sigma_init

        # Generate kernel for a random subset
        indices = np.random.choice(np.arange(len(X)), size=n, replace=False)
        kernel = self.generate([X[i] for i in indices], representation_type=representation_type)

        if representation_type == 'molecular':
            # min smallest kernel element
            smallest_kernel_element = np.min(kernel)

            # Rescale sigma such that smallest kernel element will be equal to 1/2
            self.sigma = - np.log(smallest_kernel_element) / np.log(2)
            # Update sigma if the given sigma was completely wrong
            if np.isinf(self.sigma):
                self._quick_estimate_sigma(X, representation_type, sigma_init * 10, count + 1)
            elif self.sigma <= 1e-12:
                self._quick_estimate_sigma(X, representation_type, sigma_init / 10, count + 1)
        else:
            sizes = np.asarray([len(X[i]) for i in indices])
            kernel /= sizes[:,None] * sizes[None,:]
            smallest_kernel_element = np.min(kernel)

            if smallest_kernel_element < 0.5:
                self._quick_estimate_sigma(X, representation_type, sigma_init * 2.5, count + 1)
            elif smallest_kernel_element > 1:
                self._quick_estimate_sigma(X, representation_type, sigma_init / 2.5, count + 1)



    def generate(self, X, Y=None, representation_type='molecular'):
        """
        Create a gaussian kernel from representations `X`. Optionally
        an asymmetric kernel can be constructed between representations
        `X` and `Y`.
        If `representation_type=='molecular` it is assumed that the representations
        are molecular and of shape (n_samples, representation_size).
        If `representation_type=='atomic` the representations are assumed to be atomic
        and the kernel will be computed as an atomic decomposition.

        :param X: representations
        :type X: array
        :param Y: (Optional) representations
        :type Y: array

        :return: Gaussian kernel matrix of shape (n_samplesX, n_samplesX) if \
                 Y=None else (n_samplesX, n_samplesY)
        :rtype: array
        """

        if self.sigma == 'auto':
            if representation_type == 'molecular':
                auto_sigma = True
            else:
                auto_sigma = False

            # Do a quick and dirty initial estimate of sigma
            self._quick_estimate_sigma(X, representation_type)
        else:
            auto_sigma = False


        if representation_type == 'molecular':
            kernel = self._generate_molecular(X,Y)
        elif representation_type == 'atomic':
            kernel = self._generate_atomic(X,Y)
        else:
            # Should never get here for users
            print("Error: representation_type needs to be 'molecular' or 'atomic'. Got %s" % representation_type)
            raise SystemExit

        if auto_sigma:
            # Find smallest kernel element
            smallest_kernel_element = np.min(kernel)
            largest_kernel_element = np.max(kernel)
            # Rescale kernel such that we don't have to calculate it again for a new sigma
            alpha = - np.log(2) / np.log(smallest_kernel_element/largest_kernel_element)
            self.sigma /= np.sqrt(alpha)
            return kernel ** alpha

        return kernel

    def _generate_molecular(self, X, Y=None):

        X = np.asarray(X)

        # Note: Transposed for Fortran
        n = X.shape[0]

        if Y is None:
            # Do symmetric matrix
            K = np.empty((n, n), order='F')
            fgaussian_kernel_symmetric(X.T, n, K, self.sigma)
        else:
            Y = np.asarray(Y)
            # Do asymmetric matrix
            m = Y.shape[0]
            K = np.empty((n, m), order='F')
            fgaussian_kernel(X.T, n, Y.T, m, K, self.sigma)

        return K

    def _generate_atomic(self, X, Y=None):

        n1 = np.array([len(x) for x in X], dtype=np.int32)

        max1 = np.max(n1)

        nm1 = n1.size

        rep_size = X[0].shape[1]

        x1 = np.zeros((nm1, max1, rep_size), dtype=np.float64, order="F")

        for i in range(nm1):
            x1[i,:n1[i]] = X[i]


        # Reorder for Fortran speed
        x1 = np.swapaxes(x1, 0, 2)

        sigmas = np.array([self.sigma], dtype=np.float64)
        nsigmas = sigmas.size

        if Y is None:
            # Do symmetric matrix
            return fget_vector_kernels_gaussian_symmetric(x1, n1, [self.sigma],
                nm1, nsigmas)[0]
        else:
            # Do asymmetric matrix
            n2 = np.array([len(y) for y in Y], dtype=np.int32)
            max2 = np.max(n2)
            nm2 = n2.size
            x2 = np.zeros((nm2, max2, rep_size), dtype=np.float64, order="F")
            for i in range(nm2):
                x2[i,:n2[i]] = Y[i]
            x2 = np.swapaxes(x2, 0, 2)
            return fget_vector_kernels_gaussian(x1, x2, n1, n2, [self.sigma],
                nm1, nm2, nsigmas)[0]

class LaplacianKernel(_BaseKernel):
    """
    Laplacian kernel
    """

    def __init__(self, sigma='auto', alchemy=True):
        """
        :param sigma: Scale parameter of the gaussian kernel. `sigma='auto'` will try to guess
                      a good value (very approximate).
        :type sigma: float or string
        :param alchemy: Determines if contributions from atoms of differing elements should be
                        included in the kernel.
        :type alchemy: bool
        """
        self.sigma = sigma
        self.alchemy = alchemy

        self.representations = None

    def transform(self, X):
        return self._transform(X)

    def fit_transform(self, X, y=None):
        return self._fit_transform(X)

    def _quick_estimate_sigma(self, X, representation_type, sigma_init=100, count=1):
        """
        Use 50 random points for atomic, 200 for molecular, to get an approximate guess for sigma
        """

        if count > 10:
            print("Error. Could not automatically determine parameter `sigma` in the kernel %s"
                    % self.__class__.__name__)
            raise SystemExit

        if representation_type == 'molecular':
            n = 200
        else:
            n = 50

        if len(X) < n:
            n = len(X)

        self.sigma = sigma_init

        # Generate kernel for a random subset
        indices = np.random.choice(np.arange(len(X)), size=n, replace=False)
        kernel = self.generate([X[i] for i in indices], representation_type=representation_type)

        if representation_type == 'molecular':
            # min smallest kernel element
            smallest_kernel_element = np.min(kernel)

            # Rescale sigma such that smallest kernel element will be equal to 1/2
            self.sigma = - np.log(smallest_kernel_element) / np.log(2)
            # Update sigma if the given sigma was completely wrong
            if np.isinf(self.sigma):
                self._quick_estimate_sigma(X, representation_type, sigma_init * 10, count + 1)
            elif self.sigma <= 1e-12:
                self._quick_estimate_sigma(X, representation_type, sigma_init / 10, count + 1)
        else:
            sizes = np.asarray([len(X[i]) for i in indices])
            kernel /= sizes[:,None] * sizes[None,:]
            smallest_kernel_element = np.min(kernel)

            if smallest_kernel_element < 0.5:
                self._quick_estimate_sigma(X, representation_type, sigma_init * 2.5, count + 1)
            elif smallest_kernel_element > 1:
                self._quick_estimate_sigma(X, representation_type, sigma_init / 2.5, count + 1)



    def generate(self, X, Y=None, representation_type='molecular'):
        """
        Create a gaussian kernel from representations `X`. Optionally
        an asymmetric kernel can be constructed between representations
        `X` and `Y`.
        If `representation_type=='molecular` it is assumed that the representations
        are molecular and of shape (n_samples, representation_size).
        If `representation_type=='atomic` the representations are assumed to be atomic
        and the kernel will be computed as an atomic decomposition.

        :param X: representations
        :type X: array
        :param Y: (Optional) representations
        :type Y: array

        :return: Gaussian kernel matrix of shape (n_samplesX, n_samplesX) if \
                 Y=None else (n_samplesX, n_samplesY)
        :rtype: array
        """

        if self.sigma == 'auto':
            if representation_type == 'molecular':
                auto_sigma = True
            else:
                auto_sigma = False

            # Do a quick and dirty initial estimate of sigma
            self._quick_estimate_sigma(X, representation_type)
        else:
            auto_sigma = False


        if representation_type == 'molecular':
            kernel = self._generate_molecular(X,Y)
        elif representation_type == 'atomic':
            kernel = self._generate_atomic(X,Y)
        else:
            # Should never get here for users
            print("Error: representation_type needs to be 'molecular' or 'atomic'. Got %s" % representation_type)
            raise SystemExit

        if auto_sigma:
            # Find smallest kernel element
            smallest_kernel_element = np.min(kernel)
            largest_kernel_element = np.max(kernel)
            # Rescale kernel such that we don't have to calculate it again for a new sigma
            alpha = - np.log(2) / np.log(smallest_kernel_element/largest_kernel_element)
            self.sigma /= alpha
            return kernel ** alpha

        return kernel

    def _generate_molecular(self, X, Y=None):

        X = np.asarray(X)

        # Note: Transposed for Fortran
        n = X.shape[0]

        if Y is None:
            # Do symmetric matrix
            K = np.empty((n, n), order='F')
            flaplacian_kernel_symmetric(X.T, n, K, self.sigma)
        else:
            Y = np.asarray(Y)
            # Do asymmetric matrix
            m = Y.shape[0]
            K = np.empty((n, m), order='F')
            flaplacian_kernel(X.T, n, Y.T, m, K, self.sigma)

        return K

    def _generate_atomic(self, X, Y=None):

        n1 = np.array([len(x) for x in X], dtype=np.int32)

        max1 = np.max(n1)

        nm1 = n1.size

        rep_size = X[0].shape[1]

        x1 = np.zeros((nm1, max1, rep_size), dtype=np.float64, order="F")

        for i in range(nm1):
            x1[i,:n1[i]] = X[i]


        # Reorder for Fortran speed
        x1 = np.swapaxes(x1, 0, 2)

        sigmas = np.array([self.sigma], dtype=np.float64)
        nsigmas = sigmas.size

        if Y is None:
            # Do symmetric matrix
            return fget_vector_kernels_laplacian_symmetric(x1, n1, [self.sigma],
                nm1, nsigmas)[0]
        else:
            # Do asymmetric matrix
            n2 = np.array([len(y) for y in Y], dtype=np.int32)
            max2 = np.max(n2)
            nm2 = n2.size
            x2 = np.zeros((nm2, max2, rep_size), dtype=np.float64, order="F")
            for i in range(nm2):
                x2[i,:n2[i]] = Y[i]
            x2 = np.swapaxes(x2, 0, 2)
            return fget_vector_kernels_laplacian(x1, x2, n1, n2, [self.sigma],
                nm1, nm2, nsigmas)[0]

