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

from .data import Data

from ..kernels.fkernels import fgaussian_kernel
from ..kernels.fkernels import fgaussian_kernel_symmetric
from ..kernels.fkernels import fget_vector_kernels_gaussian
from ..kernels.fkernels import fget_vector_kernels_gaussian_symmetric
from ..kernels.fkernels import flaplacian_kernel
from ..kernels.fkernels import flaplacian_kernel_symmetric
from ..kernels.fkernels import fget_vector_kernels_laplacian
from ..kernels.fkernels import fget_vector_kernels_laplacian_symmetric

from ..fchl.ffchl_module import fget_kernels_fchl
from ..fchl.ffchl_module import fget_symmetric_kernels_fchl
from ..fchl.ffchl_module import fget_global_kernels_fchl
from ..fchl.ffchl_module import fget_global_symmetric_kernels_fchl

from ..utils.alchemy import get_alchemy
from ..utils import get_unique

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

        if not hasattr(X, '_representation_short_name'):
            print("Error: No representations found in Data object")
            raise SystemExit

        if X._representation_short_name == 'fchl' and self.__class__.__name__ != 'FCHLKernel' or \
                X._representation_short_name != 'fchl' and self.__class__.__name__ == 'FCHLKernel':
            print("Error: The FCHL representation is only compatible with the FCHL kernel")
            raise SystemExit

    def _set_representations(self, rep):
        self.representations = rep

    def _set_nuclear_charges(self, charge):
        self.nuclear_charges = charge

    def transform(self, X):

        self._check_data_object(X)

        # Kernel between representation stored in fit and representation
        # given in data object. The order matters
        if X._representation_type is None:
            # For FCHL to keep the documentation tidy, since self.local overrides
            # X._representation_type
            kernel = self.generate(X._representations, self.representations)
        elif X._representation_type == 'molecular':
            # Ignore constant features
            constant_features = (np.std(X._representations, axis=0) == 0) * self._constant_features * \
                    (X._representations[0] == self.representations[0])
            kernel = self.generate(X._representations[:,~constant_features], self.representations[:,~constant_features], 'molecular')
        elif (self.alchemy == 'auto' and X._representation_alchemy) or self.alchemy:
            # Ignore constant features
            flat_representations = np.asarray([item for sublist in X._representations for item in sublist])
            constant_features = (np.std(flat_representations, axis=0) == 0) * self._constant_features * \
                    (X._representations[0][0] == self.representations[0][0])
            X_representations = [np.asarray([rep[~constant_features] for rep in mol]) for mol in X._representations]
            self_representations = [np.asarray([rep[~constant_features] for rep in mol]) for mol in self.representations]
            kernel = self.generate(X_representations, self_representations, 'atomic')
        else:
            # Find elements used in representations from X
            elements1 = get_unique(X.nuclear_charges[X._indices])
            # Get the elements common to both X and the fitted representations
            # stored in self
            elements = list(set(elements1).intersection(self.elements))
            ## Get list in the form [H_representations, C_representations, ...]
            # And ignore constant features
            rep1, rep2 = self._get_elementwise_representations_transform(X._representations, X.nuclear_charges[X._indices], elements)
            # Sum elementwise contributions to the kernel
            kernel = np.zeros((len(X._representations), len(self.representations)))
            for i in range(len(rep1)):
                kernel += self.generate(rep1[i], rep2[i], representation_type='atomic')

        X._kernel = kernel

        return X

    def _get_elementwise_representations_transform(self, representations, nuclear_charges, elements):
        """
        Create a set of lists where the each item only contain representations of a specific element
        """

        # Ignore constant features
        flat_nuclear_charges1 = np.asarray([item for sublist in nuclear_charges for item in sublist], dtype=int)
        flat_representations1 = np.asarray([item for sublist in representations for item in sublist])
        flat_nuclear_charges2 = np.asarray([item for sublist in self.nuclear_charges for item in sublist], dtype=int)
        flat_representations2 = np.asarray([item for sublist in self.representations for item in sublist])

        constant_features = {element: None for element in elements}

        for ele in elements:
            rep1 = flat_representations1[flat_nuclear_charges1 == ele]
            rep2 = flat_representations2[flat_nuclear_charges2 == ele]
            rep = np.concatenate([rep1, rep2])
            constant_features[ele] = (np.std(rep, axis=0) == 0)

        # Create the representations
        elementwise_representations1 = [[np.atleast_2d(
            [v[~constant_features[element]] for j,v in enumerate(representations[i]) 
            if nuclear_charges[i][j] == element]) for i in range(len(nuclear_charges))]
            for element in elements]

        elementwise_representations2 = [[np.atleast_2d(
            [v[~constant_features[element]] for j,v in enumerate(self.representations[i]) 
            if self.nuclear_charges[i][j] == element]) for i in range(len(self.nuclear_charges))]
            for element in elements]

        return elementwise_representations1, elementwise_representations2

    def _get_elementwise_representations_fit(self, representations, nuclear_charges):
        """
        Create a list where the each item only contain representations of a specific element
        """
        elements = get_unique(nuclear_charges)
        self.elements = elements

        # Ignore constant features
        flat_nuclear_charges = np.asarray([item for sublist in nuclear_charges for item in sublist], dtype=int)
        flat_representations = np.asarray([item for sublist in representations for item in sublist])

        constant_features = {element: None for element in elements}

        for ele in elements:
            rep = flat_representations[flat_nuclear_charges == ele]
            constant_features[ele] = (np.std(rep, axis=0) == 0)

        # Create the representations
        elementwise_representations = [[np.atleast_2d(
            [v[~constant_features[element]] for j,v in enumerate(representations[i]) 
            if nuclear_charges[i][j] == element]) for i in range(len(nuclear_charges))]
            for element in elements]

        return elementwise_representations

    def _fit_transform(self, X):

        self._check_data_object(X)

        # Store representation / nuclear_charges for future transform calls
        self._set_representations(X._representations)
        self._set_nuclear_charges(X.nuclear_charges[X._indices])

        if X._representation_type is None:
            # For FCHL to keep the documentation tidy, since self.local overrides
            # X._representation_type
            kernel = self.generate(X._representations)
        elif X._representation_type == 'molecular':
            # Ignore constant features
            self._constant_features = (np.std(X._representations, axis=0) == 0)
            kernel = self.generate(X._representations[:,~self._constant_features], representation_type='molecular')
        elif (self.alchemy == 'auto' and X._representation_alchemy) or self.alchemy:
            # Ignore constant features
            flat_representations = np.asarray([item for sublist in X._representations for item in sublist])
            self._constant_features = (np.std(flat_representations, axis=0) == 0)
            representations = [np.asarray([rep[~self._constant_features] for rep in mol]) for mol in X._representations]
            kernel = self.generate(representations, representation_type='atomic')
        else:
            # Get list in the form [H_representations, C_representations, ...]
            # And ignore constant features
            rep = self._get_elementwise_representations_fit(X._representations, X.nuclear_charges[X._indices])
            # Sum elementwise contributions to the kernel
            kernel = np.zeros((len(X._representations),)*2)
            for r in rep:
                kernel += self.generate(r, representation_type='atomic')

        # Store kernel
        X._kernel = kernel

        return X

    def fit_transform(self, X, y=None):
        return self._fit_transform(X)

class GaussianKernel(_BaseKernel):
    """
    Gaussian kernel
    """

    def __init__(self, sigma='auto', alchemy='auto'):
        """
        :param sigma: Scale parameter of the gaussian kernel. `sigma='auto'` will try to guess
                      a good value (very approximate).
        :type sigma: float
        :param alchemy: Determines if contributions from atoms of differing elements should be
                        included in the kernel. If alchemy='auto', this will be automatically
                        determined from the representation used
        :type alchemy: bool
        """
        self.sigma = sigma
        self.alchemy = alchemy

        self.representations = None
        self._elements = None
        self._constant_features = None

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
            self.sigma *= np.sqrt(-np.log(smallest_kernel_element) / np.log(2))
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
            elif smallest_kernel_element > 0.95:
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

        if Y is None or X is Y:
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

        for i in range(len(X)):
            rep_size = X[0].shape[1]
            if rep_size == 0:
                continue
            break
        else:
            if Y is None:
                return np.zeros((nm1, nm1))
            return np.zeros((nm1, len(Y)))

        x1 = np.zeros((nm1, max1, rep_size), dtype=np.float64, order="F")

        for i in range(nm1):
            if X[i].shape[1] == 0:
                n1[i] = 0
                continue
            x1[i,:n1[i]] = X[i]


        # Reorder for Fortran speed
        x1 = np.swapaxes(x1, 0, 2)

        sigmas = np.array([self.sigma], dtype=np.float64)
        nsigmas = sigmas.size

        if Y is None or X is Y:
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
                if Y[i].shape[1] == 0:
                    n2[i] = 0
                    continue
                x2[i,:n2[i]] = Y[i]
            x2 = np.swapaxes(x2, 0, 2)
            return fget_vector_kernels_gaussian(x1, x2, n1, n2, [self.sigma],
                nm1, nm2, nsigmas)[0]

class LaplacianKernel(_BaseKernel):
    """
    Laplacian kernel
    """

    def __init__(self, sigma='auto', alchemy='auto'):
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
        self._elements = None
        self._constant_features = None

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
            self.sigma *= - np.log(smallest_kernel_element) / np.log(2)
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

        if Y is None or X is Y:
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

        for i in range(len(X)):
            rep_size = X[0].shape[1]
            if rep_size == 0:
                continue
            break
        else:
            if Y is None:
                return np.zeros((nm1, nm1))
            return np.zeros((nm1, len(Y)))

        x1 = np.zeros((nm1, max1, rep_size), dtype=np.float64, order="F")

        for i in range(nm1):
            if X[i].shape[1] == 0:
                n1[i] = 0
                continue
            x1[i,:n1[i]] = X[i]

        #for i,j in zip(x1[:,:,2], n1):
        #    print(i,j)


        # Reorder for Fortran speed
        x1 = np.swapaxes(x1, 0, 2)

        sigmas = np.array([self.sigma], dtype=np.float64)
        nsigmas = sigmas.size

        if Y is None or X is Y:
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
                if Y[i].shape[1] == 0:
                    n2[i] = 0
                    continue
                x2[i,:n2[i]] = Y[i]
            x2 = np.swapaxes(x2, 0, 2)
            return fget_vector_kernels_laplacian(x1, x2, n1, n2, [self.sigma],
                nm1, nm2, nsigmas)[0]

class FCHLKernel(_BaseKernel):
    """
    FCHL kernel
    """

    def __init__(self, sigma='auto', alchemy=True, two_body_scaling=np.sqrt(8),
            three_body_scaling=1.6, two_body_width=0.2, three_body_width=np.pi,
            two_body_power=4.0, three_body_power=2.0, damping_start=1.0, cutoff=5.0,
            fourier_order=1, alchemy_period_width=1.6, alchemy_group_width=1.6,
            local=True):
        """
        :param sigma: Scale parameter of the gaussian basis functions. `sigma='auto'` will try to guess
                      a good value (very approximate).
        :type sigma: float or string
        :param alchemy: Determines if contributions from atoms of differing elements should be
                        included in the kernel.
        :type alchemy: bool
        :param two_body_scaling: Weight for 2-body terms.
        :type two_body_scaling: float
        :param three_body_scaling: Weight for 3-body terms.
        :type three_body_scaling: float
        :param two_body_width: Gaussian width for 2-body terms
        :type two_body_width: float
        :param three_body_width: Gaussian width for 3-body terms.
        :type three_body_width: float
        :param two_body_power: Powerlaw for :math:`r^{-n}` 2-body terms.
        :type two_body_power: float
        :param three_body_power: Powerlaw for Axilrod-Teller-Muto 3-body term
        :type three_body_power: float
        :param damping_start: The fraction of the cutoff radius at which cut-off damping start.
        :type damping_start: float
        :param cutoff: Cut-off radius.
        :type cutoff: float
        :param fourier_order: 3-body Fourier-expansion truncation order.
        :type fourier_order: integer
        :param alchemy_period_width: Gaussian width along periods (columns) in the periodic table.
        :type alchemy_period_width: float
        :param alchemy_group_width: Gaussian width along groups (rows) in the periodic table.
        :type alchemy_group_width: float
        :param local: Do local decomposition
        :param local: bool
        """
        self.sigma = sigma
        self.alchemy = alchemy
        self.two_body_scaling = two_body_scaling
        self.three_body_scaling = three_body_scaling
        self.two_body_width = two_body_width
        self.three_body_width = three_body_width
        self.two_body_power = two_body_power
        self.three_body_power = three_body_power
        self.damping_start = damping_start
        self.cutoff = cutoff
        self.fourier_order = fourier_order
        self.alchemy_period_width = alchemy_period_width
        self.alchemy_group_width = alchemy_group_width
        self.local = local

        self.representations = None

    def _quick_estimate_sigma(self, X, sigma_init=1, count=1):
        """
        Use 50 random points for atomic, 200 for molecular, to get an approximate guess for sigma
        """

        if count > 10:
            print("Error. Could not automatically determine parameter `sigma` in the kernel %s"
                    % self.__class__.__name__)
            raise SystemExit

        n = 50

        if len(X) < n:
            n = len(X)

        self.sigma = sigma_init

        # Generate kernel for a random subset
        indices = np.random.choice(np.arange(len(X)), size=n, replace=False)
        kernel = self.generate(X)#[indices])

        if not self.local:
            # min smallest kernel element
            smallest_kernel_element = np.min(kernel)
            if smallest_kernel_element < 0.3:
                self._quick_estimate_sigma(X, sigma_init * 2.5, count + 1)
            elif smallest_kernel_element > 0.9:
                self._quick_estimate_sigma(X, sigma_init / 2.5, count + 1)

        else:
            sizes = np.asarray([len(X[i]) for i in indices])
            kernel /= sizes[:,None] * sizes[None,:]
            smallest_kernel_element = np.min(kernel)

            if smallest_kernel_element < 0.1:
                self._quick_estimate_sigma(X, sigma_init * 2.5, count + 1)
            elif smallest_kernel_element > 0.3:
                self._quick_estimate_sigma(X, sigma_init / 2.5, count + 1)

    def generate(self, X, Y=None):
        """
        Create a kernel from representations `X`. Optionally
        an asymmetric kernel can be constructed between representations
        `X` and `Y`.

        :param X: representations
        :type X: array
        :param Y: (Optional) representations
        :type Y: array

        :return: Gaussian kernel matrix of shape (n_samplesX, n_samplesX) if \
                 Y=None else (n_samplesX, n_samplesY)
        :rtype: array
        """

        if self.sigma == 'auto':
            # Do a quick and dirty initial estimate of sigma
            self._quick_estimate_sigma(X)


        if not self.local:
            kernel = self._generate_molecular(X,Y)
        else:
            kernel = self._generate_atomic(X,Y)

        return kernel

    def _generate_molecular(self, X, Y=None):

        atoms_max = X.shape[1]
        neighbors_max = X.shape[3]
        nm1 = X.shape[0]
        N1 = np.zeros((nm1),dtype=np.int32)

        for a in range(nm1):
            N1[a] = len(np.where(X[a,:,1,0] > 0.0001)[0])

        neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)

        for a, representation in enumerate(X):
            ni = N1[a]
            for i, x in enumerate(representation[:ni]):
                neighbors1[a,i] = len(np.where(x[0] < self.cutoff)[0])

        if self.alchemy:
            alchemy = 'periodic-table'
        else:
            alchemy = 'off'

        doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=self.alchemy_group_width, c_width=self.alchemy_period_width)

        if Y is None or X is Y:
            # Do symmetric kernel
            return fget_global_symmetric_kernels_fchl(X, N1, neighbors1, [self.sigma],
                        nm1, 1, self.three_body_width, self.two_body_width, self.damping_start,
                        self.cutoff, self.fourier_order, pd, self.two_body_scaling, self.three_body_scaling,
                        doalchemy, self.two_body_power, self.three_body_power)[0]
        else:
            # Do asymmetric kernel
            nm2 = Y.shape[0]
            N2 = np.zeros((nm2),dtype=np.int32)

            for a in range(nm2):
                N2[a] = len(np.where(Y[a,:,1,0] > 0.0001)[0])

            neighbors2 = np.zeros((nm2, atoms_max), dtype=np.int32)

            for a, representation in enumerate(Y):
                ni = N2[a]
                for i, x in enumerate(representation[:ni]):
                    neighbors2[a,i] = len(np.where(x[0] < self.cutoff)[0])

            return fget_global_kernels_fchl(X, Y, N1, N2, neighbors1, neighbors2, [self.sigma],
                        nm1, nm2, 1, self.three_body_width, self.two_body_width, self.damping_start,
                        self.cutoff, self.fourier_order, pd, self.two_body_scaling, self.three_body_scaling,
                        doalchemy, self.two_body_power, self.three_body_power)[0]

    def _generate_atomic(self, X, Y=None):

        atoms_max = X.shape[1]
        neighbors_max = X.shape[3]

        nm1 = X.shape[0]
        N1 = np.zeros((nm1),dtype=np.int32)

        for a in range(nm1):
            N1[a] = len(np.where(X[a,:,1,0] > 0.0001)[0])

        neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)

        for a, representation in enumerate(X):
            ni = N1[a]
            for i, x in enumerate(representation[:ni]):
                neighbors1[a,i] = len(np.where(x[0] < self.cutoff)[0])

        if self.alchemy:
            alchemy = 'periodic-table'
        else:
            alchemy = 'off'

        doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=self.alchemy_group_width, c_width=self.alchemy_period_width)

        if Y is None or X is Y:
            return fget_symmetric_kernels_fchl(X, N1, neighbors1, [self.sigma],
                        nm1, 1, self.three_body_width, self.two_body_width, self.damping_start,
                        self.cutoff, self.fourier_order, pd, self.two_body_scaling, self.three_body_scaling,
                        doalchemy, self.two_body_power, self.three_body_power)[0]
        else:
            nm2 = Y.shape[0]
            N2 = np.zeros((nm2),dtype=np.int32)

            for a in range(nm2):
                N2[a] = len(np.where(Y[a,:,1,0] > 0.0001)[0])

            neighbors2 = np.zeros((nm2, atoms_max), dtype=np.int32)

            for a, representation in enumerate(Y):
                ni = N2[a]
                for i, x in enumerate(representation[:ni]):
                    neighbors2[a,i] = len(np.where(x[0] < self.cutoff)[0])

            return fget_kernels_fchl(X, Y, N1, N2, neighbors1, neighbors2, [self.sigma],
                        nm1, nm2, 1, self.three_body_width, self.two_body_width, self.damping_start, 
                        self.cutoff, self.fourier_order, pd, self.two_body_scaling, self.three_body_scaling, 
                        doalchemy, self.two_body_power, self.three_body_power)[0]

    def fit_transform(self, X, y=None):

        # Check that the cutoff is less than the cutoff used to make the representation
        if self.cutoff > X._representation_cutoff:
            print("Error: Cutoff used in the FCHL kernel (%.2f) must be lower or equal" % self.cutoff,
                    "to the cutoff used in the FCHL representation (%.2f)" % X._representation_cutoff)
            raise SystemExit

        return self._fit_transform(X)

