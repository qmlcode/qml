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

from qml.utils import is_none
from qml.data import Data

from .fkernels import fgaussian_kernel, fgaussian_kernel_symmetric
from .fkernels import flaplacian_kernel
from .fkernels import flinear_kernel
from .fkernels import fsargan_kernel
from .fkernels import fmatern_kernel_l2

from .fkernels import fget_local_kernels_gaussian, fget_local_kernels_gaussian_symmetric
from .fkernels import fget_local_kernels_laplacian

def laplacian_kernel(A, B, sigma):
    """ Calculates the Laplacian kernel matrix K, where :math:`K_{ij}`:

            :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - B_j\\|_1}{\sigma} \\big)`

        Where :math:`A_{i}` and :math:`B_{j}` are representation vectors.
        K is calculated using an OpenMP parallel Fortran routine.

        :param A: 2D array of representations - shape (N, representation size).
        :type A: numpy array
        :param B: 2D array of representations - shape (M, representation size).
        :type B: numpy array
        :param sigma: The value of sigma in the kernel matrix.
        :type sigma: float

        :return: The Laplacian kernel matrix - shape (N, M)
        :rtype: numpy array
    """

    na = A.shape[0]
    nb = B.shape[0]

    K = np.empty((na, nb), order='F')

    # Note: Transposed for Fortran
    flaplacian_kernel(A.T, na, B.T, nb, K, sigma)

    return K

#def gaussian_kernel(A, B, sigma):
#    """ Calculates the Gaussian kernel matrix K, where :math:`K_{ij}`:
#
#            :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - B_j\\|_2^2}{2\sigma^2} \\big)`
#
#        Where :math:`A_{i}` and :math:`B_{j}` are representation vectors.
#        K is calculated using an OpenMP parallel Fortran routine.
#
#        :param A: 2D array of representations - shape (N, representation size).
#        :type A: numpy array
#        :param B: 2D array of representations - shape (M, representation size).
#        :type B: numpy array
#        :param sigma: The value of sigma in the kernel matrix.
#        :type sigma: float
#
#        :return: The Gaussian kernel matrix - shape (N, M)
#        :rtype: numpy array
#    """
#
#    na = A.shape[0]
#    nb = B.shape[0]
#
#    K = np.empty((na, nb), order='F')
#
#    # Note: Transposed for Fortran
#    fgaussian_kernel(A.T, na, B.T, nb, K, sigma)
#
#    return K

def linear_kernel(A, B):
    """ Calculates the linear kernel matrix K, where :math:`K_{ij}`:

            :math:`K_{ij} = A_i \cdot B_j`

        VWhere :math:`A_{i}` and :math:`B_{j}` are  representation vectors. 

        K is calculated using an OpenMP parallel Fortran routine.

        :param A: 2D array of representations - shape (N, representation size).
        :type A: numpy array
        :param B: 2D array of representations - shape (M, representation size).
        :type B: numpy array

        :return: The Gaussian kernel matrix - shape (N, M)
        :rtype: numpy array
    """

    na = A.shape[0]
    nb = B.shape[0]

    K = np.empty((na, nb), order='F')

    # Note: Transposed for Fortran
    flinear_kernel(A.T, na, B.T, nb, K)

    return K

def sargan_kernel(A, B, sigma, gammas):
    """ Calculates the Sargan kernel matrix K, where :math:`K_{ij}`:

            :math:`K_{ij} = \\exp \\big( -\\frac{\\| A_i - B_j \\|_1)}{\sigma} \\big) \\big(1 + \\sum_{k} \\frac{\gamma_{k} \\| A_i - B_j \\|_1^k}{\sigma^k} \\big)`

        Where :math:`A_{i}` and :math:`B_{j}` are representation vectors.
        K is calculated using an OpenMP parallel Fortran routine.

        :param A: 2D array of representations - shape (N, representation size).
        :type A: numpy array
        :param B: 2D array of representations - shape (M, representation size).
        :type B: numpy array
        :param sigma: The value of sigma in the kernel matrix.
        :type sigma: float
        :param gammas: 1D array of parameters in the kernel matrix.
        :type gammas: numpy array

        :return: The Sargan kernel matrix - shape (N, M).
        :rtype: numpy array
    """

    ng = len(gammas)

    if ng == 0:
        return laplacian_kernel(A, B, sigma)

    na = A.shape[0]
    nb = B.shape[0]

    K = np.empty((na, nb), order='F')

    # Note: Transposed for Fortran
    fsargan_kernel(A.T, na, B.T, nb, K, sigma, gammas, ng)

    return K

def matern_kernel(A, B, sigma, order = 0, metric = "l1"):
    """ Calculates the Matern kernel matrix K, where :math:`K_{ij}`:

            for order = 0:
                :math:`K_{ij} = \\exp\\big( -\\frac{d}{\sigma} \\big)`
            for order = 1:
                :math:`K_{ij} = \\exp\\big( -\\frac{\\sqrt{3} d}{\sigma} \\big) \\big(1 + \\frac{\\sqrt{3} d}{\sigma} \\big)`
            for order = 2:
                :math:`K_{ij} = \\exp\\big( -\\frac{\\sqrt{5} d}{d} \\big) \\big( 1 + \\frac{\\sqrt{5} d}{\sigma} + \\frac{5 d^2}{3\sigma^2} \\big)`

        Where :math:`A_i` and :math:`B_j` are representation vectors, and d is a distance measure.

        K is calculated using an OpenMP parallel Fortran routine.

        :param A: 2D array of representations - shape (N, representation size).
        :type A: numpy array
        :param B: 2D array of representations - shape (M, representation size).
        :type B: numpy array
        :param sigma: The value of sigma in the kernel matrix.
        :type sigma: float
        :param order: The order of the polynomial (0, 1, 2)
        :type order: integer
        :param metric: The distance metric ('l1', 'l2')
        :type metric: string

        :return: The Matern kernel matrix - shape (N, M)
        :rtype: numpy array
    """

    if metric == "l1":
        if order == 0:
            gammas = []
        elif order == 1:
            gammas = [1]
            sigma /= np.sqrt(3)
        elif order == 2:
            gammas = [1,1/3.0]
            sigma /= np.sqrt(5)
        else:
            print("Order:%d not implemented in Matern Kernel" % order)
            raise SystemExit

        return sargan_kernel(A, B, sigma, gammas)

    elif metric == "l2":
        pass
    else:
        print("Error: Unknown distance metric %s in Matern kernel" % str(metric))
        raise SystemExit

    na = A.shape[0]
    nb = B.shape[0]

    K = np.empty((na, nb), order='F')

    # Note: Transposed for Fortran
    fmatern_kernel_l2(A.T, na, B.T, nb, K, sigma, order)

    return K

def get_local_kernels_gaussian(A, B, na, nb, sigmas):
    """ Calculates the Gaussian kernel matrix K, for a local representation where :math:`K_{ij}`:

            :math:`K_{ij} = \sum_{a \in i} \sum_{b \in j} \\exp \\big( -\\frac{\\|A_a - B_b\\|_2^2}{2\sigma^2} \\big)`

        Where :math:`A_{a}` and :math:`B_{b}` are representation vectors.

        Note that the input array is one big 2D array with all atoms concatenated along the same axis.
        Further more a series of kernels is produced (since calculating the distance matrix is expensive
        but getting the resulting kernels elements for several sigmas is not.)

        K is calculated using an OpenMP parallel Fortran routine.

        :param A: 2D array of descriptors - shape (total atoms A, representation size).
        :type A: numpy array
        :param B: 2D array of descriptors - shape (total atoms B, representation size).
        :type B: numpy array
        :param na: 1D array containing numbers of atoms in each compound.
        :type na: numpy array
        :param nb: 1D array containing numbers of atoms in each compound.
        :type nb: numpy array
        :param sigma: The value of sigma in the kernel matrix.
        :type sigma: float

        :return: The Gaussian kernel matrix - shape (nsigmas, N, M)
        :rtype: numpy array
    """

    assert np.sum(na) == A.shape[0], "Error in A input"
    assert np.sum(nb) == B.shape[0], "Error in B input"

    assert A.shape[1] == B.shape[1], "Error in representation sizes"

    nma = len(na)
    nmb = len(nb)

    sigmas = np.asarray(sigmas)
    nsigmas = len(sigmas)

    return fget_local_kernels_gaussian(A.T, B.T, na, nb, sigmas, nma, nmb, nsigmas)

def get_local_kernels_laplacian(A, B, na, nb, sigmas):
    """ Calculates the Local Laplacian kernel matrix K, for a local representation where :math:`K_{ij}`:

            :math:`K_{ij} = \sum_{a \in i} \sum_{b \in j} \\exp \\big( -\\frac{\\|A_a - B_b\\|_1}{\sigma} \\big)`

        Where :math:`A_{a}` and :math:`B_{b}` are representation vectors.

        Note that the input array is one big 2D array with all atoms concatenated along the same axis.
        Further more a series of kernels is produced (since calculating the distance matrix is expensive
        but getting the resulting kernels elements for several sigmas is not.)
        
        K is calculated using an OpenMP parallel Fortran routine.

        :param A: 2D array of descriptors - shape (N, representation size).
        :type A: numpy array
        :param B: 2D array of descriptors - shape (M, representation size).
        :type B: numpy array
        :param na: 1D array containing numbers of atoms in each compound.
        :type na: numpy array
        :param nb: 1D array containing numbers of atoms in each compound.
        :type nb: numpy array
        :param sigmas: List of the sigmas.
        :type sigmas: list

        :return: The Laplacian kernel matrix - shape (nsigmas, N, M)
        :rtype: numpy array
    """

    assert np.sum(na) == A.shape[0], "Error in A input"
    assert np.sum(nb) == B.shape[0], "Error in B input"

    assert A.shape[1] == B.shape[1], "Error in representation sizes"

    nma = len(na)
    nmb = len(nb)
     
    sigmas = np.asarray(sigmas)
    nsigmas = len(sigmas)

    return fget_local_kernels_laplacian(A.T, B.T, na, nb, sigmas, nma, nmb, nsigmas)


class BaseKernel(BaseEstimator):
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

class GaussianKernel(BaseKernel):
    """
    Gaussian kernel
    """

    def __init__(self, sigma=1):
        self.sigma = sigma

        self.representations = None

    def _set_representations(self, rep):
        self.representations = rep

    def transform(self, X):

        self._check_data_object(X)

        # Kernel between representation stored in fit and representation
        # given in data object. The order matters
        kernel = self.generate(X.representations, self.representations, X.property_type)

        X.kernel = kernel

        return X

    def fit_transform(self, X, y=None):

        self._check_data_object(X)

        # Store representation for future transform calls
        self._set_representations(X.representations)

        kernel = self.generate(X.representations, representation_type=X.representation_type)

        X.kernel = kernel

        return X

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

        if representation_type == 'molecular':
            return self._generate_molecular(X,Y)
        elif representation_type == 'atomic':
            return self._generate_atomic(X,Y)
        else:
            # Should never get here for users
            print("Error: property_type needs to equal 'molecular' \
                    or 'atomic'")
            raise SystemExit

    def _generate_molecular(self, X, Y=None):

        # Note: Transposed for Fortran
        n = X.shape[0]

        if is_none(Y):
            # Do symmetric matrix
            K = np.empty((n, n), order='F')
            fgaussian_kernel_symmetric(X.T, n, K, self.sigma)
        else:
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

        if is_none(Y):
            # Do symmetric matrix
            return fget_vector_kernels_gaussian_symmetric(x1, n1, [self.sigma],
                nm1, nsigmas)
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
                nm1, nm2, nsigmas)

