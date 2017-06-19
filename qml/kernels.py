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

from .fkernels import fgaussian_kernel
from .fkernels import flaplacian_kernel
from .fkernels import fsargan_kernel
from .fkernels import fmatern_kernel_l2

def laplacian_kernel(A, B, sigma):
    """ Calculates the Laplacian kernel matrix K, where :math:`K_{ij}`:

            :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - B_j\\|_1}{\sigma} \\big)`

        Where :math:`A_{i}` and :math:`B_{j}` are representation vectors.
        K is calculated using an OpenMP parallel Fortran routine.

        :param A: 2D array of descriptors - shape (N, representation size).
        :type A: numpy array
        :param B: 2D array of descriptors - shape (M, representation size).
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

def gaussian_kernel(A, B, sigma):
    """ Calculates the Gaussian kernel matrix K, where :math:`K_{ij}`:

            :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - B_j\\|_2^2}{2\sigma^2} \\big)`

        Where :math:`A_{i}` and :math:`B_{j}` are representation vectors.
        K is calculated using an OpenMP parallel Fortran routine.

        :param A: 2D array of descriptors - shape (N, representation size).
        :type A: numpy array
        :param B: 2D array of descriptors - shape (M, representation size).
        :type B: numpy array
        :param sigma: The value of sigma in the kernel matrix.
        :type sigma: float

        :return: The Gaussian kernel matrix - shape (N, M)
        :rtype: numpy array
    """

    na = A.shape[0]
    nb = B.shape[0]

    K = np.empty((na, nb), order='F')

    # Note: Transposed for Fortran
    fgaussian_kernel(A.T, na, B.T, nb, K, sigma)

    return K

def sargan_kernel(A, B, sigma, gammas):
    """ Calculates the Sargan kernel matrix K, where :math:`K_{ij}`:

            :math:`K_{ij} = \\exp \\big( -\\frac{\\| A_i - B_j \\|_1)}{\sigma} \\big) \\big(1 + \\sum_{k} \\frac{\gamma_{k} \\| A_i - B_j \\|_1^k}{\sigma^k} \\big)`

        Where :math:`A_{i}` and :math:`B_{j}` are representation vectors.
        K is calculated using an OpenMP parallel Fortran routine.

        :param A: 2D array of descriptors - shape (N, representation size).
        :type A: numpy array
        :param B: 2D array of descriptors - shape (M, representation size).
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

        :param A: 2D array of descriptors - shape (N, representation size).
        :type A: numpy array
        :param B: 2D array of descriptors - shape (M, representation size).
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

