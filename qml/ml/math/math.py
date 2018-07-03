# MIT License
#
# Copyright (c) 2016 Anders Steen Christensen
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

import numpy as np

from copy import deepcopy

from .fcho_solve import fcho_solve
from .fcho_solve import fcho_invert
from .fcho_solve import fbkf_solve
from .fcho_solve import fbkf_invert


def cho_invert(A):
    """ Returns the inverse of a positive definite matrix, using a Cholesky decomposition
        via calls to LAPACK dpotrf and dpotri in the F2PY module.

        :param A: Matrix (symmetric and positive definite, left-hand side).
        :type A: numpy array

        :return: The inverse matrix
        :rtype: numpy array
    """

    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    I = np.asfortranarray(A)

    fcho_invert(I)

    # Matrix to store the inverse
    i_lower = np.tril_indices_from(A)

    # Copy lower triangle to upper
    I.T[i_lower] = I[i_lower]

    return I


def cho_solve(A, y):
    """ Solves the equation

            :math:`A x = y`

        for x using a Cholesky decomposition  via calls to LAPACK dpotrf and dpotrs in the F2PY module. Preserves the input matrix A.

        :param A: Matrix (symmetric and positive definite, left-hand side).
        :type A: numpy array
        :param y: Vector (right-hand side of the equation).
        :type y: numpy array

        :return: The solution vector.
        :rtype: numpy array
        """

    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    if len(y.shape) != 1 or y.shape[0] != A.shape[1]:
        raise ValueError('expected matrix and vector of same stride size')

    n = A.shape[0]

    # Backup diagonal before Cholesky-decomposition
    A_diag = A[np.diag_indices_from(A)]

    x = np.zeros(n)
    fcho_solve(A, y, x)

    # Reset diagonal after Cholesky-decomposition
    A[np.diag_indices_from(A)] = A_diag

    # Copy lower triangle to upper
    i_lower = np.tril_indices_from(A)
    A.T[i_lower] = A[i_lower]

    return x


def bkf_invert(A):
    """ Returns the inverse of a positive definite matrix, using a Cholesky decomposition
        via calls to LAPACK dpotrf and dpotri in the F2PY module.

        :param A: Matrix (symmetric and positive definite, left-hand side).
        :type A: numpy array

        :return: The inverse matrix
        :rtype: numpy array
    """

    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    I = np.asfortranarray(A)

    fbkf_invert(I)

    # Matrix to store the inverse
    i_lower = np.tril_indices_from(A)

    # Copy lower triangle to upper
    I.T[i_lower] = I[i_lower]

    return I


def bkf_solve(A, y):
    """ Solves the equation

            :math:`A x = y`

        for x using a Cholesky decomposition  via calls to LAPACK dpotrf and dpotrs in the F2PY module. Preserves the input matrix A.

        :param A: Matrix (symmetric and positive definite, left-hand side).
        :type A: numpy array
        :param y: Vector (right-hand side of the equation).
        :type y: numpy array

        :return: The solution vector.
        :rtype: numpy array
        """

    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    if len(y.shape) != 1 or y.shape[0] != A.shape[1]:
        raise ValueError('expected matrix and vector of same stride size')

    n = A.shape[0]

    # Backup diagonal before Cholesky-decomposition
    A_diag = A[np.diag_indices_from(A)]

    x = np.zeros(n)
    fbkf_solve(A, y, x)

    # Reset diagonal after Cholesky-decomposition
    A[np.diag_indices_from(A)] = A_diag

    # Copy lower triangle to upper
    i_lower = np.tril_indices_from(A)
    A.T[i_lower] = A[i_lower]

    return x
