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

from .fsolvers import fcho_solve
from .fsolvers import fcho_invert
from .fsolvers import fbkf_solve
from .fsolvers import fbkf_invert
from .fsolvers import fsvd_solve
from .fsolvers import fqrlq_solve
from .fsolvers import fcond
from .fsolvers import fcond_ge


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
    """ Returns the inverse of a positive definite matrix, using a Bausch-Kauffman decomposition
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

        for x using a  Bausch-Kauffma  decomposition  via calls to LAPACK  in the F2PY module. Preserves the input matrix A.

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


def svd_solve(A, y, rcond=None):
    """ Solves the equation

            :math:`A x = y`

        for x using a singular-value decomposition (SVD) via calls to
        LAPACK DGELSD in the F2PY module. Preserves the input matrix A.

        :param A: Matrix (symmetric and positive definite, left-hand side).
        :type A: numpy array
        :param y: Vector (right-hand side of the equation).
        :type y: numpy array
        :param rcond: Optional parameater for lowest singular-value  
        :type rcond: float 

        :return: The solution vector.
        :rtype: numpy array
        """

    print(y.shape)
    print(A.shape)
    if len(y.shape) != 1 or y.shape[0] != A.shape[0]:
        raise ValueError('expected matrix and vector of same stride size')

    if rcond is None:
        rcond=0.0

    x_dim = A.shape[1]
    A = np.asarray(A, order="F")
    x = fsvd_solve(A, y, x_dim, rcond)

    return x


def qrlq_solve(A, y):
    """ Solves the equation

            :math:`A x = y`

        for x using a QR or LQ decomposition (depending on matrix dimensions)
        via calls to LAPACK DGELSD in the F2PY module. Preserves the input matrix A.

        :param A: Matrix (symmetric and positive definite, left-hand side).
        :type A: numpy array
        :param y: Vector (right-hand side of the equation).
        :type y: numpy array

        :return: The solution vector.
        :rtype: numpy array
        """

    if len(y.shape) != 1 or y.shape[0] != A.shape[0]:
        raise ValueError('expected matrix and vector of same stride size')
    
    x_dim = A.shape[1]
    A = np.asarray(A, order="F")
    x = fqrlq_solve(A, y, x_dim)

    return x


def condition_number(A, method="cholesky"):
    """ Returns the condition number for the square matrix A.

        Two different methods are implemented: 
        Cholesky (requires a positive-definite matrix), but barely any additional memory overhead.
        LU: Does not require a positive definite matrix, but requires additional memory.
    """

    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    if (method.lower() == "cholesky"):
        assert np.allclose(A, A.T), \
            "ERROR: Can't use a Cholesky-decomposition for a non-symmetric matrix."

        cond = fcond(A)

        return cond

    elif (method.lower() == "lu"):

        cond = fcond_ge(A)

        return cond

