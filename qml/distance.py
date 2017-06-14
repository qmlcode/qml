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

from .fdistance import fmanhattan_distance, fl2_distance
from .fdistance import fp_distance_integer, fp_distance_double

def manhattan_distance(A, B):
    """ Calculates the Manhattan distances, D,  between two
        Numpy arrays of representations.
    
            :math:`D_{ij} = \\|A_i - B_j\\|_1`

        Where :math:`A_{i}` and :math:`B_{j}` are representation vectors.
        D is calculated using an OpenMP parallel Fortran routine.

        :param A: 2D array of descriptors - shape (N, representation size).
        :type A: numpy array
        :param B: 2D array of descriptors - shape (M, representation size).
        :type B: numpy array

        :return: The Manhattan-distance matrix.
        :rtype: numpy array
    """

    if len(A.shape) != 2 or len(B.shape) != 2:
        raise ValueError('expected matrices of dimension=2')

    if B.shape[1] != A.shape[1]:
        raise ValueError('expected matrices containing vectors of same size')

    na = A.shape[0]
    nb = B.shape[0]

    D = np.empty((na, nb), order='F')

    fmanhattan_distance(A.T, B.T, D)

    return D

def l2_distance(A, B):
    """ Calculates the L2 distances, D, between two
        Numpy arrays of representations.

            :math:`D_{ij} = \\|A_i - B_j\\|_2`

        Where :math:`A_{i}` and :math:`B_{j}` are representation vectors.
        D is calculated using an OpenMP parallel Fortran routine.

        :param A: 2D array of descriptors - shape (N, representation size).
        :type A: numpy array
        :param B: 2D array of descriptors - shape (M, representation size).
        :type B: numpy array
    
        :return: The L2-distance matrix.
        :rtype: numpy array
    """

    if len(A.shape) != 2 or len(B.shape) != 2:
        raise ValueError('expected matrices of dimension=2')

    if B.shape[1] != A.shape[1]:
        raise ValueError('expected matrices containing vectors of same size')

    na = A.shape[0]
    nb = B.shape[0]

    D = np.empty((na, nb), order='F')

    fl2_distance(A.T, B.T, D)

    return D

def p_distance(A, B, p = 2):
    """ Calculates the p-norm distances between two
        Numpy arrays of representations.
        The value of the keyword argument ``p =`` sets the norm order. 
        E.g. ``p = 1.0`` and ``p = 2.0`` with yield the Manhattan and L2 distances, respectively.
    
            .. math:: D_{ij} = \|A_i - B_j\|_p

        Where :math:`A_{i}` and :math:`B_{j}` are representation vectors.
        D is calculated using an OpenMP parallel Fortran routine.

        :param A: 2D array of descriptors - shape (N, representation size).
        :type A: numpy array
        :param B: 2D array of descriptors - shape (M, representation size).
        :type B: numpy array
        :param p: The norm order
        :type p: float

        :return: The distance matrix.
        :rtype: numpy array
    """

    if len(A.shape) != 2 or len(B.shape) != 2:
        raise ValueError('expected matrices of dimension=2')

    if B.shape[1] != A.shape[1]:
        raise ValueError('expected matrices containing vectors of same size')

    na = A.shape[0]
    nb = B.shape[0]

    D = np.empty((na, nb), order='F')


    if (type(p) == type(1)):
        if (p == 2):
            fl2_distance(A, B, D)
        else:
            fp_distance_integer(A.T, B.T, D, p)

    elif (type(p) == type(1.0)):
        if p.is_integer():
            p = int(p)
            if (p == 2):
                fl2_distance(A, B, D)
            else:
                fp_distance_integer(A.T, B.T, D, p)

        else:
            fp_distance_double(A.T, B.T, D, p)
    else:
        raise ValueError('expected exponent of integer or float type')

    return D
