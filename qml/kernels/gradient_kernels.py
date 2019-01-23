# MIT License
#
# Copyright (c) 2018 Anders Steen Christensen
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

import os
import numpy as np

from .fgradient_kernels import flocal_kernel
from .fgradient_kernels import fatomic_local_kernel
from .fgradient_kernels import fatomic_local_gradient_kernel
from .fgradient_kernels import flocal_gradient_kernel
from .fgradient_kernels import fgdml_kernel
from .fgradient_kernels import fsymmetric_gdml_kernel
from .fgradient_kernels import fgaussian_process_kernel
from .fgradient_kernels import fsymmetric_gaussian_process_kernel


def get_local_kernel(X1, X2, Q1, Q2, SIGMA):
    """ Calculates the Gaussian kernel matrix K with the local decomposition where :math:`K_{ij}`:

            :math:`K_{ij} = \\sum_{I\\in i} \\sum_{J\\in j}\\exp \\big( -\\frac{\\|X_I - X_J\\|_2^2}{2\sigma^2} \\big)`

        Where :math: X_{I}` and :math:`X_{J}` are representation vectors of the atomic environments.
        For instance atom-centered symmetry functions could be used here.
        K is calculated analytically using an OpenMP parallel Fortran routine.

        :param X1: Array of representations - shape=(N1, rep_size, max_atoms).
        :type X1: numpy array
        :param X2: Array of representations - shape=(N2, rep_size, max_atoms).
        :type X2: numpy array
        
        :param Q1: List of lists containing the nuclear charges for each molecule.
        :type Q1: list
        :param Q2: List of lists containing the nuclear charges for each molecule.
        :type Q2: list

        :param SIGMA: Gaussian kernel width.
        :type SIGMA: float

        :return: 2D matrix of kernel elements shape=(N1, N2),
        :rtype: numpy array
    """
    

    N1 = np.zeros((X1.shape[0]), dtype=np.int32)
    N2 = np.zeros((X2.shape[0]), dtype=np.int32)
    
    for i, rep in enumerate(X1):
        N1[i] = sum([any(np.abs(atom) > 0) for atom in rep])
    
    for i, rep in enumerate(X2):
        N2[i] = sum([any(np.abs(atom) > 0) for atom in rep])

    Q1_input = np.zeros((max(N1), X1.shape[0]), dtype=np.int32)
    Q2_input = np.zeros((max(N2), X2.shape[0]), dtype=np.int32)
    
    for i, q in enumerate(Q1):
        Q1_input[:len(q),i] = q

    for i, q in enumerate(Q2):
        Q2_input[:len(q),i] = q

    K = flocal_kernel(
            X1, 
            X2,
            Q1_input,
            Q2_input,
            N1,
            N2,
            len(N1), 
            len(N2),
            SIGMA
    )

    return K


def get_atomic_local_kernel(X1, X2, Q1, Q2, SIGMA):


    N1 = np.zeros((X1.shape[0]), dtype=np.int32)
    N2 = np.zeros((X2.shape[0]), dtype=np.int32)
    
    for i, rep in enumerate(X1):
        N1[i] = sum([any(np.abs(atom) > 0) for atom in rep])
    
    for i, rep in enumerate(X2):
        N2[i] = sum([any(np.abs(atom) > 0) for atom in rep])

    Q1_input = np.zeros((max(N1), X1.shape[0]), dtype=np.int32)
    Q2_input = np.zeros((max(N2), X2.shape[0]), dtype=np.int32)
    
    for i, q in enumerate(Q1):
        Q1_input[:len(q),i] = q

    for i, q in enumerate(Q2):
        Q2_input[:len(q),i] = q

    K = fatomic_local_kernel(
            X1, 
            X2, 
            Q1_input,
            Q2_input,
            N1,
            N2,
            len(N1), 
            len(N2),
            np.sum(N1),     # mol2.natoms,
            # np.sum(N2),     # mol1.natoms,
            SIGMA
    )

    return K


def get_atomic_local_gradient_kernel(X1, X2, dX2, Q1, Q2, SIGMA):

    N1 = np.zeros((X1.shape[0]), dtype=np.int32)
    N2 = np.zeros((X2.shape[0]), dtype=np.int32)

    
    for i, rep in enumerate(X1):
        N1[i] = sum([any(np.abs(atom) > 0) for atom in rep])
    
    for i, rep in enumerate(X2):
        N2[i] = sum([any(np.abs(atom) > 0) for atom in rep])
    
    Q1_input = np.zeros((max(N1), X1.shape[0]), dtype=np.int32)
    Q2_input = np.zeros((max(N2), X2.shape[0]), dtype=np.int32)
    
    for i, q in enumerate(Q1):
        Q1_input[:len(q),i] = q

    for i, q in enumerate(Q2):
        Q2_input[:len(q),i] = q

    # This kernel must run with MKL_NUM_THREADS=1
    original_mkl_threads = os.environ.get('MKL_NUM_THREADS')
    os.environ["MKL_NUM_THREADS"] = "1"
    
    K = fatomic_local_gradient_kernel(
            X1, 
            X2, 
            dX2,
            Q1_input,
            Q2_input,
            N1,
            N2,
            len(N1), 
            len(N2),
            np.sum(N1),         # mol2.natoms,
            np.sum(N2)*3,         # mol1.natoms,
            SIGMA
    )

    # Reset MKL_NUM_THREADS back to its original value 
    if original_mkl_threads is None:
        del os.environ["MKL_NUM_THREADS"]
    else:
        os.environ["MKL_NUM_THREADS"] = original_mkl_threads

    return K


def get_local_gradient_kernel(X1, X2, dX2, Q1, Q2, SIGMA):

    N1 = np.zeros((X1.shape[0]), dtype=np.int32)
    N2 = np.zeros((X2.shape[0]), dtype=np.int32)
    
    for i, rep in enumerate(X1):
        N1[i] = sum([any(np.abs(atom) > 0) for atom in rep])
    
    for i, rep in enumerate(X2):
        N2[i] = sum([any(np.abs(atom) > 0) for atom in rep])
    
    Q1_input = np.zeros((max(N1), X1.shape[0]), dtype=np.int32)
    Q2_input = np.zeros((max(N2), X2.shape[0]), dtype=np.int32)
    
    for i, q in enumerate(Q1):
        Q1_input[:len(q),i] = q

    for i, q in enumerate(Q2):
        Q2_input[:len(q),i] = q
   
    # This kernel must run with MKL_NUM_THREADS=1
    original_mkl_threads = os.environ.get('MKL_NUM_THREADS')
    os.environ["MKL_NUM_THREADS"] = "1"
    
    K = flocal_gradient_kernel(
            X1, 
            X2, 
            dX2,
            Q1_input,
            Q2_input,
            N1,
            N2,
            len(N1), 
            len(N2),
            np.sum(N1),         # mol2.natoms,
            np.sum(N2)*3,         # mol1.natoms,
            SIGMA
    )

    # Reset MKL_NUM_THREADS back to its original value 
    if original_mkl_threads is None:
        del os.environ["MKL_NUM_THREADS"]
    else:
        os.environ["MKL_NUM_THREADS"] = original_mkl_threads

    return K


def get_gdml_kernel(X1, X2, dX1, dX2, Q1, Q2, SIGMA):

    N1 = np.zeros((X1.shape[0]), dtype=np.int32)
    N2 = np.zeros((X2.shape[0]), dtype=np.int32)

    
    for i, rep in enumerate(X1):
        N1[i] = sum([any(np.abs(atom) > 0) for atom in rep])
    
    for i, rep in enumerate(X2):
        N2[i] = sum([any(np.abs(atom) > 0) for atom in rep])

    Q1_input = np.zeros((max(N1), X1.shape[0]), dtype=np.int32)
    Q2_input = np.zeros((max(N2), X2.shape[0]), dtype=np.int32)
    
    for i, q in enumerate(Q1):
        Q1_input[:len(q),i] = q

    for i, q in enumerate(Q2):
        Q2_input[:len(q),i] = q
   
   
    # This kernel must run with MKL_NUM_THREADS=1
    original_mkl_threads = os.environ.get('MKL_NUM_THREADS')
    os.environ["MKL_NUM_THREADS"] = "1"

    K = fgdml_kernel(
            X1, 
            X2, 
            dX1,
            dX2,
            Q1_input,
            Q2_input,
            N1,
            N2,
            len(N1), 
            len(N2),
            np.sum(N1),         # mol2.natoms,
            np.sum(N2),         # mol1.natoms,
            SIGMA
    )
    
    # Reset MKL_NUM_THREADS back to its original value 
    if original_mkl_threads is None:
        del os.environ["MKL_NUM_THREADS"]
    else:
        os.environ["MKL_NUM_THREADS"] = original_mkl_threads

    return K


def get_symmetric_gdml_kernel(X1, dX1, Q1, SIGMA):

    N1 = np.zeros((X1.shape[0]), dtype=np.int32)

    
    for i, rep in enumerate(X1):
        N1[i] = sum([any(np.abs(atom) > 0) for atom in rep])
    
    Q1_input = np.zeros((max(N1), X1.shape[0]), dtype=np.int32)
    
    for i, q in enumerate(Q1):
        Q1_input[:len(q),i] = q
   
    # This kernel must run with MKL_NUM_THREADS=1
    original_mkl_threads = os.environ.get('MKL_NUM_THREADS')
    os.environ["MKL_NUM_THREADS"] = "1"

    K = fsymmetric_gdml_kernel(
            X1, 
            dX1,
            Q1_input,
            N1,
            len(N1), 
            np.sum(N1),         # mol2.natoms,
            SIGMA
    )
    
    # Reset MKL_NUM_THREADS back to its original value 
    if original_mkl_threads is None:
        del os.environ["MKL_NUM_THREADS"]
    else:
        os.environ["MKL_NUM_THREADS"] = original_mkl_threads

    return K


def get_gp_kernel(X1, X2, dX1, dX2, Q1, Q2, SIGMA):

    N1 = np.zeros((X1.shape[0]), dtype=np.int32)
    N2 = np.zeros((X2.shape[0]), dtype=np.int32)
    
    for i, rep in enumerate(X1):
        N1[i] = sum([any(np.abs(atom) > 0) for atom in rep])
    
    for i, rep in enumerate(X2):
        N2[i] = sum([any(np.abs(atom) > 0) for atom in rep])

    Q1_input = np.zeros((max(N1), X1.shape[0]), dtype=np.int32)
    Q2_input = np.zeros((max(N2), X2.shape[0]), dtype=np.int32)
    
    for i, q in enumerate(Q1):
        Q1_input[:len(q),i] = q

    for i, q in enumerate(Q2):
        Q2_input[:len(q),i] = q
   
    # This kernel must run with MKL_NUM_THREADS=1
    original_mkl_threads = os.environ.get('MKL_NUM_THREADS')
    os.environ["MKL_NUM_THREADS"] = "1"

    K = fgaussian_process_kernel(
            X1, 
            X2, 
            dX1,
            dX2,
            Q1_input,
            Q2_input,
            N1,
            N2,
            len(N1), 
            len(N2),
            np.sum(N1),         # mol2.natoms,
            np.sum(N2),         # mol1.natoms,
            SIGMA
    )
    
    # Reset MKL_NUM_THREADS back to its original value 
    if original_mkl_threads is None:
        del os.environ["MKL_NUM_THREADS"]
    else:
        os.environ["MKL_NUM_THREADS"] = original_mkl_threads

    return K


def get_symmetric_gp_kernel(X1, dX1, Q1, SIGMA):

    N1 = np.zeros((X1.shape[0]), dtype=np.int32)

    
    for i, rep in enumerate(X1):
        N1[i] = sum([any(np.abs(atom) > 0) for atom in rep])
   
    Q1_input = np.zeros((max(N1), X1.shape[0]), dtype=np.int32)
    
    for i, q in enumerate(Q1):
        Q1_input[:len(q),i] = q

    # This kernel must run with MKL_NUM_THREADS=1
    original_mkl_threads = os.environ.get('MKL_NUM_THREADS')
    os.environ["MKL_NUM_THREADS"] = "1"

    K = fsymmetric_gaussian_process_kernel(
            X1, 
            dX1,
            Q1_input,
            N1,
            len(N1), 
            np.sum(N1),         # mol2.natoms,
            SIGMA
    )
    
    # Reset MKL_NUM_THREADS back to its original value 
    if original_mkl_threads is None:
        del os.environ["MKL_NUM_THREADS"]
    else:
        os.environ["MKL_NUM_THREADS"] = original_mkl_threads

    return K
