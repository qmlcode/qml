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

import numpy as np
from numpy import empty, asfortranarray, ascontiguousarray, zeros

from .fkernels import fget_vector_kernels_gaussian
from .fkernels import fget_vector_kernels_laplacian

from .arad_kernels import get_atomic_kernels_arad


def get_atomic_kernels_laplacian(mols1, mols2, sigmas):

    n1 = np.array([mol.natoms for mol in mols1], dtype=np.int32)
    n2 = np.array([mol.natoms for mol in mols2], dtype=np.int32)

    amax1 = np.amax(n1)
    amax2 = np.amax(n2)

    nm1 = len(mols1)
    nm2 = len(mols2)

    cmat_size = mols1[0].local_coulomb_matrix.shape[1]

    x1 = np.zeros((nm1,amax1,cmat_size), dtype=np.float64, order="F")
    x2 = np.zeros((nm2,amax2,cmat_size), dtype=np.float64, order="F")

    for imol in range(nm1):
        x1[imol,:n1[imol],:cmat_size] = mols1[imol].local_coulomb_matrix

    for imol in range(nm2):
        x2[imol,:n2[imol],:cmat_size] = mols2[imol].local_coulomb_matrix

    # Reorder for Fortran speed
    x1 = np.swapaxes(x1,0,2)
    x2 = np.swapaxes(x2,0,2)

    nsigmas = len(sigmas)

    sigmas = np.array(sigmas, dtype=np.float64)

    return fget_vector_kernels_laplacian(x1, x2, n1, n2, sigmas, \
        nm1, nm2, nsigmas)


def get_atomic_kernels_gaussian(mols1, mols2, sigmas):

    n1 = np.array([mol.natoms for mol in mols1], dtype=np.int32)
    n2 = np.array([mol.natoms for mol in mols2], dtype=np.int32)

    amax1 = np.amax(n1)
    amax2 = np.amax(n2)

    nm1 = len(mols1)
    nm2 = len(mols2)

    cmat_size = mols1[0].local_coulomb_matrix.shape[1]

    x1 = np.zeros((nm1,amax1,cmat_size), dtype=np.float64, order="F")
    x2 = np.zeros((nm2,amax2,cmat_size), dtype=np.float64, order="F")

    for imol in range(nm1):
        x1[imol,:n1[imol],:cmat_size] = mols1[imol].local_coulomb_matrix

    for imol in range(nm2):
        x2[imol,:n2[imol],:cmat_size] = mols2[imol].local_coulomb_matrix

    # Reorder for Fortran speed
    x1 = np.swapaxes(x1,0,2)
    x2 = np.swapaxes(x2,0,2)

    nsigmas = len(sigmas)

    sigmas = np.array(sigmas, dtype=np.float64)

    return fget_vector_kernels_gaussian(x1, x2, n1, n2, sigmas, \
        nm1, nm2, nsigmas)


def arad_kernels(mols1, mols2, sigmas,
        width=0.2, cut_distance=5.0, r_width=1.0, c_width=0.5):

    amax = mol1[0].arad_descriptor.shape[1]

    nm1 = len(mols1)
    nm2 = len(mols2)

    X1 = np.array([mol.arad_descriptor for mol in mols1]).reshape((nm1,amax,5,amax))
    X2 = np.array([mol.arad_descriptor for mol in mols2]).reshape((nm2,amax,5,amax))

    Z1 = [mol.nuclear_charges for mol in mols1]
    Z2 = [mol.nuclear_charges for mol in mols2]

    K = get_atomic_kernels_arad(X1, X2, Z1, Z2, sigmas, \
        width=width, cut_distance=cut_distance, r_width=r_width, c_width=c_width)

    return K


def arad_symmetric_kernels(mols1, sigmas,
        width=0.2, cut_distance=5.0, r_width=1.0, c_width=0.5):

    amax = mol1[0].arad_descriptor.shape[1]

    nm1 = len(mols1)

    X1 = np.array([mol.arad_descriptor for mol in mols1]).reshape((nm1,amax,5,amax))

    Z1 = [mol.nuclear_charges for mol in mols1]

    K = get_symmetric_atomic_kernels_arad(X1, Z1, sigmas, \
        width=width, cut_distance=cut_distance, r_width=r_width, c_width=c_width)

    return K
