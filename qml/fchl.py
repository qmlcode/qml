# MIT License
#
# Copyright (c) 2017 Felix Faber and Anders Steen Christensen
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
import copy

from .ffchl_module import fget_kernels_fchl
from .ffchl_module import fget_symmetric_kernels_fchl
from .ffchl_module import fget_global_kernels_fchl
from .ffchl_module import fget_global_symmetric_kernels_fchl
from .ffchl_module import fget_atomic_kernels_fchl
from .ffchl_module import fget_atomic_symmetric_kernels_fchl

from .alchemy import get_alchemy

def generate_fchl_representation(coordinates, nuclear_charges,
        size=23, neighbors=23, cut_distance = 5.0, cell=None):
    """ Generates a representation for the FCHL kernel module.

    :param coordinates: Input coordinates.
    :type coordinates: numpy array
    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param size: Max number of atoms in representation.
    :type size: integer
    :param neighbors: Max number of atoms within the cut-off around an atom. (For periodic systems)
    :type neighbors: integer
    :param cell: Unit cell vectors. The presence of this keyword argument will generate a periodic representation.
    :type cell: numpy array
    :param cut_distance: Spatial cut-off distance.
    :type cut_distance: float
    :return: FCHL representation, shape = (size,5,neighbors).
    :rtype: numpy array
    """

    if cell is None:
        neighbors=size

    L = len(coordinates)
    coords = np.asarray(coordinates)
    ocupationList = np.asarray(nuclear_charges)
    M =  np.zeros((size,5,neighbors))

    if cell is not None:
        coords = np.dot(coords,cell)
        nExtend = (np.floor(cut_distance/np.linalg.norm(cell,2,axis = 0)) + 1).astype(int)

        for i in range(-nExtend[0],nExtend[0] + 1):
            for j in range(-nExtend[1],nExtend[1] + 1):
                for k in range(-nExtend[2],nExtend[2] + 1):
                    if i == -nExtend[0] and j  == -nExtend[1] and k  == -nExtend[2]:
                        coordsExt = coords + i*cell[0,:] + j*cell[1,:] + k*cell[2,:]
                        ocupationListExt = copy.copy(ocupationList)
                    else:
                        ocupationListExt = append(ocupationListExt,ocupationList)
                        coordsExt = append(coordsExt,coords + i*cell[0,:] + j*cell[1,:] + k*cell[2,:],axis = 0)
    else:
        coordsExt = copy.copy(coords)
        ocupationListExt = copy.copy(ocupationList)

    M[:,0,:] = 1E+100

    for i in range(L):
        cD = - coords[i] + coordsExt[:]

        ocExt =  np.asarray(ocupationListExt)
        D1 = np.sqrt(np.sum(cD**2, axis = 1))
        args = np.argsort(D1)
        D1 = D1[args]
        ocExt = np.asarray([ocExt[l] for l in args])
        cD = cD[args]

        args = np.where(D1 < cut_distance)[0]
        D1 = D1[args]
        ocExt = np.asarray([ocExt[l] for l in args])
        cD = cD[args]
        M[i,0,: len(D1)] = D1
        M[i,1,: len(D1)] = ocExt[:]
        M[i,2:5,: len(D1)] = cD.T
    return M


def get_local_kernels_fchl(A, B, sigmas, \
        t_width=np.pi/1.0, d_width=0.2, cut_start=1.0, cut_distance=5.0, \
        r_width=1.0, order=1, c_width=0.5, scale_distance=1.0, scale_angular=0.1,
        n_width = 1.0, m_width = 1.0, l_width = 1.0, s_width = 1.0, alchemy="periodic-table", two_body_power=6.0, three_body_power=3.0, elemental_vectors=None):
    """ Calculates the Gaussian kernel matrix K, where :math:`K_{ij}`:

            :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - B_j\\|_2^2}{2\sigma^2} \\big)`

        Where :math:`A_{i}` and :math:`B_{j}` are FCHL representation vectors.
        K is calculated analytically using an OpenMP parallel Fortran routine.
        Note, that this kernel will ONLY work with FCHL representations as input.

        :param A: Array of FCHL representation - shape=(N, maxsize, 5, maxneighbors).
        :type A: numpy array
        :param B: Array of FCHL representation - shape=(M, maxsize, 5, maxneighbors).
        :type B: numpy array
        :param sigma: List of kernel-widths.
        :type sigma: list
        :param t_width: Gaussian width for the angular (theta) terms.
        :type t_width: float
        :param d_width: Gaussian width for the distance terms.
        :type d_width: float
        :param cut_start: The fraction of the cut-off radius at which cut-off damping start
        :type cut_start: float
        :param cut_distance: Cut-off radius.
        :type cut_distance: float
        :param r_width: Gaussian width along rows in the periodic table.
        :type r_width: float
        :param c_width: Gaussian width along columns in the periodic table.
        :type c_width: float
        :param order: Fourier-expansion truncation order.
        :type order: integer
        :param scale_distance: Weight for distance-dependent terms.
        :type scale_distance: float
        :param scale_angular: Weight for angle-dependent terms.
        :type scale_angular: float

        :return: Array of FCHL kernel matrices matrix - shape=(n_sigmas, N, M),
        :rtype: numpy array
    """

    atoms_max = A.shape[1]
    neighbors_max = A.shape[3]

    assert B.shape[1] == atoms_max, "ERROR: Check FCHL representation sizes! code = 2"
    assert B.shape[3] == neighbors_max, "ERROR: Check FCHL representation sizes! code = 3"

    nm1 = A.shape[0]
    nm2 = B.shape[0]

    N1 = np.zeros((nm1),dtype=np.int32)
    N2 = np.zeros((nm2),dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a,:,1,0] > 0.0001)[0])

    for a in range(nm2):
        N2[a] = len(np.where(B[a,:,1,0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)
    neighbors2 = np.zeros((nm2, atoms_max), dtype=np.int32)

    for a, representation in enumerate(A):
        ni = N1[a]
        for i, x in enumerate(representation[:ni]):
            neighbors1[a,i] = len(np.where(x[0]< cut_distance)[0])

    for a, representation in enumerate(B):
        ni = N2[a]
        for i, x in enumerate(representation[:ni]):
            neighbors2[a,i] = len(np.where(x[0]< cut_distance)[0])

    nsigmas = len(sigmas)

    doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=r_width, c_width=c_width, 
        n_width = n_width, m_width = m_width, l_width = l_width, s_width = s_width,
        elemental_vectors=elemental_vectors)

    sigmas = np.array(sigmas)
    assert len(sigmas.shape) == 1, "Third argument (sigmas) is not a 1D list/numpy.array!"

    return fget_kernels_fchl(A, B, N1, N2, neighbors1, neighbors2, sigmas, \
                nm1, nm2, nsigmas, t_width, d_width, cut_start, cut_distance, order, pd, scale_distance, scale_angular, doalchemy, two_body_power, three_body_power)

def get_local_symmetric_kernels_fchl(A, sigmas, \
        t_width=np.pi/1.0, d_width=0.2, cut_start=1.0, cut_distance=5.0, \
        r_width=1.0, order=1, c_width=0.5, scale_distance=1.0, scale_angular=0.1,
        n_width = 1.0, m_width = 1.0, l_width = 1.0, s_width = 1.0, alchemy="periodic-table", two_body_power=6.0, three_body_power=3.0, elemental_vectors=None):
    """ Calculates the Gaussian kernel matrix K, where :math:`K_{ij}`:

            :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - A_j\\|_2^2}{2\sigma^2} \\big)`

        Where :math:`A_{i}` and :math:`A_{j}` are FCHL representation vectors.
        K is calculated analytically using an OpenMP parallel Fortran routine.
        Note, that this kernel will ONLY work with FCHL representations as input.

        :param A: Array of FCHL representation - shape=(N, maxsize, 5, maxneighbors).
        :type A: numpy array
        :param sigma: List of kernel-widths.
        :type sigma: list
        :param t_width: Gaussian width for the angular (theta) terms.
        :type t_width: float
        :param d_width: Gaussian width for the distance terms.
        :type d_width: float
        :param cut_start: The fraction of the cut-off radius at which cut-off damping start
        :type cut_start: float
        :param cut_distance: Cut-off radius.
        :type cut_distance: float
        :param r_width: Gaussian width along rows in the periodic table.
        :type r_width: float
        :param c_width: Gaussian width along columns in the periodic table.
        :type c_width: float
        :param order: Fourier-expansion truncation order.
        :type order: integer
        :param scale_distance: Weight for distance-dependent terms.
        :type scale_distance: float
        :param scale_angular: Weight for angle-dependent terms.
        :type scale_angular: float

        :return: Array of FCHL kernel matrices matrix - shape=(n_sigmas, N, N),
        :rtype: numpy array
    """

    atoms_max = A.shape[1]
    neighbors_max = A.shape[3]

    nm1 = A.shape[0]
    N1 = np.zeros((nm1),dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a,:,1,0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)

    for a, representation in enumerate(A):
        ni = N1[a]
        for i, x in enumerate(representation[:ni]):
            neighbors1[a,i] = len(np.where(x[0]< cut_distance)[0])

    nsigmas = len(sigmas)

    doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=r_width, c_width=c_width, 
        n_width = n_width, m_width = m_width, l_width = l_width, s_width = s_width,
        elemental_vectors=elemental_vectors)

    sigmas = np.array(sigmas)
    assert len(sigmas.shape) == 1, "Second argument (sigmas) is not a 1D list/numpy.array!"

    return fget_symmetric_kernels_fchl(A, N1, neighbors1, sigmas, \
                nm1, nsigmas, t_width, d_width, cut_start, cut_distance, order, pd, scale_distance, scale_angular, doalchemy, two_body_power, three_body_power)

def get_global_symmetric_kernels_fchl(A, sigmas, \
        t_width=np.pi/1.0, d_width=0.2, cut_start=1.0, cut_distance=5.0, \
        r_width=1.0, order=1, c_width=0.5, scale_distance=1.0, scale_angular=0.1,
        n_width = 1.0, m_width = 1.0, l_width = 1.0, s_width = 1.0, alchemy="periodic-table", two_body_power=6.0, three_body_power=3.0, elemental_vectors=None):
    """ Calculates the Gaussian kernel matrix K, where :math:`K_{ij}`:

            :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - A_j\\|_2^2}{2\sigma^2} \\big)`

        Where :math:`A_{i}` and :math:`A_{j}` are FCHL representation vectors.
        K is calculated analytically using an OpenMP parallel Fortran routine.
        Note, that this kernel will ONLY work with FCHL representations as input.

        :param A: Array of FCHL representation - shape=(N, maxsize, 5, maxneighbors).
        :type A: numpy array
        :param sigma: List of kernel-widths.
        :type sigma: list
        :param t_width: Gaussian width for the angular (theta) terms.
        :type t_width: float
        :param d_width: Gaussian width for the distance terms.
        :type d_width: float
        :param cut_start: The fraction of the cut-off radius at which cut-off damping start
        :type cut_start: float
        :param cut_distance: Cut-off radius.
        :type cut_distance: float
        :param r_width: Gaussian width along rows in the periodic table.
        :type r_width: float
        :param c_width: Gaussian width along columns in the periodic table.
        :type c_width: float
        :param order: Fourier-expansion truncation order.
        :type order: integer
        :param scale_distance: Weight for distance-dependent terms.
        :type scale_distance: float
        :param scale_angular: Weight for angle-dependent terms.
        :type scale_angular: float

        :return: Array of FCHL kernel matrices matrix - shape=(n_sigmas, N, N),
        :rtype: numpy array
    """

    atoms_max = A.shape[1]
    neighbors_max = A.shape[3]

    nm1 = A.shape[0]
    N1 = np.zeros((nm1),dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a,:,1,0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)

    for a, representation in enumerate(A):
        ni = N1[a]
        for i, x in enumerate(representation[:ni]):
            neighbors1[a,i] = len(np.where(x[0]< cut_distance)[0])

    nsigmas = len(sigmas)

    doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=r_width, c_width=c_width, 
        n_width = n_width, m_width = m_width, l_width = l_width, s_width = s_width,
        elemental_vectors=elemental_vectors)

    sigmas = np.array(sigmas)
    assert len(sigmas.shape) == 1, "Second argument (sigmas) is not a 1D list/numpy.array!"

    return fget_global_symmetric_kernels_fchl(A, N1, neighbors1, sigmas, \
                nm1, nsigmas, t_width, d_width, cut_start, cut_distance, order, pd, scale_distance, scale_angular, doalchemy, two_body_power, three_body_power)

    
def get_global_kernels_fchl(A, B, sigmas, \
        t_width=np.pi/1.0, d_width=0.2, cut_start=1.0, cut_distance=5.0, \
        r_width=1.0, order=1, c_width=0.5, scale_distance=1.0, scale_angular=0.1,
        n_width = 1.0, m_width = 1.0, l_width = 1.0, s_width = 1.0, alchemy="periodic-table", two_body_power=6.0, three_body_power=3.0, elemental_vectors=None):
    """ Calculates the Gaussian kernel matrix K, where :math:`K_{ij}`:

            :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - B_j\\|_2^2}{2\sigma^2} \\big)`

        Where :math:`A_{i}` and :math:`B_{j}` are FCHL representation vectors.
        K is calculated analytically using an OpenMP parallel Fortran routine.
        Note, that this kernel will ONLY work with FCHL representations as input.

        :param A: Array of FCHL representation - shape=(N, maxsize, 5, maxneighbors).
        :type A: numpy array
        :param B: Array of FCHL representation - shape=(M, maxsize, 5, maxneighbors).
        :type B: numpy array
        :param sigma: List of kernel-widths.
        :type sigma: list
        :param t_width: Gaussian width for the angular (theta) terms.
        :type t_width: float
        :param d_width: Gaussian width for the distance terms.
        :type d_width: float
        :param cut_start: The fraction of the cut-off radius at which cut-off damping start
        :type cut_start: float
        :param cut_start: The fraction of the cut-off radius at which cut-off damping start
        :type cut_start: float
        :param cut_distance: Cut-off radius.
        :type cut_distance: float
        :param r_width: Gaussian width along rows in the periodic table.
        :type r_width: float
        :param c_width: Gaussian width along columns in the periodic table.
        :type c_width: float
        :param order: Fourier-expansion truncation order.
        :type order: integer
        :param scale_distance: Weight for distance-dependent terms.
        :type scale_distance: float
        :param scale_angular: Weight for angle-dependent terms.
        :type scale_angular: float

        :return: Array of FCHL kernel matrices matrix - shape=(n_sigmas, N, M),
        :rtype: numpy array
    """

    atoms_max = A.shape[1]
    neighbors_max = A.shape[3]

    assert B.shape[1] == atoms_max, "ERROR: Check FCHL representation sizes!"
    assert B.shape[3] == neighbors_max, "ERROR: Check FCHL representation sizes!"

    nm1 = A.shape[0]
    nm2 = B.shape[0]

    N1 = np.zeros((nm1),dtype=np.int32)
    N2 = np.zeros((nm2),dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a,:,1,0] > 0.0001)[0])

    for a in range(nm2):
        N2[a] = len(np.where(B[a,:,1,0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)
    neighbors2 = np.zeros((nm2, atoms_max), dtype=np.int32)

    for a, representation in enumerate(A):
        ni = N1[a]
        for i, x in enumerate(representation[:ni]):
            neighbors1[a,i] = len(np.where(x[0]< cut_distance)[0])

    for a, representation in enumerate(B):
        ni = N2[a]
        for i, x in enumerate(representation[:ni]):
            neighbors2[a,i] = len(np.where(x[0]< cut_distance)[0])

    nsigmas = len(sigmas)

    doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=r_width, c_width=c_width, 
        n_width = n_width, m_width = m_width, l_width = l_width, s_width = s_width,
        elemental_vectors=elemental_vectors)

    sigmas = np.array(sigmas)
    assert len(sigmas.shape) == 1, "Third argument (sigmas) is not a 1D list/numpy.array!"

    return fget_global_kernels_fchl(A, B, N1, N2, neighbors1, neighbors2, sigmas, \
                nm1, nm2, nsigmas, t_width, d_width, cut_start, cut_distance, order, pd, scale_distance, scale_angular, doalchemy, two_body_power, three_body_power)

    
def get_atomic_kernels_fchl(A, B, sigmas, \
        t_width=np.pi/1.0, d_width=0.2, cut_start=1.0, cut_distance=5.0, \
        r_width=1.0, order=1, c_width=0.5, scale_distance=1.0, scale_angular=0.1,
        n_width = 1.0, m_width = 1.0, l_width = 1.0, s_width = 1.0, alchemy="periodic-table", two_body_power=6.0, three_body_power=3.0, elemental_vectors=None):
    """ Calculates the Gaussian kernel matrix K, where :math:`K_{ij}`:

            :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - B_j\\|_2^2}{2\sigma^2} \\big)`

        Where :math:`A_{i}` and :math:`B_{j}` are FCHL representation vectors.
        K is calculated analytically using an OpenMP parallel Fortran routine.
        Note, that this kernel will ONLY work with FCHL representations as input.

        :param A: Array of FCHL representation - shape=(N, maxsize, 5, size).
        :type A: numpy array
        :param B: Array of FCHL representation - shape=(M, maxsize, 5, size).
        :type B: numpy array
        :param sigma: List of kernel-widths.
        :type sigma: list
        :param t_width: Gaussian width for the angular (theta) terms.
        :type t_width: float
        :param d_width: Gaussian width for the distance terms.
        :type d_width: float
        :param cut_start: The fraction of the cut-off radius at which cut-off damping start
        :type cut_start: float
        :param cut_distance: Cut-off radius.
        :type cut_distance: float
        :param r_width: Gaussian width along rows in the periodic table.
        :type r_width: float
        :param c_width: Gaussian width along columns in the periodic table.
        :type c_width: float
        :param order: Fourier-expansion truncation order.
        :type order: integer
        :param scale_distance: Weight for distance-dependent terms.
        :type scale_distance: float
        :param scale_angular: Weight for angle-dependent terms.
        :type scale_angular: float

        :return: Array of FCHL kernel matrices matrix - shape=(n_sigmas, N, M),
        :rtype: numpy array
    """

    assert len(A.shape) == 3
    assert len(B.shape) == 3

    # assert B.shape[1] == atoms_max, "ERROR: Check FCHL representation sizes! code = 2"
    # assert B.shape[3] == neighbors_max, "ERROR: Check FCHL representation sizes! code = 3"

    na1 = A.shape[0]
    na2 = B.shape[0]

    neighbors1 = np.zeros((na1), dtype=np.int32)
    neighbors2 = np.zeros((na2), dtype=np.int32)

    for i, x in enumerate(A):
        neighbors1[i] = len(np.where(x[0]< cut_distance)[0])

    for i, x in enumerate(B):
        neighbors2[i] = len(np.where(x[0]< cut_distance)[0])

    nsigmas = len(sigmas)

    doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=r_width, c_width=c_width, 
        n_width = n_width, m_width = m_width, l_width = l_width, s_width = s_width,
        elemental_vectors=elemental_vectors)

    sigmas = np.array(sigmas)
    assert len(sigmas.shape) == 1

    return fget_atomic_kernels_fchl(A, B, neighbors1, neighbors2, sigmas, \
                na1, na2, nsigmas, t_width, d_width, cut_start, cut_distance, order, pd, scale_distance, scale_angular, doalchemy, two_body_power, three_body_power)

    
def get_atomic_symmetric_kernels_fchl(A, sigmas, \
        t_width=np.pi/1.0, d_width=0.2, cut_start=1.0, cut_distance=5.0, \
        r_width=1.0, order=1, c_width=0.5, scale_distance=1.0, scale_angular=0.1,
        n_width = 1.0, m_width = 1.0, l_width = 1.0, s_width = 1.0, alchemy="periodic-table",
        two_body_power=6.0, three_body_power=3.0, elemental_vectors=None):
    """ Calculates the Gaussian kernel matrix K, where :math:`K_{ij}`:

            :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - B_j\\|_2^2}{2\sigma^2} \\big)`

        Where :math:`A_{i}` and :math:`B_{j}` are FCHL representation vectors.
        K is calculated analytically using an OpenMP parallel Fortran routine.
        Note, that this kernel will ONLY work with FCHL representations as input.

        :param A: Array of FCHL representation - shape=(N, maxsize, 5, size).
        :type A: numpy array
        :param sigma: List of kernel-widths.
        :type sigma: list
        :param t_width: Gaussian width for the angular (theta) terms.
        :type t_width: float
        :param d_width: Gaussian width for the distance terms.
        :type d_width: float
        :param cut_start: The fraction of the cut-off radius at which cut-off damping start
        :type cut_start: float
        :param cut_distance: Cut-off radius.
        :type cut_distance: float
        :param r_width: Gaussian width along rows in the periodic table.
        :type r_width: float
        :param c_width: Gaussian width along columns in the periodic table.
        :type c_width: float
        :param order: Fourier-expansion truncation order.
        :type order: integer
        :param scale_distance: Weight for distance-dependent terms.
        :type scale_distance: float
        :param scale_angular: Weight for angle-dependent terms.
        :type scale_angular: float

        :return: Array of FCHL kernel matrices matrix - shape=(n_sigmas, N, M),
        :rtype: numpy array
    """

    assert len(A.shape) == 3

    na1 = A.shape[0]

    neighbors1 = np.zeros((na1), dtype=np.int32)

    for i, x in enumerate(A):
        neighbors1[i] = len(np.where(x[0]< cut_distance)[0])

    nsigmas = len(sigmas)

    doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=r_width, c_width=c_width, 
        n_width = n_width, m_width = m_width, l_width = l_width, s_width = s_width,
        elemental_vectors=elemental_vectors)

    sigmas = np.array(sigmas)
    assert len(sigmas.shape) == 1, "Second argument (sigmas) is not a 1D list/numpy.array!"

    return fget_atomic_symmetric_kernels_fchl(A, neighbors1, sigmas, \
                na1, nsigmas, t_width, d_width, cut_start, cut_distance, order, pd, scale_distance, scale_angular, doalchemy, two_body_power, three_body_power)
