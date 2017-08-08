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

from .ffchl_kernels import fget_kernels_fchl
from .ffchl_kernels import fget_symmetric_kernels_fchl
from .ffchl_kernels import fget_global_kernels_fchl
from .ffchl_kernels import fget_global_symmetric_kernels_fchl
from .ffchl_kernels import fget_atomic_kernels_fchl
from .ffchl_kernels import fget_atomic_symmetric_kernels_fchl
from .ffchl_kernels import fget_atomic_force_alphas_fchl
from .ffchl_kernels import fget_atomic_force_kernels_fchl

PTP = {\
         1  :[1,1] ,2:  [1,8]#Row1

        ,3  :[2,1] ,4:  [2,2]#Row2\
        ,5  :[2,3] ,6:  [2,4] ,7  :[2,5] ,8  :[2,6] ,9  :[2,7] ,10 :[2,8]\

        ,11 :[3,1] ,12: [3,2]#Row3\
        ,13 :[3,3] ,14: [3,4] ,15 :[3,5] ,16 :[3,6] ,17 :[3,7] ,18 :[3,8]\

        ,19 :[4,1] ,20: [4,2]#Row4\
        ,31 :[4,3] ,32: [4,4] ,33 :[4,5] ,34 :[4,6] ,35 :[4,7] ,36 :[4,8]\
        ,21 :[4,9] ,22: [4,10],23 :[4,11],24 :[4,12],25 :[4,13],26 :[4,14],27 :[4,15],28 :[4,16],29 :[4,17],30 :[4,18]\

        ,37 :[5,1] ,38: [5,2]#Row5\
        ,49 :[5,3] ,50: [5,4] ,51 :[5,5] ,52 :[5,6] ,53 :[5,7] ,54 :[5,8]\
        ,39 :[5,9] ,40: [5,10],41 :[5,11],42 :[5,12],43 :[5,13],44 :[5,14],45 :[5,15],46 :[5,16],47 :[5,17],48 :[5,18]\

        ,55 :[6,1] ,56: [6,2]#Row6\
        ,81 :[6,3] ,82: [6,4] ,83 :[6,5] ,84 :[6,6] ,85 :[6,7] ,86 :[6,8]
               ,72: [6,10],73 :[6,11],74 :[6,12],75 :[6,13],76 :[6,14],77 :[6,15],78 :[6,16],79 :[6,17],80 :[6,18]\
        ,57 :[6,19],58: [6,20],59 :[6,21],60 :[6,22],61 :[6,23],62 :[6,24],63 :[6,25],64 :[6,26],65 :[6,27],66 :[6,28],67 :[6,29],68 :[6,30],69 :[6,31],70 :[6,32],71 :[6,33]\

        ,87 :[7,1] ,88: [7,2]#Row7\
        ,113:[7,3] ,114:[7,4] ,115:[7,5] ,116:[7,6] ,117:[7,7] ,118:[7,8]\
               ,104:[7,10],105:[7,11],106:[7,12],107:[7,13],108:[7,14],109:[7,15],110:[7,16],111:[7,17],112:[7,18]\
        ,89 :[7,19],90: [7,20],91 :[7,21],92 :[7,22],93 :[7,23],94 :[7,24],95 :[7,25],96 :[7,26],97 :[7,27],98 :[7,28],99 :[7,29],100:[7,30],101:[7,31],101:[7,32],102:[7,14],103:[7,33]}

QtNm = {
    #Row1
    1  :[1,0,0,1./2.]
    ,2:  [1,0,0,-1./2.]


    #Row2
    ,3  :[2,0,0,1./2.]
    ,4: [2,0,0,-1./2.]

    ,5  :[2,-1,1,1./2.],   6: [2,0,1,1./2.]  , 7  : [2,1,1,1./2.]
    ,8  : [2,-1,1,-1./2.] ,9: [2,0,1,-1./2.] ,10 :[2,1,1,-1./2.]


    #Row3
    ,11 :[3,0,0,1./2.]
    ,12: [3,0,0,-1./2.]

    ,13 :[3,-1,1,1./2.] ,  14: [3,0,1,1./2.] ,   15 :[3,1,1,1./2.]
    ,16 :[3,-1,1,-1./2.]  ,17 :[3,0,1,-1./2.]  ,18 :[3,1,1,-1./2.]


    #Row3
    ,19 :[4,0,0,1./2.]
    ,20: [4,0,0,-1./2.]

    ,31 :[4,-1,2,1./2.] , 32: [4,0,1,1./2.]  , 33 :[4,1,1,1./2.]
    ,34 :[4,-1,1,-1./2.] ,35 :[4,0,1,-1./2.] ,36 :[4,1,1,-1./2.]

    ,21 :[4,-2,2,1./2.],  22:[4,-1,2,1./2.],  23 :[4,0,2,1./2.], 24 :[4,1,2,1./2.], 25 :[4,2,2,1./2.]
    ,26 :[4,-2,2,-1./2.], 27:[4,-1,2,-1./2.], 28 :[4,0,2,-1./2.],29 :[4,1,2,-1./2.],30 :[4,2,2,-1./2.]


    #Row5
    ,37 :[5,0,0,1./2.]
    ,38: [5,0,0,-1./2.]

    ,49 :[5,-1,1,1./2.] ,  50: [5,0,1,1./2.]  ,  51 :[5,1,1,1./2.]
    ,52 :[5,-1,1,-1./2.]  ,53 :[5,0,1,-1./2.]  ,54 :[5,1,1,-1./2.]


    ,39 :[5,-2,2,1./2.], 40:[5,-1,2,1./2.],   41 :[5,0,2,1./2.], 42 :[5,1,2,1./2.], 43 :[5,2,2,1./2.]
    ,44 :[5,-2,2,-1./2.],45 :[5,-1,2,-1./2.],46 :[5,0,2,-1./2.],47 :[5,1,2,-1./2.],48 :[5,2,2,-1./2.]


    #Row6
    ,55 :[6,0,0,1./2.]
    ,56: [6,0,0,-1./2.]

    ,81 :[6,-1,1,1./2.] ,82: [6,0,1,1./2.] ,83: [6,1,1,1./2.]
    ,84 :[6,-1,1,-1./2.] ,85 :[6,0,1,-1./2.] ,86 :[6,1,1,-1./2.]

    ,71 :[6,-2,2,1./2.], 72: [6,-1,2,1./2.],  73 :[6,0,2,1./2.], 74 :[6,1,2,1./2.], 75 :[6,2,2,1./2.]
    ,76 :[6,-2,2,-1./2.],77 :[6,-1,2,-1./2.],78 :[6,0,2,-1./2.],79 :[6,1,2,-1./2.],80 :[6,2,2,-1./2.]

    ,57 :[6,-3,3,1./2.], 58: [6,-2,3,1./2.],  59 :[6,-1,3,1./2.], 60 :[6,0,3,1./2.],  61 :[6,1,3,1./2.], 62 :[6,2,3,1./2.], 63 :[6,3,3,1./2.]
    ,64 :[6,-3,3,-1./2.],65 :[6,-2,3,-1./2.],66 :[6,-1,3,-1./2.],67 :[6,0,3,-1./2.],68 :[6,1,3,-1./2.],69 :[6,2,3,-1./2.],70 :[6,3,3,-1./2.]


    #Row7
    ,87 :[7,0,0,1./2.]
    ,88: [7,0,0,-1./2.]

    ,113:[7,-1,1,1./2.] , 114:[7,0,1,1./2.] ,  115:[7,1,1,1./2.]
    ,116:[7,-1,1,-1./2.] ,117:[7,0,1,-1./2.] ,118:[7,1,1,-1./2.]

    ,103:[7,-2,2,1./2.], 104:[7,-1,2,1./2.],  105:[7,0,2,1./2.], 106:[7,1,2,1./2.],  107:[7,2,2,1./2.]
    ,108:[7,-2,2,-1./2.],109:[7,-1,2,-1./2.],110:[7,0,2,-1./2.],111:[7,1,2,-1./2.],112:[7,2,2,-1./2.]

    ,89 :[7,-3,3,1./2.], 90: [7,-2,3,1./2.],  91 :[7,-1,3,1./2.], 92 :[7,0,3,1./2.],  93 :[7,1,3,1./2.],  94 :[7,2,3,1./2.],  95 :[7,3,3,1./2.]
    ,96 :[7,-3,3,-1./2.],97 :[7,-2,3,-1./2.],98 :[7,-1,3,-1./2.],99 :[7,0,3,-1./2.],100:[7,1,3,-1./2.],101:[7,2,3,-1./2.],102:[7,3,3,-1./2.]}







def QNum_distance(a,b, n_width, m_width, l_width, s_width):
    """ Calculate stochiometric distance
        a -- nuclear charge of element a
        b -- nuclear charge of element b
        r_width -- sigma in row-direction
        c_width -- sigma in column direction
    """

    na = QtNm[int(a)][0]
    nb = QtNm[int(b)][0]

    ma = QtNm[int(a)][1]
    mb = QtNm[int(b)][1]

    la = QtNm[int(a)][2]
    lb = QtNm[int(b)][2]

    sa = QtNm[int(a)][3]
    sb = QtNm[int(b)][3]

    return  np.exp(-(na - nb)**2/(4 * n_width**2)
                   -(ma - mb)**2/(4 * m_width**2)
                   -(la - lb)**2/(4 * l_width**2)
                   -(sa - sb)**2/(4 * s_width**2))

def gen_QNum_distances(emax=100, n_width = 0.001, m_width = 0.001, l_width = 0.001, s_width = 0.001):
    """ Generate stochiometric ditance matrix
        emax -- Largest element
        r_width -- sigma in row-direction
        c_width -- sigma in column direction
    """

    pd = np.zeros((emax,emax))

    for i in range(emax):
        for j in range(emax):

            pd[i,j] = QNum_distance(i+1, j+1, n_width, m_width, l_width, s_width)

    return pd

def periodic_distance(a, b, r_width, c_width):
    """ Calculate stochiometric distance

        a -- nuclear charge of element a
        b -- nuclear charge of element b
        r_width -- sigma in row-direction
        c_width -- sigma in column direction
    """

    ra = PTP[int(a)][0]
    rb = PTP[int(b)][0]
    ca = PTP[int(a)][1]
    cb = PTP[int(b)][1]

    # return (r_width**2 * c_width**2) / ((r_width**2 + (ra - rb)**2) * (c_width**2 + (ca - cb)**2))

    return np.exp(-(ra - rb)**2/(4 * r_width**2)-(ca - cb)**2/(4 * c_width**2))


def gen_pd(emax=100, r_width=0.001, c_width=0.001):
    """ Generate stochiometric ditance matrix

        emax -- Largest element
        r_width -- sigma in row-direction
        c_width -- sigma in column direction
    """

    pd = np.zeros((emax,emax))

    for i in range(emax):
        for j in range(emax):

            pd[i,j] = periodic_distance(i+1, j+1, r_width, c_width)

    return pd


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
        t_width=np.pi/1.0, d_width=0.2, cut_distance=5.0, \
        r_width=1.0, order=1, c_width=0.5, scale_distance=1.0, scale_angular=0.1,
        n_width = 1.0, m_width = 1.0, l_width = 1.0, s_width = 1.0, alchemy="periodic-table", two_body_power=6.0, three_body_power=3.0):
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

    pd = None
    doalchemy = True
    if alchemy == "periodic-table":
        pd = gen_pd(emax=103, r_width=r_width, c_width=c_width)

    elif alchemy == "quantum-numbers":
        pd = gen_QNum_distances(emax=101, n_width = n_width, m_width = m_width,
                                          l_width = l_width, s_width = s_width)
    elif alchemy == "off":
        pd = np.eye(103)
        doalchemy = False

    else:
        print("ERROR: Unknown alchemy specified:", alchemy)
        exit(1)

    sigmas = np.array(sigmas)
    assert len(sigmas.shape) == 1, "Third argument (sigmas) is not a 1D list/numpy.array!"

    return fget_kernels_fchl(A, B, N1, N2, neighbors1, neighbors2, sigmas, \
                nm1, nm2, nsigmas, t_width, d_width, cut_distance, order, pd, scale_distance, scale_angular, doalchemy, two_body_power, three_body_power)

def get_local_symmetric_kernels_fchl(A, sigmas, \
        t_width=np.pi/1.0, d_width=0.2, cut_distance=5.0, \
        r_width=1.0, order=1, c_width=0.5, scale_distance=1.0, scale_angular=0.1,
        n_width = 1.0, m_width = 1.0, l_width = 1.0, s_width = 1.0, alchemy="periodic-table", two_body_power=6.0, three_body_power=3.0):
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

    pd = None
    doalchemy = True
    if alchemy == "periodic-table":
        pd = gen_pd(emax=103, r_width=r_width, c_width=c_width)

    elif alchemy == "quantum-numbers":
        pd = gen_QNum_distances(emax=101, n_width = n_width, m_width = m_width,
                                          l_width = l_width, s_width = s_width)
    elif alchemy == "off":
        pd = np.eye(103)
        doalchemy = False
    else:
        print("ERROR: Unknown alchemy specified:", alchemy)
        exit(1)


    sigmas = np.array(sigmas)
    assert len(sigmas.shape) == 1, "Second argument (sigmas) is not a 1D list/numpy.array!"

    return fget_symmetric_kernels_fchl(A, N1, neighbors1, sigmas, \
                nm1, nsigmas, t_width, d_width, cut_distance, order, pd, scale_distance, scale_angular, doalchemy, two_body_power, three_body_power)

def get_global_symmetric_kernels_fchl(A, sigmas, \
        t_width=np.pi/1.0, d_width=0.2, cut_distance=5.0, \
        r_width=1.0, order=1, c_width=0.5, scale_distance=1.0, scale_angular=0.1,
        n_width = 1.0, m_width = 1.0, l_width = 1.0, s_width = 1.0, alchemy="periodic-table", two_body_power=6.0, three_body_power=3.0):
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

    pd = None
    doalchemy = True
    if alchemy == "periodic-table":
        pd = gen_pd(emax=103, r_width=r_width, c_width=c_width)

    elif alchemy == "quantum-numbers":
        pd = gen_QNum_distances(emax=101, n_width = n_width, m_width = m_width,
                                          l_width = l_width, s_width = s_width)
    elif alchemy == "off":
        pd = np.eye(103)
        doalchemy = False
    else:
        print("ERROR: Unknown alchemy specified:", alchemy)
        exit(1)


    sigmas = np.array(sigmas)
    assert len(sigmas.shape) == 1, "Second argument (sigmas) is not a 1D list/numpy.array!"

    return fget_global_symmetric_kernels_fchl(A, N1, neighbors1, sigmas, \
                nm1, nsigmas, t_width, d_width, cut_distance, order, pd, scale_distance, scale_angular, doalchemy, two_body_power, three_body_power)

    
def get_global_kernels_fchl(A, B, sigmas, \
        t_width=np.pi/1.0, d_width=0.2, cut_distance=5.0, \
        r_width=1.0, order=1, c_width=0.5, scale_distance=1.0, scale_angular=0.1,
        n_width = 1.0, m_width = 1.0, l_width = 1.0, s_width = 1.0, alchemy="periodic-table", two_body_power=6.0, three_body_power=3.0):
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

    pd = None
    doalchemy = True
    if alchemy == "periodic-table":
        pd = gen_pd(emax=103, r_width=r_width, c_width=c_width)

    elif alchemy == "quantum-numbers":
        pd = gen_QNum_distances(emax=101, n_width = n_width, m_width = m_width,
                                          l_width = l_width, s_width = s_width)
    elif alchemy == "off":
        pd = np.eye(103)
        doalchemy = False
    else:
        print("ERROR: Unknown alchemy specified:", alchemy)
        exit(1)

    sigmas = np.array(sigmas)
    assert len(sigmas.shape) == 1, "Third argument (sigmas) is not a 1D list/numpy.array!"

    return fget_global_kernels_fchl(A, B, N1, N2, neighbors1, neighbors2, sigmas, \
                nm1, nm2, nsigmas, t_width, d_width, cut_distance, order, pd, scale_distance, scale_angular, doalchemy, two_body_power, three_body_power)

    
def get_atomic_kernels_fchl(A, B, sigmas, \
        t_width=np.pi/1.0, d_width=0.2, cut_distance=5.0, \
        r_width=1.0, order=1, c_width=0.5, scale_distance=1.0, scale_angular=0.1,
        n_width = 1.0, m_width = 1.0, l_width = 1.0, s_width = 1.0, alchemy="periodic-table", two_body_power=6.0, three_body_power=3.0):
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

    pd = None
    doalchemy = True
    if alchemy == "periodic-table":
        pd = gen_pd(emax=103, r_width=r_width, c_width=c_width)

    elif alchemy == "quantum-numbers":
        pd = gen_QNum_distances(emax=101, n_width = n_width, m_width = m_width,
                                          l_width = l_width, s_width = s_width)
    elif alchemy == "off":
        pd = np.eye(103)
        doalchemy = False
    else:
        print("ERROR: Unknown alchemy specified:", alchemy)
        exit(1)

    sigmas = np.array(sigmas)
    assert len(sigmas.shape) == 1

    return fget_atomic_kernels_fchl(A, B, neighbors1, neighbors2, sigmas, \
                na1, na2, nsigmas, t_width, d_width, cut_distance, order, pd, scale_distance, scale_angular, doalchemy, two_body_power, three_body_power)

    
def get_atomic_symmetric_kernels_fchl(A, sigmas, \
        t_width=np.pi/1.0, d_width=0.2, cut_distance=5.0, \
        r_width=1.0, order=1, c_width=0.5, scale_distance=1.0, scale_angular=0.1,
        n_width = 1.0, m_width = 1.0, l_width = 1.0, s_width = 1.0, alchemy="periodic-table", two_body_power=6.0, three_body_power=3.0):
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

    pd = None
    doalchemy = True
    if alchemy == "periodic-table":
        pd = gen_pd(emax=103, r_width=r_width, c_width=c_width)

    elif alchemy == "quantum-numbers":
        pd = gen_QNum_distances(emax=101, n_width = n_width, m_width = m_width,
                                          l_width = l_width, s_width = s_width)
    elif alchemy == "off":
        pd = np.eye(103)
        doalchemy = False
    else:
        print("ERROR: Unknown alchemy specified:", alchemy)
        exit(1)

    sigmas = np.array(sigmas)
    assert len(sigmas.shape) == 1, "Second argument (sigmas) is not a 1D list/numpy.array!"

    return fget_atomic_symmetric_kernels_fchl(A, neighbors1, sigmas, \
                na1, nsigmas, t_width, d_width, cut_distance, order, pd, scale_distance, scale_angular, doalchemy, two_body_power, three_body_power)

    
def get_atomic_force_alphas_fchl(A, F, sigmas, \
        t_width=np.pi/1.0, d_width=0.2, cut_distance=5.0, \
        r_width=1.0, order=1, c_width=0.5, scale_distance=1.0, scale_angular=0.1,
        n_width = 1.0, m_width = 1.0, l_width = 1.0, s_width = 1.0, alchemy="periodic-table", two_body_power=6.0, three_body_power=3.0):
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

    pd = None
    doalchemy = True
    if alchemy == "periodic-table":
        pd = gen_pd(emax=103, r_width=r_width, c_width=c_width)

    elif alchemy == "quantum-numbers":
        pd = gen_QNum_distances(emax=101, n_width = n_width, m_width = m_width,
                                          l_width = l_width, s_width = s_width)
    elif alchemy == "off":
        pd = np.eye(103)
        doalchemy = False
    else:
        print("ERROR: Unknown alchemy specified:", alchemy)
        exit(1)

    sigmas = np.array(sigmas)
    assert len(sigmas.shape) == 1, "Second argument (sigmas) is not a 1D list/numpy.array!"

    return fget_atomic_force_alphas_fchl(A, F, neighbors1, sigmas, \
                na1, nsigmas, t_width, d_width, cut_distance, order, pd, scale_distance, scale_angular, doalchemy, two_body_power, three_body_power)

 
def get_atomic_force_kernels_fchl(A, B, sigmas, \
        t_width=np.pi/1.0, d_width=0.2, cut_distance=5.0, \
        r_width=1.0, order=1, c_width=0.5, scale_distance=1.0, scale_angular=0.1,
        n_width = 1.0, m_width = 1.0, l_width = 1.0, s_width = 1.0, alchemy="periodic-table", two_body_power=6.0, three_body_power=3.0):
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

    pd = None
    doalchemy = True
    if alchemy == "periodic-table":
        pd = gen_pd(emax=103, r_width=r_width, c_width=c_width)

    elif alchemy == "quantum-numbers":
        pd = gen_QNum_distances(emax=101, n_width = n_width, m_width = m_width,
                                          l_width = l_width, s_width = s_width)
    elif alchemy == "off":
        pd = np.eye(103)
        doalchemy = False
    else:
        print("ERROR: Unknown alchemy specified:", alchemy)
        exit(1)

    sigmas = np.array(sigmas)
    assert len(sigmas.shape) == 1

    return fget_atomic_force_kernels_fchl(A, B, neighbors1, neighbors2, sigmas, \
                na1, na2, nsigmas, t_width, d_width, cut_distance, order, pd, scale_distance, scale_angular, doalchemy, two_body_power, three_body_power)
