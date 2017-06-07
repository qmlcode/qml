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

from .farad_kernels import fget_kernels_arad
from .farad_kernels import fget_symmetric_kernels_arad

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


def get_atomic_kernels_arad(X1, X2, Z1, Z2, sigmas, \
        width=0.2, cut_distance=5.0, r_width=1.0, c_width=0.5):
    """ Calculates the Gaussian kernel matrix K for atomic ARAD
        descriptors for a list of different sigmas.

        K is calculated using an OpenMP parallel Fortran routine.

        Arguments:
        ==============
        X1 -- np.array of ARAD descriptors for molecules in set 1.
        X2 -- np.array of ARAD descriptors for molecules in set 2.
        Z1 -- List of lists of nuclear charges for molecules in set 1.
        Z2 -- List of lists of nuclear charges for molecules in set 2.
        sigmas -- List of sigma for which to calculate the Kernel matrices.

        Returns:
        ==============
        K -- The kernel matrices for each sigma (3D-array, Ns x N1 x N2)
    """

    amax = X1.shape[1]

    assert X1.shape[3] == amax, "ERROR: Check ARAD decriptor sizes! code = 1"
    assert X2.shape[1] == amax, "ERROR: Check ARAD decriptor sizes! code = 2"
    assert X2.shape[3] == amax, "ERROR: Check ARAD decriptor sizes! code = 3"

    nm1 = len(Z1)
    nm2 = len(Z2)

    assert X1.shape[0] == nm1,  "ERROR: Check ARAD decriptor sizes! code = 4"
    assert X2.shape[0] == nm2,  "ERROR: Check ARAD decriptor sizes! code = 5"

    N1 = []
    for Z in Z1:
        N1.append(len(Z))

    N2 = []
    for Z in Z2:
        N2.append(len(Z))

    N1 = np.array(N1,dtype=np.int32)
    N2 = np.array(N2,dtype=np.int32)

    nsigmas = len(sigmas)
    
    c1 = []
    for charges in Z1:
        c1.append(np.array([PTP[int(q)] for q in charges], dtype=np.int32))

    Z1_arad = np.zeros((nm1,amax,2))

    for i in range(nm1):
        for j, z in enumerate(c1[i]):
            Z1_arad[i,j] = z

    c2 = []
    for charges in Z2:
        c2.append(np.array([PTP[int(q)] for q in charges], dtype=np.int32))

    Z2_arad = np.zeros((nm2,amax,2))

    for i in range(nm2):
        for j, z in enumerate(c2[i]):
            Z2_arad[i,j] = z

    sigmas = np.array(sigmas)

    return fget_kernels_arad(X1, X2, Z1_arad, Z2_arad, N1, N2, sigmas, \
                nm1, nm2, nsigmas, width, cut_distance, r_width, c_width)


def get_atomic_symmetric_kernels_arad(X1, Z1, sigmas, \
        width=0.2, cut_distance=5.0, r_width=1.0, c_width=0.5):
    """ Calculates the Gaussian kernel matrix K for atomic ARAD
        descriptors for a list of different sigmas.

        K is calculated using an OpenMP parallel Fortran routine.

        Arguments:
        ==============
        X1 -- np.array of ARAD descriptors for molecules in set 1.
        Z1 -- List of lists of nuclear charges for molecules in set 1.
        sigmas -- List of sigma for which to calculate the Kernel matrices.

        Returns:
        ==============
        K -- The kernel matrices for each sigma (3D-array, Ns x N1 x N2)
    """

    amax = X1.shape[1]

    assert X1.shape[3] == amax, "ERROR: Check ARAD decriptor sizes! code = 1"

    nm1 = len(Z1)

    assert X1.shape[0] == nm1,  "ERROR: Check ARAD decriptor sizes! code = 4"

    N1 = []
    for Z in Z1:
        N1.append(len(Z))

    N1 = np.array(N1,dtype=np.int32)

    nsigmas = len(sigmas)
    
    c1 = []
    for charges in Z1:
        c1.append(np.array([PTP[int(q)] for q in charges], dtype=np.int32))

    Z1_arad = np.zeros((nm1,amax,2))

    for i in range(nm1):
        for j, z in enumerate(c1[i]):
            Z1_arad[i,j] = z

    sigmas = np.array(sigmas)

    return fget_symmetric_kernels_arad(X1, Z1_arad, N1, sigmas, \
                nm1, nsigmas, width, cut_distance, r_width, c_width)
