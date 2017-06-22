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

from .farad_kernels import fget_local_kernels_arad
from .farad_kernels import fget_local_symmetric_kernels_arad

from .farad_kernels import fget_atomic_kernels_arad
from .farad_kernels import fget_atomic_symmetric_kernels_arad

from .data import PTP

def getAngle(sp,norms):
    epsilon = 10.* np.finfo(float).eps
    angles = np.zeros(sp.shape)
    mask1 = np.logical_and(np.abs(sp - norms) > epsilon ,np.abs(norms) > epsilon)
    angles[mask1] = np.arccos(np.clip(sp[mask1]/norms[mask1], -1.0, 1.0))
    return angles


def generate_arad_representation(coordinates, nuclear_charges, size=23, cut_distance=5.0):
    """ Generates a representation for the ARAD kernel module.

    :param coordinates: Input coordinates.
    :type coordinates: numpy array
    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param size: Max number of atoms in representation.
    :type size: integer
    :param cut_distance: Spatial cut-off distance.
    :type cut_distance: float
    :return: ARAD representation, shape = (size,5,size).
    :rtype: numpy array
    """

    # PBC is not supported by the kernels currently
    cell = None
    maxAts = size
    maxMolSize = size

    coords = coordinates
    occupationList = nuclear_charges
    cut = cut_distance

    L = coords.shape[0]
    occupationList = np.asarray(occupationList)
    M = np.zeros((maxMolSize, 5, maxAts))

    if cell is not None:
        coords = np.dot(coords, cell)
        nExtend = (np.floor(cut/np.linalg.norm(cell, 2, axis = 0)) + 1).astype(int)
        for i in range(-nExtend[0], nExtend[0] + 1):
            for j in range(-nExtend[1], nExtend[1] + 1):
                for k in range(-nExtend[2], nExtend[2] + 1):
                    if i == -nExtend[0] and j  == -nExtend[1] and k  == -nExtend[2]:
                        coordsExt = coords + i*cell[0,:] + j*cell[1,:] + k*cell[2,:]
                        occupationListExt = occupationList.copy()
                    else:
                        occupationListExt = np.append(occupationListExt,occupationList)
                        coordsExt = np.append(coordsExt, coords + i*cell[0,:] + j*cell[1,:] + k*cell[2,:], axis = 0)

    else:
        coordsExt = coords.copy()
        occupationListExt = occupationList.copy()

    M[:,0,:] = 1E+100

    for i in range(L):
        #Calculate Distance
        cD = coordsExt[:] - coords[i]
        ocExt = np.asarray([PTP[o] for o in occupationListExt])

        #Obtaining angles
        sp = np.sum(cD[:,np.newaxis] * cD[np.newaxis,:], axis = 2)
        D1 = np.sqrt(np.sum(cD**2, axis = 1))
        D2 = D1[:, np.newaxis] * D1[np.newaxis, :]
        angs = getAngle(sp, D2)

        #Obtaining cos and sine terms
        cosAngs = np.cos(angs) * (1. - np.sin(np.pi * D1[np.newaxis,:]/(2. * cut)))
        sinAngs = np.sin(angs) * (1. - np.sin(np.pi * D1[np.newaxis,:]/(2. * cut)))

        args = np.argsort(D1)

        D1 = D1[args]

        ocExt = np.asarray([ocExt[l] for l in args])

        sub_indices = np.ix_(args, args)
        cosAngs = cosAngs[sub_indices]
        sinAngs = sinAngs[sub_indices]

        args = np.where(D1 < cut)[0]

        D1 = D1[args]

        ocExt = np.asarray([ocExt[l] for l in args])

        sub_indices = np.ix_(args, args)
        cosAngs = cosAngs[sub_indices]
        sinAngs = sinAngs[sub_indices]

        norm = np.sum(1.0 - np.sin(np.pi * D1[np.newaxis, :] / (2.0 * cut)))
        M[i, 0, :len(D1)] = D1
        M[i, 1, :len(D1)] = ocExt[:, 0]
        M[i, 2, :len(D1)] = ocExt[:, 1]
        M[i, 3, :len(D1)] = np.sum(cosAngs,axis = 1) / norm
        M[i, 4, :len(D1)] = np.sum(sinAngs,axis = 1) / norm

    return M


def get_local_kernels_arad(X1, X2, sigmas, 
        width=0.2, cut_distance=5.0, r_width=1.0, c_width=0.5):
    """ Calculates the Gaussian kernel matrix K for atomic ARAD
        descriptors for a list of different sigmas. Each kernel element
        is the sum of all kernel elements between pairs of atoms in two molecules.

        K is calculated using an OpenMP parallel Fortran routine.

        :param X1: ARAD descriptors for molecules in set 1.
        :type X1: numpy array
        :param X2: Array of ARAD descriptors for molecules in set 2.
        :type X2: numpy array
        :param sigmas: List of sigmas for which to calculate the Kernel matrices.
        :type sigmas: list

        :return: The kernel matrices for each sigma - shape (number_sigmas, number_molecules1, number_molecules2)
        :rtype: numpy array
    """

    amax = X1.shape[1]

    assert X1.shape[3] == amax, "ERROR: Check ARAD decriptor sizes! code = 1"
    assert X2.shape[1] == amax, "ERROR: Check ARAD decriptor sizes! code = 2"
    assert X2.shape[3] == amax, "ERROR: Check ARAD decriptor sizes! code = 3"

    nm1 = X1.shape[0]
    nm2 = X2.shape[0]

    N1 = np.empty(nm1, dtype = np.int32)
    Z1_arad = np.zeros((nm1, amax, 2))
    for i in range(nm1):
        N1[i] = len(np.where(X1[i,:,2,0] > 0)[0])
        Z1_arad[i] = X1[i,:,1:3,0]

    N2 = np.empty(nm2, dtype = np.int32)
    Z2_arad = np.zeros((nm2, amax, 2))
    for i in range(nm2):
        N2[i] = len(np.where(X2[i,:,2,0] > 0)[0])
        Z2_arad[i] = X2[i,:,1:3,0]

    sigmas = np.array(sigmas)
    nsigmas = sigmas.size

    return fget_local_kernels_arad(X1, X2, Z1_arad, Z2_arad, N1, N2, sigmas, 
                nm1, nm2, nsigmas, width, cut_distance, r_width, c_width)


def get_local_symmetric_kernels_arad(X1, sigmas, 
    width=0.2, cut_distance=5.0, r_width=1.0, c_width=0.5):
    """ Calculates the Gaussian kernel matrix K for atomic ARAD
        descriptors for a list of different sigmas. Each kernel element
        is the sum of all kernel elements between pairs of atoms in two molecules.

        K is calculated using an OpenMP parallel Fortran routine.

        :param X1: ARAD descriptors for molecules in set 1.
        :type X1: numpy array
        :param sigmas: List of sigmas for which to calculate the Kernel matrices.
        :type sigmas: list

        :return: The kernel matrices for each sigma - shape (number_sigmas, number_molecules1, number_molecules1)
        :rtype: numpy array
    """

    nm1 = X1.shape[0]
    amax = X1.shape[1]

    def getAngle(self, sp, norms):
        angles = np.zeros(sp.shape)
        mask1 = np.logical_and(np.abs(sp - norms) > self.epsilon ,np.abs(norms) > self.epsilon)
        angles[mask1] = np.arccos(sp[mask1]/norms[mask1])
        return angles

    def __init__(self, maxMolSize = 30, maxAts = 30, cut = 5., debug = False):
        self.tag = 'coords'
        self.debug = debug
        self.maxMolSize = maxMolSize
        self.maxAts = maxAts
        self.cut = cut
        self.epsilon = 100.0 * np.finfo(float).eps
        self.PTP = {
            1  :[1,1] ,2:  [1,8] #Row1

            ,3  :[2,1] ,4:  [2,2] #Row2
            ,5  :[2,3] ,6:  [2,4] ,7  :[2,5] ,8  :[2,6] ,9  :[2,7] ,10 :[2,8]\

            ,11 :[3,1] ,12: [3,2] #Row3
            ,13 :[3,3] ,14: [3,4] ,15 :[3,5] ,16 :[3,6] ,17 :[3,7] ,18 :[3,8]\

            ,19 :[4,1] ,20: [4,2] #Row4
            ,31 :[4,3] ,32: [4,4] ,33 :[4,5] ,34 :[4,6] ,35 :[4,7] ,36 :[4,8]\
            ,21 :[4,9] ,22: [4,10],23 :[4,11],24 :[4,12],25 :[4,13],26 :[4,14],27 :[4,15],28 :[4,16],29 :[4,17],30 :[4,18]\

            ,37 :[5,1] ,38: [5,2] #Row5
            ,49 :[5,3] ,50: [5,4] ,51 :[5,5] ,52 :[5,6] ,53 :[5,7] ,54 :[5,8]\
            ,39 :[5,9] ,40: [5,10],41 :[5,11],42 :[5,12],43 :[5,13],44 :[5,14],45 :[5,15],46 :[5,16],47 :[5,17],48 :[5,18]\

            ,55 :[6,1] ,56: [6,2] #Row6
            ,81 :[6,3] ,82: [6,4] ,83 :[6,5] ,84 :[6,6] ,85 :[6,7] ,86 :[6,8]
                ,72: [6,10],73 :[6,11],74 :[6,12],75 :[6,13],76 :[6,14],77 :[6,15],78 :[6,16],79 :[6,17],80 :[6,18]\
            ,57 :[6,19],58: [6,20],59 :[6,21],60 :[6,22],61 :[6,23],62 :[6,24],63 :[6,25],64 :[6,26],65 :[6,27],66 :[6,28],67 :[6,29],68 :[6,30],69 :[6,31],70 :[6,32],71 :[6,33]\

            ,87 :[7,1] ,88: [7,2] #Row7
            ,113:[7,3] ,114:[7,4] ,115:[7,5] ,116:[7,6] ,117:[7,7] ,118:[7,8]\
                ,104:[7,10],105:[7,11],106:[7,12],107:[7,13],108:[7,14],109:[7,15],110:[7,16],111:[7,17],112:[7,18]\
            ,89 :[7,19],90: [7,20],91 :[7,21],92 :[7,22],93 :[7,23],94 :[7,24],95 :[7,25],96 :[7,26],97 :[7,27],98 :[7,28],99 :[7,29],100:[7,30],101:[7,31],101:[7,32],102:[7,14],103:[7,33]}

    def describe(self, coords, occupationList, cell = None):
        L = coords.shape[0]
        occupationList = np.asarray(occupationList)
        M = np.zeros((self.maxMolSize, 5, self.maxAts))

        if cell is not None:
            coords = np.dot(coords, cell)
            nExtend = (np.floor(self.cut/np.linalg.norm(cell, 2, axis = 0)) + 1).astype(int)
            for i in range(-nExtend[0], nExtend[0] + 1):
                for j in range(-nExtend[1], nExtend[1] + 1):
                    for k in range(-nExtend[2], nExtend[2] + 1):
                        if i == -nExtend[0] and j  == -nExtend[1] and k  == -nExtend[2]:
                            coordsExt = coords + i*cell[0,:] + j*cell[1,:] + k*cell[2,:]
                            occupationListExt = occupationList.copy()
                        else:
                            occupationListExt = np.append(occupationListExt,occupationList)
                            coordsExt = np.append(coordsExt, coords + i*cell[0,:] + j*cell[1,:] + k*cell[2,:], axis = 0)

        else:
            coordsExt = coords.copy()
            occupationListExt = occupationList.copy()

    for i in range(na1):
        N1[i] = len(np.where(X1[i,0,:] < cut_distance)[0])
        Z1_arad[i] = X1[i,1:3,0]

    for i in range(na2):
        N2[i] = len(np.where(X2[i,0,:] < cut_distance)[0])
        Z2_arad[i] = X2[i,1:3,0]

    sigmas = np.array(sigmas)
    nsigmas = sigmas.size

    return fget_atomic_kernels_arad(X1, X2, Z1_arad, Z2_arad, N1, N2, sigmas, 
                na1, na2, nsigmas, width, cut_distance, r_width, c_width)

    
def get_atomic_symmetric_kernels_arad(X1, sigmas, 
        width=0.2, cut_distance=5.0, r_width=1.0, c_width=0.5):
    """ Calculates the Gaussian kernel matrix K for atomic ARAD
        descriptors for a list of different sigmas. For atomic properties, e.g. 
        partial charges, chemical shifts, etc.

        K is calculated using an OpenMP parallel Fortran routine.

        :param X1: ARAD descriptors for molecules in set 1. shape=(number_atoms,5,size)
        :type X1: numpy array
        :param sigmas: List of sigmas for which to calculate the Kernel matrices.
        :type sigmas: list

        :return: The kernel matrices for each sigma - shape (number_sigmas, number_atoms1, number_atoms1)
        :rtype: numpy array
    """

    assert len(X1.shape) == 3
    na1 = X1.shape[0]

    N1 = np.empty(na1, dtype = np.int32)
    Z1_arad = np.zeros((na1, 2))

    for i in range(na1):
        N1[i] = len(np.where(X1[i,0,:] < cut_distance)[0])
        Z1_arad[i] = X1[i,1:3,0]

    sigmas = np.array(sigmas)
    nsigmas = sigmas.size

    return fget_atomic_symmetric_kernels_arad(X1, Z1_arad, N1, sigmas, 
                na1, nsigmas, width, cut_distance, r_width, c_width)
