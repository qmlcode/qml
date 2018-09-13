# MIT License
#
# Copyright (c) 2017-2018 Felix Faber and Anders Steen Christensen
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
import copy

from qml.utils.alchemy import get_alchemy
from qml.utils import ELEMENT_NAME

def generate_representation(coordinates, nuclear_charges,
        max_size=23, neighbors=23, cut_distance = 5.0, cell=None):
    """ Generates a representation for the FCHL kernel module.

    :param coordinates: Input coordinates.
    :type coordinates: numpy array
    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param max_size: Max number of atoms in representation.
    :type max_size: integer
    :param neighbors: Max number of atoms within the cut-off around an atom. (For periodic systems)
    :type neighbors: integer
    :param cell: Unit cell vectors. The presence of this keyword argument will generate a periodic representation.
    :type cell: numpy array
    :param cut_distance: Spatial cut-off distance - must be the same as used in the kernel function call.
    :type cut_distance: float
    :return: FCHL representation, shape = (size,5,neighbors).
    :rtype: numpy array
    """

    size = max_size

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
                        ocupationListExt = np.append(ocupationListExt,ocupationList)
                        coordsExt = np.append(coordsExt,coords + i*cell[0,:] + j*cell[1,:] + k*cell[2,:],axis = 0)
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


def generate_displaced_representations(coordinates, nuclear_charges,
                    max_size=23, neighbors=23, cut_distance = 5.0, cell=None, dx=0.005): 
    """ Generates displaced representations for the FCHL kernel module.

    :param coordinates: Input coordinates.
    :type coordinates: numpy array
    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param max_size: Max number of atoms in representation.
    :type max_size: integer
    :param neighbors: Max number of atoms within the cut-off around an atom. (For periodic systems)
    :type neighbors: integer
    :param cell: Unit cell vectors. The presence of this keyword argument will generate a periodic representation.
    :type cell: numpy array
    :param dx: Real-space displacement for numerical derivatives, in units of angstrom.
    :type dx: float
    :param cut_distance: Spatial cut-off distance - must be the same as used in the kernel function call.
    :type cut_distance: float
    :return: FCHL representation, shape = (size,5,neighbors).
    :rtype: numpy array
    """
    size = max_size
    if cell is None:
        neighbors=size
    reps = np.zeros((3,2,size,size,5,neighbors))

    compound_size = len(nuclear_charges)

    for xyz in range(3):

        for i in range(compound_size):
            for idisp, disp in enumerate([-dx, dx]):

                displaced_coordinates = copy.deepcopy(coordinates)
                displaced_coordinates[i,xyz] += disp

                rep = generate_representation(displaced_coordinates, nuclear_charges,
                    max_size=size, neighbors=neighbors, cut_distance=cut_distance, cell=cell)

                reps[xyz,idisp,i,:,:,:] = rep[:,:,:]

    return reps


def generate_displaced_representations_5point(coordinates, nuclear_charges,
                    max_size=23, neighbors=23, cut_distance = 5.0, cell=None, dx=0.005): 
    """ Generates displaced representations for the FCHL kernel module, using a 5-point stencil.

    :param coordinates: Input coordinates.
    :type coordinates: numpy array
    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param max_size: Max number of atoms in representation.
    :type max_size: integer
    :param neighbors: Max number of atoms within the cut-off around an atom. (For periodic systems)
    :type neighbors: integer
    :param cell: Unit cell vectors. The presence of this keyword argument will generate a periodic representation.
    :type cell: numpy array
    :param dx: Real-space displacement for numerical derivatives, in units of angstrom.
    :type dx: float
    :param cut_distance: Spatial cut-off distance - must be the same as used in the kernel function call.
    :type cut_distance: float
    :return: FCHL representation, shape = (size,5,neighbors).
    :rtype: numpy array
    """
    size = max_size
    if cell is None:
        neighbors=size
    reps = np.zeros((3,5,size,size,5,neighbors))

    compound_size = len(nuclear_charges)

    for xyz in range(3):

        for i in range(compound_size):
            for idisp, disp in enumerate([-2*dx, -dx, 0.0, dx, 2*dx]):

                displaced_coordinates = copy.deepcopy(coordinates)
                displaced_coordinates[i,xyz] += disp

                rep = generate_representation(displaced_coordinates, nuclear_charges,
                    max_size=size, neighbors=neighbors, cut_distance=cut_distance, cell=cell)

                reps[xyz,idisp,i,:,:,:] = rep[:,:,:]

    return reps


def generate_representation_electric_field(coordinates, nuclear_charges, 
        fictitious_charges="gasteiger", max_size=23, neighbors=23, cut_distance = 5.0):
    """ Generates a representation for the FCHL kernel module, including fictitious partial charges.
        For use with fist-order electric field-dependent properties, such as dipole moments and IR intensity.
        
        Good choices are charges from e.g. a force field model or a QM charge model, e.g. Mulliken, etc.

    :param coordinates: Input coordinates.
    :type coordinates: numpy array
    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param partial_charges:  Either list of fictitious partial charges, or a string representing the charge model from which these are to be calculated (the latter requires Open Babel with Pybel installed). Default option is "gasteiger". Please see the OpenBabel documentation for list of supported charges models.
    :type partial_charges: numpy array or string.
    :param max_size: Max number of atoms in representation.
    :type max_size: integer
    :param neighbors: Max number of atoms within the cut-off around an atom. (For periodic systems)
    :type neighbors: integer
    :param cell: Unit cell vectors. The presence of this keyword argument will generate a periodic representation.
    :type cell: numpy array
    :param cut_distance: Spatial cut-off distance - must be the same as used in the kernel function call.
    :type cut_distance: float
    :return: FCHL representation, shape = (size,5,neighbors).
    :rtype: numpy array
    """

    partial_charges = None

    # If a list is given, assume these are the fictitious charges
    if isinstance(fictitious_charges, (list,)) or \
       isinstance(fictitious_charges, (np.ndarray,)):
        
        assert(len(fictitious_charges) == len(nuclear_charges)), "Error: incorrect length of fictitious charge list"
        
        partial_charges = fictitious_charges

    # Otherwise, if a string is given, assume this is the name of a charge model
    # in Open Babel//Pybel.
    elif isinstance(fictitious_charges, (basestring,)):

        # Dirty hack for now.
        try:
            import pybel
            import openbabel
        except ImportError:
            print("QML ERROR: Could not generate fictitious charges because OpenBabel/Pybel was not found.")
            exit()

        temp_xyz = "%i\n\n" % len(nuclear_charges)

        for i, nuc in enumerate(nuclear_charges):
            temp_xyz += "%s %f %f %f\n" \
                % (ELEMENT_NAME[nuc], coordinates[i][0],  coordinates[i][1], coordinates[i][2])

        mol = pybel.readstring("xyz", temp_xyz)
        
        this_charge_model = openbabel.OBChargeModel.FindType(fictitious_charges)
        this_charge_model.ComputeCharges(mol.OBMol)

        partial_charges = [atom.partialcharge for atom in mol]

    else:
        print("QML ERROR: Unable to parse argument for fictitious charges", fictitious_charges)
        exit()

    size = max_size
    neighbors=size

    L = len(coordinates)
    coords = np.asarray(coordinates)
    ocupationList = np.asarray(nuclear_charges)
    partial_charges = np.asarray(partial_charges)
    M =  np.zeros((size,6,neighbors))

    coordsExt = copy.copy(coords)
    partialExt = copy.copy(partial_charges)
    ocupationListExt = copy.copy(ocupationList)

    M[:,0,:] = 1E+100

    for i in range(L):
        cD = - coords[i] + coordsExt[:]

        ocExt =  np.asarray(ocupationListExt)
        qExt =  np.asarray(partialExt)
        
        D1 = np.sqrt(np.sum(cD**2, axis = 1))
        args = np.argsort(D1)
        D1 = D1[args]
        
        ocExt = np.asarray([ocExt[l] for l in args])
        qExt = np.asarray([qExt[l] for l in args])

        cD = cD[args]

        args = np.where(D1 < cut_distance)[0]
        D1 = D1[args]
        ocExt = np.asarray([ocExt[l] for l in args])
        qExt = np.asarray([qExt[l] for l in args])

        cD = cD[args]
        M[i,0,: len(D1)] = D1
        M[i,1,: len(D1)] = ocExt[:]
        M[i,2:5,: len(D1)] = cD.T
        M[i,5,: len(D1)] = qExt[:]


    return M
