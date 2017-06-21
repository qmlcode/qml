from __future__ import print_function

import sys
import time
import numpy as np
import qml
from qml.kernels import laplacian_kernel
from qml.math import cho_solve
PTP = {
         1  :[1,1] ,2:  [1,8] #Row1

        ,3  :[2,1] ,4:  [2,2] #Row2
        ,5  :[2,3] ,6:  [2,4] ,7  :[2,5] ,8  :[2,6] ,9  :[2,7] ,10 :[2,8]

        ,11 :[3,1] ,12: [3,2] #Row3
        ,13 :[3,3] ,14: [3,4] ,15 :[3,5] ,16 :[3,6] ,17 :[3,7] ,18 :[3,8]

        ,19 :[4,1] ,20: [4,2] #Row4
        ,31 :[4,3] ,32: [4,4] ,33 :[4,5] ,34 :[4,6] ,35 :[4,7] ,36 :[4,8]
        ,21 :[4,9] ,22: [4,10],23 :[4,11],24 :[4,12],25 :[4,13],26 :[4,14],27 :[4,15],28 :[4,16],29 :[4,17],30 :[4,18]

        ,37 :[5,1] ,38: [5,2] #Row5
        ,49 :[5,3] ,50: [5,4] ,51 :[5,5] ,52 :[5,6] ,53 :[5,7] ,54 :[5,8]
        ,39 :[5,9] ,40: [5,10],41 :[5,11],42 :[5,12],43 :[5,13],44 :[5,14],45 :[5,15],46 :[5,16],47 :[5,17],48 :[5,18]

        ,55 :[6,1] ,56: [6,2] #Row6
        ,81 :[6,3] ,82: [6,4] ,83 :[6,5] ,84 :[6,6] ,85 :[6,7] ,86 :[6,8]
               ,72: [6,10],73 :[6,11],74 :[6,12],75 :[6,13],76 :[6,14],77 :[6,15],78 :[6,16],79 :[6,17],80 :[6,18]
        ,57 :[6,19],58: [6,20],59 :[6,21],60 :[6,22],61 :[6,23],62 :[6,24],63 :[6,25],64 :[6,26],65 :[6,27],66 :[6,28],67 :[6,29],68 :[6,30],69 :[6,31],70 :[6,32],71 :[6,33]

        ,87 :[7,1] ,88: [7,2] #Row7
        ,113:[7,3] ,114:[7,4] ,115:[7,5] ,116:[7,6] ,117:[7,7] ,118:[7,8]
               ,104:[7,10],105:[7,11],106:[7,12],107:[7,13],108:[7,14],109:[7,15],110:[7,16],111:[7,17],112:[7,18]
        ,89 :[7,19],90: [7,20],91 :[7,21],92 :[7,22],93 :[7,23],94 :[7,24],95 :[7,25],96 :[7,26],97 :[7,27],98 :[7,28],99 :[7,29],100:[7,30],101:[7,31],101:[7,32],102:[7,14],103:[7,33]}

def getAngle(sp, norms):
    epsilon = 100.0 * np.finfo(float).eps
    angles = np.zeros(sp.shape)
    mask1 = np.logical_and(np.abs(sp - norms) > epsilon ,np.abs(norms) > epsilon)
    angles[mask1] = np.arccos(sp[mask1]/norms[mask1])
    return angles

def generate_arad_representation(coordinates, nuclear_charges,
        cell=None, size = 23, neighbors = 23, cut = 5., debug = False):

    coords = coordinates
    occupationList = nuclear_charges

    L = coords.shape[0]
    maxAts = size
    maxMolSize = neighbors
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

def get_energies(filename):
    """ Returns a dictionary with heats of formation for each xyz-file.
    """

    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    energies = dict()

    for line in lines:
        tokens = line.split()

        xyz_name = tokens[0]
        hof = float(tokens[1])

        energies[xyz_name] = hof

    return energies

if __name__ == "__main__":
    # Parse file containing PBE0/def2-TZVP heats of formation and xyz filenames
    data = get_energies("hof_qm7.txt")

    # Generate a list of qml.Compound() objects
    mols = []

    for xyz_file in sorted(data.keys())[:10]:

        # Initialize the qml.Compound() objects
        mol = qml.Compound(xyz="qm7/" + xyz_file)

        # Associate a property (heat of formation) with the object
        mol.properties = data[xyz_file]

        # This is a Molecular Coulomb matrix sorted by row norm
        mol.generate_coulomb_matrix(size=20, sorting="row-norm")

        print(generate_arad_representation(mol.coordinates, mol.nuclear_charges, size=11, neighbors=11))
        mols.append(mol)
