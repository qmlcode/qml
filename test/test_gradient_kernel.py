from __future__ import print_function

import sys
sys.path.insert(0, "/home/andersx/dev/qml/andersx/fchl_acsf/build/lib.linux-x86_64-3.5")

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from fgradient_kernels import fatomic_local_gradient_kernel
from fgradient_kernels import fgdml_kernel

import os

from copy import deepcopy

import numpy as np

np.set_printoptions(linewidth=999, edgeitems=10, suppress=True)

import qml
from qml.representations import generate_acsf

SIGMA = 2.0

def get_mols(files):
    path = os.path.dirname(os.path.realpath(__file__))
    return [qml.data.Compound(xyz=path + "/" + f) for f in files]


def calc_kernel(q1, q2):
    return np.exp(-np.linalg.norm(q1 - q2)**2 / (2*SIGMA**2))

def calc_local_kernel(rep1, rep2, n1, n2):
    
    k = 0.0
   
    for i in range(n1):
        for j in range(n2):

            k += np.exp(-np.linalg.norm(rep1[i] - rep2[j])**2 / (2*SIGMA**2))

    return k


def get_kernel_gradient(mol1, mol2, dx=1e-6):

    true_coords = deepcopy(mol1.coordinates)
    kernel = np.zeros((3*mol1.natoms, mol2.natoms))

    rep2 = generate_acsf(mol2.nuclear_charges, mol2.coordinates)


    idx1 = 0
    for n, coord in enumerate(true_coords):
        for xyz, x in enumerate(coord):

            temp_coords = deepcopy(true_coords)
            temp_coords[n,xyz] = x - dx
    
            rep1 = generate_acsf(mol1.nuclear_charges, temp_coords)
   
            for j in range(mol1.natoms):
                for i in range(mol2.natoms):

                    kernel[idx1, i] += calc_kernel(rep1[j], rep2[i])
            
            temp_coords[n,xyz] = x + dx

            rep1 = generate_acsf(mol1.nuclear_charges, temp_coords)
   
            for j in range(mol1.natoms):
                for i in range(mol2.natoms):

                    kernel[idx1, i] -= calc_kernel(rep1[j], rep2[i])
                 

            idx1 += 1

    kernel /= (2 * dx)

    return kernel


def get_kernel_hessian(mol1, mol2, dx=5e-5):

    coords1 = deepcopy(mol1.coordinates)
    coords2 = deepcopy(mol2.coordinates)

    kernel = np.zeros((3*mol1.natoms, 3*mol2.natoms))



    idx1 = 0

    for n1, coord1 in enumerate(coords1):
        for xyz1, x1 in enumerate(coord1):


            temp1 = deepcopy(coords1)
            temp1[n1,xyz1] = x1 - dx
            rep1 = generate_acsf(mol1.nuclear_charges, temp1)
   
            idx2 = 0
             
            for n2, coord2 in enumerate(coords2):
                for xyz2, x2 in enumerate(coord2):

                    temp2 = deepcopy(coords2)
                    temp2[n2,xyz2] = x2 - dx
                    rep2 = generate_acsf(mol2.nuclear_charges, temp2)

                    kernel[idx1, idx2] += calc_local_kernel(rep1, rep2, mol1.natoms, mol2.natoms)
                    
                    temp2[n2,xyz2] = x2 + dx
                    rep2 = generate_acsf(mol2.nuclear_charges, temp2)

                    kernel[idx1, idx2] -= calc_local_kernel(rep1, rep2, mol1.natoms, mol2.natoms)

                    idx2 += 1

            
            temp1[n1,xyz1] = x1 + dx
            rep1 = generate_acsf(mol1.nuclear_charges, temp1)
   
            idx2 = 0
             
            for n2, coord2 in enumerate(coords2):
                for xyz2, x2 in enumerate(coord2):

                    temp2 = deepcopy(coords2)
                    temp2[n2,xyz2] = x2 - dx
                    rep2 = generate_acsf(mol2.nuclear_charges, temp2)

                    kernel[idx1, idx2] -= calc_local_kernel(rep1, rep2, mol1.natoms, mol2.natoms)
                    
                    temp2[n2,xyz2] = x2 + dx
                    rep2 = generate_acsf(mol2.nuclear_charges, temp2)

                    kernel[idx1, idx2] += calc_local_kernel(rep1, rep2, mol1.natoms, mol2.natoms)

                    idx2 += 1




            idx1 += 1

    kernel /= (4 * dx**2)

    return kernel


def get_analytical_kernel_gradient(mol1, mol2):

    x1, dx1 = generate_acsf(mol1.nuclear_charges, mol1.coordinates, gradients=True)
    x2 = generate_acsf(mol2.nuclear_charges, mol2.coordinates)

    # print(x2.shape)
    # print(x1.shape)
    # print(dx1.shape)

    k = fatomic_local_gradient_kernel(np.array([x2]), np.array([x1]), np.array([dx1]), 
            np.array([mol2.natoms]), 
            np.array([mol1.natoms]),
            1,
            1,
            mol2.natoms,
            mol1.natoms*3,
            SIGMA
        )

    return k

def get_analytical_kernel_hessian(mol1, mol2):

    print("REPS")
    x1, dx1 = generate_acsf(mol1.nuclear_charges, mol1.coordinates, gradients=True)
    x2, dx2 = generate_acsf(mol2.nuclear_charges, mol2.coordinates, gradients=True)


    print("ker")
    k = fgdml_kernel(
            np.array([x2]), 
            np.array([x1]), 
            np.array([dx2]), 
            np.array([dx1]), 
            np.array([mol2.natoms]), 
            np.array([mol1.natoms]),
            1,
            1,
            mol2.natoms,
            mol1.natoms,
            SIGMA
        )

    return k




if __name__ == "__main__":

    files = [
            "ammonia.xyz",
            "2wat.xyz",
            "water2.xyz",
            "qm7/0101.xyz",
            "qm7/0102.xyz",
            "qm7/0103.xyz",
            "qm7/0104.xyz",
            "qm7/0105.xyz",
            "qm7/0106.xyz",
            "qm7/0107.xyz",
            "qm7/0108.xyz",
            "qm7/0109.xyz",
            "qm7/0110.xyz"
            ]

    mols = get_mols(files)

    max_size = max([mol.natoms for mol in mols])

    X = []
    dX = []
    N = []

    paddy = np.zeros((16, 150))
    dpaddy = np.zeros((16, 150, 16, 3))

    for mol in mols:

        x1, dx1 = generate_acsf(mol.nuclear_charges, mol.coordinates, gradients=True)

        n = mol.natoms

        rep = deepcopy(paddy)
        rep[:n,:] += x1

        
        drep = deepcopy(dpaddy)
        drep[:n,:,:n,:] += dx1



        X.append(rep)
        dX.append(drep)
        N.append(mol.natoms)


    X = np.array(X)
    dX = np.array(dX)
        
    # print(dX.shape)
    # 
    # dX = np.swapaxes(dX, 0, 2)
    # print(dX.shape)
    # 
    # dX = np.swapaxes(dX, 1, 3)
    # print(dX.shape)
    # 
    # dX = np.swapaxes(dX, 2, 4)
    # print(dX.shape)


    N = np.array(N, dtype=np.int32)
    na = np.sum(N)
   
    # print(X)


    k = fgdml_kernel(
            X,          # np.array([x2]), 
            X,          # np.array([x1]), 
            dX,         # np.array([dx2]), 
            dX,         # np.array([dx1]), 
            N,          # np.array([mol2.natoms]), 
            N,          # np.array([mol1.natoms]),
            len(mols),  # 1,
            len(mols),  # 1,
            na,         # mol2.natoms,
            na,         # mol1.natoms,
            10.0
        )
    
    print(k)
    print(k.shape)

    # for i, mol1 in enumerate(mols):
    #     for j, mol2 in enumerate(mols):
    # 
    #         # k = get_kernel_hessian(mol2, mol1)
    #         ka = get_analytical_kernel_hessian(mol1, mol2)
    #         diff = ka#- k
    #         print(i,j, np.amax(diff))
