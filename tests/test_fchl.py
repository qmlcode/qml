#!/usr/bin/env python2

import numpy as np

from numpy import cos, arccos, sin, arctan2, sqrt, cross, dot
from numpy.linalg import norm, inv

import qml
from qml.fchl import generate_fchl_representation
from qml.fchl import get_local_symmetric_kernels_fchl
from qml.fchl import get_local_kernels_fchl
from qml.fchl import get_global_symmetric_kernels_fchl
from qml.fchl import get_global_kernels_fchl
from qml.fchl import get_atomic_kernels_fchl
from qml.fchl import get_atomic_symmetric_kernels_fchl
from qml.fchl import get_atomic_force_alphas_fchl

np.set_printoptions(linewidth=99999999999)

TO_DEG = 180.0 / np.pi


def csv_to_reps(csv_filename, N=None, size=23, elements=None):


    df = pandas.read_csv(csv_filename)

    Nmax = df.shape[0]
    idx = range(Nmax)

    np.random.shuffle(idx)

    if N is not None:
        idx = idx[:N]

    reps = []
    y = []

    for i in idx:

        coordinates = np.array(ast.literal_eval(df["coordinates"][i]))
        nuclear_charges = np.array(ast.literal_eval(df["nuclear_charges"][i]), dtype=np.int32)
        atomtypes = ast.literal_eval(df["atomtypes"][i])

        force = np.array(ast.literal_eval(df["forces"][i]))

        rep = generate_fchl_representation(coordinates, nuclear_charges, \
                size=size, cut_distance=CUT_DISTANCE)

        for j, atomtype in enumerate(atomtypes):

            # if atomtype in elements or elements is None:
            if elements is None:

                reps.append(rep[j])
                y.append(force[j])
            elif atomtype in elements:
                reps.append(rep[j])
                y.append(force[j])


    return np.array(reps), np.array(y)

if __name__ == "__main__":

    mol = qml.Compound(xyz="qm7/0004.xyz")
    X1 = generate_fchl_representation(mol.coordinates, mol.nuclear_charges, size=9, neighbors=9)

    np.save("/home/andersx/X1.npy", X1)

    # mol2 = qml.Compound(xyz="NHClF.xyz")
    mol2 = qml.Compound(xyz="qm7/0005.xyz")
    X2 = generate_fchl_representation(mol2.coordinates, mol2.nuclear_charges, size=9, neighbors=9)

    X = np.array([X1, X2])
    Z = [mol.nuclear_charges, mol2.nuclear_charges]

    sigmas = [25.0]

    K1 = get_local_symmetric_kernels_fchl(X, sigmas, two_body_power=6.0, three_body_power=3.0)[0]
    K2 = get_local_kernels_fchl(X, X, sigmas, two_body_power=1.0, three_body_power=1.0)[0]
    print K1
    print K2
    K2 = get_global_symmetric_kernels_fchl(X, sigmas,two_body_power=1.0, three_body_power=1.0)[0]
    print K2
    K3 = get_global_kernels_fchl(X, X, sigmas, two_body_power=1.0, three_body_power=1.0)[0]
    print K3

    A = X1[:4]
    B = X2[:4]

    K4 = get_atomic_kernels_fchl(A, A, sigmas)[0]
    print K4

    K4 = get_atomic_symmetric_kernels_fchl(A, sigmas, two_body_power=1.0, three_body_power=1.0,
            alchemy="custom", elemental_vectors={1:[2,5,1],6:[2,5,2],7:[2,5,2],8:[1,1,4]})[0]
    print K4
    
    Xf = generate_fchl_representation(mol2.coordinates, \
           mol2.nuclear_charges, size=4, neighbors=4)

    np.set_printoptions(suppress=True)

    exit() 
    K4 = get_atomic_symmetric_kernels_fchl(Xf, sigmas)[0]
    K4 = get_atomic_force_alphas_fchl(Xf, sigmas, alchemy="off")[0]

    # [[ 15.96092603  15.93389651]
    #  [ 15.93389651  15.94988783]]

