#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append("/home/andersx/dev/qml/fchl_forceenergy/build/lib.linux-x86_64-2.7")

import ast

import numpy as np
import pandas as pd

import qml
from qml.math import cho_solve
from qml.fchl import generate_fchl_representation
from qml.fchl import get_atomic_force_alphas_fchl
from qml.fchl import get_atomic_force_kernels_fchl
from qml.fchl import get_scalar_vector_alphas_fchl
from qml.fchl import get_scalar_vector_kernels_fchl
from qml.fchl import get_local_symmetric_kernels_fchl
from qml.fchl import get_local_kernels_fchl
from qml.fchl import get_atomic_kernels_fchl

np.set_printoptions(linewidth=19999999999, suppress=True, edgeitems=10)

CUT_DISTANCE = 1e6
# SIGMAS = [5.0]
SIGMAS = [0.25]
# SIGMAS = [0.01 * 2**i for i in range(20)]
SIZE = 19

TRAINING = 100
TEST =  20

def csv_to_atomic_reps(csv_filename):


    df = pd.read_csv(csv_filename)

    reps = []
    y = []

    for i in range(len(df)):

        coordinates = np.array(ast.literal_eval(df["coordinates"][i]))
        nuclear_charges = np.array(ast.literal_eval(df["nuclear_charges"][i]), dtype=np.int32)
        atomtypes = ast.literal_eval(df["atomtypes"][i])

        force = np.array(ast.literal_eval(df["forces"][i]))

        rep = generate_fchl_representation(coordinates, nuclear_charges,
                                            size=SIZE,
                                            cut_distance=CUT_DISTANCE,
                                        )

        for j, atomtype in enumerate(atomtypes):

            reps.append(rep[j])
            y.append(force[j])


    return np.array(reps), np.array(y)


def csv_to_molecular_reps(csv_filename, force_key="forces", energy_key="energy"):


    df = pd.read_csv(csv_filename)

    x = []
    f = []
    e = []
    for i in range(len(df)):

        coordinates = np.array(ast.literal_eval(df["coordinates"][i]))
        nuclear_charges = np.array(ast.literal_eval(df["nuclear_charges"][i]), dtype=np.int32)
        atomtypes = ast.literal_eval(df["atomtypes"][i])

        force = np.array(ast.literal_eval(df[force_key][i]))
        energy = df[energy_key][i]

        rep = generate_fchl_representation(coordinates, nuclear_charges,
                                            size=SIZE,
                                            cut_distance=CUT_DISTANCE,
                                        )
        x.append(rep)
        f.append(force)
        e.append(energy)

    return np.array(x), f, e


def test_old_forces():

    Xall, Yall = csv_to_atomic_reps("data/molecule_300.csv")

    train = SIZE*TRAINING
    test = SIZE*TEST

    X = Xall[:train]
    Y = Yall[:train]

    Xs = Xall[-test:]
    Ys = Yall[-test:]

    alphas = get_atomic_force_alphas_fchl(X, Y, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy="off"
                )[0]

    # print(alphas)

    np.save("X1_old.npy", X)
    np.save("X2_old.npy", Xs)

    np.save("alpha_old.npy", alphas)
    Ks  = get_atomic_force_kernels_fchl(X, Xs, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy="off"
                )[0]
    # print(Ks)
    np.save("Ks_old.npy", Ks)

    Yss = np.einsum('jkl,l->kj', Ks, alphas)

    print("RMSE FORCE COMPONENT", np.mean(np.abs(Yss - Ys)))

    # print(Ys[:19])
    # print(Yss[:19])

def test_old_energy():
    
    # Xall, Fall, Eall = csv_to_molecular_reps("data/02.csv",
    #                            force_key="orca_forces", energy_key="orca_energy")
    Xall, Fall, Eall = csv_to_molecular_reps("data/molecule_300.csv", 
                                force_key="forces", energy_key="om2_energy")

    X = Xall[:train]
    F = Fall[:train]
    E = Eall[:train]
    
    Xs = Xall[-test:]
    Fs = Fall[-test:]
    Es = Eall[-test:]

    K = get_local_symmetric_kernels_fchl(X, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy="off"
                )[0]

    Ks = get_local_kernels_fchl(Xs, X, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy="off"
                )[0]
    E = np.array(E)
    alphas = cho_solve(K, E)

    Ess = np.dot(Ks, alphas)

    print("RMSE ENERGY", np.mean(np.abs(Ess - Es)))
    print(Ess)
    print(Es)

def test_new_forces():
    
    # Xall, Fall, Eall = csv_to_molecular_reps("data/02.csv",
    #                            force_key="orca_forces", energy_key="orca_energy")
    Xall, Fall, Eall = csv_to_molecular_reps("data/molecule_300.csv", 
                                force_key="forces", energy_key="om2_energy")
    Eall = np.array(Eall)
    train = TRAINING
    test = TEST

    X = Xall[:train]
    F = Fall[:train]
    E = Eall[:train]
    
    Xs = Xall[-test:]
    Fs = Fall[-test:]
    Es = Eall[-test:]

    alphas = get_scalar_vector_alphas_fchl(X, F, E, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy="off"
                )[0]

    # print(alphas)

    # np.save("alpha_new.npy", alphas)
    Ks = get_scalar_vector_kernels_fchl(X, Xs, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy="off"
                )[0]
    # print(Ks)
    # np.save("Ks_new.npy", Ks)

    Fss = np.einsum('jkl,l->kj', Ks, alphas)

    Fs = np.array(Fs)
    Fs = np.reshape(Fs, (Fss.shape[0], Fss.shape[1]))

    # print(Fs.shape)
    # print(Fss.shape)
    print("RMSE FORCE COMPONENT", np.mean(np.abs(Fss - Fs)))

    #print(Fs[:19])
    #print(Fss[:19])
   
    # Xs = np.reshape(Xs, (TEST*SIZE,5,SIZE))
    # X = np.reshape(X, (TRAINING*SIZE,5,SIZE))
    # Ks = get_atomic_kernels_fchl(Xs, X, SIGMAS,
    #                 cut_distance=CUT_DISTANCE, 
    #                 alchemy="off"
    #             )[0]

    Ks_small = get_local_kernels_fchl(X, Xs, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy="off"
                )[0]



    Ks = np.zeros((TRAINING*SIZE,TEST*SIZE))

    # print(Ks_small.shape)  
    # print(Ks.shape)

    for i in range(TEST):
       for j in range(TRAINING):

           Ks[j*SIZE:(j+1)*SIZE,i*SIZE:(i+1)*SIZE] = Ks_small[j,i]
           # print(i, j, j*TRAINING, (j+1)*TRAINING, i*TEST, (i+1)*TEST)
           # print(Ks[j*TRAINING:(j+1)*TRAINING,i*TEST:(i+1)*TEST])


    # print(Ks)
    # print(Ks_small)

    Ess = np.dot(Ks.T, alphas)    
    Ess = np.reshape(Ess, (TEST,SIZE))


    alpha_print = np.reshape(alphas, (SIZE,TRAINING))
    # print(alpha_print)

    print(Ess)
    Ess = np.sum(Ess, axis=1)
    print(Ess)
    print(Es)

    print("RMSE ENERGY", np.mean(np.abs(Ess - Es)))
    

if __name__ == "__main__":

    Xall, Fall, Eall = csv_to_molecular_reps("data/molecule_300.csv", 
                                force_key="forces", energy_key="om2_energy")

    train = TRAINING
    test = TEST

    Eall = np.array(Eall)
    print("STD ENERGY", Eall.std())

    test_old_forces()
    test_old_energy()
    test_new_forces()

