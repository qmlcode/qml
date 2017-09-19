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
from qml.fchl import get_force_energy_alphas_fchl
from qml.fchl import get_local_symmetric_kernels_fchl

np.set_printoptions(linewidth=100, suppress=True)

CUT_DISTANCE = 1e6
# SIGMAS = [5.0]
SIGMAS = [0.25]
# SIGMAS = [0.01 * 2**i for i in range(20)]
SIZE = 19

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

def test_old_forces():

    Xall, Yall = csv_to_atomic_reps("data/molecule_300.csv")

    train = SIZE*40
    test = SIZE*20

    X = Xall[:train]
    Y = Yall[:train]

    Xs = Xall[-test:]
    Ys = Yall[-test:]

    alphas = get_atomic_force_alphas_fchl(X, Y, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy="off"
                )[0]

    print(alphas)

    Ks  = get_atomic_force_kernels_fchl(X, Xs, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy="off"
                )[0]

    Yss = np.einsum('jkl,l->kj', Ks, alphas)

    print("RMSE FORCE COMPONENT", np.mean(np.abs(Yss - Ys)))

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


def test_new_forces():
    
    # Xall, Fall, Eall = csv_to_molecular_reps("data/02.csv",
    #                            force_key="orca_forces", energy_key="orca_energy")
    Xall, Fall, Eall = csv_to_molecular_reps("data/molecule_300.csv", 
                                force_key="forces", energy_key="om2_energy")

    train = 40
    test = 20

    X = Xall[:train]
    F = Fall[:train]
    E = Eall[:train]

    alphas = get_force_energy_alphas_fchl(X, F, E, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy="off"
                )[0]

    print(alphas)
    K = get_local_symmetric_kernels_fchl(X, SIGMAS,
                    cut_distance=CUT_DISTANCE, 
                    alchemy="off"
                )[0]
    print(K)

if __name__ == "__main__":

    # test_old_forces()
    test_new_forces()


