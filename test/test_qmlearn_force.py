#!/usr/bin/env python3

import sys

sys.path.insert(0, "/home/andersx/dev/qml/gradient_kernel/build/lib.linux-x86_64-3.5")

import pickle

import glob
import os
import ast
import numpy as np
import pandas as pd
from qml import qmlearn
import sklearn.pipeline
import sklearn.model_selection

import qml

np.random.seed(666)

TEST_DIR = "/home/andersx/dev/qml/develop/test/"


def parse_qm7_data(num=100):
    
    energies = np.loadtxt(TEST_DIR + "/data/hof_qm7.txt", usecols=1)

    indexes = list(range(len(energies)))
    np.random.shuffle(indexes)
    indexes = indexes[:num]

    energies = energies[indexes]

    filenames = sorted(glob.glob(TEST_DIR + "/qm7/*.xyz"))# [indexes]
    filenames = [filenames[i] for i in indexes]

    data = qmlearn.Data(filenames)
    data.set_energies(energies)

    return data


def parse_qm9_data(num=100):
    
    energies = np.loadtxt("/project/andersx/qm9_dataset/data_characterized.txt", usecols=13) * 627.51
    filenames = np.loadtxt("/project/andersx/qm9_dataset/data_characterized.txt", usecols=0, dtype='str')

    indexes = list(range(len(energies)))
    np.random.shuffle(indexes)
    indexes = indexes[:num]

    energies = energies[indexes]
    filenames = ["/project/andersx/qm9_dataset/characterized_xyz/" + filenames[i] for i in indexes]

    data = qmlearn.Data(filenames)
    data.set_energies(energies)

    return data

def csv2data(csv_filename, num=10):
    
    
    df = pd.read_csv(csv_filename, delimiter=";")
    # print(df_train["forces"])

    e = []
    f = []
    n = []
    coords = []
    q = []
   


    for i in indexes[:num]:

        coordinates = np.array(ast.literal_eval(df["coordinates"][i]))
        nuclear_charges = np.array(ast.literal_eval(df["nuclear_charges"][i]), dtype=np.int32)
        atomtypes = df["atomtypes"][i]

        force = np.array(ast.literal_eval(df["forces"][i])) # * -1
        energy = float(df["atomization_energy"][i])

        n.append(len(nuclear_charges))
        f.append(force)
        e.append(energy)
        q.append(nuclear_charges)
        coords.append(coordinates)


    # data = qmlearn.Data()
    # 
    # data.set_energies(np.array(e))
    # data.set_forces(np.array(f))
    # data.natoms = np.array(n , dtype=np.int32)
    # data.coordinates =np.array(coords)
    # data.nuclear_charges = np.array(q)
   
    e = np.array(e)
    f = np.array(f)
    coords = np.array(coords)
    n = np.array(n)
    q = np.array(q)
    
    print(e.shape)
    print(f.shape)
    print(coords.shape)
    print(n.shape)
    print(q.shape)

    data = qmlearn.Data()

    data.set_energies(e)
    data.set_forces(f)
    data.natoms = n
    data.coordinates = coords
    data.nuclear_charges = q

    # rescaled_energies = qmlearn.preprocessing.AtomScaler().fit_transform(data.nuclear_charges, e)
    # data.set_energies(rescaled_energies)

    return data


def npz2data(npy_filename, num=10):
   
    print("PARSING NPY FILE:", npy_filename) 
    
    df = np.load(npy_filename)
    # print(df_train["forces"])

    print([k for k in df.keys()])
    
   

    indexes = list(range(len(df["E"])))
    np.random.shuffle(indexes)
    indexes = indexes[:num]
    
    # print(indexes)
    print(df["z"])
  
    nuclear_charges = df["z"]
    mol_size = len(nuclear_charges)

    e = df["E"][indexes].flatten()
    f = df["F"][indexes]
    coords = df["R"][indexes]
    n = np.array([mol_size for _ in e], dtype=np.int32)
    q = np.array([nuclear_charges for _ in e])

    # print(e.shape)
    # print(f.shape)
    # print(coords.shape)
    # print(n.shape)
    # print(q.shape)

    data = qmlearn.Data()

    data.set_energies(e)
    data.set_forces(f)
    data.natoms = n
    data.coordinates = coords
    data.nuclear_charges = q

    # data.set_energies(rescaled_energies)

    # rescaled_energies = qmlearn.preprocessing.AtomScaler().fit_transform(data.nuclear_charges, e)
    # data.set_energies(rescaled_energies)

    return data

def pipeline():

    test_dir = os.path.dirname(os.path.realpath(__file__))
    
    DATA_SIZE = 5200

    # data = csv2data(test_dir+"/data/force_train.csv", num=DATA_SIZE)
    # data = npz2data("/project/andersx/sgdml_npy/toluene_dft.npz", num=DATA_SIZE)
    # data = npz2data("/project/andersx/sgdml_npy/naphthalene_dft.npz", num=DATA_SIZE)

    npy_names = [
            "aspirin_dft.npz",       #  0
            "azobenzene_dft.npz",    #  1
            "benzene_dft.npz",       #  2
            "benzene_old_dft.npz",   #  3
            "ethanol_dft.npz",       #  4
            "malonaldehyde_dft.npz", #  5
            "naphthalene_dft.npz",   #  6
            "paracetamol_dft.npz",   #  7
            "salicylic_dft.npz",     #  8
            "toluene_dft.npz",       #  9
            "uracil_dft.npz",        # 10
        ]

    # data = npz2data("/project/andersx/sgdml_npy/" + npy_names[0], num=DATA_SIZE)



    # data = parse_qm7_data(num=DATA_SIZE)
    data = parse_qm9_data(num=DATA_SIZE)



    indices = np.arange(len(data.energies))
    np.random.shuffle(indices)

    # indices = indices[:DATA_SIZE]

    # model.fit(indices[:n])
    # scores = model.score(indices[-n:])
    # print("Negative MAE:", scores)

    model = sklearn.pipeline.Pipeline([
            ('preprocess', qmlearn.preprocessing.AtomScaler(data)),
            ('representations', qmlearn.representations.FCHL_ACSF()),
            # ('representations', qmlearn.representations.FCHL_ACSF(data)),
            # ('kernel', qmlearn.kernels.OQMLForceKernel()),
            # ('model', qmlearn.models.OQMLRegression())
            # ('kernel', qmlearn.kernels.GPRForceKernel()),
            # ('model', qmlearn.models.GPRRegression())
            ('kernel', qmlearn.kernels.GPREnergyKernel()),
            ('model', qmlearn.models.KernelRidgeRegression(scoring="neg_mae"))
            ],
            # memory='/dev/shm/' ### This will cache the previous steps to the virtual memory and might speed up gridsearch
            )

    pickle.dump(model, open('pipeline_qm9.pickle', 'wb'))

    exit()

    # Doing a grid search over hyper parameters
    # including which kernel to use
    params = {
              # 'kernel__sigma': [2.0, 4.0, 8.0],
              # 'model__l2_reg': [1e-14, 1e-12, 1e-10, 1e-9, 1e-8],
              # 'model__l2_reg': [1e-9, 1e-8, 1e-7],
              'kernel__sigma': [4.0],
              'model__l2_reg': [1e-14],
             }

    # from skopt import BayesSearchCV
    grid = sklearn.model_selection.GridSearchCV(model, cv=2, refit=False, param_grid=params, verbose=10000)
    grid.fit(indices)

    means = grid.cv_results_['mean_test_score']
    stds  = grid.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    
    print("Best hyper parameters:", grid.best_params_)
    print("Best score:", grid.best_score_)

    print("*** End CV examples ***")


if __name__ == "__main__":

    pipeline()

    
