#!/usr/bin/env python3

import sys

sys.path.insert(0, "/home/andersx/dev/qml/gradient_kernel/build/lib.linux-x86_64-3.5")

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


# def parse_data():
# 
#     filenames = sorted(glob.glob(TEST_DIR + "/qm7/*.xyz"))[:10]
#     data = qmlearn.Data(filenames)
#     energies = np.loadtxt(TEST_DIR + "/data/hof_qm7.txt", usecols=1)[:10]
#     rescaled_energies = qmlearn.preprocessing.AtomScaler().fit_transform(data.nuclear_charges, energies)
#     data.set_energies(rescaled_energies)
#     # data.set_energies(energies)
# 
#     return data

def csv2data(csv_filename, num=10):
    
    
    df = pd.read_csv(csv_filename, delimiter=";").head(num)
    # print(df_train["forces"])

    e = []
    f = []
    n = []
    coords = []
    q = []
   

    indexes = list(range(len(df)))
    np.random.shuffle(indexes)

    for i in indexes[:num]:

        coordinates = np.array(ast.literal_eval(df["coordinates"][i]))
        nuclear_charges = np.array(ast.literal_eval(df["nuclear_charges"][i]), dtype=np.int32)
        atomtypes = df["atomtypes"][i]

        force = np.array(ast.literal_eval(df["forces"][i])) * -1
        energy = float(df["atomization_energy"][i])

        n.append(len(nuclear_charges))
        f.append(force)
        e.append(energy)
        q.append(nuclear_charges)
        coords.append(coordinates)


    data = qmlearn.Data()
    # data.set_energies(e)
    data.set_forces(np.array(f))
    data.natoms = np.array(n , dtype=np.int32)
    data.coordinates =np.array(coords)
    data.nuclear_charges = np.array(q)
    # data.set_energies(rescaled_energies)

    rescaled_energies = qmlearn.preprocessing.AtomScaler().fit_transform(data.nuclear_charges, e)
    data.set_energies(rescaled_energies)

    return data

def pipeline():

    test_dir = os.path.dirname(os.path.realpath(__file__))

    data = csv2data(test_dir+"/data/force_train.csv", num=100)


    indices = np.arange(len(data.energies))
    np.random.shuffle(indices)


    # model.fit(indices[:n])
    # scores = model.score(indices[-n:])
    # print("Negative MAE:", scores)

    model = sklearn.pipeline.Pipeline([
            ('preprocess', qmlearn.preprocessing.AtomScaler(data)),
            ('representations', qmlearn.representations.FCHL_ACSF()),
            # ('kernel', qmlearn.kernels.OQMLForceKernel()),
            # ('model', qmlearn.models.OQMLRegression())
            ('kernel', qmlearn.kernels.GPRForceKernel()),
            ('model', qmlearn.models.GPRRegression())
            ],
            # memory='/dev/shm/' ### This will cache the previous steps to the virtual memory and might speed up gridsearch
            )

    # Doing a grid search over hyper parameters
    # including which kernel to use
    params = {
              'kernel__sigma': [2, 4, 8, 16, 32, 64],
              'model__l2_reg': [1e-14, 1e-12, 1e-10],
              # 'kernel__sigma': [8],
              # 'model__l2_reg': [1e-14],
             }

    # from skopt import BayesSearchCV
    grid = sklearn.model_selection.GridSearchCV(model, cv=2, refit=False, param_grid=params)
    grid.fit(indices)
    
    print("Best hyper parameters:", grid.best_params_)
    print("Best score:", grid.best_score_)

    print("*** End CV examples ***")


if __name__ == "__main__":

    pipeline()

    
