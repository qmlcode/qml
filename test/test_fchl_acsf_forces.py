#!/usr/bin/env python3
from __future__ import print_function


from time import time
import ast

import qml
from qml.math import cho_solve
from qml.math import svd_solve
from qml.math import qrlq_solve

from qml.representations import generate_fchl_acsf

from qml.kernels import get_local_kernels_gaussian
from qml.kernels import get_atomic_local_gradient_kernel
from qml.kernels import get_atomic_local_kernel
from qml.kernels import get_local_kernel
from qml.kernels import get_gdml_kernel
from qml.kernels import get_symmetric_gdml_kernel
from qml.kernels import get_local_gradient_kernel
from qml.kernels import get_gp_kernel
from qml.kernels import get_symmetric_gp_kernel

import scipy
import scipy.stats

import ast

import os

from copy import deepcopy

import numpy as np
import pandas as pd
np.set_printoptions(linewidth=999, edgeitems=10, suppress=True)

import csv

TRAINING = 11
TEST     = 5
VALID    = 3

ELEMENTS = [1, 6, 7, 8]

CUT_DISTANCE = 8.0

TEST_DIR = os.path.dirname(os.path.realpath(__file__))

DF_TRAIN = pd.read_csv(TEST_DIR+"/data/force_train.csv", delimiter=";").head(TRAINING)
DF_VALID = pd.read_csv(TEST_DIR+"/data/force_valid.csv", delimiter=";").head(VALID)
DF_TEST  = pd.read_csv(TEST_DIR+"/data/force_test.csv", delimiter=";").head(TEST)

SIGMA = 21.2

LAMBDA_ENERGY = 1e-8
LAMBDA_FORCE = 1e-8

np.random.seed(666)


def get_reps(df):

    x = []
    f = []
    e = []
    disp_x = []
    q = []

    # x_atoms = max([len(df["atomtypes"][i]) for i in range(len(df))])
    max_atoms = 27

    for i in range(len(df)):

        # print(i, "===========")
        coordinates = np.array(ast.literal_eval(df["coordinates"][i]))
        #print(coordinates)

        nuclear_charges = np.array(ast.literal_eval(df["nuclear_charges"][i]), dtype=np.int32)
        #print(nuclear_charges)

        atomtypes = df["atomtypes"][i]
        #print(atomtypes)

        force = np.array(ast.literal_eval(df["forces"][i]))
        #print(force)
        energy = float(df["atomization_energy"][i])
        #print(energy)
        #print("===========")

        ### BEGIN ACSF ###

        # (x1, dx1) = generate_acsf(nuclear_charges, coordinates,
        #         gradients=True, pad=max_atoms, **rep_params)
        (x1, dx1) = generate_fchl_acsf(nuclear_charges, coordinates,
               gradients=True, pad=max_atoms)

        x.append(x1)
        f.append(force)
        e.append(energy)

        disp_x.append(dx1)
        q.append(nuclear_charges)

    e = np.array(e)
    # e -= np.mean(e)# - 10 #

    f = np.array(f)
    f *= -1
    x = np.array(x)

    return x, f, e, np.array(disp_x), q


def test_fchl_acsf_operator():

    print("Representations ...")
    X, F, E, dX, Q = get_reps(DF_TRAIN)
    Xs, Fs, Es, dXs, Qs = get_reps(DF_TEST)
    Xv, Fv, Ev, dXv, Qv = get_reps(DF_VALID)

    F = np.concatenate(F)
    Fs = np.concatenate(Fs)
    Fv = np.concatenate(Fv)

    print("Kernels ...")
    Kte = get_atomic_local_kernel(X,  X, Q,  Q,  SIGMA)
    Kse = get_atomic_local_kernel(X, Xs, Q, Qs, SIGMA)
    Kve = get_atomic_local_kernel(X, Xv, Q, Qv, SIGMA)

    Kt = get_atomic_local_gradient_kernel(X,  X, dX,  Q,  Q, SIGMA)
    Ks = get_atomic_local_gradient_kernel(X, Xs, dXs, Q, Qs, SIGMA)
    Kv = get_atomic_local_gradient_kernel(X, Xv, dXv, Q, Qv, SIGMA)

    C = np.concatenate((Kte, Kt))

    Y = np.concatenate((E, F.flatten()))

    print("Alphas operator ...")
    alpha = svd_solve(C, Y, rcond=1e-12)

    eYt = np.dot(Kte, alpha)
    eYs = np.dot(Kse, alpha)
    eYv = np.dot(Kve, alpha)

    fYt = np.dot(Kt, alpha)
    fYs = np.dot(Ks, alpha)
    fYv = np.dot(Kv, alpha)


    print("===============================================================================================")
    print("====  OPERATOR, FORCE + ENERGY  ===============================================================")
    print("===============================================================================================")

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(E, eYt)
    print("TRAINING ENERGY   MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
            (np.mean(np.abs(E - eYt)), slope, intercept, r_value ))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(F.flatten(), fYt.flatten())
    print("TRAINING FORCE    MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
             (np.mean(np.abs(F.flatten() - fYt.flatten())), slope, intercept, r_value ))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Es.flatten(), eYs.flatten())
    print("TEST     ENERGY   MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
            (np.mean(np.abs(Es - eYs)), slope, intercept, r_value ))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Fs.flatten(), fYs.flatten())
    print("TEST     FORCE    MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
            (np.mean(np.abs(Fs.flatten() - fYs.flatten())), slope, intercept, r_value ))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Ev.flatten(), eYv.flatten())
    print("VALID    ENERGY   MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
            (np.mean(np.abs(Ev - eYv)), slope, intercept, r_value ))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Fv.flatten(), fYv.flatten())
    print("VALID    FORCE    MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
            (np.mean(np.abs(Fv.flatten() - fYv.flatten())), slope, intercept, r_value ))

    
def test_fchl_acsf_gaussian_process():

    print("Representations ...")
    X , F , E , dX,  Q  = get_reps(DF_TRAIN)
    Xs, Fs, Es, dXs, Qs = get_reps(DF_TEST)
    Xv, Fv, Ev, dXv, Qv = get_reps(DF_VALID)

    F  = np.concatenate(F)
    Fs = np.concatenate(Fs)
    Fv = np.concatenate(Fv)

    print("Kernels ...")
    Kt_gp = get_symmetric_gp_kernel(X, dX, Q, SIGMA)
    Ks_gp = get_gp_kernel(X, Xs, dX, dXs, Q, Qs, SIGMA)
    Kv_gp = get_gp_kernel(X, Xv, dX, dXv, Q, Qv, SIGMA)

    print("Alphas GP ...")
    C = deepcopy(Kt_gp)

    for i in range(TRAINING):
        C[i,i] += LAMBDA_ENERGY

    for i in range(TRAINING,C.shape[0]):
        C[i,i] += LAMBDA_FORCE
    
    Y = np.concatenate((E, F.flatten()))

    alpha = cho_solve(C, Y)

    Yt = np.dot(Kt_gp, alpha)
    Ys = np.dot(Ks_gp, alpha)
    Yv = np.dot(Kv_gp, alpha)

    eYt = Yt[:TRAINING]
    eYs = Ys[:TEST]
    eYv = Yv[:VALID]

    fYt = Yt[TRAINING:]
    fYs = Ys[TEST:]
    fYv = Yv[VALID:]
   
    print("===============================================================================================")
    print("====  GAUSSIAN PROCESS, FORCE + ENERGY  =======================================================")
    print("===============================================================================================")

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(E, eYt)
    print("TRAINING ENERGY   MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
            (np.mean(np.abs(E - eYt)), slope, intercept, r_value ))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(F.flatten(), fYt.flatten())
    print("TRAINING FORCE    MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
             (np.mean(np.abs(F.flatten() - fYt.flatten())), slope, intercept, r_value ))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Es.flatten(), eYs.flatten())
    print("TEST     ENERGY   MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
            (np.mean(np.abs(Es - eYs)), slope, intercept, r_value ))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Fs.flatten(), fYs.flatten())
    print("TEST     FORCE    MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
            (np.mean(np.abs(Fs.flatten() - fYs.flatten())), slope, intercept, r_value ))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Ev.flatten(), eYv.flatten())
    print("VALID    ENERGY   MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
            (np.mean(np.abs(Ev - eYv)), slope, intercept, r_value ))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Fv.flatten(), fYv.flatten())
    print("VALID    FORCE    MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
            (np.mean(np.abs(Fv.flatten() - fYv.flatten())), slope, intercept, r_value ))


if __name__ == "__main__":

    test_fchl_acsf_operator()
    test_fchl_acsf_gaussian_process()
