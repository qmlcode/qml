#!/usr/bin/env python3
from __future__ import print_function


from time import time

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
from ase.io import extxyz

TRAINING = 7
TEST     = 5
VALID    = 3

ELEMENTS = [1, 6, 7, 8]

CUT_DISTANCE = 8.0

TEST_DIR = os.path.dirname(os.path.realpath(__file__))

DF_TRAIN = pd.read_csv(TEST_DIR+"/data/force_train.csv", delimiter=";").head(TRAINING)
DF_VALID = pd.read_csv(TEST_DIR+"/data/force_valid.csv", delimiter=";").head(VALID)
DF_TEST  = pd.read_csv(TEST_DIR+"/data/force_test.csv", delimiter=";").head(TEST)
print(TRAINING, TEST, VALID)

SIGMA = 21.2

LAMBDA_ENERGY = 1e-6
LAMBDA_FORCE = 1e-6

np.random.seed(666)

def MAE(model_output, reference_vals):
    return np.mean(np.abs(model_output-reference_vals))

def get_reps(df):

    x = []
    f = []
    e = []
    disp_x = []
    q = []

    max_atoms = 27

    for i in range(len(df)):

        coordinates = np.array(ast.literal_eval(df["coordinates"][i]))
        nuclear_charges = np.array(ast.literal_eval(df["nuclear_charges"][i]), dtype=np.int32)
        atomtypes = df["atomtypes"][i]

        force = np.array(ast.literal_eval(df["forces"][i]))
        energy = float(df["atomization_energy"][i])

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


def get_reps_pbc(df_pbc):

    x = []
    f_list = []
    e = []
    disp_x = []
    q = []

    for i in df_pbc:

        coordinates = i.get_positions()
        nuclear_charges = i.get_atomic_numbers() 
        atomtypes = [1, 3]

        force = np.array(i.get_forces())
        energy = float(i.get_total_energy())
        cell = i.cell[:]

        (x1, dx1) = generate_fchl_acsf(nuclear_charges, coordinates, gradients=True, \
                    rcut=1.95, elements = atomtypes, cell = cell)

        x.append(x1)
        f_list.append(force)
        e.append(energy)

        disp_x.append(dx1)
        q.append(nuclear_charges)

    e = np.array(e)
    # e -= np.mean(e)# - 10 #

    f = np.stack(f_list)
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

    Y = np.concatenate((E, F.flatten()))

    print("Kernels ...")
    Kte = get_atomic_local_kernel(X, X,  Q, Q,  SIGMA)
    Kse = get_atomic_local_kernel(X, Xs, Q, Qs, SIGMA)
    Kve = get_atomic_local_kernel(X, Xv, Q, Qv, SIGMA)

    Kt = get_atomic_local_gradient_kernel(X, X,  dX,  Q, Q,  SIGMA)
    Ks = get_atomic_local_gradient_kernel(X, Xs, dXs, Q, Qs, SIGMA)
    Kv = get_atomic_local_gradient_kernel(X, Xv, dXv, Q, Qv, SIGMA)

    C = np.concatenate((Kte, Kt))

    print("Alphas operator ...")
    alpha = svd_solve(C, Y, rcond=1e-11)
    # alpha = qrlq_solve(C, Y)

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


def test_fchl_acsf_operator_pbc():

    SIGMA = 20.0
    TRAINING = 10
    TEST = 8


    print("Representations (pbc) ...")

    xyz_pbc_training = open(TEST_DIR+"/data/LiH_crystal_training.xyz", 'r')
    DF_TRAIN_PBC = list(extxyz.read_extxyz(xyz_pbc_training, index=slice(None, None, 1)))
    xyz_pbc_training.close()

    xyz_pbc_test = open(TEST_DIR+"/data/LiH_crystal_test.xyz", 'r')
    DF_TEST_PBC = list(extxyz.read_extxyz(xyz_pbc_test, index=slice(None, None, 1)))
    xyz_pbc_test.close()

    X, F, E, dX, Q = get_reps_pbc(DF_TRAIN_PBC)
    Xs, Fs, Es, dXs, Qs = get_reps_pbc(DF_TEST_PBC)

    F = np.concatenate(F)
    Fs = np.concatenate(Fs)

    Y = np.concatenate((E, F.flatten()))

    print("Kernels (pbc) ...")
    Kte = get_atomic_local_kernel(X, X,  Q, Q,  SIGMA)
    Kse = get_atomic_local_kernel(X, Xs, Q, Qs, SIGMA)

    Kt = get_atomic_local_gradient_kernel(X, X,  dX,  Q, Q,  SIGMA)
    Ks = get_atomic_local_gradient_kernel(X, Xs, dXs, Q, Qs, SIGMA)

    C = np.concatenate((Kte, Kt))

    print("Alphas operator (pbc) ...")
    alpha = svd_solve(C, Y, rcond=1e-9)
    # alpha = qrlq_solve(C, Y)

    eYt = np.dot(Kte, alpha)
    eYs = np.dot(Kse, alpha)

    fYt = np.dot(Kt, alpha)
    fYs = np.dot(Ks, alpha)


    print("===============================================================================================")
    print("====  OPERATOR, FORCE + ENERGY (PBC) ==========================================================")
    print("===============================================================================================")

    print("TEST  ENERGY   MAE = %10.6f      MAE (expected) = %10.6f " % (np.mean(np.abs(Es - eYs)), 2.363580))
    print("TEST  FORCE    MAE = %10.6f      MAE (expected) = %10.6f " % (np.mean(np.abs(Fs.flatten() - fYs.flatten())), 0.981332))



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
