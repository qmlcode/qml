from __future__ import print_function

import os

import csv
import ast

import numpy as np
from copy import deepcopy

import scipy
from scipy.linalg import lstsq

import qml

import qml.fchl
from qml.fchl import generate_representation
from qml.fchl import generate_representation_electric_field
from qml.fchl import generate_displaced_representations

from qml.fchl import get_atomic_local_gradient_kernels
from qml.fchl import get_atomic_local_kernels
from qml.fchl import get_atomic_local_electric_field_gradient_kernels
from qml.fchl import get_gaussian_process_electric_field_kernels

from qml.math import cho_solve

DEBYE_TO_EAA = 0.20819434
DEBYE_TO_AU = 0.393456
DEBYE_TO_EAA = 0.20819434
HARTREE_TO_KCAL_MOL = 627.509474
KCAL_MOL_TO_EV = 1.0 / 23.06035

HARTREE_TO_EV = HARTREE_TO_KCAL_MOL * KCAL_MOL_TO_EV
BOHR_TO_ANGS = 0.529177249

CUT_DISTANCE = 1e6
MAX_SIZE = 5

REP_ARGS = {
        "cut_distance": CUT_DISTANCE,
        "max_size": 5
    }

DX = 0.01
DF = 0.01
EF_SCALING = 0.01

SIGMAS = [0.64]

KERNEL_ARGS = {
            "kernel": "gaussian",
            "kernel_args": {
                "sigma": SIGMAS,
            },
            "cut_distance": CUT_DISTANCE,
            "alchemy": "off",
    }

def parse_energy(filename):

    f = open(filename)
    lines = f.readlines()
    f.close()

    energy = dict()

    for line in lines:

        tokens = line.split()
        e = (float(tokens[1]))# - -99.624524268 - -0.499821176)
        angle = ang2ang(float(tokens[0]))

        energy[angle] = e

    e = np.array([energy[key] for key in sorted(energy.keys())])
    offset = 0.0#np.amin(e)

    for key in sorted(energy.keys()):
        energy[key] = (energy[key] - offset)# * HARTREE_TO_KCAL_MOL * KCAL_MOL_TO_EV

    return energy 


def ang2ang(angle):

    out = angle - 90.0

    if out < -180.0:
        out += 360

    return out

def parse_dipole(filename):

    f = open(filename)
    lines = f.readlines()
    f.close()

    dipole = dict()

    for line in lines:

        tokens = line.split()

        mu = np.array([float(tokens[-3]), float(tokens[-2]),float(tokens[-1])])
        angle = ang2ang(float(tokens[0]))

        dipole[angle] = mu#  * DEBYE_TO_EAA

    return dipole



def parse_csv(filename):

    X = []
    X_gradient = []
    X_dipole = []

    E = []
    G = []
    D = []

    with open(filename, 'r') as csvfile:

        csvlines = csv.reader(csvfile, delimiter=";")

        for i, row in enumerate(csvlines):

            nuclear_charges = np.array(ast.literal_eval(row[6]), dtype=np.int32)

            # Gradients (from force in hartree/borh to gradients in eV/angstrom)
            gradient = np.array(ast.literal_eval(row[5])) * HARTREE_TO_EV/BOHR_TO_ANGS * -1

            # SCF energy (eV)
            energy = float(row[4]) * HARTREE_TO_EV
        
            # Dipole moment (Debye -> eV/Angs)
            dipole = np.array(ast.literal_eval(row[3])) * DEBYE_TO_EAA
            
            # Coordinates (Angstrom)
            coords = np.array(ast.literal_eval(row[2]))

            rep          = generate_representation(           coords, nuclear_charges, **REP_ARGS)
            rep_gradient = generate_displaced_representations(coords, nuclear_charges, dx=DX, **REP_ARGS)
            rep_dipole   = generate_representation_electric_field(  coords, nuclear_charges, 
                                fictitious_charges="Gasteiger", **REP_ARGS)

            E.append(energy)
            D.append(dipole)
            G.append(gradient)

            X.append(rep)
            X_gradient.append(rep_gradient)
            X_dipole.append(rep_dipole)
    
    X = np.array(X)
    X_gradient = np.array(X_gradient)
    X_dipole = np.array(X_dipole)
    
    E = np.array(E)
    G = np.array(G)
    D = np.array(D)

    return X, X_gradient, X_dipole, E, G, D


def test_multiple_operators():
    
    
    try:
        import pybel
        import openbabel
    except ImportError:
        return
        

    test_dir = os.path.dirname(os.path.realpath(__file__))

    X, X_gradient, X_dipole, E, G, D = parse_csv(test_dir + "/data/dichloromethane_mp2_test.csv")

    K          = get_atomic_local_kernels(X, X, **KERNEL_ARGS)[0]
    K_gradient = get_atomic_local_gradient_kernels(X, X_gradient, dx=DX, **KERNEL_ARGS)[0]
    K_dipole   = get_atomic_local_electric_field_gradient_kernels(X, X_dipole, df=DF, ef_scaling=EF_SCALING, **KERNEL_ARGS)[0]
    
    Xs, Xs_gradient, Xs_dipole, Es, Gs, Ds = parse_csv(test_dir + "/data/dichloromethane_mp2_train.csv")
    
    Ks          = get_atomic_local_kernels(X, Xs, **KERNEL_ARGS)[0]
    Ks_gradient = get_atomic_local_gradient_kernels(X, Xs_gradient, dx=DX, **KERNEL_ARGS)[0]
    Ks_dipole   = get_atomic_local_electric_field_gradient_kernels(X, Xs_dipole, df=DF, ef_scaling=EF_SCALING, **KERNEL_ARGS)[0]

    offset = E.mean()
    E -= offset
    Es -= offset

    Y = np.concatenate((E, G.flatten(), D.flatten()))
    C = np.concatenate((K.T , K_gradient.T, K_dipole.T))
    
    
    alpha, residuals, rank, sing = lstsq(C, Y, cond=-1, lapack_driver="gelsd")

    Et = np.dot(K.T, alpha)
    Gt = np.dot(K_gradient.T, alpha)
    Dt = np.dot(K_dipole.T, alpha)


    mae = np.mean(np.abs(Et - E)) / KCAL_MOL_TO_EV
    mae_gradient = np.mean(np.abs(Gt - G.flatten())) / KCAL_MOL_TO_EV * BOHR_TO_ANGS
    mae_dipole = np.mean(np.abs(Dt - D.flatten())) / DEBYE_TO_EAA
    
    assert mae < 0.8, "Error in multiple operator training energy"
    assert mae_gradient < 0.1, "Error in multiple operator training energy"
    assert mae_dipole < 0.01, "Error in multiple operator training dipole"
   
    # print(mae)
    # print(mae_gradient)
    # print(mae_dipole)

    Ess = np.dot(Ks.T, alpha)
    Gss = np.dot(Ks_gradient.T, alpha)
    Dss = np.dot(Ks_dipole.T, alpha)
    
    mae = np.mean(np.abs(Ess - Es)) / KCAL_MOL_TO_EV
    mae_gradient = np.mean(np.abs(Gss - Gs.flatten())) / KCAL_MOL_TO_EV * BOHR_TO_ANGS 
    mae_dipole = np.mean(np.abs(Dss - Ds.flatten())) / DEBYE_TO_EAA
    
    assert mae < 0.8, "Error in multiple operator test energy"
    assert mae_gradient < 0.1, "Error in multiple operator test force"
    assert mae_dipole < 0.02, "Error in multiple operator test dipole"
    
    # print(mae)
    # print(mae_gradient)
    # print(mae_dipole)

def test_generate_representation():

   
    coords = np.array([[1.464, 0.707, 1.056],
                       [0.878, 1.218, 0.498],
                       [2.319, 1.126, 0.952]])

    nuclear_charges = np.array([8, 1, 1], dtype=np.int32)

    rep_ref = np.loadtxt("data/fchl_ef_rep.txt").reshape((3, 6, 3))

    # Test with fictitious charges from a numpy array
    fic_charges1 = np.array([-0.41046649,  0.20523324,  0.20523324])

    rep1 = generate_representation_electric_field(  coords, nuclear_charges,
        fictitious_charges=fic_charges1, max_size=3)

    assert np.allclose(rep1, rep_ref), "Error generating representation for electric fields"


    # Test with fictitious charges from a list
    fic_charges2 = [-0.41046649,  0.20523324,  0.20523324]

    rep2 = generate_representation_electric_field(  coords, nuclear_charges,
        fictitious_charges=fic_charges2, max_size=3)

    assert np.allclose(rep2, rep_ref), "Error generating representation for electric fields"


    # Test with fictitious charges from Open Babel (Gasteiger). 
    # Skip test if there is no pybel/openbabel
    try:
        import pybel
        import openbabel
    except ImportError:
        return

    rep3 = generate_representation_electric_field(  coords, nuclear_charges,
            fictitious_charges="Gasteiger", max_size=3)

    assert np.allclose(rep3, rep_ref), "Error assigning partial charges: Check Openbabel/Pybel installation"


def test_gaussian_process():
    
    
    try:
        import pybel
        import openbabel
    except ImportError:
        return
        

    test_dir = os.path.dirname(os.path.realpath(__file__))

    X, X_gradient, X_dipole, E, G, D = parse_csv(test_dir + "/data/dichloromethane_mp2_test.csv")

    K          = get_gaussian_process_electric_field_kernels(X_dipole, X_dipole, **KERNEL_ARGS)[0]
    
    Xs, Xs_gradient, Xs_dipole, Es, Gs, Ds = parse_csv(test_dir + "/data/dichloromethane_mp2_train.csv")
    
    Ks  = get_gaussian_process_electric_field_kernels(X_dipole, Xs_dipole, **KERNEL_ARGS)[0]

    offset = E.mean()
    E -= offset
    Es -= offset

    Y = np.concatenate((E, D.flatten()))
    Ys = np.concatenate((Es, Ds.flatten()))
    
    C = deepcopy(K); 
    C[np.diag_indices_from(C)] += 1e-9
    alpha = cho_solve(C, Y)

    Yt = np.dot(K.T, alpha)
    Yss = np.dot(Ks.T, alpha)

    n = len(E)

    mae = np.mean(np.abs(Yt[:n] - E)) / KCAL_MOL_TO_EV
    mae_dipole = np.mean(np.abs(Yt[n:] - D.flatten())) / DEBYE_TO_EAA
    
    print(mae)
    print(mae_dipole)

    assert mae < 0.002, "Error in multiple operator training energy"
    assert mae_dipole < 0.001, "Error in multiple operator training dipole"

    mae = np.mean(np.abs(Yss[:n] - Es)) / KCAL_MOL_TO_EV
    mae_dipole = np.mean(np.abs(Yss[n:] - Ds.flatten())) / DEBYE_TO_EAA
    
    print(mae)
    print(mae_dipole)
    
    assert mae < 0.3, "Error in multiple operator test energy"
    assert mae_dipole < 0.02, "Error in multiple operator test dipole"

def test_gaussian_process_field_dependent():
    
    
    try:
        import pybel
        import openbabel
    except ImportError:
        return
        

    test_dir = os.path.dirname(os.path.realpath(__file__))

    dipole = parse_dipole(test_dir + "/data/hf_dipole.txt")
    energy = parse_energy(test_dir + "/data/hf_energy.txt")

    # Get energies, dipole moments and angles from datafiles
    e = np.array([energy[key] for key in sorted(energy.keys())])
    mu = np.array([dipole[key] for key in sorted(dipole.keys())])
    a = np.array([key for key in sorted(dipole.keys())])

    # Generate dummy coordinates
    coordinates = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0]])
    nuclear_charges = np.array([9,1],dtype=np.int32)

    # Which angle to put in the training set
    train_angles = np.array([135])
    
    reps = []
    fields = []
    eY = []
    dY = []
    
    # Make training set
    for ang in train_angles:

        ang_rad = ang/180.0*np.pi

        field = np.array([np.cos(ang_rad), np.sin(ang_rad), 0.0]) * 0.001
        rep = generate_representation_electric_field(coordinates, nuclear_charges, max_size=2,
                neighbors=2, cut_distance=1e6)
        fields.append(field)
        reps.append(rep)

        eY.append(energy[ang])

        dY.append(dipole[ang])

    F = [fields,fields]
    X = np.array(reps)
    eY = np.array(eY)
    dY = np.concatenate(dY)
    
    eYs = []
    dYs = []

    reps = []
    fields_test = []

    # Make test set
    test_angles = range(-180,180,20)
    for ang in test_angles:

        ang_rad = ang/180.0*np.pi

        field = np.array([np.cos(ang_rad), np.sin(ang_rad), 0.0]) * 0.001

        rep = generate_representation_electric_field(coordinates, nuclear_charges, max_size=2,
                neighbors=2, cut_distance=1e6)

        fields_test.append(field)
        reps.append(rep)

        eYs.append(energy[ang])
        dYs.append(dipole[ang])
    
    
    Fs = [fields,fields_test]
    Xs = np.array(reps)
    eYs = np.array(eYs)
    dYs = np.concatenate(dYs)

    Y = np.concatenate((eY, dY))

    sigmas = [2.5]
    kernel_args = {
            "kernel": "gaussian",
            "kernel_args": {
                "sigma": sigmas,
            },
            "cut_distance": 1e6,
            "alchemy": "off",
    }

    # Get Gaussian Process kernels for training and test
    K   = get_gaussian_process_electric_field_kernels(X, X,  fields=F,  **kernel_args)[0]
    Ks  = get_gaussian_process_electric_field_kernels(X, Xs, fields=Fs, **kernel_args)[0]

    n_train = len(eY)
    n_test = len(eYs)


    # Solve
    C = deepcopy(K)
    C[np.diag_indices_from(C)] += 1e-14
    alpha = cho_solve(C,Y)
    
    Cs = deepcopy(Ks)

    Yt = np.dot(K.T, alpha)
    
    eYt = Yt[:n_train]
    dYt = Yt[n_train:]
    
    # get predictions
    Yss = np.dot(Cs.T, alpha)

    eYss = Yss[:n_test]
    dYss = Yss[n_test:]

    dtmae = np.mean(np.abs(dYt - dY))
    etmae = np.mean(np.abs(eYt - eY))

    dmae = np.mean(np.abs(dYs - dYss))
    emae = np.mean(np.abs(eYs - eYss))

    # print("%7.2f    %10.4f  %14.4f  %14.4f  %14.4f" % (sigmas[0], dmae, emae,dtmae,etmae))

    assert dmae < 0.005, "Error in test dipole"
    assert dtmae < 0.002, "Error in training dipole"
    assert emae < 0.0001, "Error in test energy"
    assert etmae < 0.0001, "Error in training energy"


if __name__ == "__main__":

    test_multiple_operators()
    test_generate_representation()
    test_gaussian_process()
    test_gaussian_process_field_dependent()
    test_explicit_electric_field()

