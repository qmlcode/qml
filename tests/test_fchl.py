from __future__ import print_function

import os
import numpy as np
import qml

import qml

from qml.math import cho_solve

from qml.fchl import generate_representation
from qml.fchl import get_local_symmetric_kernels
from qml.fchl import get_local_kernels
from qml.fchl import get_global_symmetric_kernels
from qml.fchl import get_global_kernels
from qml.fchl import get_atomic_kernels
from qml.fchl import get_atomic_symmetric_kernels

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

def test_krr_fchl_local():

    # Test that all kernel arguments work
    kernel_args = {
            "cut_distance": 1e6,
            "cut_start": 0.5,
            "two_body_width": 0.1,
            "two_body_scaling": 2.0,
            "two_body_power": 6.0,
            "three_body_width": 3.0,
            "three_body_scaling": 2.0,
            "three_body_power": 3.0,
            "alchemy": "periodic-table",
            "alchemy_period_width": 1.0,
            "alchemy_group_width": 1.0,
            "fourier_order": 2,
            }

    test_dir = os.path.dirname(os.path.realpath(__file__))

    # Parse file containing PBE0/def2-TZVP heats of formation and xyz filenames
    data = get_energies(test_dir + "/data/hof_qm7.txt")

    # Generate a list of qml.Compound() objects"
    mols = []


    for xyz_file in sorted(data.keys())[:100]:

        # Initialize the qml.Compound() objects
        mol = qml.Compound(xyz=test_dir + "/qm7/" + xyz_file)

        # Associate a property (heat of formation) with the object
        mol.properties = data[xyz_file]

        # This is a Molecular Coulomb matrix sorted by row norm
        mol.representation = generate_representation(mol.coordinates, \
                                mol.nuclear_charges, cut_distance=1e6)
        mols.append(mol)

    # Shuffle molecules
    np.random.seed(666)
    np.random.shuffle(mols)

    # Make training and test sets
    n_test  = len(mols) // 3
    n_train = len(mols) - n_test

    training = mols[:n_train]
    test  = mols[-n_test:]

    X = np.array([mol.representation for mol in training])
    Xs = np.array([mol.representation for mol in test])

    # List of properties
    Y = np.array([mol.properties for mol in training])
    Ys = np.array([mol.properties for mol in test])

    # Set hyper-parameters
    sigma = 2.5
    llambda = 1e-8

    K_symmetric = get_local_symmetric_kernels(X, [sigma], **kernel_args)[0]
    K = get_local_kernels(X, X, [sigma], **kernel_args)[0]

    assert np.allclose(K, K_symmetric), "Error in FCHL symmetric local kernels"
    assert np.invert(np.all(np.isnan(K_symmetric))), "FCHL local symmetric kernel contains NaN"
    assert np.invert(np.all(np.isnan(K))), "FCHL local kernel contains NaN"

    # Solve alpha
    K[np.diag_indices_from(K)] += llambda
    alpha = cho_solve(K,Y)

    # Calculate prediction kernel
    Ks = get_local_kernels(Xs, X , [sigma], **kernel_args)[0]
    assert np.invert(np.all(np.isnan(Ks))), "FCHL local testkernel contains NaN"

    Yss = np.dot(Ks, alpha)

    mae = np.mean(np.abs(Ys - Yss))
    assert abs(2 - mae) < 1.0, "Error in FCHL local kernel-ridge regression"


def test_krr_fchl_global():

    test_dir = os.path.dirname(os.path.realpath(__file__))

    # Parse file containing PBE0/def2-TZVP heats of formation and xyz filenames
    data = get_energies(test_dir + "/data/hof_qm7.txt")

    # Generate a list of qml.Compound() objects"
    mols = []


    for xyz_file in sorted(data.keys())[:100]:

        # Initialize the qml.Compound() objects
        mol = qml.Compound(xyz=test_dir + "/qm7/" + xyz_file)

        # Associate a property (heat of formation) with the object
        mol.properties = data[xyz_file]

        # This is a Molecular Coulomb matrix sorted by row norm
        mol.representation = generate_representation(mol.coordinates, \
                                mol.nuclear_charges, cut_distance=1e6)
        mols.append(mol)

    # Shuffle molecules
    np.random.seed(666)
    np.random.shuffle(mols)

    # Make training and test sets
    n_test  = len(mols) // 3
    n_train = len(mols) - n_test

    training = mols[:n_train]
    test  = mols[-n_test:]

    X = np.array([mol.representation for mol in training])
    Xs = np.array([mol.representation for mol in test])

    # List of properties
    Y = np.array([mol.properties for mol in training])
    Ys = np.array([mol.properties for mol in test])

    # Set hyper-parameters
    sigma = 100.0
    llambda = 1e-8

    K_symmetric = get_global_symmetric_kernels(X, [sigma])[0]
    K = get_global_kernels(X, X, [sigma])[0]

    assert np.allclose(K, K_symmetric), "Error in FCHL symmetric global kernels"
    assert np.invert(np.all(np.isnan(K_symmetric))), "FCHL global symmetric kernel contains NaN"
    assert np.invert(np.all(np.isnan(K))), "FCHL global kernel contains NaN"

    # Solve alpha
    K[np.diag_indices_from(K)] += llambda
    alpha = cho_solve(K,Y)

    # # Calculate prediction kernel
    Ks = get_global_kernels(Xs, X , [sigma])[0]
    assert np.invert(np.all(np.isnan(Ks))), "FCHL global testkernel contains NaN"

    Yss = np.dot(Ks, alpha)

    mae = np.mean(np.abs(Ys - Yss))
    assert abs(2 - mae) < 1.0, "Error in FCHL global kernel-ridge regression"


def test_krr_fchl_atomic():

    test_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Parse file containing PBE0/def2-TZVP heats of formation and xyz filenames
    data = get_energies(test_dir + "/data/hof_qm7.txt")

    # Generate a list of qml.Compound() objects"
    mols = []

    for xyz_file in sorted(data.keys())[:10]:

        # Initialize the qml.Compound() objects
        mol = qml.Compound(xyz=test_dir + "/qm7/" + xyz_file)

        # Associate a property (heat of formation) with the object
        mol.properties = data[xyz_file]

        # This is a Molecular Coulomb matrix sorted by row norm
        mol.representation = generate_representation(mol.coordinates, \
                                mol.nuclear_charges, cut_distance=1e6)
        mols.append(mol)

    X = np.array([mol.representation for mol in mols])

    # Set hyper-parameters
    sigma = 2.5

    K = get_local_symmetric_kernels(X, [sigma])[0]

    K_test = np.zeros((len(mols),len(mols)))

    for i, Xi in enumerate(X):
        for j, Xj in enumerate(X):


            K_atomic = get_atomic_kernels(Xi[:mols[i].natoms], Xj[:mols[j].natoms], [sigma])[0]
            K_test[i,j] = np.sum(K_atomic)
            
            assert np.invert(np.all(np.isnan(K_atomic))), "FCHL atomic kernel contains NaN"

            if (i == j):
                K_atomic_symmetric = get_atomic_symmetric_kernels(Xi[:mols[i].natoms], [sigma])[0]
                assert np.allclose(K_atomic, K_atomic_symmetric), "Error in FCHL symmetric atomic kernels"
                assert np.invert(np.all(np.isnan(K_atomic_symmetric))), "FCHL atomic symmetric kernel contains NaN"

    assert np.allclose(K, K_test), "Error in FCHL atomic kernels"


if __name__ == "__main__":

    test_krr_fchl_local()
    test_krr_fchl_global()
    test_krr_fchl_atomic()
