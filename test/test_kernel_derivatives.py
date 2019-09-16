# MIT License
#
# Copyright (c) 2017-2018 Anders Steen Christensen, Lars Andersen Bratholm
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from copy import deepcopy

import numpy as np
import ast
import csv

import qml
from qml.representations import generate_acsf

from qml.kernels import get_global_kernel
from qml.kernels import get_local_kernel
from qml.kernels import get_local_kernels
from qml.kernels import get_local_symmetric_kernel
from qml.kernels import get_local_symmetric_kernels
from qml.kernels import get_atomic_local_kernel
from qml.kernels import get_atomic_local_gradient_kernel
from qml.kernels import get_local_gradient_kernel
from qml.kernels import get_gdml_kernel
from qml.kernels import get_symmetric_gdml_kernel
from qml.kernels import get_gp_kernel
from qml.kernels import get_symmetric_gp_kernel

np.set_printoptions(linewidth=666, edgeitems=10)

CSV_FILE = "data/amons_small.csv"

TRAINING = 7
TEST     = 5

SIGMA = 2.5
MAX_ATOMS = 20
DX = 1e-4

def csv_to_molecular_reps(csv_filename):

    np.random.seed(667)

    x = []
    dx = []
    f = []
    e = []
    n = []
    q = []

    disp_x = [[] for _ in range(4)]

    with open(csv_filename, 'r') as csvfile:

        df = csv.reader(csvfile, delimiter=";", quotechar='#')

        for i, row in enumerate(df):

            if i > TEST+TRAINING: break

            coordinates = np.array(ast.literal_eval(row[2]))
            nuclear_charges = ast.literal_eval(row[5])
            atomtypes = ast.literal_eval(row[1])
            force = np.array(ast.literal_eval(row[3]))
            energy = float(row[6])

            (x1, dx1) = generate_acsf(nuclear_charges, coordinates, pad=MAX_ATOMS, gradients=True)

            x.append(x1)
            dx.append(dx1)
            f.append(force)
            e.append(energy)
            n.append(len(nuclear_charges))
            q.append(nuclear_charges)


            for j in range(len(nuclear_charges)):
                for xyz in range(3):

                    for k, disp in enumerate([2*DX, DX, -DX, -2*DX]):
                        disp_coords = deepcopy(coordinates)

                        disp_coords[j,xyz] += disp

                        # dx1 = generate_fchl_acsf(nuclear_charges, disp_coords, **REP_PARAMS)
                        # print(dx1.shape)
                        dx1 = generate_acsf(nuclear_charges, disp_coords, pad=MAX_ATOMS)

                        drep = dx1

                        disp_x[k].append(drep)

    return np.array(x), np.array(dx), e, f, np.array(disp_x), n, q


def test_global_kernel():

    Xall, dXall, Eall, Fall, dispXall, Nall, Qall= csv_to_molecular_reps(CSV_FILE)

    X  = Xall[:TRAINING]
    Q  = Qall[:TRAINING]


    Xs = Xall[-TEST:]
    Qs  = Qall[-TEST:]

    K1 = get_global_kernel(X, Xs, Q, Qs, SIGMA)
    K2 = get_global_kernel(X, X, Q, Q, SIGMA)


def test_atomic_local_kernel():

    Xall, dXall, Eall, Fall, dispXall, Nall, Qall= csv_to_molecular_reps(CSV_FILE)

    X  = Xall[:TRAINING]
    dX = dXall[:TRAINING]
    N  = Nall[:TRAINING]
    dispX  = dispXall[:,:sum(N)*3,:,:]
    Q  = Qall[:TRAINING]

    Xs = Xall[-TEST:]
    dXs = dXall[-TEST:]
    Ns = Nall[-TEST:]
    dispXs = dispXall[:,-sum(Ns)*3:,:,:]
    Qs  = Qall[-TEST:]

    K = get_atomic_local_kernel(X, Xs, Q, Qs, SIGMA)

    K_numm = np.zeros(K.shape)

    idx = 0

    for i in range(TRAINING):
        for n1 in range(N[i]):

            for j in range(TEST):
                for n2 in range(Ns[j]):

                    if (Q[i][n1] == Qs[j][n2]):
                        d = np.linalg.norm(X[i,n1] - Xs[j,n2])
                        gauss = np.exp(-d**2 / (2 * SIGMA**2))
                        K_numm[j, idx] += gauss

            idx += 1

    assert np.allclose(K, K_numm), "Error in get_local_kernel()"

def test_local_kernel():

    Xall, dXall, Eall, Fall, dispXall, Nall, Qall= csv_to_molecular_reps(CSV_FILE)

    X  = Xall[:TRAINING]
    dX = dXall[:TRAINING]
    N  = Nall[:TRAINING]
    dispX  = dispXall[:,:sum(N)*3,:,:]
    Q  = Qall[:TRAINING]


    Xs = Xall[-TEST:]
    dXs = dXall[-TEST:]
    Ns = Nall[-TEST:]
    dispXs = dispXall[:,-sum(Ns)*3:,:,:]
    Qs  = Qall[-TEST:]

    K = get_local_kernel(X, X, Q, Q, SIGMA)
    assert np.allclose(K, K.T), "Error in symmetric get_local_kernel()"
    
    Ksymm = get_local_symmetric_kernel(X, Q, SIGMA)
    assert np.allclose(Ksymm, K), "Error in get_local_symmetric_kernel()"

    K = get_local_kernel(X, Xs, Q, Qs, SIGMA)

    K_numm = np.zeros(K.shape)

    for i in range(TRAINING):
        for n1 in range(N[i]):

            for j in range(TEST):
                for n2 in range(Ns[j]):
                    if (Q[i][n1] == Qs[j][n2]):

                        d = np.linalg.norm(X[i,n1] - Xs[j,n2])
                        gauss = np.exp(-d**2 / (2 * SIGMA**2))
                        K_numm[j, i] += gauss


    assert np.allclose(K, K_numm), "Error in get_local_kernel()"


def test_atomic_local_kernel():

    Xall, dXall, Eall, Fall, dispXall, Nall, Qall= csv_to_molecular_reps(CSV_FILE)

    X  = Xall[:TRAINING]
    dX = dXall[:TRAINING]
    N  = Nall[:TRAINING]
    dispX  = dispXall[:,:sum(N)*3,:,:]
    Q  = Qall[:TRAINING]

    Xs = Xall[-TEST:]
    dXs = dXall[-TEST:]
    Ns = Nall[-TEST:]
    dispXs = dispXall[:,-sum(Ns)*3:,:,:]
    Qs  = Qall[-TEST:]

    K = get_atomic_local_kernel(X, Xs, Q, Qs, SIGMA)

    K_numm = np.zeros(K.shape)

    idx = 0

    for i in range(TRAINING):
        for n1 in range(N[i]):

            for j in range(TEST):
                for n2 in range(Ns[j]):

                    if (Q[i][n1] == Qs[j][n2]):
                        d = np.linalg.norm(X[i,n1] - Xs[j,n2])
                        gauss = np.exp(-d**2 / (2 * SIGMA**2))
                        K_numm[j, idx] += gauss

            idx += 1

    assert np.allclose(K, K_numm), "Error in get_local_kernel()"


def test_atomic_local_gradient():

    Xall, dXall, Eall, Fall, dispXall, Nall, Qall= csv_to_molecular_reps(CSV_FILE)

    X  = Xall[:TRAINING]
    dX = dXall[:TRAINING]
    N  = Nall[:TRAINING]
    dispX  = dispXall[:,:sum(N)*3,:,:]
    Q  = Qall[:TRAINING]

    Xs = Xall[-TEST:]
    dXs = dXall[-TEST:]
    Ns = Nall[-TEST:]
    dispXs = dispXall[:,-sum(Ns)*3:,:,:]
    Qs  = Qall[-TEST:]

    Kt_gradient = get_atomic_local_gradient_kernel(X, X, dX, Q, Q, SIGMA)

    idx1 = 0

    K_numm = np.zeros((4, Kt_gradient.shape[0], Kt_gradient.shape[1]))

    for i in range(TRAINING):
        for n1 in range(N[i]):


            idx2 = 0
            for j in range(TRAINING):
                for n2 in range(N[j]):
                    for xyz in range(3):

                        for n_diff in range(N[j]):
                            for k in range(4):
                                if (Q[i][n1] == Q[j][n_diff]):
                                    d = np.linalg.norm(X[i,n1] - dispX[k,idx2,n_diff])
                                    gauss = np.exp(-d**2 / (2 * SIGMA**2))
                                    K_numm[k,idx2,idx1] += gauss
                        idx2 += 1
            idx1 += 1

    K_numm = -(-K_numm[0] + 8*K_numm[1] - 8*K_numm[2] + K_numm[3]) / (12 * DX)

    assert np.allclose(Kt_gradient, K_numm), "Error in get_atomic_local_gradient_kernel()"


def test_local_gradient():

    Xall, dXall, Eall, Fall, dispXall, Nall, Qall= csv_to_molecular_reps(CSV_FILE)

    X  = Xall[:TRAINING]
    dX = dXall[:TRAINING]
    N  = Nall[:TRAINING]
    dispX  = dispXall[:,:sum(N)*3,:,:]
    Q  = Qall[:TRAINING]

    Xs = Xall[-TEST:]
    dXs = dXall[-TEST:]
    Ns = Nall[-TEST:]
    dispXs = dispXall[:,-sum(Ns)*3:,:,:]
    Qs  = Qall[-TEST:]

    Kt_gradient = get_local_gradient_kernel(X, X, dX, Q, Q, SIGMA)

    idx2 = 0

    K_numm = np.zeros((4, Kt_gradient.shape[0], Kt_gradient.shape[1]))

    for i in range(TRAINING):
        for n1 in range(N[i]):

            idx1 = 0

            for j in range(TRAINING):
                for n2 in range(N[j]):
                    for xyz in range(3):
                        for n_diff in range(N[j]):
                            for k in range(4):

                                if (Q[i][n1] == Q[j][n_diff]):
                                    d = np.linalg.norm(X[i,n1] - dispX[k,idx1,n_diff])
                                    gauss = np.exp(-d**2 / (2 * SIGMA**2))
                                    K_numm[k,idx1,idx2] += gauss

                        idx1 += 1
        idx2 += 1

    K_numm = -(-K_numm[0] + 8*K_numm[1] - 8*K_numm[2] + K_numm[3]) / (12 * DX)

    assert np.allclose(Kt_gradient, K_numm), "Error in get_local_gradient_kernel()"


def test_gdml_kernel():

    Xall, dXall, Eall, Fall, dispXall, Nall, Qall= csv_to_molecular_reps(CSV_FILE)

    X  = Xall[:TRAINING]
    dX = dXall[:TRAINING]
    N  = Nall[:TRAINING]
    dispX  = dispXall[:,:sum(N)*3,:,:]
    Q  = Qall[:TRAINING]

    Xs = Xall[-TEST:]
    dXs = dXall[-TEST:]
    Ns = Nall[-TEST:]
    dispXs = dispXall[:,-sum(Ns)*3:,:,:]
    Qs  = Qall[-TEST:]

    Kt_gdml = get_gdml_kernel(X, Xs, dX, dXs, Q, Qs, SIGMA)

    idx1 = 0

    K_numm = np.zeros(Kt_gdml.shape)

    for i in range(TRAINING):
        for n1 in range(N[i]):
            for xyz1 in range(3):
                for n_diff1 in range(N[i]):

                    idx2 = 0

                    for j in range(TEST):
                        for n2 in range(Ns[j]):
                            for xyz2 in range(3):
                                for n_diff2 in range(Ns[j]):


                                    if (Q[i][n_diff1] == Qs[j][n_diff2]):
                                        # displacements = [2*DX, DX, -DX, -2*DX]

                                        d = np.linalg.norm(dispX[1,idx1,n_diff1] - dispXs[1,idx2,n_diff2])
                                        gauss1 = np.exp(-d**2 / (2 * SIGMA**2))

                                        d = np.linalg.norm(dispX[1,idx1,n_diff1] - dispXs[2,idx2,n_diff2])
                                        gauss2 = np.exp(-d**2 / (2 * SIGMA**2))

                                        d = np.linalg.norm(dispX[2,idx1,n_diff1] - dispXs[1,idx2,n_diff2])
                                        gauss3 = np.exp(-d**2 / (2 * SIGMA**2))

                                        d = np.linalg.norm(dispX[2,idx1,n_diff1] - dispXs[2,idx2,n_diff2])
                                        gauss4 = np.exp(-d**2 / (2 * SIGMA**2))

                                        gauss = (gauss1 - gauss2 - gauss3 + gauss4 )/ (4*DX*DX)
                                        K_numm[idx2,idx1] += gauss

                                idx2 += 1

                idx1 += 1

    # Numerical hessian has quite large error, probably around 1e-7, so setting atol to 1e-6.
    assert np.allclose(Kt_gdml, K_numm, atol=1e-6), "Error in get_gdml_kernel()"


def test_symmetric_gdml_kernel():

    Xall, dXall, Eall, Fall, dispXall, Nall, Qall= csv_to_molecular_reps(CSV_FILE)

    X  = Xall[:TRAINING]
    dX = dXall[:TRAINING]
    N  = Nall[:TRAINING]
    dispX  = dispXall[:,:sum(N)*3,:,:]
    Q  = Qall[:TRAINING]

    Xs = Xall[-TEST:]
    dXs = dXall[-TEST:]
    Ns = Nall[-TEST:]
    dispXs = dispXall[:,-sum(Ns)*3:,:,:]
    Qs  = Qall[-TEST:]

    K = get_gdml_kernel(X, X, dX, dX, Q, Q, SIGMA)
    K_symm = get_symmetric_gdml_kernel(X, dX, Q,  SIGMA)

    assert np.allclose(K, K_symm), "Error in get_symmetric_gdml_kernel()"


def test_gp_kernel():

    Xall, dXall, Eall, Fall, dispXall, Nall, Qall= csv_to_molecular_reps(CSV_FILE)

    X  = Xall[:TRAINING]
    dX = dXall[:TRAINING]
    N  = Nall[:TRAINING]
    dispX  = dispXall[:,:sum(N)*3,:,:]
    Q  = Qall[:TRAINING]

    Xs = Xall[-TEST:]
    dXs = dXall[-TEST:]
    Ns = Nall[-TEST:]
    dispXs = dispXall[:,-sum(Ns)*3:,:,:]
    Qs  = Qall[-TEST:]

    K = get_gp_kernel(X, Xs, dX, dXs, Q, Qs, SIGMA)

    K_uu = get_local_kernel(X, Xs, Q, Qs, SIGMA)
    assert np.allclose(K[:TEST,:TRAINING], K_uu), "Error: Fail in Gaussian Process kernel K_uu"

    K_ug = get_local_gradient_kernel(X, Xs, dXs,  Q, Qs, SIGMA)
    assert np.allclose(K[TEST:,:TRAINING], K_ug), "Error: Fail in Gaussian Process kernel K_ug"

    K_gu = get_local_gradient_kernel(Xs, X, dX,  Qs, Q, SIGMA)
    assert np.allclose(K[:TEST,TRAINING:], K_gu.T), "Error: Fail in Gaussian Process kernel K_gu"

    K_gg = get_gdml_kernel(X, Xs, dX, dXs,  Q, Qs, SIGMA)

    assert np.allclose(K[TEST:,TRAINING:], K_gg), "Error: Fail in Gaussian Process kernel K_gg"

    K1_symm = get_gp_kernel(X, X, dX, dX, Q, Q, SIGMA)
    K2_symm = get_symmetric_gp_kernel(X, dX, Q, SIGMA)
    assert np.allclose(K1_symm, K2_symm), "Error: Fail in Gaussian Process symmetric kernel"


def test_local_kernels():

    Xall, dXall, Eall, Fall, dispXall, Nall, Qall= csv_to_molecular_reps(CSV_FILE)

    X  = Xall[:TRAINING]
    dX = dXall[:TRAINING]
    N  = Nall[:TRAINING]
    dispX  = dispXall[:,:sum(N)*3,:,:]
    Q  = Qall[:TRAINING]


    Xs = Xall[-TEST:]
    dXs = dXall[-TEST:]
    Ns = Nall[-TEST:]
    dispXs = dispXall[:,-sum(Ns)*3:,:,:]
    Qs  = Qall[-TEST:]

    sigma1 = 2.5
    sigma2 = 25.0

    K1 = get_local_kernel(X, Xs, Q, Qs, sigma1)
    K2 = get_local_kernel(X, Xs, Q, Qs, sigma2)
    K3 = get_local_kernel(X, Xs, Q, Qs, 10.0)
    K4 = get_local_kernel(X, Xs, Q, Qs, 2.0)

    K5 = get_local_kernel(X, Xs, Q, Qs, 3.0)

    K = get_local_kernels(X, Xs, Q, Qs, [sigma1, sigma2,10.0,2.0,3.0])


    assert np.allclose(K1, K[0]), "Error in get_local_kernels() 1"
    assert np.allclose(K2, K[1]), "Error in get_local_kernels() 2"

    K1 = get_local_kernel(X, X, Q, Q, sigma1)
    K2 = get_local_symmetric_kernel(X, Q, sigma2)

    K  = get_local_symmetric_kernels(X, Q, [sigma1, sigma2])

    assert np.allclose(K1, K[0]), "Error in get_local_symmetric_kernels() 1"
    assert np.allclose(K2, K[1]), "Error in get_local_symmetric_kernels() 2"

if __name__ == "__main__":

    test_local_kernel()
    test_atomic_local_kernel()
    test_atomic_local_gradient()
    test_local_gradient()
    test_gdml_kernel()
    test_symmetric_gdml_kernel()
    test_gp_kernel()
    test_global_kernel()
    test_local_kernels()
