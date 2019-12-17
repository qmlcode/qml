# MIT License
#
# Copyright (c) 2017-2019 Anders Steen Christensen, Jakub Wagner
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

from __future__ import print_function

import sys
import os
import numpy as np
import scipy

import qml
from qml.helpers import get_BoB_groups, compose_BoB_sigma_vector
from qml.kernels import gaussian_kernel, laplacian_kernel
from qml.kernels import gaussian_sigmas_kernel, laplacian_sigmas_kernel

def test_indices_getter():
    """
    Test if indices of BoB groups are correctly returned.
    """
    asize = {"C":1, "O":2, "H":2}
    asize = {k: asize[k] for k in sorted(asize, key=asize.get)}   # Sort asize
    correct_low_indices = {'CC': 0, 'CO': 1, 'CH': 3, 'OO': 5, 'OH': 8, 'HH': 12}
    correct_high_indices = {'CC': 1, 'CO': 3, 'CH': 5, 'OO': 8, 'OH': 12, 'HH': 15}
    low_indices, high_indices = get_BoB_groups(asize)
    assert low_indices == correct_low_indices
    assert high_indices == correct_high_indices
    asize = {"O":5, "C":9, "N":7, "H":20, "F":6}
    asize = {k: asize[k] for k in sorted(asize, key=asize.get)}   # Sort asize
    correct_low_indices = {'OO': 0  , 'OF': 15 , 'ON': 45,  'OC': 80, 'OH': 125,
                           'FF': 225, 'FN': 246, 'FC': 288, 'FH': 342,
                           'NN': 462, 'NC': 490, 'NH': 553,
                           'CC': 693, 'CH': 738,
                           'HH': 918}
    correct_high_indices = {'OO': 15,  'OF': 45,  'ON': 80,  'OC': 125, 'OH': 225,
                            'FF': 246, 'FN': 288, 'FC': 342, 'FH': 462,
                            'NN': 490, 'NC': 553, 'NH': 693,
                            'CC': 738, 'CH': 918,
                            'HH': 1128}
    low_indices, high_indices = get_BoB_groups(asize)
    assert low_indices == correct_low_indices
    assert high_indices == correct_high_indices

def test_sigma_vector_composition():
    """
    Test if vector of per-feature sigmas is correctly composed.
    """
    asize = {"C":1, "O":2, "H":2}
    asize = {k: asize[k] for k in sorted(asize, key=asize.get)}   # Sort asize
    low_indices, high_indices = get_BoB_groups(asize)
    sigmas_for_bags = {'CC': 1, 'CO': 2, 'CH': 3, 'OO': 4, 'OH': 5, 'HH': 6}
    sigmas = compose_BoB_sigma_vector(sigmas_for_bags, low_indices, high_indices)
    correct_sigmas = np.array([1., 2., 2., 3., 3., 4., 4., 4., 5., 5., 5., 5., 6., 6., 6.])
    assert np.allclose(sigmas, correct_sigmas)

def test_gaussian_sigmas_kernel():
    """
    Test if Gaussian kernel with per-feature sigmas work correctly.
    """
    np.random.seed(666)
    n_train = 25
    n_test = 20
    n_features = 1000
    # List of dummy representations
    X_train = np.random.rand(n_train, n_features)
    X_test = np.random.rand(n_test, n_features)
    sigmas = np.random.rand(n_features)
    K_test = np.ones((n_train, n_test))
    for i in range(n_train):
        for j in range(n_test):
            for k in range(n_features):
                K_test[i,j] *= np.exp(-np.abs((X_train[i,k]-X_test[j,k])**2/(2.0*sigmas[k]**2)))
    K = gaussian_sigmas_kernel(X_train, X_test, sigmas)
    assert np.allclose(K, K_test)

def test_laplacian_sigmas_kernel():
    """
    Test if Laplacian kernel with per-feature sigmas work correctly.
    """
    np.random.seed(666)
    n_train = 25
    n_test = 20
    n_features = 1000
    # List of dummy representations
    X_train = np.random.rand(n_train, n_features)
    X_test = np.random.rand(n_test, n_features)
    sigmas = np.random.rand(n_features)
    K_test = np.ones((n_train, n_test))
    for i in range(n_train):
        for j in range(n_test):
            for k in range(n_features):
                K_test[i,j] *= np.exp(-np.abs((X_train[i,k]-X_test[j,k])/(sigmas[k])))
    K = laplacian_sigmas_kernel(X_train, X_test, sigmas)
    assert np.allclose(K, K_test)

def test_single_sigma_gaussian():
    """
    Test if gaussian_sigmas_kernel gives the same result as gaussian_kernel if
    all local sigmas are set to the global sigma.
    """
    np.random.seed(666)
    n_train = 25
    n_test = 20
    # List of dummy representations
    X_train = np.random.rand(n_train, 1000)
    X_test = np.random.rand(n_test, 1000)
    global_sigma = 1.0
    sigmas = np.ones(1000)*global_sigma
    K = gaussian_sigmas_kernel(X_train, X_test, sigmas)
    K_test = gaussian_kernel(X_train, X_test, global_sigma)
    assert np.allclose(K, K_test)
    K_symm = gaussian_sigmas_kernel(X_train, X_train, sigmas)
    assert np.allclose(K_symm, K_symm.T)

def test_single_sigma_laplacian():
    """
    Test if laplacian_sigmas_kernel gives the same result as laplacian_kernel
    if all local sigmas are set to the global sigma.
    """
    np.random.seed(666)
    n_train = 25
    n_test = 20
    # List of dummy representations
    X_train = np.random.rand(n_train, 1000)
    X_test = np.random.rand(n_test, 1000)
    global_sigma = 1.0
    sigmas = np.ones(1000)*global_sigma
    K = laplacian_sigmas_kernel(X_train, X_test, sigmas)
    K_test = gaussian_kernel(X_train, X_test, global_sigma)
    assert np.allclose(K, K_test)
    K_symm = laplacian_sigmas_kernel(X_train, X_train, sigmas)
    assert np.allclose(K_symm, K_symm.T)

if __name__ == "__main__":
    test_indices_getter()
    test_sigma_vector_composition()
    test_gaussian_kernel()
    test_laplacian_kernel()
    test_single_sigma_gaussian()
    test_single_sigma_laplacian()
