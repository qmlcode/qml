# MIT License
#
# Copyright (c) 2017 Anders Steen Christensen
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
import numpy as np
import qml
from qml.kernels import laplacian_kernel
from qml.kernels import gaussian_kernel

def test_laplacian_kernel(n_train, n_test, X, Xs):

    n_train = 25
    n_test = 20

    # List of dummy representations
    X = np.random.rand(n_train, 1000)
    Xs = np.random.rand(n_test, 1000)

    sigma = 100.0

    Ltest = np.zeros((n_train, n_test))

    for i in range(n_train):
        for j in range(n_test):
            Ltest[i,j] = np.exp( np.sum(np.abs(X[i] - Xs[j])) / (-1.0 * sigma))

    L = laplacian_kernel(X, Xs, sigma)

    # Compare two implementations:
    assert np.allclose(L, Ltest), "Error in Laplacian kernel"

    Lsymm = laplacian_kernel(X, X, sigma)

    # Check for symmetry:
    assert np.allclose(Lsymm, Lsymm.T), "Error in Laplacian kernel"

def test_gaussian_kernel(n_train, n_test, X, Xs):

    n_train = 25
    n_test = 20

    # List of dummy representations
    X = np.random.rand(n_train, 1000)
    Xs = np.random.rand(n_test, 1000)

    sigma = 100.0

    Gtest = np.zeros((n_train, n_test))

    for i in range(n_train):
        for j in range(n_test):
            Gtest[i,j] = np.exp( np.sum(np.square(X[i] - Xs[j])) / (-2.0 * sigma**2))

    G = gaussian_kernel(X, Xs, sigma)

    # Compare two implementations:
    assert np.allclose(G, Gtest), "Error in Gaussian kernel"

    Gsymm = gaussian_kernel(X, X, sigma)

    # Check for symmetry:
    assert np.allclose(Gsymm, Gsymm.T), "Error in Gaussian kernel"

def test_kernels():

    test_laplacian_kernel()
    test_gaussian_kernel()

if __name__ == "__main__":
    test_kernels()

