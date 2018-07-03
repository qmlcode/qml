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

import os

import numpy as np

from copy import deepcopy

import qml
import qml.ml.math

def test_cho_solve():

    test_dir = os.path.dirname(os.path.realpath(__file__))

    A_ref = np.loadtxt(test_dir + "/data/K_local_gaussian.txt")
    y_ref = np.loadtxt(test_dir + "/data/y_cho_solve.txt")

    A = deepcopy(A_ref) 
    y = deepcopy(y_ref) 
    x_qml   = qml.ml.math.cho_solve(A,y)

    # Check arrays are unchanged
    assert np.allclose(y, y_ref)
    assert np.allclose(A, A_ref)

    A = deepcopy(A_ref) 
    x_scipy = np.linalg.solve(A, y)

    # Check for correct solution
    assert np.allclose(x_qml, x_scipy)


def test_cho_invert():

    test_dir = os.path.dirname(os.path.realpath(__file__))

    A_ref = np.loadtxt(test_dir + "/data/K_local_gaussian.txt")

    A = deepcopy(A_ref) 
    Ai_qml = qml.ml.math.cho_invert(A)

    # Check A is unchanged
    assert np.allclose(A, A_ref)

    A = deepcopy(A_ref) 
    one = np.eye(A.shape[0])
    
    # Check that it is a true inverse
    assert np.allclose(np.matmul(A, Ai_qml), one, atol=1e-7)


def test_bkf_invert():

    test_dir = os.path.dirname(os.path.realpath(__file__))

    A_ref = np.loadtxt(test_dir + "/data/K_local_gaussian.txt")

    A = deepcopy(A_ref) 
    Ai_qml = qml.ml.math.bkf_invert(A)

    # Check A is unchanged
    assert np.allclose(A, A_ref)

    A = deepcopy(A_ref) 
    one = np.eye(A.shape[0])
   
    np.set_printoptions(linewidth=20000)
    assert np.allclose(np.matmul(A, Ai_qml), one, atol=1e-7)


def test_bkf_solve():

    test_dir = os.path.dirname(os.path.realpath(__file__))

    A_ref = np.loadtxt(test_dir + "/data/K_local_gaussian.txt")
    y_ref = np.loadtxt(test_dir + "/data/y_cho_solve.txt")

    A = deepcopy(A_ref) 
    y = deepcopy(y_ref)
    x_qml   = qml.ml.math.bkf_solve(A,y)

    # Check arrays are unchanged
    assert np.allclose(y, y_ref)
    assert np.allclose(A, A_ref)

    A = deepcopy(A_ref) 
    y = deepcopy(y_ref) 
    x_scipy = np.linalg.solve(A, y)

    # Check for correct solution
    assert np.allclose(x_qml, x_scipy)


if __name__ == "__main__":
    test_cho_solve()
    test_cho_invert()
    test_bkf_invert()
    test_bkf_solve()
