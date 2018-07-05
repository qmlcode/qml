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

import numpy as np

from qml.ml.kernels.distance import *


def test_manhattan():

    nfeatures = 5
    n1 = 7
    n2 = 9

    v1 = np.random.random((n1, nfeatures))
    v2 = np.random.random((n2, nfeatures))

    D = manhattan_distance(v1, v2)

    Dtest = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            for k in range(nfeatures):
                Dtest[i,j] += abs(v1[i, k] - v2[j, k])

    assert np.allclose(D, Dtest), "Error in manhattan distance"

def test_l2():

    nfeatures = 5
    n1 = 7
    n2 = 9

    v1 = np.random.random((n1, nfeatures))
    v2 = np.random.random((n2, nfeatures))

    D = l2_distance(v1, v2)

    Dtest = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            for k in range(nfeatures):
                Dtest[i,j] += (v1[i, k] - v2[j, k])**2

    np.sqrt(Dtest, out=Dtest)

    assert np.allclose(D, Dtest), "Error in l2 distance"

def test_p():

    nfeatures = 5
    n1 = 7
    n2 = 9

    v1 = np.random.random((n1, nfeatures))
    v2 = np.random.random((n2, nfeatures))

    D = p_distance(v1, v2, 3)


    Dtest = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            for k in range(nfeatures):
                Dtest[i,j] += abs(v1[i, k] - v2[j, k])**3

    Dtest = Dtest**(1.0/3)

    assert np.allclose(D, Dtest), "Error in p-distance"

    Dfloat = p_distance(v1, v2, 3.0)
    assert np.allclose(D, Dfloat), "Floatingpoint Error in p-distance"

if __name__ == "__main__":
    test_manhattan()
    test_l2()
    test_p()
