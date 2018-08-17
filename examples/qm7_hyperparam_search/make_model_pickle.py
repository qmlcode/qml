# MIT License
#
# Copyright (c) 2018 Silvia Amabilino
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


import pickle
import numpy as np
from qml.ml.representations import CoulombMatrix
from qml.ml.kernels import GaussianKernel
from qml.models import KernelRidgeRegression
from qml.data import Data
from sklearn.pipeline import make_pipeline

data = Data(filenames="../../test/qm7/*.xyz")
energies = np.loadtxt('../../test/data/hof_qm7.txt', usecols=[1])
data.set_energies(energies)
estimator = make_pipeline(CoulombMatrix(size=23, data=data), GaussianKernel(), KernelRidgeRegression())

pickle.dump(estimator, open('model.pickle', 'wb'))
with open('idx.csv', 'w') as f:
    for i in range(energies.size):
        f.write('%s\n' % i)
