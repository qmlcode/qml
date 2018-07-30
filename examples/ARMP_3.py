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

"""
This script shows how to set up the ARMP estimator where the data to be fitted is passed directly to the fit function.
"""

from qml.aglaia.aglaia import ARMP
import os
import numpy as np

## ------------- ** Loading the data ** ---------------

current_dir = os.path.dirname(os.path.realpath(__file__))
data = np.load(current_dir + '/../test/data/local_slatm_ch4cn_light.npz')

representation = data["arr_0"]
zs = data["arr_1"]
energies = data["arr_2"]

## ------------- ** Setting up the estimator ** ---------------

estimator = ARMP(iterations=150, l2_reg=0.0, learning_rate=0.005, hidden_layer_sizes=(40, 20, 10))

##  ------------- ** Fitting to the data ** ---------------

estimator.fit(x=representation, y=energies, classes=zs)

##  ------------- ** Predicting and scoring ** ---------------

score = estimator.score(x=representation, y=energies, classes=zs)

print("The mean absolute error is %s kJ/mol." % (str(-score)))

energies_predict = estimator.predict(x=representation, classes=zs)
