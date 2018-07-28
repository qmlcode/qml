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
This script shows how to set up the ARMP estimator where the XYZ data is used to make QML compounds and the local
descriptors are generated from the QML compounds and then stored.
"""

from qml.aglaia.aglaia import ARMP
import glob
import numpy as np
import os

## ------------- ** Loading the data ** ---------------

current_dir = os.path.dirname(os.path.realpath(__file__))
filenames = glob.glob(current_dir + '/../test/CN_isobutane/*.xyz')
energies = np.loadtxt(current_dir + '/../test/CN_isobutane/prop_kjmol_training.txt', usecols=[1])
filenames.sort()

## ------------- ** Setting up the estimator ** ---------------

estimator = ARMP(iterations=10, representation='acsf', representation_params={"radial_rs": np.arange(0, 10, 1), "angular_rs": np.arange(0.5, 10.5, 1),
"theta_s": np.arange(0, 5, 1)}, tensorboard=False)

estimator.generate_compounds(filenames)
estimator.set_properties(energies)

estimator.generate_representation()

##  ------------- ** Fitting to the data ** ---------------

idx = np.arange(0,100)

estimator.fit(idx)


##  ------------- ** Predicting and scoring ** ---------------

score = estimator.score(idx)

print("The mean absolute error is %s kJ/mol." % (str(-score)))

energies_predict = estimator.predict(idx)
