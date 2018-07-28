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
