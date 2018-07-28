"""
This script shows how to set up the ARMP estimator where the descriptor is set directly and stored in the class.
"""

from qml.aglaia.aglaia import ARMP
import numpy as np
import os

## ------------- ** Loading the data ** ---------------

current_dir = os.path.dirname(os.path.realpath(__file__))
data = np.load(current_dir + '/../test/data/local_slatm_ch4cn_light.npz')

descriptor = data["arr_0"]
zs = data["arr_1"]
energies = data["arr_2"]

## ------------- ** Setting up the estimator ** ---------------

estimator = ARMP(iterations=100, l2_reg=0.0)

estimator.set_representations(representations=descriptor)
estimator.set_classes(zs)
estimator.set_properties(energies)

##  ------------- ** Fitting to the data ** ---------------

idx = np.arange(0,100)

estimator.fit(idx)

##  ------------- ** Predicting and scoring ** ---------------

score = estimator.score(idx)

print("The mean absolute error is %s kJ/mol." % (str(-score)))

energies_predict = estimator.predict(idx)

