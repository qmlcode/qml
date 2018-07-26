"""
This script shows how to set up the MRMP estimator where the descriptor is set directly and stored in the class.
"""

from qml.aglaia.aglaia import MRMP
import numpy as np
import os

## ------------- ** Loading the data ** ---------------

# The data loaded contains 100 samples of the CN + isobutane data set in unsorted CM representation
current_dir = os.path.dirname(os.path.realpath(__file__))
data = np.load(current_dir + '/../test/data/CN_isopent_light_UCM.npz')

representation = data["arr_0"]
energies = data["arr_1"]

## ------------- ** Setting up the estimator ** ---------------

estimator = MRMP(iterations=7000, l2_reg=0.0)

estimator.set_representations(representations=representation)
estimator.set_properties(energies)

##  ------------- ** Fitting to the data ** ---------------

idx = np.arange(0,100)

estimator.fit(idx)

##  ------------- ** Predicting and scoring ** ---------------

score = estimator.score(idx)

print("The mean absolute error is %s kJ/mol." % (str(-score)))

energies_predict = estimator.predict(idx)