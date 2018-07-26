"""
This script shows how to set up the MRMP estimator where the data to be fitted is passed directly to the fit function.
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

estimator = MRMP()

##  ------------- ** Fitting to the data ** ---------------

estimator.fit(representation, energies)

##  ------------- ** Predicting and scoring ** ---------------

score = estimator.score(representation, energies)

print("The mean absolute error is %s kJ/mol." % (str(-score)))

energies_predict = estimator.predict(representation)
