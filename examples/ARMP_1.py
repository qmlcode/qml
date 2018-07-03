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

estimator = ARMP(iterations=100, representation='acsf', descriptor_params={"radial_rs": np.arange(0,10, 0.1), "angular_rs": np.arange(0.5, 10.5, 0.1),
"theta_s": np.arange(0, 5, 0.1)}, tensorboard=False)

estimator.generate_compounds(filenames[:100])
estimator.set_properties(energies[:100])

estimator.generate_descriptors()

##  ------------- ** Fitting to the data ** ---------------

idx = np.arange(0,100)

estimator.fit(idx)


##  ------------- ** Predicting and scoring ** ---------------

score = estimator.score(idx)

print("The mean absolute error is %s kJ/mol." % (str(-score)))

energies_predict = estimator.predict(idx)