"""
This script shows how to set up the MRMP estimator where the XYZ data is used to make QML compounds and global descriptors
are generated from the QML compounds and stored.
"""


from qml.aglaia.aglaia import MRMP
import glob
import numpy as np
import os

## ------------- ** Loading the data ** ---------------

current_dir = os.path.dirname(os.path.realpath(__file__))
filenames = glob.glob(current_dir + '/../test/CN_isobutane/*.xyz')
energies = np.loadtxt(current_dir + '/../test/CN_isobutane/prop_kjmol_training.txt', usecols=[1])
filenames.sort()

## ------------- ** Setting up the estimator ** ---------------

estimator = MRMP(representation='slatm', representation_params={'slatm_dgrid2': 0.06, 'slatm_dgrid1': 0.06})

estimator.generate_compounds(filenames[:100])
estimator.set_properties(energies[:100])

estimator.generate_representation()

##  ------------- ** Fitting to the data ** ---------------

idx = np.arange(0,100)

estimator.fit(idx)

##  ------------- ** Predicting and scoring ** ---------------

score = estimator.score(idx)

print("The mean absolute error is %s kJ/mol." % (str(-score)))

energies_predict = estimator.predict(idx)