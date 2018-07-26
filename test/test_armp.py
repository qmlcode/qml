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
This test checks if all the ways of setting up the estimator ARMP work.
"""


import numpy as np
from qml.aglaia.aglaia import ARMP
from qml.aglaia.utils import InputError
import glob
import os

def test_set_representation():
    """
    This function tests the function _set_representation.
    """
    try:
        ARMP(representation='slatm', representation_params={'slatm_sigma12': 0.05})
        raise Exception
    except InputError:
        pass

    try:
        ARMP(representation='coulomb_matrix')
        raise Exception
    except InputError:
        pass

    try:
        ARMP(representation='slatm', representation_params={'slatm_alchemy': 0.05})
        raise Exception
    except InputError:
        pass

    parameters = {'slatm_sigma1': 0.07, 'slatm_sigma2': 0.04, 'slatm_dgrid1': 0.02, 'slatm_dgrid2': 0.06,
                  'slatm_rcut': 5.0, 'slatm_rpower': 7, 'slatm_alchemy': True}

    estimator = ARMP(representation='slatm', representation_params=parameters)

    assert estimator.representation_name == 'slatm'
    assert estimator.slatm_parameters == parameters

def test_set_properties():
    """
    This test checks that the set_properties function sets the correct properties.
    :return:
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))

    energies = np.loadtxt(test_dir + '/CN_isobutane/prop_kjmol_training.txt',
                          usecols=[1])

    estimator = ARMP(representation='slatm')

    assert estimator.properties == None

    estimator.set_properties(energies)

    assert np.all(estimator.properties == energies)

def test_set_descriptor():
    """
    This test checks that the set_descriptor function works as expected.
    :return:
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))

    data_incorrect = np.load(test_dir + "/data/CN_isopent_light_UCM.npz")
    data_correct = np.load(test_dir + "/data/local_slatm_ch4cn_light.npz")
    descriptor_correct = data_correct["arr_0"]
    descriptor_incorrect = data_incorrect["arr_0"]


    estimator = ARMP()

    assert estimator.representation == None

    estimator.set_representations(representations=descriptor_correct)

    assert np.all(estimator.representation == descriptor_correct)

    # Pass a descriptor with the wrong shape
    try:
        estimator.set_representations(representations=descriptor_incorrect)
        raise Exception
    except InputError:
        pass

def test_fit_1():
    """
    This function tests the first way of fitting the descriptor: the data is passed by first creating compounds and then
    the descriptors are created from the compounds.
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))

    filenames = glob.glob(test_dir + "/CN_isobutane/*.xyz")
    energies = np.loadtxt(test_dir + '/CN_isobutane/prop_kjmol_training.txt',
                          usecols=[1])
    filenames.sort()

    estimator = ARMP(representation="acsf")
    estimator.generate_compounds(filenames[:50])
    estimator.set_properties(energies[:50])
    estimator.generate_representation()

    idx = np.arange(0, 50)
    estimator.fit(idx)

def test_fit_2():
    """
    This function tests the second way of fitting the descriptor: the data is passed by storing the compounds in the
    class.
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))

    data = np.load(test_dir + "/data/local_slatm_ch4cn_light.npz")
    descriptor = data["arr_0"]
    classes = data["arr_1"]
    energies = data["arr_2"]

    estimator = ARMP()
    estimator.set_representations(representations=descriptor)
    estimator.set_classes(classes=classes)
    estimator.set_properties(energies)

    idx = np.arange(0, 100)
    estimator.fit(idx)

def test_fit_3():
    """
    This function tests the thrid way of fitting the descriptor: the data is passed directly to the fit function.
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))

    data = np.load(test_dir + "/data/local_slatm_ch4cn_light.npz")
    descriptor = data["arr_0"]
    classes = data["arr_1"]
    energies = data["arr_2"]

    estimator = ARMP()
    estimator.fit(x=descriptor, y=energies, classes=classes)

def test_score_3():
    """
    This function tests that all the scoring functions work.
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))

    data = np.load(test_dir + "/data/local_slatm_ch4cn_light.npz")
    descriptor = data["arr_0"]
    classes = data["arr_1"]
    energies = data["arr_2"]

    estimator_1 = ARMP(scoring_function='mae')
    estimator_1.fit(x=descriptor, y=energies, classes=classes)
    estimator_1.score(x=descriptor, y=energies, classes=classes)

    estimator_2 = ARMP(scoring_function='r2')
    estimator_2.fit(x=descriptor, y=energies, classes=classes)
    estimator_2.score(x=descriptor, y=energies, classes=classes)

    estimator_3 = ARMP(scoring_function='rmse')
    estimator_3.fit(x=descriptor, y=energies, classes=classes)
    estimator_3.score(x=descriptor, y=energies, classes=classes)

def test_predict_3():
    test_dir = os.path.dirname(os.path.realpath(__file__))

    data = np.load(test_dir + "/data/local_slatm_ch4cn_light.npz")
    descriptor = data["arr_0"]
    classes = data["arr_1"]
    energies = data["arr_2"]

    estimator = ARMP()
    estimator.fit(x=descriptor, y=energies, classes=classes)
    energies_pred = estimator.predict(x=descriptor, classes=classes)

    assert energies.shape == energies_pred.shape

if __name__ == "__main__":
    test_set_representation()
    test_set_properties()
    test_set_descriptor()
    test_fit_1()
    test_fit_2()
    test_fit_3()
    test_score_3()
    test_predict_3()
