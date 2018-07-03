"""
This test checks if all the ways of setting up the estimator ARMP work.
"""


import numpy as np
from qml.aglaia.aglaia import ARMP
import joblib
from qml.aglaia.utils import InputError
import glob
import os

def test_set_representation():
    """
    This function tests the function _set_representation.
    """
    try:
        ARMP(representation='slatm', descriptor_params={'slatm_sigma12': 0.05})
        raise Exception
    except InputError:
        pass

    try:
        ARMP(representation='coulomb_matrix')
        raise Exception
    except InputError:
        pass

    try:
        ARMP(representation='slatm', descriptor_params={'slatm_alchemy': 0.05})
        raise Exception
    except InputError:
        pass

    parameters = {'slatm_sigma1': 0.07, 'slatm_sigma2': 0.04, 'slatm_dgrid1': 0.02, 'slatm_dgrid2': 0.06,
                  'slatm_rcut': 5.0, 'slatm_rpower': 7, 'slatm_alchemy': True}

    estimator = ARMP(representation='slatm', descriptor_params=parameters)

    assert estimator.representation == 'slatm'
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
    data_correct = joblib.load(test_dir + "/data/local_slatm_ch4cn_light.bz")
    descriptor_correct = data_correct["descriptor"]
    descriptor_incorrect = data_incorrect["arr_0"]


    estimator = ARMP()

    assert estimator.descriptor == None

    estimator.set_descriptors(descriptors=descriptor_correct)

    assert np.all(estimator.descriptor == descriptor_correct)

    # Pass a descriptor with the wrong shape
    try:
        estimator.set_descriptors(descriptors=descriptor_incorrect)
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
    estimator.generate_descriptors()

    idx = np.arange(0, 50)
    estimator.fit(idx)

def test_fit_2():
    """
    This function tests the second way of fitting the descriptor: the data is passed by storing the compounds in the
    class.
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))

    data = joblib.load(test_dir + "/data/local_slatm_ch4cn_light.bz")
    descriptor = data["descriptor"]
    classes = data["zs"]
    energies = data["energies"]

    estimator = ARMP()
    estimator.set_descriptors(descriptors=descriptor)
    estimator.set_classes(classes=classes)
    estimator.set_properties(energies)

    idx = np.arange(0, 100)
    estimator.fit(idx)

def test_fit_3():
    """
    This function tests the thrid way of fitting the descriptor: the data is passed directly to the fit function.
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))

    data = joblib.load(test_dir + "/data/local_slatm_ch4cn_light.bz")
    descriptor = data["descriptor"]
    classes = data["zs"]
    energies = data["energies"]

    estimator = ARMP()
    estimator.fit(x=descriptor, y=energies, classes=classes)

def test_score_3():
    """
    This function tests that all the scoring functions work.
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))

    data = joblib.load(test_dir + "/data/local_slatm_ch4cn_light.bz")
    descriptor = data["descriptor"]
    classes = data["zs"]
    energies = data["energies"]

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

    data = joblib.load(test_dir + "/data/local_slatm_ch4cn_light.bz")
    descriptor = data["descriptor"]
    classes = data["zs"]
    energies = data["energies"]

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