"""
This test checks if all the ways of setting up the estimator MRMP work.
"""


import numpy as np
from qml.aglaia.aglaia import MRMP
import joblib
from qml.aglaia.utils import InputError
import glob
import os

def test_set_representation():
    """
    This function tests the method MRMP._set_representation.
    """
    try:
        MRMP(representation='unsorted_coulomb_matrix', descriptor_params={'slatm_sigma1': 0.05})
        raise Exception
    except InputError:
        pass

    try:
        MRMP(representation='coulomb_matrix')
        raise Exception
    except InputError:
        pass

    try:
        MRMP(representation='slatm', descriptor_params={'slatm_alchemy': 0.05})
        raise Exception
    except InputError:
        pass

    parameters ={'slatm_sigma1': 0.07, 'slatm_sigma2': 0.04, 'slatm_dgrid1': 0.02, 'slatm_dgrid2': 0.06,
                                'slatm_rcut': 5.0, 'slatm_rpower': 7, 'slatm_alchemy': True}

    estimator = MRMP(representation='slatm', descriptor_params=parameters)

    assert estimator.representation == 'slatm'
    assert estimator.slatm_parameters == parameters

def test_set_properties():
    """
    This test checks that the MRMP.set_properties method works.
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))

    energies = np.loadtxt(test_dir + '/CN_isobutane/prop_kjmol_training.txt',
                          usecols=[1])

    estimator = MRMP(representation='unsorted_coulomb_matrix')

    assert estimator.properties == None

    estimator.set_properties(energies)

    assert np.all(estimator.properties == energies)

def test_set_descriptor():
    """
    This test checks that the set_descriptor function works as expected.
    """

    test_dir = os.path.dirname(os.path.realpath(__file__))

    data_correct = np.load(test_dir + "/data/CN_isopent_light_UCM.npz")
    data_incorrect = joblib.load(test_dir + "/data/local_slatm_ch4cn_light.bz")
    descriptor_correct = data_correct["arr_0"]
    descriptor_incorrect = data_incorrect["descriptor"]

    estimator = MRMP()

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
    This function tests the first way of preparing for fitting the neural network: 
    Compounds are created from xyz files and the energies are stored in the estimator.
    The fit method is called with the indices of the molecules we want to fit.
    """

    test_dir = os.path.dirname(os.path.realpath(__file__))

    filenames = glob.glob(test_dir + "/CN_isobutane/*.xyz")
    energies = np.loadtxt(test_dir + '/CN_isobutane/prop_kjmol_training.txt',
                          usecols=[1])
    filenames.sort()

    available_representations = ['sorted_coulomb_matrix', 'unsorted_coulomb_matrix', 'bag_of_bonds', 'slatm']

    for rep in available_representations:
        estimator = MRMP(representation=rep)
        estimator.generate_compounds(filenames[:100])
        estimator.set_properties(energies[:100])
        estimator.generate_descriptors()

        idx = np.arange(0, 100)
        estimator.fit(idx)

def test_fit_2():
    """
    This function tests a second way of fitting the descriptor:
    The premade descriptors are stored in the estimator together with the energies.
    The fit method is called with the indices of the molecules we want to fit.
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))

    data = np.load(test_dir + "/data/CN_isopent_light_UCM.npz")
    descriptor = data["arr_0"]
    energies = data["arr_1"]

    estimator = MRMP()
    estimator.set_descriptors(descriptors=descriptor)
    estimator.set_properties(energies)

    idx = np.arange(0, 100)
    estimator.fit(idx)

def test_fit_3():
    """
    This function tests a third way of fitting the descriptor: 
    The data is passed directly to the fit function.
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))

    data = np.load(test_dir + "/data/CN_isopent_light_UCM.npz")
    descriptor = data["arr_0"]
    energies = data["arr_1"]

    estimator = MRMP()
    estimator.fit(descriptor, energies)

def test_score():
    """
    This function tests that all the scoring functions work.
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))

    data = np.load(test_dir + "/data/CN_isopent_light_UCM.npz")
    descriptor = data["arr_0"]
    energies = data["arr_1"]

    estimator_1 = MRMP(scoring_function='mae')
    estimator_1.fit(descriptor, energies)
    estimator_1.score(descriptor, energies)

    estimator_2 = MRMP(scoring_function='r2')
    estimator_2.fit(descriptor, energies)
    estimator_2.score(descriptor, energies)

    estimator_3 = MRMP(scoring_function='rmse')
    estimator_3.fit(descriptor, energies)
    estimator_3.score(descriptor, energies)


if __name__ == "__main__":

    test_set_properties()
    test_set_descriptor()
    test_set_representation()
    test_fit_1()
    test_fit_2()
    test_fit_3()
    test_score()
