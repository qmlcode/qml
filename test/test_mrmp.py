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
This test checks if all the ways of setting up the estimator MRMP work.
"""

import numpy as np
from qml.aglaia.aglaia import MRMP
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
    data_incorrect = np.load(test_dir + "/data/local_slatm_ch4cn_light.npz")
    descriptor_correct = data_correct["arr_0"]
    descriptor_incorrect = data_incorrect["arr_0"]

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

def test_save_local():
    """
    This function tests the saving and the loading of a trained model.
    """

    x = np.linspace(-10.0, 10.0, 2000)
    y = x ** 2

    x = np.reshape(x, (x.shape[0], 1))

    estimator = MRMP()
    estimator.fit(x=x, y=y)

    score_after_training = estimator.score(x, y)
    estimator.save_nn()

    estimator.load_nn()
    score_after_loading = estimator.score(x, y)

    assert score_after_loading == score_after_training

def test_load_external():
    """
    This function tests if a model that has been trained on a different computer can be loaded and used on a different
    computer.
    """

    x = np.linspace(-10.0, 10.0, 2000)
    y = x ** 2
    x = np.reshape(x, (x.shape[0], 1))

    estimator = MRMP()
    estimator.load_nn("saved_model")

    score_after_loading = estimator.score(x, y)
    score_on_other_machine = -24.101043

    assert np.isclose(score_after_loading, score_on_other_machine)


if __name__ == "__main__":

    # test_set_properties()
    # test_set_descriptor()
    # test_set_representation()
    # test_fit_1()
    # test_fit_2()
    # test_fit_3()
    # test_score()

    test_load_external()
