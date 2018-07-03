"""
This file contains tests for the atom centred symmetry function module.
"""

import tensorflow as tf
import numpy as np

from qml.aglaia import symm_funct
from qml.aglaia import np_symm_funct
from qml.aglaia import tensormol_symm_funct
import os


def test_acsf_1():
    """
    This test compares the atom centred symmetry functions generated with tensorflow and numpy.
    The test system consists of 5 configurations of CH4 + CN radical.

    :return: None
    """

    radial_cutoff = 10.0
    angular_cutoff = 10.0
    radial_rs = [0.0, 0.1, 0.2]
    angular_rs = [0.0, 0.1, 0.2]
    theta_s = [0.0, 0.1, 0.2]
    zeta = 3.0
    eta = 2.0

    test_dir = os.path.dirname(os.path.realpath(__file__))
    input_data = test_dir + "/data/data_test_acsf.npz"
    data = np.load(input_data)

    xyzs = data["arr_0"]
    zs = data["arr_1"]
    elements = data["arr_2"]
    element_pairs = data["arr_3"]

    n_samples = xyzs.shape[0]
    n_atoms = zs.shape[1]

    with tf.name_scope("Inputs"):
        zs_tf = tf.placeholder(shape=[n_samples, n_atoms], dtype=tf.int32, name="zs")
        xyz_tf = tf.placeholder(shape=[n_samples, n_atoms, 3], dtype=tf.float32, name="xyz")

    acsf_tf_t = symm_funct.generate_parkhill_acsf(xyz_tf, zs_tf, elements, element_pairs, radial_cutoff, angular_cutoff, radial_rs, angular_rs, theta_s, zeta, eta)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    acsf_tf = sess.run(acsf_tf_t, feed_dict={xyz_tf: xyzs, zs_tf: zs})

    acsf_np = np_symm_funct.generate_acsf(xyzs, zs, elements, element_pairs, radial_cutoff, angular_cutoff, radial_rs, angular_rs, theta_s, zeta, eta)

    n_samples = xyzs.shape[0]
    n_atoms = xyzs.shape[1]

    for i in range(n_samples):
        for j in range(n_atoms):
            acsf_np_sort = np.sort(acsf_np[i][j])
            acsf_tf_sort = np.sort(acsf_tf[i][j])
            np.testing.assert_array_almost_equal(acsf_np_sort, acsf_tf_sort, decimal=4)

def test_acsf_2():
    """
    This test compares the atom centred symmetry functions generated with tensorflow and tensormol.
    The test system consists of 4 atoms at very predictable positions.

    :return:
    """

    radial_cutoff = 10.0
    angular_cutoff = 10.0
    radial_rs = [0.0, 0.1, 0.2]
    angular_rs = [0.0, 0.1, 0.2]
    theta_s = [0.0, 0.1, 0.2]
    zeta = 3.0
    eta = 2.0

    test_dir = os.path.dirname(os.path.realpath(__file__))
    input_data = test_dir + "/data/data_test_acsf_01.npz"
    data = np.load(input_data)

    xyzs = data["arr_0"]
    zs = data["arr_1"]
    elements = data["arr_2"]
    element_pairs = data["arr_3"]

    n_samples = xyzs.shape[0]
    n_atoms = zs.shape[1]

    with tf.name_scope("Inputs"):
        zs_tf = tf.placeholder(shape=[n_samples, n_atoms], dtype=tf.int32, name="zs")
        xyz_tf = tf.placeholder(shape=[n_samples, n_atoms, 3], dtype=tf.float32, name="xyz")

    acsf_tf_t = symm_funct.generate_parkhill_acsf(xyz_tf, zs_tf, elements, element_pairs, radial_cutoff, angular_cutoff, radial_rs, angular_rs, theta_s, zeta, eta)
    acsf_tm_t = tensormol_symm_funct.tensormol_acsf(xyz_tf, zs_tf, elements, element_pairs, radial_cutoff, angular_cutoff, radial_rs, angular_rs, theta_s, zeta, eta)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    acsf_tf = sess.run(acsf_tf_t, feed_dict={xyz_tf: xyzs, zs_tf: zs})
    acsf_tm = sess.run(acsf_tm_t, feed_dict={xyz_tf: xyzs, zs_tf: zs})

    acsf_tf = np.reshape(acsf_tf, acsf_tm.shape)

    for i in range(acsf_tm.shape[0]):
        acsf_tm_sort = np.sort(acsf_tm[i])
        acsf_tf_sort = np.sort(acsf_tf[i])
        np.testing.assert_array_almost_equal(acsf_tm_sort, acsf_tf_sort, decimal=1)

def test_acsf_3():
    """
    This test compares the atom centred symmetry functions generated with tensorflow and numpy.
    The test system consists of 10 molecules from the QM7 data set.

    :return: None
    """
    radial_cutoff = 10.0
    angular_cutoff = 10.0
    radial_rs = [0.0, 0.1, 0.2]
    angular_rs = [0.0, 0.1, 0.2]
    theta_s = [0.0, 0.1, 0.2]
    zeta = 3.0
    eta = 2.0

    test_dir = os.path.dirname(os.path.realpath(__file__))
    input_data = test_dir + "/data/qm7_testdata.npz"
    data = np.load(input_data)

    xyzs = data["arr_0"]
    zs = data["arr_1"]
    elements = data["arr_2"]
    element_pairs = data["arr_3"]

    n_samples = xyzs.shape[0]
    max_n_atoms = zs.shape[1]

    with tf.name_scope("Inputs"):
        zs_tf = tf.placeholder(shape=[n_samples, max_n_atoms], dtype=tf.int32, name="zs")
        xyz_tf = tf.placeholder(shape=[n_samples, max_n_atoms, 3], dtype=tf.float32, name="xyz")

    acsf_tf_t = symm_funct.generate_parkhill_acsf(xyz_tf, zs_tf, elements, element_pairs, radial_cutoff, angular_cutoff, radial_rs, angular_rs, theta_s, zeta, eta)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    acsf_tf = sess.run(acsf_tf_t, feed_dict={xyz_tf: xyzs, zs_tf: zs})

    acsf_np = np_symm_funct.generate_acsf(xyzs, zs, elements, element_pairs, radial_cutoff, angular_cutoff, radial_rs, angular_rs, theta_s, zeta, eta)

    for i in range(n_samples):
        for j in range(max_n_atoms):
            if zs[i][j] == 0:
                continue
            else:
                acsf_np_sort = np.sort(acsf_np[i][j])
                acsf_tf_sort = np.sort(acsf_tf[i][j])
                np.testing.assert_array_almost_equal(acsf_np_sort, acsf_tf_sort, decimal=4)

if __name__ == "__main__":
    test_acsf_1()
    test_acsf_2()
    test_acsf_3()
