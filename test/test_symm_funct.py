# MIT License
#
# Copyright (c) 2018 Silvia Amabilino, Lars Andersen Bratholm
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
This file contains tests for the atom centred symmetry function module.
"""
from __future__ import print_function

import tensorflow as tf
import numpy as np

import qml
from qml.ml.representations import generate_acsf
from qml.aglaia import symm_funct
import os

def pad(size, coordinates, nuclear_charges):

    n_samples = len(coordinates)

    padded_coordinates = np.zeros((n_samples, size, 3))
    padded_nuclear_charges = np.zeros((n_samples, size), dtype = int)

    for i in range(n_samples):
        natoms = coordinates[i].shape[0]
        if natoms > size:
            print("natoms larger than padded size")
            quit()
        padded_coordinates[i,:natoms] = coordinates[i]
        padded_nuclear_charges[i,:natoms] = nuclear_charges[i]

    return padded_coordinates, padded_nuclear_charges

def test_acsf():
    files = ["qm7/0101.xyz",
             "qm7/0102.xyz",
             "qm7/0103.xyz",
             "qm7/0104.xyz",
             "qm7/0105.xyz",
             "qm7/0106.xyz",
             "qm7/0107.xyz",
             "qm7/0108.xyz",
             "qm7/0109.xyz",
             "qm7/0110.xyz"]


    path = test_dir = os.path.dirname(os.path.realpath(__file__))

    mols = []
    for xyz_file in files:
        mol = qml.data.Compound(xyz=path + "/" + xyz_file)
        mols.append(mol)

    elements = set()
    for mol in mols:
        elements = elements.union(mol.nuclear_charges)

    elements = list(elements)

    fort_acsf(mols, path, elements)
    fort_acsf_gradients(mols, path, elements)
    tf_acsf(mols, path, elements)

def fort_acsf(mols, path, elements):

    # Generate atom centered symmetry functions representation
    # using the Compound class
    for i, mol in enumerate(mols):
        mol.generate_acsf(elements = elements)

    X_test = np.concatenate([mol.representation for mol in mols])
    X_ref = np.loadtxt(path + "/data/acsf_representation.txt")
    assert np.allclose(X_test, X_ref), "Error in ACSF representation"

    # Generate atom centered symmetry functions representation
    # directly from the representations module
    rep = []
    for i, mol in enumerate(mols):
        rep.append(generate_acsf(coordinates = mol.coordinates,
                nuclear_charges = mol.nuclear_charges, 
                elements = elements))

    X_test = np.concatenate(rep)
    X_ref = np.loadtxt(path + "/data/acsf_representation.txt")
    assert np.allclose(X_test, X_ref), "Error in ACSF representation"

def tf_acsf(mols, path, elements):
    radial_cutoff = 5
    angular_cutoff = 5
    radial_rs = np.linspace(0, radial_cutoff, 3)
    angular_rs = np.linspace(0, angular_cutoff, 3)
    theta_s = np.linspace(0, np.pi, 3)
    zeta = 1.0
    eta = 1.0

    element_pairs = [[1,1], [6,1], [7,1], [7,7], [6,6], [7,6]]

    xyzs, zs = pad(16, [mol.coordinates for mol in mols],
            [mol.nuclear_charges for mol in mols])


    n_samples = xyzs.shape[0]
    max_n_atoms = zs.shape[1]

    with tf.name_scope("Inputs"):
        zs_tf = tf.placeholder(shape=[n_samples, max_n_atoms], dtype=tf.int32, name="zs")
        xyz_tf = tf.placeholder(shape=[n_samples, max_n_atoms, 3], dtype=tf.float32, name="xyz")

    acsf_tf_t = symm_funct.generate_parkhill_acsf(xyz_tf, zs_tf, elements, element_pairs, radial_cutoff, angular_cutoff, radial_rs, angular_rs, theta_s, zeta, eta)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    X_test = sess.run(acsf_tf_t, feed_dict={xyz_tf: xyzs, zs_tf: zs})[zs > 0]
    X_ref = np.loadtxt(path + "/data/acsf_representation.txt")
    assert np.allclose(X_test, X_ref, atol = 1e-4), "Error in ACSF representation"

def fort_acsf_gradients(mols, path, elements):

    # Generate atom centered symmetry functions representation
    # and gradients using the Compound class
    for i, mol in enumerate(mols):
        mol.generate_acsf(elements = elements, gradients = True)

    X_test = np.concatenate([mol.representation for mol in mols])
    X_ref = np.loadtxt(path + "/data/acsf_representation.txt")
    assert np.allclose(X_test, X_ref), "Error in ACSF representation"

    Xgrad_test = np.concatenate([mol.gradients.reshape(mol.natoms**2, mol.gradients.shape[1]*3)
        for mol in mols])
    Xgrad_ref = np.loadtxt(path + "/data/acsf_gradients.txt")
    assert np.allclose(Xgrad_test, Xgrad_ref), "Error in ACSF gradients"

    # Generate atom centered symmetry functions representation
    # and gradients directly from the representations module
    rep = []
    grad = []
    for i, mol in enumerate(mols):
        r, g = generate_acsf(coordinates = mol.coordinates, nuclear_charges = mol.nuclear_charges,
                elements = elements, gradients = True)
        rep.append(r)
        grad.append(g)

    # Reshape the gradients to fit the test format
    for i, mol in enumerate(mols):
        g = grad[i]
        natoms = mol.natoms
        repsize = g.shape[1]
        grad[i] = g.reshape(natoms ** 2, repsize * 3)

    X_test = np.concatenate(rep)
    X_ref = np.loadtxt(path + "/data/acsf_representation.txt")
    assert np.allclose(X_test, X_ref), "Error in ACSF representation"

    Xgrad_test = np.concatenate(grad, axis = 0)
    Xgrad_ref = np.loadtxt(path + "/data/acsf_gradients.txt")
    assert np.allclose(Xgrad_test, Xgrad_ref), "Error in ACSF gradients"

if __name__ == "__main__":
    test_acsf()

