# MIT License
#
# Copyright (c) 2017 Anders Steen Christensen
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

import os

import numpy as np

import qml
from qml.representations import get_slatm_mbtypes


def test_slatm_global_representation():

    files = ["qm7/0001.xyz",
             "qm7/0002.xyz",
             "qm7/0003.xyz",
             "qm7/0004.xyz",
             "qm7/0005.xyz",
             "qm7/0006.xyz",
             "qm7/0007.xyz",
             "qm7/0008.xyz",
             "qm7/0009.xyz",
             "qm7/0010.xyz"]

    path = os.path.dirname(os.path.realpath(__file__))

    mols = []
    for xyz_file in files:

        mol = qml.data.Compound(xyz=path + "/" + xyz_file)
        mols.append(mol)

    mbtypes = get_slatm_mbtypes(np.array([mol.nuclear_charges for mol in mols]))

    for i, mol in enumerate(mols): 

        mol.generate_slatm(mbtypes, local=False)

    X_qml = np.array([mol.representation for mol in mols])
    X_ref = np.loadtxt(path + "/data/slatm_global_representation.txt")

    assert np.allclose(X_qml, X_ref), "Error in SLATM generation"


def test_slatm_local_representation():

    files = ["qm7/0001.xyz",
             "qm7/0002.xyz",
             "qm7/0003.xyz",
             "qm7/0004.xyz",
             "qm7/0005.xyz",
             "qm7/0006.xyz",
             "qm7/0007.xyz",
             "qm7/0008.xyz",
             "qm7/0009.xyz",
             "qm7/0010.xyz"]

    path = os.path.dirname(os.path.realpath(__file__))

    mols = []
    for xyz_file in files:

        mol = qml.data.Compound(xyz=path + "/" + xyz_file)
        mols.append(mol)

    mbtypes = get_slatm_mbtypes(np.array([mol.nuclear_charges for mol in mols]))

    for i, mol in enumerate(mols): 
        mol.generate_slatm(mbtypes, local=True)

    X_qml = []
    for i, mol in enumerate(mols): 
        for rep in mol.representation: 
            X_qml.append(rep)

    X_qml = np.asarray(X_qml)
    X_ref = np.loadtxt(path + "/data/slatm_local_representation.txt")

    assert np.allclose(X_qml, X_ref), "Error in SLATM generation"

if __name__ == "__main__":

    test_slatm_global_representation()
    test_slatm_local_representation()

