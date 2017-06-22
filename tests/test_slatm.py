import os

import numpy as np

import qml
from qml.representations import get_slatm_mbtypes


def test_slatm_representation():

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

    path = test_dir = os.path.dirname(os.path.realpath(__file__))

    print(path)
    mols = []
    for xyz_file in files:

        mol = qml.Compound(xyz=path + "/" + xyz_file)
        mols.append(mol)

    mbtypes = get_slatm_mbtypes(np.array([mol.nuclear_charges for mol in mols]))

    for i, mol in enumerate(mols): 
        mol.generate_slatm(mbtypes)

    X_qml = np.array([mol.representation for mol in mols])
    X_ref = np.loadtxt(path + "/data/slatm_representation.txt")

    assert np.allclose(X_qml, X_ref), "Error in SLATM generation"
