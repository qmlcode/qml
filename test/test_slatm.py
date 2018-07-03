import os

import numpy as np

import qml
from qml.ml.representations import get_slatm_mbtypes


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

