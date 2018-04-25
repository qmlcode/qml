from __future__ import print_function

import os
import qml
import numpy as np
from qml.representations import generate_acsf


def test_representations():
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
        mol = qml.Compound(xyz=path + "/" + xyz_file)
        mols.append(mol)

    acsf(mols)

def acsf(mols):

    # Generate coulomb matrix representation, sorted by row-norm
    for i, mol in enumerate(mols): 
        nuclear_charges = mol.nuclear_charges
        coordinates = mol.coordinates
        mol.representation = generate_acsf(nuclear_charges, coordinates)

    print(mols[-1].representation[-1])

if __name__ == "__main__":
    test_representations()
