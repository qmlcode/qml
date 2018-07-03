import qml
import numpy as np
import os
import sys

from qml.wrappers import arad_local_kernels, arad_local_symmetric_kernels

def get_energies(filename):
    """ Returns a dictionary with heats of formation for each xyz-file.
    """

    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    energies = dict()

    for line in lines:
        tokens = line.split()

        xyz_name = tokens[0]
        hof = float(tokens[1])

        energies[xyz_name] = hof

    return energies

# def test_arad_wrapper():
# 
#     test_dir = os.path.dirname(os.path.realpath(__file__))
# 
#     # Parse file containing PBE0/def2-TZVP heats of formation and xyz filenames
#     data = get_energies("%s/data/hof_qm7.txt" % test_dir)
# 
#     # Generate a list of qml.Compound() objects
#     mols = []
# 
#     for xyz_file in sorted(data.keys())[:50]:
# 
#         # Initialize the qml.Compound() objects
#         mol = qml.Compound(xyz="%s/qm7/" % test_dir + xyz_file)
# 
#         # Associate a property (heat of formation) with the object
#         mol.properties = data[xyz_file]
# 
#         # This is a Molecular Coulomb matrix sorted by row norm
#         mol.generate_arad_representation(size=23)
# 
#         mols.append(mol)
# 
# 
#     # Shuffle molecules
#     np.random.seed(666)
#     np.random.shuffle(mols)
# 
#     # Make training and test sets
#     n_test  = 10
#     n_train = 40
# 
#     training = mols[:n_train]
#     test  = mols[-n_test:]
# 
#     sigmas = [10.0, 100.0]
#    
#     
#     K1 = arad_local_symmetric_kernels(training, sigmas)
#     assert np.all(K1 > 0.0), "ERROR: ARAD symmetric kernel negative"
#     assert np.invert(np.all(np.isnan(K1))), "ERROR: ARAD symmetric kernel contains NaN"
#     
# 
#     K2 = arad_local_kernels(training, test, sigmas)
#     assert np.all(K2 > 0.0), "ERROR: ARAD symmetric kernel negative"
#     assert np.invert(np.all(np.isnan(K2))), "ERROR: ARAD symmetric kernel contains NaN"

if __name__ == "__main__":

    pass 
    # test_arad_wrapper()
