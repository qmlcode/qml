from __future__ import print_function

import os
import qml
import numpy as np
from qml.representations import generate_acsf
import glob


def test_representations():
    #files = ["qm7/0101.xyz",
    #         "qm7/0102.xyz"]#,
    #         #"qm7/0103.xyz",
    #         #"qm7/0104.xyz",
    #         #"qm7/0105.xyz",
    #         #"qm7/0106.xyz",
    #         #"qm7/0107.xyz",
    #         #"qm7/0108.xyz",
    #         #"qm7/0109.xyz",
    #         #"qm7/0110.xyz"]

    path = test_dir = os.path.dirname(os.path.realpath(__file__))
    files = glob.glob(path + "/qm7/*.xyz")

    #files = ["/home/lb17101/dev/qml/tests/qm7/0101.xyz",
    #         "/home/lb17101/dev/qml/tests/qm7/0102.xyz"]

    mols = []
    for xyz_file in files:
        mol = qml.Compound(xyz=xyz_file)
        mols.append(mol)

    acsf(mols)

def acsf(mols):

    # Generate coulomb matrix representation, sorted by row-norm
    for i, mol in enumerate(mols): 
        nuclear_charges = mol.nuclear_charges
        coordinates = mol.coordinates
        mol.representation = generate_acsf(nuclear_charges, coordinates, nRs2 = 20, nRs3 = 20, nTs = 10)

    #coords = [mol.coordinates for mol in mols]
    #nuclear_charges = [mol.nuclear_charges for mol in mols]
    #maxsize = max([x.shape[0] for x in coords])
    #xyz = []
    #for coord in coords:
    #    c = np.zeros((maxsize, 3))
    #    c[:coord.shape[0]] = coord
    #    xyz.append(c)

    #Zs = []
    #for charges in nuclear_charges:
    #    z = np.zeros(maxsize, dtype = int)
    #    z[:charges.shape[0]] = charges
    #    Zs.append(z)

    #Zs = np.asarray(Zs)
    #xyzs = np.asarray(xyz)
    #Rs = np.linspace(0,5,10)
    #Ts = np.linspace(0,np.pi,10)

    #rep = generate_pyacsf(xyzs, Zs, [1,6,7], [[1,1],[1,6],[1,7],[6,6],[6,7],[7,7]], 5, 5, Rs, Rs, Ts, 1, 1)
    #for i in range(len(mols)):
    #    for j in range(len(mols[i].representation)):
    #        print(sum(np.asarray(sorted(rep[i][j]))-np.asarray(sorted(mols[i].representation[j]))))
    #quit()

def distance(r1, r2):
    """
    This function calculates the norm of two vectors.

    :param r1: a numpy array of shape (n,)
    :param r2: a numpy array of shape (n,)
    :return: a scalar
    """
    diff = r2-r1
    return np.linalg.norm(diff)

def fc(r_ij, r_c):
    """
    This function calculates the f_c term of the symmetry functions. r_ij is the distance between atom i and j. r_c is
    the cut off distance.

    :param r_ij: scalar
    :param r_c: scalar
    :return: sclar
    """
    if r_ij < r_c:
        f_c = 0.5 * (np.cos(np.pi * r_ij / r_c) + 1.0)
    else:
        f_c = 0.0
    return f_c

def get_costheta(xyz_i, xyz_j, xyz_k):
    """
    This function calculates the cosine of the angle in radiants between three points i, j and k. It requires the
    cartesian coordinates of the three points.

    :param xyz_i: numpy array of shape (3, )
    :param xyz_j: numpy array of shape (3, )
    :param xyz_k: numpy array of shape (3, )
    :return: scalar
    """
    r_ij = xyz_j - xyz_i
    r_ik = xyz_k - xyz_i
    numerator = np.dot(r_ij, r_ik)
    denominator = np.linalg.norm(r_ij) * np.linalg.norm(r_ik)
    costheta = numerator/denominator
    return costheta

def acsf_rad(xyzs, Zs, elements, radial_cutoff, radial_rs, eta):
    """
    This function calculates the radial part of the symmetry functions.

    :param xyzs: cartesian coordinates of the atoms. Numpy array of shape (n_samples, n_atoms, 3)
    :param Zs: atomic numpers of the atoms. Numpy array of shape (n_samples, n_atoms)
    :param elements: array of all the atomic numbers present sorted in descending order. Numpy array of shape (n_elements, )
    :param radial_cutoff: cut off length. scalar
    :param radial_rs: list of all the radial Rs parameters. Numpy array of shape (n_rad_rs,)
    :param eta: parameter. scalar.
    :return: numpy array of shape (n_samples, n_atoms, n_rad_rs*n_elements)
    """
    n_samples = xyzs.shape[0]
    n_atoms = xyzs.shape[1]
    n_rs = len(radial_rs)
    n_elements = len(elements)

    total_descriptor = []
    for sample in range(n_samples):
        sample_descriptor = []
        for main_atom in range(n_atoms):
            atom_descriptor = np.zeros((n_rs* n_elements,))
            if Zs[sample][main_atom] == 0:
                sample_descriptor.append(atom_descriptor)
                continue
            else:
                for i, rs_value in enumerate(radial_rs):
                    for neighb_atom in range(n_atoms):
                        if main_atom == neighb_atom:
                            continue
                        elif Zs[sample][neighb_atom] == 0:
                            continue
                        else:
                            r_ij = distance(xyzs[sample, main_atom, :], xyzs[sample, neighb_atom, :])
                            cut_off_term = fc(r_ij, radial_cutoff)
                            exponent_term = np.exp(-eta * (r_ij - rs_value) ** 2)
                            g2_term = exponent_term * cut_off_term
                            # Compare the current neighbouring atom to the list of possible neighbouring atoms and then
                            # split the terms accordingly
                            for j in range(len(elements)):
                                if Zs[sample][neighb_atom] == elements[j]:
                                    atom_descriptor[i * n_elements + j] += g2_term

            sample_descriptor.append(atom_descriptor)
        total_descriptor.append(sample_descriptor)

    total_descriptor = np.asarray(total_descriptor)

    return total_descriptor

def acsf_ang(xyzs, Zs, element_pairs, angular_cutoff, angular_rs, theta_s, zeta, eta):
    """
    This function calculates the angular part of the symmetry functions.

    :param xyzs: cartesian coordinates of the atoms. Numpy array of shape (n_samples, max_n_atoms, 3)
    :param Zs: atomic numpers of the atoms. Numpy array of shape (n_samples, max_n_atoms)
    :param element_pairs: array of all the possible pairs of atomic numbers, sorted in descending order. Numpy array of shape (n_element_pairs, 2)
    :param angular_cutoff: cut off length. scalar
    :param angular_rs: list of all the angular Rs parameters. Numpy array of shape (n_ang_rs,)
    :param theta_s: list of all the thetas parameters. Numpy array of shape (n_thetas,)
    :param zeta: parameter. scalar.
    :param eta: parameter. scalar.
    :return: numpy array of shape (n_samples, max_n_atoms, n_ang_rs*n_thetas*n_element_pairs)
    """
    n_samples = xyzs.shape[0]
    max_n_atoms = xyzs.shape[1]
    n_rs = len(angular_rs)
    n_theta = len(theta_s)
    n_elements_pairs = len(element_pairs)

    total_descriptor = []

    for sample in range(n_samples):
        lab_count = 0
        sample_descriptor = []
        for i in range(max_n_atoms):  # Loop over main atom
            atom_descriptor = np.zeros((n_rs*n_theta*n_elements_pairs, ))
            if Zs[sample][i] == 0:  # Making sure the main atom is not a dummy atom
                sample_descriptor.append(atom_descriptor)
                continue
            else:
                counter = 0
                for rid, rs_value in enumerate(angular_rs):  # Loop over parameters
                    for tid, theta_value in enumerate(theta_s):
                        for j in range(max_n_atoms):  # Loop over 1st neighbour
                            if j == i:
                                continue
                            elif Zs[sample][j] == 0:    # Making sure the neighbours are not dummy atoms
                                continue
                            for k in range(j+1, max_n_atoms):  # Loop over 2nd neighbour
                                if k == j or k == i:
                                    continue
                                elif Zs[sample][k] == 0: # Making sure the neighbours are not dummy atoms
                                    continue

                                r_ij = distance(xyzs[sample, i, :], xyzs[sample, j, :])
                                r_ik = distance(xyzs[sample, i, :], xyzs[sample, k, :])
                                # LAB EDIT
                                #if r_ij > angular_cutoff:
                                #    continue
                                #if r_ik > angular_cutoff:
                                #    continue
                                cos_theta_ijk = get_costheta(xyzs[sample, i, :], xyzs[sample, j, :], xyzs[sample, k, :])
                                theta_ijk = np.arccos(cos_theta_ijk)

                                term_1 = np.power((1.0 + np.cos(theta_ijk - theta_value)), zeta)
                                term_2 = np.exp(- eta * np.power(0.5*(r_ij + r_ik) - rs_value, 2))
                                term_3 = fc(r_ij, angular_cutoff) * fc(r_ik, angular_cutoff)
                                g_term = term_1 * term_2 * term_3 * np.power(2.0, 1.0 - zeta)
                                # Compare the pair of neighbours to all the possible element pairs, then summ accordingly
                                current_pair = np.sort([Zs[sample][j], Zs[sample][k]])
                                for m, pair in enumerate(element_pairs):
                                    if np.all(current_pair == pair):
                                        atom_descriptor[counter * n_elements_pairs + m] += g_term
                                        lab_count += 1
                                        if i == 0 and j == 1 and k == 2 and rid == 0 and tid == 0:
                                            print(theta_ijk, g_term, counter * n_elements_pairs + m)
                                        break
                                else:
                                    #print(current_pair, element_pairs)
                                    quit()

                        counter += 1

            sample_descriptor.append(atom_descriptor)
        #print(lab_count)
        total_descriptor.append(sample_descriptor)

    return np.asarray(total_descriptor)

def generate_pyacsf(xyzs, Zs, elements, element_pairs, radial_cutoff, angular_cutoff, radial_rs,
                  angular_rs, theta_s, zeta, eta):
    """
    This function calculates the symmetry functions used in the tensormol paper.

    :param xyzs: cartesian coordinates of the atoms. Numpy array of shape (n_samples, n_atoms, 3)
    :param Zs: atomic numpers of the atoms. Numpy array of shape (n_samples, n_atoms)
    :param elements: array of all the atomic numbers present sorted in descending order. Numpy array of shape (n_elements, )
    :param element_pairs: array of all the possible pairs of atomic numbers, sorted in descending order. Numpy array of shape (n_element_pairs, 2)
    :param radial_cutoff: cut off length for radial part. scalar
    :param angular_cutoff: cut off length for angular part. scalar
    :param radial_rs: list of all the radial Rs parameters. Numpy array of shape (n_rad_rs,)
    :param angular_rs: list of all the angular Rs parameters. Numpy array of shape (n_ang_rs,)
    :param theta_s: list of all the thetas parameters. Numpy array of shape (n_thetas,)
    :param zeta: parameter. scalar.
    :param eta: parameter. scalar.
    :return: numpy array of shape (n_samples, n_atoms, n_rad_rs*n_elements + n_ang_rs*n_thetas*n_element_pairs)
    """

    rad_term = acsf_rad(xyzs, Zs, elements, radial_cutoff, radial_rs, eta)
    ang_term = acsf_ang(xyzs, Zs, element_pairs, angular_cutoff, angular_rs, theta_s, zeta, eta)

    acsf = np.concatenate((rad_term, ang_term), axis=-1)

    return acsf


if __name__ == "__main__":
    test_representations()
