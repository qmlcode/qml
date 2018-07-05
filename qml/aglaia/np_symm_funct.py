"""
This module contains an implementation of the symmetry functions used by Parkhill in the Tensormol paper:
https://arxiv.org/pdf/1711.06385.pdf
Here they are implemented with Numpy so that the Tensorflow implementation has something to be compared against.
"""

import numpy as np

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
        sample_descriptor = []
        for i in range(max_n_atoms):  # Loop over main atom
            atom_descriptor = np.zeros((n_rs*n_theta*n_elements_pairs, ))
            if Zs[sample][i] == 0:  # Making sure the main atom is not a dummy atom
                sample_descriptor.append(atom_descriptor)
                continue
            else:
                counter = 0
                for rs_value in angular_rs:  # Loop over parameters
                    for theta_value in theta_s:
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
                                cos_theta_ijk = get_costheta(xyzs[sample, i, :], xyzs[sample, j, :], xyzs[sample, k, :])
                                theta_ijk = np.arccos(cos_theta_ijk)

                                term_1 = np.power((1.0 + np.cos(theta_ijk - theta_value)), zeta)
                                term_2 = np.exp(- eta * np.power(0.5*(r_ij + r_ik) - rs_value, 2))
                                term_3 = fc(r_ij, angular_cutoff) * fc(r_ik, angular_cutoff)
                                g_term = term_1 * term_2 * term_3 * np.power(2.0, 1.0 - zeta)
                                # Compare the pair of neighbours to all the possible element pairs, then summ accordingly
                                current_pair = np.flip(np.sort([Zs[sample][j], Zs[sample][k]]), axis=0)     # Sorting the pair in descending order
                                for m, pair in enumerate(element_pairs):
                                    if np.all(current_pair == pair):
                                        atom_descriptor[counter * n_elements_pairs + m] += g_term
                        counter += 1

            sample_descriptor.append(atom_descriptor)
        total_descriptor.append(sample_descriptor)

    return np.asarray(total_descriptor)

def generate_acsf(xyzs, Zs, elements, element_pairs, radial_cutoff, angular_cutoff, radial_rs,
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
    xyzs = np.array([[[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]]])

    zs = [[7, 2, 1, 1]]

    elements = [1, 2, 7]
    element_pairs = [[1, 1], [2, 1], [7, 1], [7, 2]]

    # # input_data = "/Volumes/Transcend/repositories/Aglaia/aglaia/tests/data_test_acsf.npz"
    # input_data = "/Volumes/Transcend/repositories/Aglaia/aglaia/tests/qm7_testdata.npz"
    # data = np.load(input_data)
    #
    # xyzs = data["arr_0"]
    # zs = data["arr_1"]
    # elements = data["arr_2"]
    # element_pairs = data["arr_3"]

    radial_cutoff = 1000.0
    angular_cutoff = 1000.0
    radial_rs = [0.0, 1.0]
    angular_rs = [0.0, 1.0]
    theta_s = [np.pi, np.pi * 0.5]
    # radial_rs = [0.0]
    # angular_rs = [0.0]
    # theta_s = [3.0]
    zeta = 0.0
    eta = 1.0

    rad_term = acsf_rad(xyzs, zs, elements, radial_cutoff, radial_rs, eta)
    ang_term = acsf_ang(xyzs, zs, element_pairs, angular_cutoff, angular_rs, theta_s, zeta, eta)
    print(rad_term)