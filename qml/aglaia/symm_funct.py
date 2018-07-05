"""
This module contains an implementation of the the symmetry functions used in the Parkhill paper https://arxiv.org/pdf/1711.06385.pdf.
This implementation is different. It works for both data sets where all the molecules are the same but in different configurations and
for datasets with all different molecules.

Note: it is all in single precision.
"""

import tensorflow as tf
import numpy as np

def acsf_rad(xyzs, Zs, radial_cutoff, radial_rs, eta):
    """
    This does the radial part of the symmetry function (G2 function in Behler's papers). It works only for datasets where
    all samples are the same molecule but in different configurations.

    :param xyzs: tf tensor of shape (n_samples, n_atoms, 3) contaning the coordinates of each atom in each data sample
    :param Zs: tf tensor of shape (n_samples, n_atoms) containing the atomic number of each atom in each data sample
    :param radial_cutoff: scalar tensor
    :param radial_rs: tf tensor of shape (n_rs,) with the R_s values
    :param eta: tf scalar

    :return: tf tensor of shape (n_samples, n_atoms, n_atoms, n_rs)
    """

    # Calculating the distance matrix between the atoms of each sample
    with tf.name_scope("Distances"):
        dxyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
        dist_tensor = tf.cast(tf.norm(dxyzs, axis=3), dtype=tf.float32)  # (n_samples, n_atoms, n_atoms)

    # Indices of terms that need to be zero (diagonal elements)
    mask_0 = tf.zeros(tf.shape(dist_tensor))
    mask_1 = tf.ones(tf.shape(Zs))
    where_eq_idx = tf.cast(tf.matrix_set_diag(mask_0, mask_1), dtype=tf.bool)

    # Calculating the exponential term
    with tf.name_scope("Exponential_term"):
        expanded_rs = tf.expand_dims(tf.expand_dims(tf.expand_dims(radial_rs, axis=0), axis=0), axis=0) # (1, 1, 1, n_rs)
        expanded_dist = tf.expand_dims(dist_tensor, axis=-1) # (n_samples, n_atoms, n_atoms, 1)
        exponent = - eta * tf.square(tf.subtract(expanded_dist, expanded_rs))
        exp_term = tf.exp(exponent) # (n_samples, n_atoms, n_atoms, n_rs)

    # Calculating the fc terms
    with tf.name_scope("fc_term"):
        # Finding where the distances are less than the cutoff
        where_less_cutoff = tf.less(dist_tensor, radial_cutoff)
        # Calculating all of the fc function terms
        fc = 0.5 * (tf.cos(3.14159265359 * dist_tensor / radial_cutoff) + 1.0)
        # Setting to zero the terms where the distance is larger than the cutoff
        zeros = tf.zeros(tf.shape(dist_tensor), dtype=tf.float32)
        cut_off_fc = tf.where(where_less_cutoff, fc, zeros)  # (n_samples, n_atoms, n_atoms)
        # Cleaning up diagonal terms
        clean_fc_term = tf.where(where_eq_idx, zeros, cut_off_fc)
        # Cleaning up dummy atoms terms
        dummy_atoms = tf.logical_not(tf.equal(Zs, tf.constant(0, dtype=tf.int32)))  # False where there are dummy atoms
        dummy_mask = tf.logical_and(tf.expand_dims(dummy_atoms, axis=1), tf.expand_dims(dummy_atoms, axis=-1))
        cleaner_fc_term = tf.where(dummy_mask, clean_fc_term, zeros)

        # Multiplying exponential and fc terms
        expanded_fc = tf.expand_dims(cleaner_fc_term, axis=-1) # (n_samples, n_atoms, n_atoms, 1)

    with tf.name_scope("Rad_term"):
        presum_term = tf.multiply(expanded_fc, exp_term) # (n_samples, n_atoms, n_atoms, n_rs)

    return presum_term

def acsf_ang(xyzs, Zs, angular_cutoff, angular_rs, theta_s, zeta, eta):
    """
    This does the angular part of the symmetry function as mentioned here: https://arxiv.org/pdf/1711.06385.pdf
    It only works for systems where all the samples are the same molecule but in different configurations.

    :param xyzs: tf tensor of shape (n_samples, n_atoms, 3) contaning the coordinates of each atom in each data sample
    :param Zs: tf tensor of shape (n_samples, n_atoms) containing the atomic number of each atom in each data sample
    :param angular_cutoff: scalar tensor
    :param angular_rs: tf tensor of shape (n_ang_rs,) with the equivalent of the R_s values from the G2
    :param theta_s: tf tensor of shape (n_thetas,)
    :param zeta: tf tensor of shape (1,)
    :param eta: tf tensor of shape (1,)
    :return: tf tensor of shape (n_samples, n_atoms, n_atoms, n_atoms, n_ang_rs * n_thetas)
    """

    # Finding the R_ij + R_ik term
    with tf.name_scope("Sum_distances"):
        dxyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
        dist_tensor = tf.cast(tf.norm(dxyzs, axis=3), dtype=tf.float32)  # (n_samples, n_atoms, n_atoms)

        # This is the tensor where element sum_dist_tensor[0,1,2,3] is the R_12 + R_13 in the 0th data sample
        sum_dist_tensor = tf.expand_dims(dist_tensor, axis=3) + tf.expand_dims(dist_tensor,
                                                                           axis=2)  # (n_samples, n_atoms, n_atoms, n_atoms)

    # Problem with the above tensor: we still have the R_ii + R_ik distances which are non zero and could be summed
    # These need to be set to zero
    n_atoms = Zs.get_shape().as_list()[1]

    zarray = np.zeros((n_atoms, n_atoms, n_atoms))

    for i in range(n_atoms):
        for j in range(n_atoms):
            for k in range(n_atoms):
                if i == j or i == k or j == k:
                    zarray[i, j, k] = 1

    # Make a bool tensor of the indices
    where_eq_idx = tf.tile(tf.expand_dims(tf.convert_to_tensor(zarray, dtype=tf.bool), axis=0),
                           multiples=[tf.shape(sum_dist_tensor)[0], 1, 1, 1])

    # For all the elements that are true in where_eq_idx, turn the elements of sum_dist_tensor to zero
    zeros_1 = tf.zeros(tf.shape(sum_dist_tensor), dtype=tf.float32)

    # Now finding the fc terms
    with tf.name_scope("Fc_term"):
        # 1. Find where Rij and Rik are < cutoff
        where_less_cutoff = tf.less(dist_tensor, angular_cutoff)
        # 2. Calculate the fc on the Rij and Rik tensors
        fc_1 = 0.5 * (tf.cos(3.14159265359 * dist_tensor / angular_cutoff) + 1.0)
        # 3. Apply the mask calculated in 1.  to zero the values for where the distances are > than the cutoff
        zeros_2 = tf.zeros(tf.shape(dist_tensor), dtype=tf.float32)
        cut_off_fc = tf.where(where_less_cutoff, fc_1, zeros_2)  # (n_samples, n_atoms, n_atoms)
        # 4. Multiply the two tensors elementwise
        fc_term = tf.multiply(tf.expand_dims(cut_off_fc, axis=3),
                              tf.expand_dims(cut_off_fc, axis=2))  # (n_samples,  n_atoms, n_atoms, n_atoms)
        # 5. Cleaning up the terms that should be zero because there are equal indices
        clean_fc_term = tf.where(where_eq_idx, zeros_1, fc_term)
        # 6. Cleaning up the terms due to the dummy atoms
        dummy_atoms = tf.logical_not(tf.equal(Zs, tf.constant(0, dtype=tf.int32)))  # False where there are dummy atoms
        dummy_mask_2d = tf.logical_and(tf.expand_dims(dummy_atoms, axis=1), tf.expand_dims(dummy_atoms, axis=-1))
        dummy_mask_3d = tf.logical_and(tf.expand_dims(dummy_mask_2d, axis=1), tf.expand_dims(tf.expand_dims(dummy_atoms, axis=-1), axis=-1))
        cleaner_fc_term = tf.where(dummy_mask_3d, clean_fc_term, zeros_1)


    # Now finding the theta_ijk term
    with tf.name_scope("Theta"):
        # Doing the dot products of all the possible vectors
        dots_dxyzs = tf.cast(tf.reduce_sum(tf.multiply(tf.expand_dims(dxyzs, axis=3), tf.expand_dims(dxyzs, axis=2)),
                                   axis=4), dtype=tf.float32)  # (n_samples,  n_atoms, n_atoms, n_atoms)

        # Doing the products of the magnitudes
        dist_prod = tf.multiply(tf.expand_dims(dist_tensor, axis=3),
                                tf.expand_dims(dist_tensor, axis=2))  # (n_samples,  n_atoms, n_atoms, n_atoms)
        # Dividing the dot products by the magnitudes to obtain cos theta
        cos_theta = tf.divide(dots_dxyzs, dist_prod)
        # Taking care of the values that due numerical error are just above 1.0 or below -1.0
        cut_cos_theta = tf.clip_by_value(cos_theta, tf.constant(-1.0), tf.constant(1.0))
        # Applying arc cos to find the theta value
        theta = tf.acos(cut_cos_theta)  # (n_samples,  n_atoms, n_atoms, n_atoms)
        # Removing the NaNs created by dividing by zero
        clean_theta = tf.where(where_eq_idx, zeros_1, theta)
        # cleaning up NaNs due by dummy atoms
        dummy_atoms = tf.logical_not(tf.equal(Zs, tf.constant(0, dtype=tf.int32)))  # False where there are dummy atoms
        dummy_mask_2d = tf.logical_and(tf.expand_dims(dummy_atoms, axis=1), tf.expand_dims(dummy_atoms, axis=-1))
        dummy_mask_3d = tf.logical_and(tf.expand_dims(dummy_mask_2d, axis=1),
                                      tf.expand_dims(tf.expand_dims(dummy_atoms, axis=-1), axis=-1))
        cleaner_theta = tf.where(dummy_mask_3d, clean_theta, zeros_1)

    # Finding the (0.5 * clean_sum_dist - R_s) term
    with tf.name_scope("Exp_term"):
        # Augmenting the dims of angular_rs
        expanded_rs = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(angular_rs, axis=0), axis=0), axis=0),
                                     axis=0)  # (1, 1, 1, 1, n_rs)
        # Augmenting the dim of clean_sum_dist *0.5
        # expanded_sum = tf.expand_dims(clean_sum_dist * 0.5, axis=-1)
        expanded_sum = tf.expand_dims(sum_dist_tensor * 0.5, axis=-1)
        # Combining them
        brac_term = tf.subtract(expanded_sum, expanded_rs)
        # Finally making the exponential term
        exponent = - eta * tf.square(brac_term)
        exp_term = tf.exp(exponent)  # (n_samples,  n_atoms, n_atoms, n_atoms, n_rs)

    # Finding the cos(theta - theta_s) term
    with tf.name_scope("Cos_term"):
        # Augmenting the dimensions of theta_s
        expanded_theta_s = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(theta_s, axis=0), axis=0), axis=0),
                                          axis=0)
        # Augmenting the dimensions of theta
        expanded_theta = tf.expand_dims(cleaner_theta, axis=-1)
        # Subtracting them and do the cos
        cos_theta_term = tf.cos(
            tf.subtract(expanded_theta, expanded_theta_s))  # (n_samples,  n_atoms, n_atoms, n_atoms, n_theta_s)
        # Make the whole cos term  of the sum
        cos_term = tf.pow(tf.add(tf.ones(tf.shape(cos_theta_term), dtype=tf.float32), cos_theta_term),
                          zeta)  # (n_samples,  n_atoms, n_atoms, n_atoms, n_theta_s)

    # Final product of terms inside the sum time by 2^(1-zeta)
    expanded_fc = tf.expand_dims(tf.expand_dims(cleaner_fc_term, axis=-1), axis=-1, name="Expanded_fc")
    expanded_cos = tf.expand_dims(cos_term, axis=-2, name="Expanded_cos")
    expanded_exp = tf.expand_dims(exp_term, axis=-1, name="Expanded_exp")

    const = tf.pow(tf.constant(2.0, dtype=tf.float32), (1.0 - zeta))

    with tf.name_scope("Ang_term"):
        prod_of_terms = const * tf.multiply(tf.multiply(expanded_cos, expanded_exp),
                                            expanded_fc)  # (n_samples,  n_atoms, n_atoms, n_atoms, n_rs, n_theta_s)

        # Reshaping to shape (n_samples,  n_atoms, n_atoms, n_atoms, n_rs*n_theta_s)
        presum_term = tf.reshape(prod_of_terms,
                                 [tf.shape(prod_of_terms)[0], n_atoms, n_atoms, n_atoms,
                                  theta_s.shape[0] * angular_rs.shape[0]])

    return presum_term

def sum_rad(pre_sum, Zs, elements_list, radial_rs):
    """
    Sum of the terms in the radial part of the symmetry function. The terms corresponding to the same neighbour identity
    are summed together.

    :param pre_sum: tf tensor of shape (n_samples, n_atoms, n_atoms, n_rs)
    :param Zs: tf tensor of shape (n_samples, n_atoms)
    :param elements_list: np.array of shape (n_elements,)
    :param radial_rs: tf tensor of shape (n_rad_rs,)
    :return: tf tensor of shape (n_samples, n_atoms, n_rad_rd * n_elements)
    """
    n_atoms = Zs.get_shape().as_list()[1]
    n_elements = len(elements_list)
    n_rs = radial_rs.get_shape().as_list()[0]

    ## Making a matrix of all the possible neighbouring atoms
    # No need to clean up diagonal elements because they are already set to zero in the presum term
    neighb_atoms = tf.tile(tf.expand_dims(tf.expand_dims(Zs, axis=1), axis=-1),
                           multiples=[1, n_atoms, 1, n_rs])  # (n_samples, n_atoms, n_atoms, n_rs)
    zeros = tf.zeros(tf.shape(pre_sum), dtype=tf.float32)

    # Looping over all the possible elements in the system and extracting the relevant terms from the pre_sum term
    pre_sum_terms = []

    for i in range(n_elements):
        element = tf.constant(elements_list[i], dtype=tf.int32)
        equal_elements = tf.equal(neighb_atoms, element)
        slice_presum = tf.where(equal_elements, pre_sum, zeros)
        slice_sum = tf.reduce_sum(slice_presum, axis=[2])
        pre_sum_terms.append(slice_sum)

    # Concatenating the extracted terms.
    final_term = tf.concat(pre_sum_terms, axis=-1, name="sum_rad")

    # Cleaning up the dummy atoms descriptors
    dummy_atoms = tf.logical_not(tf.equal(Zs, tf.constant(0, dtype=tf.int32)))  # False where there are dummy atoms
    mask = tf.tile(tf.expand_dims(dummy_atoms, axis=-1), multiples=[1, 1, n_elements*n_rs])
    # clean_final_term = tf.where(mask, final_term, tf.zeros(final_term.shape, dtype=tf.float32))
    clean_final_term = tf.where(mask, final_term, tf.zeros(tf.shape(final_term), dtype=tf.float32))

    return clean_final_term

def sum_ang(pre_sumterm, Zs, element_pairs_list, angular_rs, theta_s):
    """
    This function does the sum of the terms in the radial part of the symmetry function. Three body interactions where
    the two neighbours are the same elements are summed together.

    :param pre_sumterm: tf tensor of shape (n_samples, n_atoms, n_ang_rs * n_thetas)
    :param Zs: tf tensor of shape (n_samples, n_atoms)
    :param element_pairs_list: np array of shape (n_elementpairs, 2)
    :param angular_rs: tf tensor of shape (n_ang_rs,)
    :param theta_s: tf tensor of shape (n_thetas,)
    :return: tf tensor of shape (n_samples, n_atoms, n_ang_rs * n_thetas * n_elementpairs)
    """

    n_atoms = Zs.get_shape().as_list()[1]
    n_pairs = len(element_pairs_list)
    n_rs = angular_rs.get_shape().as_list()[0]
    n_thetas = theta_s.get_shape().as_list()[0]

    # Making the pair matrix
    Zs_exp_1 = tf.expand_dims(tf.tile(tf.expand_dims(Zs, axis=1), multiples=[1, n_atoms, 1]), axis=-1)
    Zs_exp_2 = tf.expand_dims(tf.tile(tf.expand_dims(Zs, axis=-1), multiples=[1, 1, n_atoms]), axis=-1)
    neighb_pairs = tf.concat([Zs_exp_1, Zs_exp_2], axis=-1)  # (n_samples, n_atoms, n_atoms, 2)

    # Cleaning up diagonal elements
    zarray = np.zeros((n_atoms, n_atoms, 2))

    for i in range(n_atoms):
        zarray[i, i, :] = 1

    # Make a bool tensor of the indices
    where_eq_idx = tf.tile(tf.expand_dims(tf.convert_to_tensor(zarray, dtype=tf.bool), axis=0),
                           multiples=[tf.shape(Zs)[0], 1, 1, 1]) # (n_samples, n_atoms, n_atoms, 2)

    zeros = tf.zeros(tf.shape(neighb_pairs), dtype=tf.int32)
    clean_pairs = tf.where(where_eq_idx, zeros, neighb_pairs)

    # Sorting the pairs in descending order so that for example pair [7, 1] is the same as [1, 7]
    sorted_pairs, _ = tf.nn.top_k(clean_pairs, k=2, sorted=True)  # (n_samples, n_atoms, n_atoms, 2)

    # Preparing to clean the sorted pairs from where there will be self interactions in the three-body-terms
    oarray = np.ones((n_atoms, n_atoms, n_atoms))

    for i in range(n_atoms):
        for j in range(n_atoms):
            for k in range(n_atoms):
                if i == j or i == k or j == k:
                    oarray[i, j, k] = 0

    # Make a bool tensor of the indices
    where_self_int = tf.tile(tf.expand_dims(tf.convert_to_tensor(oarray, dtype=tf.bool), axis=0),
                       multiples=[tf.shape(Zs)[0], 1, 1, 1]) # (n_samples, n_atoms, n_atoms, n_atoms)

    exp_self_int = tf.expand_dims(where_self_int, axis=-1)  # (n_samples, n_atoms, n_atoms, n_atoms, 1)

    zeros_large = tf.zeros(tf.shape(pre_sumterm), dtype=tf.float32, name="zero_large")
    presum_terms = []

    with tf.name_scope("Extract"):
        for i in range(n_pairs):
            # Making a tensor where all the elements are the pair under consideration
            pair = tf.constant(element_pairs_list[i], dtype=tf.int32)
            expanded_pair = tf.tile(
                tf.expand_dims(tf.expand_dims(tf.expand_dims(pair, axis=0), axis=0), axis=0),
                multiples=[tf.shape(Zs)[0], n_atoms, n_atoms, 1], name="expand_pair")  # (n_samples, n_atoms, n_atoms, 2)
            # Comparing which neighbour pairs correspond to the pair under consideration
            equal_pair_mix = tf.equal(expanded_pair, sorted_pairs)
            equal_pair_split1, equal_pair_split2 = tf.split(equal_pair_mix, 2, axis=-1)
            equal_pair = tf.tile(tf.expand_dims(tf.logical_and(equal_pair_split1, equal_pair_split2), axis=[1]),
                                 multiples=[1, n_atoms, 1, 1, 1])  # (n_samples, n_atoms, n_atoms, n_atoms, 1)
            # Removing the pairs where the same atom is present more than once
            int_to_keep = tf.logical_and(equal_pair, exp_self_int)
            exp_int_to_keep = tf.tile(int_to_keep, multiples=[1, 1, 1, 1, n_rs * n_thetas])
            # Extracting the terms that correspond to the pair under consideration
            slice_presum = tf.where(exp_int_to_keep, pre_sumterm, zeros_large, name="sl_pr_s")
            slice_sum = 0.5 * tf.reduce_sum(slice_presum, axis=[2, 3], name="sum_ang")
            presum_terms.append(slice_sum)

    # Concatenating all of the terms corresponding to different pair neighbours
    final_term = tf.concat(presum_terms, axis=-1, name="concat_presum")

    # Cleaning up the dummy atoms descriptors
    dummy_atoms = tf.logical_not(tf.equal(Zs, tf.constant(0, dtype=tf.int32)))  # False where there are dummy atoms
    mask = tf.tile(tf.expand_dims(dummy_atoms, axis=-1), multiples=[1, 1, n_thetas * n_rs * n_pairs])
    clean_final_term = tf.where(mask, final_term, tf.zeros(tf.shape(final_term)))

    return clean_final_term

def generate_parkhill_acsf(xyzs, Zs, elements, element_pairs, radial_cutoff, angular_cutoff,
                           radial_rs, angular_rs, theta_s, zeta, eta):
    """
    This function generates the atom centred symmetry function as used in the Tensormol paper. Currently only tested for
    single systems with many conformations. It requires the coordinates of all the atoms in each data sample, the atomic
    charges for each atom (in the same order as the xyz), the overall elements and overall element pairs. Then it
    requires the parameters for the ACSF that are used in the Tensormol paper: https://arxiv.org/pdf/1711.06385.pdf

    :param xyzs: tensor of shape (n_samples, n_atoms, 3)
    :param Zs: tensor of shape (n_samples, n_atoms)
    :param elements: np.array of shape (n_elements,)
    :param element_pairs: np.array of shape (n_elementpairs, 2)
    :param radial_cutoff: scalar float
    :param angular_cutoff: scalar float
    :param radial_rs: np.array of shape (n_rad_rs,)
    :param angular_rs: np.array of shape (n_ang_rs,)
    :param theta_s: np.array of shape (n_thetas,)
    :param zeta: scalar float
    :param eta: scalar float
    :return: a tf tensor of shape (n_samples, n_atoms, n_rad_rs * n_elements + n_ang_rs * n_thetas * n_elementpairs)
    """

    with tf.name_scope("acsf_params"):
        rad_cutoff = tf.constant(radial_cutoff, dtype=tf.float32)
        ang_cutoff = tf.constant(angular_cutoff, dtype=tf.float32)
        rad_rs = tf.constant(radial_rs, dtype=tf.float32)
        ang_rs = tf.constant(angular_rs, dtype=tf.float32)
        theta_s = tf.constant(theta_s, dtype=tf.float32)
        zeta_tf = tf.constant(zeta, dtype=tf.float32)
        eta_tf = tf.constant(eta, dtype=tf.float32)

    ##  Calculating the radial part of the symmetry function
    # First obtaining all the terms in the sum
    with tf.name_scope("Radial_part"):
        pre_sum_rad = acsf_rad(xyzs, Zs, rad_cutoff, rad_rs, eta_tf)  # (n_samples, n_atoms, n_atoms, n_rad_rs)
    with tf.name_scope("Sum_rad"):
        # Then summing based on the identity of the atoms interacting
        rad_term = sum_rad(pre_sum_rad, Zs, elements, rad_rs) # (n_samples, n_atoms, n_rad_rs*n_elements)

    ## Calculating the angular part of the symmetry function
    # First obtaining all the terms in the sum
    with tf.name_scope("Angular_part"):
        pre_sum_ang = acsf_ang(xyzs, Zs, ang_cutoff, ang_rs, theta_s, zeta_tf, eta_tf) # (n_samples, n_atoms, n_atoms, n_atoms, n_thetas * n_ang_rs)
    with tf.name_scope("Sum_ang"):
        # Then doing the sum based on the neighbrouing pair identity
        ang_term = sum_ang(pre_sum_ang, Zs, element_pairs, ang_rs, theta_s) # (n_samples, n_atoms, n_thetas * n_ang_rs*n_elementpairs)

    with tf.name_scope("ACSF"):
        acsf = tf.concat([rad_term, ang_term], axis=-1, name="acsf") # (n_samples, n_atoms, n_rad_rs*n_elements + n_thetas * n_ang_rs*n_elementpairs)

    return acsf