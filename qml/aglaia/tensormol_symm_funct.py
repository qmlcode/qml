"""
This module contains the implementation of the symmetry functions from the Tensormol package. They have been extracted
from the package so that the implementation in Aglaia can be compared to them.
"""

import tensorflow as tf
import numpy as np

def tf_symmetry_functions_radial_grid(xyzs, Zs, radial_cutoff, radial_rs, eta, prec=tf.float64):
    """
    Encodes the radial grid portion of the symmetry functions. Should be called by tf_symmetry_functions_2()

    Args:
        xyzs (tf.float): NMol x MaxNAtoms x 3 coordinates tensor
        Zs (tf.int32): NMol x MaxNAtoms atomic number tensor
        num_atoms (np.int32): NMol number of atoms numpy array
        radial_cutoff (tf.float): scalar tensor with the cutoff for radial pairs
        radial_rs (tf.float): NRadialGridPoints tensor with R_s values for the radial grid
        eta (tf.float): scalar tensor with the eta parameter for the symmetry functions

    Returns:
        radial_embedding (tf.float): tensor of radial embeddings for all atom pairs within the radial_cutoff
        pair_indices (tf.int32): tensor of the molecule, atom, and pair atom indices
        pair_elements (tf.int32): tensor of the atomic numbers for the atom and its pair atom
    """
    dxyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
    dist_tensor = tf.norm(dxyzs + 1.e-16,axis=3)
    padding_mask = tf.not_equal(Zs, 0)
    pair_indices = tf.where(tf.logical_and(tf.logical_and(tf.less(dist_tensor, radial_cutoff),
                    tf.expand_dims(padding_mask, axis=1)), tf.expand_dims(padding_mask, axis=-1)))
    identity_mask = tf.where(tf.not_equal(pair_indices[:,1], pair_indices[:,2]))
    pair_indices = tf.cast(tf.squeeze(tf.gather(pair_indices, identity_mask)), tf.int32)
    pair_distances = tf.gather_nd(dist_tensor, pair_indices)
    pair_elements = tf.cast(tf.stack([tf.gather_nd(Zs, pair_indices[:,0:2]), tf.gather_nd(Zs, pair_indices[:,0:3:2])], axis=-1), dtype=tf.int32)
    gaussian_factor = tf.exp(-eta * tf.square(tf.expand_dims(pair_distances, axis=-1) - tf.expand_dims(radial_rs, axis=0)))
    cutoff_factor = tf.expand_dims(0.5 * (tf.cos(3.14159265359 * pair_distances / radial_cutoff) + 1.0), axis=-1)
    radial_embedding = gaussian_factor * cutoff_factor
    return radial_embedding, pair_indices, pair_elements

def tf_symmetry_function_angular_grid(xyzs, Zs, angular_cutoff, angular_rs, theta_s, zeta, eta):
    """
    Encodes the radial grid portion of the symmetry functions. Should be called by tf_symmetry_functions_2()

    Args:
        xyzs (tf.float): NMol x MaxNAtoms x 3 coordinates tensor
        Zs (tf.int32): NMol x MaxNAtoms atomic number tensor
        angular_cutoff (tf.float): scalar tensor with the cutoff for the angular triples
        angular_rs (tf.float): NAngularGridPoints tensor with the R_s values for the radial part of the angular grid
        theta_s (tf.float): NAngularGridPoints tensor with the theta_s values for the angular grid
        zeta (tf.float): scalar tensor with the zeta parameter for the symmetry functions
        eta (tf.float): scalar tensor with the eta parameter for the symmetry functions

    Returns:
        angular_embedding (tf.float): tensor of radial embeddings for all atom pairs within the radial_cutoff
        triples_indices (tf.int32): tensor of the molecule, atom, and triples atom indices
        triples_elements (tf.int32): tensor of the atomic numbers for the atom
        sorted_triples_element_pairs (tf.int32): sorted tensor of the atomic numbers of triples atoms
    """
    num_mols = Zs.get_shape().as_list()[0]
    dxyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
    dist_tensor = tf.norm(dxyzs + 1.e-16, axis=3)
    padding_mask = tf.not_equal(Zs, 0)
    pair_indices = tf.cast(tf.where(tf.logical_and(tf.logical_and(tf.less(dist_tensor, angular_cutoff),
                    tf.expand_dims(padding_mask, axis=1)), tf.expand_dims(padding_mask, axis=-1))), tf.int32)
    identity_mask = tf.where(tf.not_equal(pair_indices[:,1], pair_indices[:,2]))
    pair_indices = tf.cast(tf.squeeze(tf.gather(pair_indices, identity_mask)), tf.int32)
    mol_pair_indices = tf.dynamic_partition(pair_indices, pair_indices[:,0], num_mols)
    triples_indices = []
    tmp = []
    for i in range(num_mols):
        mol_common_pair_indices = tf.where(tf.equal(tf.expand_dims(mol_pair_indices[i][:,1], axis=1),
                                    tf.expand_dims(mol_pair_indices[i][:,1], axis=0)))
        mol_triples_indices = tf.concat([tf.gather(mol_pair_indices[i], mol_common_pair_indices[:,0]),
                                tf.gather(mol_pair_indices[i], mol_common_pair_indices[:,1])[:,-1:]], axis=1)
        permutation_identity_pairs_mask = tf.where(tf.less(mol_triples_indices[:,2], mol_triples_indices[:,3]))
        mol_triples_indices = tf.squeeze(tf.gather(mol_triples_indices, permutation_identity_pairs_mask))
        triples_indices.append(mol_triples_indices)
    triples_indices = tf.concat(triples_indices, axis=0)

    triples_elements = tf.gather_nd(Zs, triples_indices[:,0:2])
    triples_element_pairs, _ = tf.nn.top_k(tf.stack([tf.gather_nd(Zs, triples_indices[:,0:3:2]),
                            tf.gather_nd(Zs, triples_indices[:,0:4:3])], axis=-1), k=2)
    sorted_triples_element_pairs = tf.cast(tf.reverse(triples_element_pairs, axis=[-1]), dtype=tf.int32)

    triples_distances = tf.stack([tf.gather_nd(dist_tensor, triples_indices[:,:3]), tf.gather_nd(dist_tensor,
                        tf.concat([triples_indices[:,:2], triples_indices[:,3:]], axis=1))], axis=1)
    r_ijk_s = tf.square(tf.expand_dims(tf.reduce_sum(triples_distances, axis=1) / 2.0, axis=-1) - tf.expand_dims(angular_rs, axis=0))
    exponential_factor = tf.exp(-eta * r_ijk_s)

    xyz_ij_ik = tf.reduce_sum(tf.gather_nd(dxyzs, triples_indices[:,:3]) * tf.gather_nd(dxyzs,
                        tf.concat([triples_indices[:,:2], triples_indices[:,3:]], axis=1)), axis=1)
    cos_theta = xyz_ij_ik / (triples_distances[:,0] * triples_distances[:,1])
    cos_theta = tf.where(tf.greater_equal(cos_theta, 1.0), tf.ones_like(cos_theta) - 1.0e-16, cos_theta)
    cos_theta = tf.where(tf.less_equal(cos_theta, -1.0), -1.0 * tf.ones_like(cos_theta) - 1.0e-16, cos_theta)
    triples_angle = tf.acos(cos_theta)
    theta_ijk_s = tf.expand_dims(triples_angle, axis=-1) - tf.expand_dims(theta_s, axis=0)
    cos_factor = tf.pow((1 + tf.cos(theta_ijk_s)), zeta)

    cutoff_factor = 0.5 * (tf.cos(3.14159265359 * triples_distances / angular_cutoff) + 1.0)
    scalar_factor = tf.pow(2.0, 1.0-zeta)

    angular_embedding = tf.reshape(scalar_factor * tf.expand_dims(cos_factor * tf.expand_dims(cutoff_factor[:,0] * cutoff_factor[:,1], axis=-1), axis=-1) \
                        * tf.expand_dims(exponential_factor, axis=-2), [tf.shape(triples_indices)[0], tf.shape(theta_s)[0] * tf.shape(angular_rs)[0]])
    return angular_embedding, triples_indices, triples_elements, sorted_triples_element_pairs

def tensormol_acsf(xyzs, Zs, elements, element_pairs, radial_cutoff, angular_cutoff,
                           radial_rs, angular_rs, theta_s, zeta, eta):
    """
    This function uses the tensormol atom centred symmetry functions.

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


    # The data
    with tf.name_scope("Params"):
        elements = tf.constant(elements, dtype=tf.int32)
        element_pairs = tf.constant(np.flip(element_pairs, axis=1), dtype=tf.int32)

        radial_cutoff = tf.constant(radial_cutoff, dtype=tf.float32)
        angular_cutoff = tf.constant(angular_cutoff, dtype=tf.float32)
        radial_rs = tf.constant(radial_rs, dtype=tf.float32)
        angular_rs = tf.constant(angular_rs, dtype=tf.float32)
        theta_s = tf.constant(theta_s, dtype=tf.float32)
        zeta = tf.constant(zeta, dtype=tf.float32)
        eta = tf.constant(eta, dtype=tf.float32)

        num_molecules = Zs.get_shape().as_list()[0]
        num_elements = elements.get_shape().as_list()[0]
        num_element_pairs = element_pairs.get_shape().as_list()[0]

    with tf.name_scope("Radial"):
        radial_embedding, pair_indices_rad, pair_elements = tf_symmetry_functions_radial_grid(xyzs, Zs, radial_cutoff,
                                                                                              radial_rs, eta)
    with tf.name_scope("Angular"):
        angular_embedding, triples_indices, triples_element, sorted_triples_element_pairs = tf_symmetry_function_angular_grid(
            xyzs, Zs, angular_cutoff, angular_rs, theta_s, zeta, eta)

    with tf.name_scope("Sum_rad"):
        pair_element_indices = tf.cast(
            tf.where(tf.equal(tf.expand_dims(pair_elements[:, 1], axis=-1), tf.expand_dims(elements, axis=0))), tf.int32)[:,
                               1]
        triples_elements_indices = tf.cast(tf.where(
            tf.reduce_all(tf.equal(tf.expand_dims(sorted_triples_element_pairs, axis=-2), element_pairs), axis=-1)),
                                           tf.int32)[:, 1]

        radial_scatter_indices = tf.concat([pair_indices_rad, tf.expand_dims(pair_element_indices, axis=1)], axis=1)
        angular_scatter_indices = tf.concat([triples_indices, tf.expand_dims(triples_elements_indices, axis=1)], axis=1)

        radial_molecule_embeddings = tf.dynamic_partition(radial_embedding, pair_indices_rad[:, 0], num_molecules)
        radial_atom_indices = tf.dynamic_partition(radial_scatter_indices[:, 1:], pair_indices_rad[:, 0], num_molecules)
        angular_molecule_embeddings = tf.dynamic_partition(angular_embedding, triples_indices[:, 0], num_molecules)
        angular_atom_indices = tf.dynamic_partition(angular_scatter_indices[:, 1:], triples_indices[:, 0], num_molecules)

    with tf.name_scope("Sum_ang"):
        embeddings = []
        mol_atom_indices = []
        for molecule in range(num_molecules):
            atom_indices = tf.cast(tf.where(tf.not_equal(Zs[molecule], 0)), tf.int32)
            molecule_atom_elements = tf.gather_nd(Zs[molecule], atom_indices)
            num_atoms = tf.shape(molecule_atom_elements)[0]
            radial_atom_embeddings = tf.reshape(
                tf.reduce_sum(tf.scatter_nd(radial_atom_indices[molecule], radial_molecule_embeddings[molecule],
                                            [num_atoms, num_atoms, num_elements, tf.shape(radial_rs)[0]]), axis=1),
                [num_atoms, -1])
            angular_atom_embeddings = tf.reshape(
                tf.reduce_sum(tf.scatter_nd(angular_atom_indices[molecule], angular_molecule_embeddings[molecule],
                                            [num_atoms, num_atoms, num_atoms, num_element_pairs,
                                             tf.shape(angular_rs)[0] * tf.shape(theta_s)[0]]),
                              axis=[1, 2]), [num_atoms, -1])
            embeddings.append(tf.concat([radial_atom_embeddings, angular_atom_embeddings], axis=1))
            mol_atom_indices.append(tf.concat([tf.fill([num_atoms, 1], molecule), atom_indices], axis=1))

        embeddings = tf.concat(embeddings, axis=0)
        mol_atom_indices = tf.concat(mol_atom_indices, axis=0)
        atom_Zs = tf.cast(tf.gather_nd(Zs, tf.where(tf.not_equal(Zs, 0))), dtype=tf.int32)
        atom_Z_indices = tf.cast(tf.where(tf.equal(tf.expand_dims(atom_Zs, axis=1),
                                                   tf.expand_dims(elements, axis=0)))[:, 1], tf.int32)

    with tf.name_scope("Result"):
        element_embeddings = tf.dynamic_partition(embeddings, atom_Z_indices, num_elements)
        mol_indices = tf.dynamic_partition(mol_atom_indices, atom_Z_indices, num_elements)

    return embeddings