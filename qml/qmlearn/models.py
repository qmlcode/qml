# MIT License
#
# Copyright (c) 2018 Lars A. Bratholm
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

from __future__ import division, absolute_import, print_function

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error

from ..utils import *
from .data import Data
from ..math import cho_solve

class _BaseModel(BaseEstimator):
    """
    Base class for all regression models
    """

    _estimator_type = "regressor"

    def fit(self, X):
        raise NotImplementedError

    def predict(self, X):
        return NotImplementedError

    def score(self, X, y=None):
        """
        Make predictions on `X` and return a score

        :param X: Data object
        :type X: object
        :param y: Energies
        :type y: array
        :return: score
        :rtype: float
        """

        # Make predictions
        y_pred = self.predict(X)

        # Get the true values
        if is_numeric_array(y):
            pass

        elif isinstance(X, Data):
            try:
                y = X.energies[X._indices]
            except:
                print("No kernel energies found in data object in module %s" % self.__class__.__name__)
                raise SystemExit

        else:
            print("Expected variable 'X' to be Data object. Got %s" % str(X))
            raise SystemExit

        # Return the score
        if self.scoring == 'mae':
            return mean_absolute_error(y, y_pred)
        elif self.scoring == 'neg_mae':
            return - mean_absolute_error(y, y_pred)
        elif self.scoring == 'rmsd':
            return np.sqrt(mean_squared_error(y, y_pred))
        elif self.scoring == 'neg_rmsd':
            return - np.sqrt(mean_squared_error(y, y_pred))
        elif self.scoring == 'neg_log_mae':
            return - np.log(mean_absolute_error(y, y_pred))

class KernelRidgeRegression(_BaseModel):
    """
    Standard Kernel Ridge Regression using a cholesky solver
    """

    def __init__(self, l2_reg=1e-10, scoring='neg_mae'):
        """
        :param l2_reg: l2 regularization
        :type l2_reg: float
        :param scoring: Metric used for scoring ('mae', 'neg_mae', 'rmsd', 'neg_rmsd', 'neg_log_mae')
        :type scoring: string
        """
        self.l2_reg = l2_reg
        self.scoring = scoring

        self.alpha = None

    def fit(self, X, y=None):
        """
        Fit the Kernel Ridge Regression model using a cholesky solver

        :param X: Data object or kernel
        :type X: object or array
        :param y: Energies
        :type y: array
        """

        if isinstance(X, Data):
            try:
                K, y = X._kernel, X.energies[X._indices]
            except:
                print("No kernel matrix and/or energies found in data object in module %s" % self.__class__.__name__)
                raise SystemExit
        elif is_numeric_array(X) and X.ndim == 2 and X.shape[0] == X.shape[1] and y is not None:
            K = X
        else:
            print("Expected variable 'X' to be kernel matrix or Data object. Got %s" % str(X))
            raise SystemExit


        K[np.diag_indices_from(K)] += self.l2_reg

        self.alpha = cho_solve(K, y)

    def predict(self, X):
        """
        Fit the Kernel Ridge Regression model using a cholesky solver

        :param X: Data object
        :type X: object
        :param y: Energies
        :type y: array
        """

        # Check if model has been fit
        if self.alpha is None:
            print("Error: The %s model has not been trained yet" % self.__class__.__name__)
            raise SystemExit

        if isinstance(X, Data):
            try:
                K = X._kernel
            except:
                print("No kernel matrix found in data object in module %s" % self.__class__.__name__)
                raise SystemExit
        elif is_numeric_array(X) and X.ndim == 2 and X.shape[1] == self.alpha.size:
            K = X
        elif is_numeric_array(X) and X.ndim == 2 and X.shape[0] == self.alpha.size:
            K = X.T
        else:
            print("Expected variable 'X' to be kernel matrix or Data object. Got %s" % str(X))
            raise SystemExit

        return np.dot(K, self.alpha)

class NeuralNetwork(_BaseModel):
    """
    Feed forward neural network that takes molecular or atomic representations for predicting molecular properties.
    """

    def __init__(self, hl1=20, hl2=10, hl3=5, hl4=0, batch_size=200, learning_rate=0.001, iterations=500, l1_reg=0.0,
                 l2_reg=0.0, scoring="neg_mae"):
        """

        :param hl1: number of nodes in the 1st hidden layer
        :type hl1: int
        :param hl2: number of nodes in the 2nd hidden layer
        :type hl2: int
        :param hl3: number of nodes in the 3rd hidden layer
        :type hl3: int
        :param hl4: number of nodes in the 4th hidden layer
        :type hl4: int
        :param batch_size: Number of samples to have in each batch during training
        :type batch_size: int
        :param learning_rate: step size in optimisation algorithm
        :type learning_rate: float
        :param iterations: number of iterations to do during training
        :type iterations: int
        :param l1_reg: L1 regularisation parameter
        :type l1_reg: float
        :param l2_reg: L2 regularisation parameter
        :type l2_reg: float
        :param scoring: What function to use for scoring. Available options are "neg_mae", "mae", "rmsd", "neg_rmsd",
        "neg_log_mae"
        :type scoring: string
        """

        self.hl1, self.hl2, self.hl3, self.hl4 = check_hl(hl1, hl2, hl3, hl4)
        self.batch_size = check_batchsize(batch_size)
        self.learning_rate = check_learningrate(learning_rate)
        self.iterations = check_iterations(iterations)
        self.l1_reg, self.l2_reg = check_reg(l1_reg, l2_reg)
        self.scoring = check_scoring(scoring)

    def fit(self, X, y=None):
        """
        Fit a feedforward neural network.

        :param X: Data object
        :type X: object from the class Data
        :param y: Energies
        :type y: numpy array of shape (n_samples,)
        """

        # obtaining the representations and the energies from the data object
        if isinstance(X, Data):
            g, zs, ene = X._representations, X.nuclear_charges, X.energies[X._indices]
            ene = np.reshape(ene, (ene.shape[0], 1))
        else:
            raise NotImplementedError

        if X._representation_type == 'molecular':

            self._generate_molecular_model(g.shape[-1])
            self._train(g, ene, zs, "molecular")

        elif X._representation_type == 'atomic':

            g, zs = self._padding(g, zs)
            self._generate_atomic_model(g.shape[1], g.shape[-1], X.unique_elements)
            self._train(g, ene, zs, "atomic")

    def predict(self, X):
        """
        Function that predicts the molecular properties from a trained model.

        :param X: Data object
        :type X: object from class Data
        :return: predicted properties
        :rtype: numpy array of shape (n_samples,)
        """

        if isinstance(X, Data):
            g, zs = X._representations, X.nuclear_charges

        if X._representation_type == "molecular":
            y_pred = self._predict(g, zs, "molecular")
        elif X._representation_type == "atomic":
            g, zs = self._padding(g, zs)
            y_pred = self._predict(g, zs, "atomic")

        return y_pred

    def _generate_molecular_model(self, n_features):
        """
        This function generates the tensorflow graph for the molecular feed forward neural net.

        :param n_features: number of features in the representation
        :type n_features: int
        """

        tf.reset_default_graph()

        hidden_layers = []
        for item in [self.hl1, self.hl2, self.hl3, self.hl4]:
            if item != 0:
                hidden_layers.append(item)

        hidden_layers = tuple(hidden_layers)

        # Initial set up of the NN
        with tf.name_scope("Data"):
            ph_x = tf.placeholder(tf.float32, [None, n_features], name="Representation")
            ph_y = tf.placeholder(tf.float32, [None, 1], name="True_energies")
            batch_size_tf = tf.placeholder(dtype=tf.int64, name="Batch_size")
            buffer_tf = tf.placeholder(dtype=tf.int64, name="Buffer")

            dataset = tf.data.Dataset.from_tensor_slices((ph_x, ph_y))
            dataset = dataset.shuffle(buffer_size=buffer_tf)
            dataset = dataset.batch(batch_size_tf)

            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
            tf_x, tf_y = iterator.get_next()

        with tf.name_scope("Weights"):
            weights, biases = generate_weights(n_in=n_features, n_out=1, hl=hidden_layers)

        with tf.name_scope("Model"):
            z = tf.add(tf.matmul(tf_x, tf.transpose(weights[0])), biases[0])
            h = tf.sigmoid(z)

            # Calculate the activation of the remaining hidden layers
            for i in range(1, len(weights) - 1):
                z = tf.add(tf.matmul(h, tf.transpose(weights[i])), biases[i])
                h = tf.sigmoid(z)

            # Calculating the output of the last layer
            y_pred = tf.add(tf.matmul(h, tf.transpose(weights[-1])), biases[-1], name="Predicted_energies")

        with tf.name_scope("Cost_func"):
            cost = self._cost(y_pred, tf_y, weights)

        with tf.name_scope("Optimiser"):
            optimisation_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        iter_init_op = iterator.make_initializer(dataset, name="dataset_init")

    def _generate_atomic_model(self, n_atoms, n_features, elements):
        """
        This function generates the atomic feed forward neural network.

        :param n_atoms: maximum number of atoms in the molecules
        :type n_atoms: int
        :param n_features: number of features in the representation
        :type n_features: int
        :param elements: unique elements in the data set
        :type elements: array of ints
        """
        #TODO actually make these into different graphs
        tf.reset_default_graph()

        hidden_layers = []
        for item in [self.hl1, self.hl2, self.hl3, self.hl4]:
            if item != 0:
                hidden_layers.append(item)

        hidden_layers = tuple(hidden_layers)

        # Initial set up of the NN
        with tf.name_scope("Data"):
            ph_x = tf.placeholder(tf.float32, [None, n_atoms, n_features], name="Representation")
            ph_y = tf.placeholder(tf.float32, [None, 1], name="True_energies")
            ph_zs = tf.placeholder(dtype=tf.int32, shape=[None, n_atoms], name="Atomic-numbers")
            batch_size_tf = tf.placeholder(dtype=tf.int64, name="Batch_size")
            buffer_tf = tf.placeholder(dtype=tf.int64, name="Buffer")

            dataset = tf.data.Dataset.from_tensor_slices((ph_x, ph_zs, ph_y))
            dataset = dataset.shuffle(buffer_size=buffer_tf)
            dataset = dataset.batch(batch_size_tf)

            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
            tf_x, tf_zs, tf_y = iterator.get_next()

        element_weights = {}
        element_biases = {}

        with tf.name_scope("Weights"):
            for i in range(len(elements)):
                weights, biases = generate_weights(n_in=n_features, n_out=1, hl=hidden_layers)
                element_weights[elements[i]] = weights
                element_biases[elements[i]] = biases

        with tf.name_scope("Model"):
            all_atomic_energies = tf.zeros_like(tf_zs, dtype=tf.float32)

            for el in elements:
                # Obtaining the indices of where in Zs there is the current element
                current_element = tf.expand_dims(tf.constant(el, dtype=tf.int32), axis=0)
                where_element = tf.cast(tf.where(tf.equal(tf_zs, current_element)), dtype=tf.int32)

                # Extract the descriptor corresponding to the right element
                current_element_in_x = tf.gather_nd(tf_x, where_element)

                # Calculate the atomic energy of all the atoms of type equal to the current element
                atomic_ene = self._atomic_nn(current_element_in_x, hidden_layers, element_weights[el],
                                                element_biases[el])

                # Put the atomic energies in a zero array with shape equal to zs and then add it to all the atomic energies
                updates = tf.scatter_nd(where_element, atomic_ene, tf.shape(tf_zs))
                all_atomic_energies = tf.add(all_atomic_energies, updates)

            # Summing the energies of all the atoms
            total_energies = tf.reduce_sum(all_atomic_energies, axis=-1, name="Predicted_energies", keepdims=True)

        with tf.name_scope("Cost_func"):
            cost = self._cost(total_energies, tf_y, element_weights)

        with tf.name_scope("Optimiser"):
            optimisation_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        iter_init_op = iterator.make_initializer(dataset, name="dataset_init")

    def _atomic_nn(self, x, hidden_layer_sizes, weights, biases):
        """
        Constructs the atomic part of the network. It calculates the atomic property for all atoms and then sums them
        together to obtain the molecular property.

        :param x: Atomic representation
        :type x: tf tensor of shape (n_samples, n_atoms, n_features)
        :param weights: Weights used in the network for a particular element.
        :type weights: list of tf.Variables of length hidden_layer_sizes.size + 1
        :param biases: Biases used in the network for a particular element.
        :type biases: list of tf.Variables of length hidden_layer_sizes.size + 1
        :return: Output
        :rtype: tf.Variable of size (n_samples, n_atoms)
        """

        # Calculate the activation of the first hidden layer
        z = tf.add(tf.tensordot(x, tf.transpose(weights[0]), axes=1), biases[0])
        h = tf.sigmoid(z)

        # Calculate the activation of the remaining hidden layers
        for i in range(len(hidden_layer_sizes) - 1):
            z = tf.add(tf.tensordot(h, tf.transpose(weights[i + 1]), axes=1), biases[i + 1])
            h = tf.sigmoid(z)

        # Calculating the output of the last layer
        z = tf.add(tf.tensordot(h, tf.transpose(weights[-1]), axes=1), biases[-1])

        z_squeezed = tf.squeeze(z, axis=[-1])

        return z_squeezed

    def _train(self, representations, energies, zs, representation_type):
        """
        This function trains the neural network.

        :param representations: the representations of the molecules (either atomic or molecular)
        :type representations: numpy array of shape (n_samples, max_n_atoms, n_features) or (n_samples, n_features)
        :param energies: the true molecular properties to be learnt
        :type energies: numpy array of shape (n_samples, )
        :param zs: the nuclear charges of all the atoms in the molecules
        :type zs: numpy array of shape (n_samples, max_n_atoms)
        :param representation_type: flag setting whether the representation is "atomic" or "molecular".
        :type representation_type: string
        """

        graph = tf.get_default_graph()

        with graph.as_default():
            tf_x = graph.get_tensor_by_name("Data/Representation:0")
            tf_y = graph.get_tensor_by_name("Data/True_energies:0")
            if representation_type == "atomic":
                tf_zs = graph.get_tensor_by_name("Data/Atomic-numbers:0")
            batch_size_tf = graph.get_tensor_by_name("Data/Batch_size:0")
            buffer_tf = graph.get_tensor_by_name("Data/Buffer:0")

            optimisation_op = graph.get_operation_by_name("Optimiser/Adam")
            iter_init_op = graph.get_operation_by_name("dataset_init")

        batch_size = get_batch_size(self.batch_size, representations.shape[0])

        self.session = tf.Session(graph=graph)

        self.session.run(tf.global_variables_initializer())

        for i in range(self.iterations):

            if i % 2 == 0:
                buff = int(3.5 * batch_size)
            else:
                buff = int(4.5 * batch_size)

            if representation_type == "atomic":
                self.session.run(iter_init_op, feed_dict={tf_x: representations, tf_y: energies, tf_zs: zs, buffer_tf: buff,
                                                          batch_size_tf: batch_size})
            else:
                self.session.run(iter_init_op, feed_dict={tf_x: representations, tf_y: energies, buffer_tf: buff,
                                  batch_size_tf:batch_size})

            while True:
                try:
                    self.session.run(optimisation_op)
                except tf.errors.OutOfRangeError:
                    break

    def _cost(self, y_pred, y, weights):
        """
        Constructs the cost function

        :param y_pred: Predicted molecular properties
        :type y_pred: tf.Variable of size (n_samples, 1)
        :param y: True molecular properties
        :type y: tf.placeholder of shape (n_samples, 1)
        :param weights: Weights used in the network.
        :type weights: list of tf.Variables of length hidden_layer_sizes.size + 1
        :return: Cost
        :rtype: tf.Variable of size (1,)
        """

        err = tf.square(tf.subtract(y, y_pred))
        loss = tf.reduce_mean(err, name="loss")

        if isinstance(weights, dict):
            cost = loss
            if self.l2_reg > 0:
                l2_loss = 0
                for element in weights:
                    l2_loss += self._l2_loss(weights[element])
                cost += l2_loss
            if self.l1_reg > 0:
                l1_loss = 0
                for element in weights:
                    l1_loss += self._l1_loss(weights[element])
                cost += l1_loss
        else:
            cost = loss
            if self.l2_reg > 0:
                l2_loss = self._l2_loss(weights)
                cost = cost + l2_loss
            if self.l1_reg > 0:
                l1_loss = self._l1_loss(weights)
                cost = cost + l1_loss

        return cost

    def _l2_loss(self, weights):
        """
        Creates the expression for L2-regularisation on the weights

        :param weights: tensorflow tensors representing the weights
        :type weights: list of tf tensors
        :return: tensorflow scalar representing the regularisation contribution to the cost function
        :rtype: tf.float32
        """

        reg_term = tf.zeros([], name="l2_loss")

        for i in range(len(weights)):
            reg_term += tf.reduce_sum(tf.square(weights[i]))

        return self.l2_reg * reg_term

    def _l1_loss(self, weights):
        """
        Creates the expression for L1-regularisation on the weights

        :param weights: tensorflow tensors representing the weights
        :type weights: list of tf tensors
        :return: tensorflow scalar representing the regularisation contribution to the cost function
        :rtype: tf.float32
        """

        reg_term = tf.zeros([], name="l1_loss")

        for i in range(len(weights)):
            reg_term += tf.reduce_sum(tf.abs(weights[i]))

        return self.l1_reg * reg_term

    def _predict(self, representation, zs, representation_type):
        """
        This function predicts the molecular properties from the representations.

        :param representation: representation of the molecules, can be atomic or molecular
        :type representation: numpy array of shape (n_samples, n_max_atoms, n_features) or (n_samples, n_features)
        :param zs: nuclear charges of the molecules
        :type zs: numpy array of shape (n_samples, n_max_atoms)
        :param representation_type: flag saying whether the representation is the "molecular" or "atomic"
        :type representation_type: string
        :return: molecular properties predictions
        :rtype: numpy array of shape (n_samples,)
        """

        graph = tf.get_default_graph()

        batch_size = get_batch_size(self.batch_size, representation.shape[0])

        with graph.as_default():
            tf_x = graph.get_tensor_by_name("Data/Representation:0")
            tf_y = graph.get_tensor_by_name("Data/True_energies:0")
            if representation_type == "atomic":
                tf_zs = graph.get_tensor_by_name("Data/Atomic-numbers:0")
            batch_size_tf = graph.get_tensor_by_name("Data/Batch_size:0")
            buffer_tf = graph.get_tensor_by_name("Data/Buffer:0")
            iter_init_op = graph.get_operation_by_name("dataset_init")

            y_pred = graph.get_tensor_by_name("Model/Predicted_energies:0")

            if representation_type == "atomic":
                self.session.run(iter_init_op, feed_dict={tf_x: representation, tf_y: np.empty((representation.shape[0], 1)),
                                                          tf_zs: zs, buffer_tf: 1, batch_size_tf: batch_size})
            else:
                self.session.run(iter_init_op,
                                 feed_dict={tf_x: representation, tf_y: np.empty((representation.shape[0], 1)),
                                            batch_size_tf: batch_size, buffer_tf:1})

        tot_y_pred = []

        while True:
            try:
                ene_predictions = self.session.run(y_pred)
                tot_y_pred.append(ene_predictions)
            except tf.errors.OutOfRangeError:
                break

        return np.concatenate(tot_y_pred, axis=0).ravel()

    def _padding(self, representation, nuclear_charges):
        """
        This function  takes atomic representations for molecules of different sizes and pads them with 0 so that they
        all have the same size.

        :param representation: list of numby arrays of shape (n_atoms, n_features)
        :param nuclear_charges: list of numpy arrays of shape (n_atoms,)
        :return: numpy array of shape (n_samples, max_n_atoms, n_features) and (n_samples, max_n_atoms)
        """

        max_n_atoms = 0

        for i in range(len(representation)):
            n_atoms = representation[i].shape[0]
            if n_atoms > max_n_atoms:
                max_n_atoms = n_atoms

        padded_rep = np.zeros((len(representation), max_n_atoms, representation[0].shape[1]))
        padded_zs = np.zeros((len(representation), max_n_atoms))

        for i in range(len(representation)):
            n_atoms = representation[i].shape[0]
            padded_rep[i, :n_atoms] = representation[i]
            padded_zs[i, :n_atoms] = nuclear_charges[i]

        return padded_rep, padded_zs






