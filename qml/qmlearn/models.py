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

class MolecularNeuralNetwork(_BaseModel):
    """
    Feed forward neural network that takes molecular representations and molecular properties.
    """

    def __init__(self, hl1=20, hl2=10, hl3=5, hl4=0, batch_size=200, learning_rate=0.001, iterations=500, l1_reg=0.0,
                 l2_reg=0.0, scoring="neg_mae"):
        """

        :param hl1: number of nodes in the 1st hidden layer
        :param hl2: number of nodes in the 2nd hidden layer
        :param hl3: number of nodes in the 3rd hidden layer
        :param hl4: number of nodes in the 4th hidden layer
        :param batch_size: Number of samples to have in each batch during training
        :param learning_rate: step size in optimisation algorithm
        :param iterations: number of iterations to do during training
        :param l1_reg: L1 regularisation parameter
        :param l2_reg: L2 regularisation parameter
        :param scoring: What function to use for scoring
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
        :param y: Energies
        """

        # obtaining the representations and the energies from the data object
        if isinstance(X, Data):
            g, ene = X._representations, X.energies[X._indices]
            ene = np.reshape(ene, (ene.shape[0], 1))


        # Generating the model
        self._generate_model(g.shape[1])

        # Training the model
        self._train(g, ene)

    def predict(self, X):

        if isinstance(X, Data):
            g = X._representations

        y_pred = self._predict(g)

        return y_pred

    def _generate_model(self, n_features):
        """
        This function generates the tensorflow graph.
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

    def _train(self, representations, energies):

        graph = tf.get_default_graph()

        with graph.as_default():
            tf_x = graph.get_tensor_by_name("Data/Representation:0")
            tf_y = graph.get_tensor_by_name("Data/True_energies:0")
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

        err = tf.square(tf.subtract(y,y_pred))
        loss = tf.reduce_mean(err, name="loss")
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

    def _predict(self, representation):

        graph = tf.get_default_graph()

        batch_size = get_batch_size(self.batch_size, representation.shape[0])

        with graph.as_default():
            tf_x = graph.get_tensor_by_name("Data/Representation:0")
            tf_y = graph.get_tensor_by_name("Data/True_energies:0")
            batch_size_tf = graph.get_tensor_by_name("Data/Batch_size:0")
            buffer_tf = graph.get_tensor_by_name("Data/Buffer:0")
            iter_init_op = graph.get_operation_by_name("dataset_init")

            y_pred = graph.get_tensor_by_name("Model/Predicted_energies:0")

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






