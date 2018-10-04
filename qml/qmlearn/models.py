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
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error

try:
    import tensorflow as tf
    from ..aglaia.tf_utils import generate_weights
except ImportError:
    tf = None

from ..utils import is_numeric_array, is_numeric, is_numeric_1d_array, is_positive_integer_1d_array
from ..utils import get_unique, is_positive_integer_or_zero, get_batch_size, is_positive_integer
from ..utils import is_positive, is_positive_or_zero
from ..utils import InputError
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
        if is_numeric_1d_array(y):
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

    def _set_scoring(self, scoring):
        if not scoring in ['mae', 'neg_mae', 'rmsd', 'neg_rmsd', 'neg_log_mae']:
            raise InputError("Unknown scoring function")

        self.scoring = scoring

class KernelRidgeRegression(_BaseModel):
    """
    Standard Kernel Ridge Regression using a Cholesky solver
    :param l2_reg: l2 regularization
    :type l2_reg: float
    :param scoring: Metric used for scoring ('mae', 'neg_mae', 'rmsd', 'neg_rmsd', 'neg_log_mae')
    :type scoring: string
    """

    def __init__(self, l2_reg=1e-10, scoring='neg_mae'):
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

    :param hl1: number of hidden nodes in the 1st hidden layer
    :type hl1: int
    :param hl2: number of hidden nodes in the 2nd hidden layer
    :type hl2: int
    :param hl3: number of hidden nodes in the 3rd hidden layer
    :type hl3: int
    :param hl4: number of hidden nodes in the 4th hidden layer
    :type hl4: int
    :param batch_size: Number of samples to have in each batch during training
    :type batch_size: int
    :param learning_rate: Step size in optimisation algorithm
    :type learning_rate: float
    :param iterations: Number of iterations to do during training
    :type iterations: int
    :param l1_reg: L1 regularisation parameter
    :type l1_reg: float
    :param l2_reg: L2 regularisation parameter
    :type l2_reg: float
    :param scoring: What function to use for scoring. Available options are "neg_mae", "mae", "rmsd", "neg_rmsd",
        "neg_log_mae"
    :type scoring: string
    :param size: Maximum number of atoms in a molecule to support. 'auto' will determine this automatically.
        Is ignored when using molecular representations.
    :type size: int
    """

    def __init__(self, hl1=20, hl2=10, hl3=5, hl4=0, batch_size=200, learning_rate=0.001, iterations=500, l1_reg=0.0,
                 l2_reg=0.0, scoring="neg_mae", size='auto'):

        # Check if tensorflow is available
        self.tf_check()

        self._set_hl(hl1, hl2, hl3, hl4)
        self._set_batch_size(batch_size)
        self._set_learning_rate(learning_rate)
        self._set_iterations(iterations)
        self._set_reg(l1_reg, l2_reg)
        self._set_scoring(scoring)
        self._set_size(size)

        # Will be overwritten at fit time
        self.elements = None
        self._representation_type = None
        self._constant_features = None

    def _set_hl(self, hl1, hl2, hl3, hl4):

        for hl in hl1, hl2, hl3, hl4:
            if not is_positive_integer_or_zero(hl1):
                print("Error: Expected the number of hidden nodes in a layer to be positive. Got %s" % str(hl))
                raise SystemExit

        self.hl1 = hl1
        self.hl2 = hl2
        self.hl3 = hl3
        self.hl4 = hl4

    def _set_batch_size(self, bs):
        if not is_positive_integer(bs) or int(bs) == 1:
            print("'batch_size' should be larger than 1.")
            raise SystemExit

        self.batch_size = bs

    def _set_size(self, size):
        if not is_positive_integer(size) and size != 'auto':
            print("Variable 'size' should be a positive integer. Got %s" % str(size))
            raise SystemExit

        self.size = size

    def _set_learning_rate(self, lr):
        if not is_positive(lr):
            print("Expected positive float value for variable 'learning_rate'. Got %s" % str(lr))
            raise SystemExit

        self.learning_rate = lr

    def _set_iterations(self, it):
        if not is_positive_integer(it):
            print("Expected positive integer value for variable 'iterations'. Got %s" % str(it))
            raise SystemExit

        self.iterations = it

    def _set_reg(self, l1_reg, l2_reg):
        if not is_positive_or_zero(l1_reg) or not is_positive_or_zero(l2_reg):
            raise InputError("Expected positive float value for regularisation variables 'l1_reg' and 'l2_reg. Got %s and %s" % (str(l1_reg), str(l2_reg)))

        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    def fit(self, X, y=None, nuclear_charges=None):
        """
        Fit the neural network. A molecular/atomic network will be fit if
        molecular/atomic representations are given.

        :param X: Data object or representations
        :type X: Data object or array
        :param y: Energies. Only used if X is representations.
        :type y: numpy array of shape (n_samples,)
        :param nuclear_charges: Nuclear charges. Only used if X is representations.
        :type nuclear_charges: array
        """

        # Obtaining the representations and the energies from the data object
        if isinstance(X, Data):
            if y is not None or nuclear_charges is not None:
                print("Parameter 'y' and 'nuclear_charges' are expected to be 'None'",
                      "when 'X' is a Data object")
                raise SystemExit

            representations = X._representations
            nuclear_charges = X.nuclear_charges[X._indices]
            # 2D for tensorflow
            energies = np.reshape(X.energies[X._indices], (-1,1))
            self._representation_type = X._representation_type
        else:
            # y must be an array
            if not is_numeric_1d_array(y):
                print("Expected parameter 'y' to be a numeric 1d-array. Got %s" % type(y))
                raise SystemExit

            # 2D for tensorflow
            energies = np.reshape(y, (-1,1))
            representations = np.asarray(X)

            # Check if the representations are molecular or atomic
            if len(representations) > 0 and len(representations[0]) > 0:
                item = representations[0][0]
                if is_numeric(item):
                    self._representation_type = 'molecular'
                elif is_numeric_1d_array(item):
                    self._representation_type = 'atomic'
                else:
                    print("Could not recognise format of representations (parameter 'X')")
                    raise SystemExit
            else:
                print("Could not recognise format of representations (parameter 'X')")
                raise SystemExit

            # Check that nuclear charges are also provided if the representation is atomic
            if self._representation_type == 'atomic' and nuclear_charges is None:
                print("Expected parameter 'nuclear_charges' to be a numeric array.")
                raise SystemExit

        if self._representation_type == 'atomic':
            representations, nuclear_charges = self._padding(representations, nuclear_charges)

        # Get all unique elements
        self.elements = get_unique(nuclear_charges)

        # Remove constant features
        representations = self._remove_constant_features(representations)

        # Generate the model
        self._generate_model(representations.shape)
        # Train the model
        self._train(representations, energies, nuclear_charges)

    def _remove_constant_features(self, representations):
        """
        Removes constant features of the representations, and stores
        which features should be removed at predict time.
        """

        if self._representation_type == 'atomic':
            # Due to how the atomic neural network iś constructed,
            # this cannot be done elementwise
            rep = representations.reshape(-1, representations.shape[-1])
            if self._constant_features is None:
                self._constant_features = np.all(rep == rep[0], axis=0)

            rep = rep[:,~self._constant_features]
            rep = rep.reshape(representations.shape[0:2] + (-1,))

            return rep

        else:
            if self._constant_features is None:
                self._constant_features = np.all(representations == representations[0], axis=0)

            return representations[:,~self._constant_features]

    def predict(self, X, nuclear_charges=None):
        """
        Predict the molecular properties from a trained model.

        :param X: Data object
        :type X: object from class Data
        :param nuclear_charges: Nuclear charges. Only used if X is representations.
        :type nuclear_charges: array
        :return: predicted properties
        :rtype: numpy array of shape (n_samples,)
        """

        if self._representation_type is None:
            print("The model have not been fitted")
            raise SystemExit

        if isinstance(X, Data):
            if self._representation_type == 'molecular' and nuclear_charges is not None:
                print("Parameter 'nuclear_charges' is expected to be 'None'",
                      "when 'X' is a Data object")
                raise SystemExit

            representations, nuclear_charges = X._representations, X.nuclear_charges[X._indices]
        else:
            representations = X

            if self._representation_type == 'atomic' and nuclear_charges is None:
                print("Parameter 'nuclear_charges' is expected to be numeric array")
                raise SystemExit

        if self._representation_type == "atomic":
            representations, nuclear_charges = self._padding(representations, nuclear_charges)

        # Remove constant features
        representations = self._remove_constant_features(representations)

        y_pred = self._predict(representations, nuclear_charges)

        return y_pred

    # TODO support for different activation functions
    def _generate_model(self, shape):
        """
        Generate the Tensorflow graph for the molecular feed forward neural net.

        :param shape: Shape of representations
        :type shape: tuple
        """

        n_features = shape[-1]

        self.graph = tf.Graph()

        hidden_layers = []
        for item in [self.hl1, self.hl2, self.hl3, self.hl4]:
            if item != 0:
                hidden_layers.append(int(item))
            else:
                break

        # Initial set up of the NN
        with self.graph.as_default():
            with tf.name_scope("Data"):
                ph_y = tf.placeholder(tf.float32, [None, 1], name="True_energies")
                if self._representation_type == 'atomic':
                    n_atoms = shape[1]
                    ph_zs = tf.placeholder(dtype=tf.int32, shape=[None, n_atoms], name="Atomic-numbers")
                    ph_x = tf.placeholder(tf.float32, [None, n_atoms, n_features], name="Representation")
                else:
                    ph_x = tf.placeholder(tf.float32, [None, n_features], name="Representation")
                # Has to be int64 for tf.data.Dataset
                batch_size_tf = tf.placeholder(dtype=tf.int64, name="Batch_size")
                buffer_tf = tf.placeholder(dtype=tf.int64, name="Buffer")

                if self._representation_type == 'atomic':
                    dataset = tf.data.Dataset.from_tensor_slices((ph_x, ph_zs, ph_y))
                else:
                    dataset = tf.data.Dataset.from_tensor_slices((ph_x, ph_y))

                dataset = dataset.shuffle(buffer_size=buffer_tf)
                dataset = dataset.batch(batch_size_tf)

                iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

                if self._representation_type == 'atomic':
                    tf_x, tf_zs, tf_y = iterator.get_next()
                else:
                    tf_x, tf_y = iterator.get_next()

            with tf.name_scope("Weights"):
                if self._representation_type == 'atomic':
                    weights = {}
                    biases = {}

                    for i, element in enumerate(self.elements):
                        w, b = generate_weights(n_in=n_features, n_out=1, hl=hidden_layers)
                        weights[element] = w
                        biases[element] = b
                else:
                    weights, biases = generate_weights(n_in=n_features, n_out=1, hl=hidden_layers)

            with tf.name_scope("Model"):
                if self._representation_type == 'atomic':
                    all_atomic_energies = tf.zeros_like(tf_zs, dtype=tf.float32)

                    for el in self.elements:
                        # Obtaining the indices where the current element occurs
                        current_element = tf.expand_dims(tf.constant(el, dtype=tf.int32), axis=0)
                        where_element = tf.cast(tf.where(tf.equal(tf_zs, current_element)), dtype=tf.int32)

                        # Extract the descriptor corresponding to the right element
                        current_element_in_x = tf.gather_nd(tf_x, where_element)

                        # Calculate the atomic energy of all the atoms of type equal to the current element
                        atomic_ene = self._atomic_nn(current_element_in_x, hidden_layers, weights[el],
                                                        biases[el])

                        # Put the atomic energies in a zero array with shape equal to zs and then add it to all the atomic energies
                        updates = tf.scatter_nd(where_element, atomic_ene, tf.shape(tf_zs))
                        all_atomic_energies = tf.add(all_atomic_energies, updates)

                    # Summing the energies of all the atoms
                    y_pred = tf.reduce_sum(all_atomic_energies, axis=-1, name="Predicted_energies", keepdims=True)

                else:
                    if len(hidden_layers) > 0:
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

    def _train(self, representations, energies, nuclear_charges):
        """
        Train the neural network.

        :param representations: the representations of the molecules (either atomic or molecular)
        :type representations: numpy array of shape (n_samples, max_n_atoms, n_features) or (n_samples, n_features)
        :param energies: the true molecular properties to be learnt
        :type energies: numpy array of shape (n_samples, )
        :param nuclear_charges: the nuclear charges of all the atoms in the molecules
        :type nuclear_charges: numpy array of shape (n_samples, max_n_atoms)
        :param representation_type: flag setting whether the representation is "atomic" or "molecular".
        :type representation_type: string
        """

        batch_size = get_batch_size(self.batch_size, representations.shape[0])

        with self.graph.as_default():
            tf_x = self.graph.get_tensor_by_name("Data/Representation:0")
            tf_y = self.graph.get_tensor_by_name("Data/True_energies:0")
            if self._representation_type == "atomic":
                tf_zs = self.graph.get_tensor_by_name("Data/Atomic-numbers:0")
            batch_size_tf = self.graph.get_tensor_by_name("Data/Batch_size:0")
            buffer_tf = self.graph.get_tensor_by_name("Data/Buffer:0")

            optimisation_op = self.graph.get_operation_by_name("Optimiser/Adam")
            iter_init_op = self.graph.get_operation_by_name("dataset_init")

            self.session = tf.Session(graph=self.graph)

            self.session.run(tf.global_variables_initializer())

            for i in range(self.iterations):

                # Alternate buffer size to improve global shuffling (Not sure that this is needed)
                if i % 2 == 0:
                    buff = int(3.5 * batch_size)
                else:
                    buff = int(4.5 * batch_size)

                if self._representation_type == "atomic":
                    self.session.run(iter_init_op, feed_dict={tf_x: representations, tf_y: energies, tf_zs: nuclear_charges, buffer_tf: buff,
                                                              batch_size_tf: batch_size})
                else:
                    self.session.run(iter_init_op, feed_dict={tf_x: representations, tf_y: energies, buffer_tf: buff,
                                      batch_size_tf:batch_size})

                # Iterate to the end of the given data
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

        if self._representation_type == 'atomic':
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
                cost += l2_loss
            if self.l1_reg > 0:
                l1_loss = self._l1_loss(weights)
                cost += l1_loss

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

    def _predict(self, representations, nuclear_charges=None):
        """
        Predicts the molecular properties from the representations.

        :param representations: representations of the molecules, can be atomic or molecular
        :type representations: numpy array of shape (n_samples, n_max_atoms, n_features) or (n_samples, n_features)
        :param nuclear_charges: nuclear charges of the molecules. Only required for atomic representations.
        :type nuclear_charges: numpy array of shape (n_samples, n_max_atoms)
        :return: molecular properties predictions
        :rtype: numpy array of shape (n_samples,)
        """

        batch_size = get_batch_size(self.batch_size, representations.shape[0])

        with self.graph.as_default():
            tf_x = self.graph.get_tensor_by_name("Data/Representation:0")
            tf_y = self.graph.get_tensor_by_name("Data/True_energies:0")
            if self._representation_type == "atomic":
                tf_zs = self.graph.get_tensor_by_name("Data/Atomic-numbers:0")
            batch_size_tf = self.graph.get_tensor_by_name("Data/Batch_size:0")
            buffer_tf = self.graph.get_tensor_by_name("Data/Buffer:0")
            iter_init_op = self.graph.get_operation_by_name("dataset_init")

            y_pred = self.graph.get_tensor_by_name("Model/Predicted_energies:0")

            if self._representation_type == "atomic":
                self.session.run(iter_init_op, feed_dict={tf_x: representations, tf_y: np.empty((representations.shape[0], 1)),
                                                          tf_zs: nuclear_charges, buffer_tf: 1, batch_size_tf: batch_size})
            else:
                self.session.run(iter_init_op,
                                 feed_dict={tf_x: representations, tf_y: np.empty((representations.shape[0], 1)),
                                            batch_size_tf: batch_size, buffer_tf:1})

        tot_y_pred = []

        # Predict until reaching the end of the iterator
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

        if self.size == 'auto':
            self.size = max_n_atoms

        elif max_n_atoms > self.size:
            print("Trying to predict on larger molecules than given by the 'size' parameter at initialization")
            raise SystemExit

        padded_rep = np.zeros((len(representation), max_n_atoms, representation[0].shape[1]))
        padded_zs = np.zeros((len(representation), max_n_atoms))

        for i in range(len(representation)):
            n_atoms = representation[i].shape[0]
            padded_rep[i, :n_atoms] = representation[i]
            padded_zs[i, :n_atoms] = nuclear_charges[i]

        return padded_rep, padded_zs

    def tf_check(self):
        if tf is None:
            print("Tensorflow not found but is needed for %s" % self.__class__.__name__)
            raise SystemExit
