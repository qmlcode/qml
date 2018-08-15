# MIT License
#
# Copyright (c) 2018 Silvia Amabilino, Lars Andersen Bratholm
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

"""
Module containing the general neural network class and the child classes for the molecular and atomic neural networks.
"""

from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator
from qml.aglaia.symm_funct import generate_parkhill_acsf
from qml.aglaia.utils import InputError, ceil, is_positive_or_zero, is_positive_integer, is_positive, \
        is_bool, is_positive_integer_or_zero, is_string, is_positive_integer_array, is_array_like, is_none, \
        check_global_representation, check_y, check_sizes, check_dy, check_classes, is_numeric_array, is_non_zero_integer, \
    is_positive_integer_or_zero_array, check_local_representation

from qml.aglaia.tf_utils import TensorBoardLogger

try:
    from qml.data import Compound
    from qml import representations as qml_rep
except ImportError:
    raise ImportError("The module qml is required")

try:
    import tensorflow
except ImportError:
    raise ImportError("Tensorflow 1.8 is required to run neural networks.")

class _NN(BaseEstimator):

    """
    Parent class for training multi-layered neural networks on molecular or atomic properties via Tensorflow
    """

    def __init__(self, hidden_layer_sizes, l1_reg, l2_reg, batch_size, learning_rate,
                 iterations, tensorboard, store_frequency, tf_dtype, scoring_function,
                 activation_function, optimiser, beta1, beta2, epsilon,
                 rho, initial_accumulator_value, initial_gradient_squared_accumulator_value,
                 l1_regularization_strength,l2_regularization_strength, tensorboard_subdir):

        """
        :param hidden_layer_sizes: Number of hidden layers. The n'th element represents the number of neurons in the n'th
            hidden layer.
        :type hidden_layer_size: Tuple of integers
        :param l1_reg: L1-regularisation parameter for the neural network weights
        :type l1_reg: float
        :param l2_reg: L2-regularisation parameter for the neural network weights
        :type l2_reg: float
        :param batch_size: Size of minibatches for the ADAM optimizer. If set to 'auto' ``batch_size = min(200,n_samples)``
        :type batch_size: integer
        :param learning_rate: The learning rate in the numerical minimisation.
        :type learning_rate: float
        :param iterations: Total number of iterations that will be carried out during the training process.
        :type iterations: integer
        :param tf_dtype: Accuracy to use for floating point operations in tensorflow. 64 and 'float64' is recognised as tf.float64
            and similar for tf.float32 and tf.float16.
        :type tf_dtype: Tensorflow datatype
        :param scoring_function: Scoring function to use. Available choices are 'mae', 'rmse', 'r2'.
        :type scoring_function: string
        :param activation_function: Activation function to use in the neural network. Currently 'sigmoid', 'tanh', 'elu', 'softplus',
            'softsign', 'relu', 'relu6', 'crelu' and 'relu_x' is supported.
        :type activation_function: Tensorflow datatype
        :param beta1: parameter for AdamOptimizer
        :type beta1: float
        :param beta2: parameter for AdamOptimizer
        :type beta2: float
        :param epsilon: parameter for AdadeltaOptimizer
        :type epsilon: float
        :param rho: parameter for AdadeltaOptimizer
        :type rho: float
        :param initial_accumulator_value: parameter for AdagradOptimizer
        :type initial_accumulator_value: float
        :param initial_gradient_squared_accumulator_value: parameter for AdagradDAOptimizer
        :type initial_gradient_squared_accumulator_value: float
        :param l1_regularization_strength: parameter for AdagradDAOptimizer
        :type l1_regularization_strength: float
        :param l2_regularization_strength: parameter for AdagradDAOptimizer
        :type l2_regularization_strength: float
        :param tensorboard: Store summaries to tensorboard or not
        :type tensorboard: boolean
        :param store_frequency: How often to store summaries to tensorboard.
        :type store_frequency: integer
        :param tensorboard_subdir: Directory to store tensorboard data
        :type tensorboard_subdir: string
        """

        super(_NN,self).__init__()

        # Initialising the parameters
        self._set_hidden_layers_sizes(hidden_layer_sizes)
        self._set_l1_reg(l1_reg)
        self._set_l2_reg(l2_reg)
        self._set_batch_size(batch_size)
        self._set_learning_rate(learning_rate)
        self._set_iterations(iterations)
        self._set_tf_dtype(tf_dtype)
        self._set_scoring_function(scoring_function)
        self._set_tensorboard(tensorboard, store_frequency, tensorboard_subdir)
        self._set_activation_function(activation_function)

        # Placeholder variables
        self.n_features = None
        self.n_samples = None
        self.training_cost = []
        self.session = None

        # Setting the optimiser
        self._set_optimiser_param(beta1, beta2, epsilon, rho, initial_accumulator_value,
                                  initial_gradient_squared_accumulator_value, l1_regularization_strength,
                                  l2_regularization_strength)

        self.optimiser = self._set_optimiser_type(optimiser)

        # Placholder variables for data
        self.compounds = None
        self.representation = None
        self.properties = None
        self.gradients = None
        self.classes = None

    def _set_activation_function(self, activation_function):
        """
        This function sets which activation function will be used in the model.

        :param activation_function: name of the activation function to use
        :type activation_function: string or tf class
        :return: None
        """
        if activation_function in ['sigmoid', tf.nn.sigmoid]:
            self.activation_function = tf.nn.sigmoid
        elif activation_function in ['tanh', tf.nn.tanh]:
            self.activation_function = tf.nn.tanh
        elif activation_function in ['elu', tf.nn.elu]:
            self.activation_function = tf.nn.elu
        elif activation_function in ['softplus', tf.nn.softplus]:
            self.activation_function = tf.nn.softplus
        elif activation_function in ['softsign', tf.nn.softsign]:
            self.activation_function = tf.nn.softsign
        elif activation_function in ['relu', tf.nn.relu]:
            self.activation_function = tf.nn.relu
        elif activation_function in ['relu6', tf.nn.relu6]:
            self.activation_function = tf.nn.relu6
        elif activation_function in ['crelu', tf.nn.crelu]:
            self.activation_function = tf.nn.crelu
        elif activation_function in ['relu_x', tf.nn.relu_x]:
            self.activation_function = tf.nn.relu_x
        else:
            raise InputError("Unknown activation function. Got %s" % str(activation_function))

    def _set_l1_reg(self, l1_reg):
        """
        This function sets the value of the l1 regularisation that will be used on the weights in the model.

        :param l1_reg: l1 regularisation on the weights
        :type l1_reg: float
        :return: None
        """
        if not is_positive_or_zero(l1_reg):
            raise InputError("Expected positive float value for variable 'l1_reg'. Got %s" % str(l1_reg))
        self.l1_reg = l1_reg

    def _set_l2_reg(self, l2_reg):
        """
         This function sets the value of the l2 regularisation that will be used on the weights in the model.

         :param l2_reg: l2 regularisation on the weights
         :type l2_reg: float
         :return: None
         """
        if not is_positive_or_zero(l2_reg):
            raise InputError("Expected positive float value for variable 'l2_reg'. Got %s" % str(l2_reg))
        self.l2_reg = l2_reg

    def _set_batch_size(self, batch_size):
        """
        This function sets the value of the batch size. The value of the batch size will be checked again once the data
        set is available, to make sure that it is sensible. This will be done by the function _get_batch_size.

        :param batch_size: size of the batch size.
        :type batch_size: float
        :return: None
        """
        if batch_size != "auto":
            if not is_positive_integer(batch_size):
                raise InputError("Expected 'batch_size' to be a positive integer. Got %s" % str(batch_size))
            elif batch_size == 1:
                raise InputError("batch_size must be larger than 1. Got %s" % str(batch_size))
            self.batch_size = int(batch_size)
        else:
            self.batch_size = batch_size

    def _set_learning_rate(self, learning_rate):
        """
        This function sets the value of the learning that will be used by the optimiser.

        :param learning_rate: step size in the optimisation algorithms
        :type l1_reg: float
        :return: None
        """
        if not is_positive(learning_rate):
            raise InputError("Expected positive float value for variable learning_rate. Got %s" % str(learning_rate))
        self.learning_rate = float(learning_rate)

    def _set_iterations(self, iterations):
        """
        This function sets the number of iterations that will be carried out by the optimiser.

        :param iterations: number of iterations
        :type l1_reg: int
        :return: None
        """
        if not is_positive_integer(iterations):
            raise InputError("Expected positive integer value for variable iterations. Got %s" % str(iterations))
        self.iterations = int(iterations)

    # TODO check that the estimators actually use this
    def _set_tf_dtype(self, tf_dtype):
        """
        This sets what data type will be used in the model.

        :param tf_dtype: data type
        :type tf_dtype: string or tensorflow class or int
        :return: None
        """
        # 2 == tf.float64 and 1 == tf.float32 for some reason
        # np.float64 recognised as tf.float64 as well
        if tf_dtype in ['64', 64, 'float64', tf.float64]:
            self.tf_dtype = tf.float64
        elif tf_dtype in ['32', 32, 'float32', tf.float32]:
            self.tf_dtype = tf.float32
        elif tf_dtype in ['16', 16, 'float16', tf.float16]:
            self.tf_dtype = tf.float16
        else:
            raise InputError("Unknown tensorflow data type. Got %s" % str(tf_dtype))

    def _set_optimiser_param(self, beta1, beta2, epsilon, rho, initial_accumulator_value, initial_gradient_squared_accumulator_value,
                             l1_regularization_strength, l2_regularization_strength):
        """
        This function sets all the parameters that are required by all the optimiser functions. In the end, only the parameters
        for the optimiser chosen will be used.

        :param beta1: parameter for AdamOptimizer
        :type beta1: float
        :param beta2: parameter for AdamOptimizer
        :type beta2: float
        :param epsilon: parameter for AdadeltaOptimizer
        :type epsilon: float
        :param rho: parameter for AdadeltaOptimizer
        :type rho: float
        :param initial_accumulator_value: parameter for AdagradOptimizer
        :type initial_accumulator_value: float
        :param initial_gradient_squared_accumulator_value: parameter for AdagradDAOptimizer
        :type initial_gradient_squared_accumulator_value: float
        :param l1_regularization_strength: parameter for AdagradDAOptimizer
        :type l1_regularization_strength: float
        :param l2_regularization_strength: parameter for AdagradDAOptimizer
        :type l2_regularization_strength: float
        :return: None
        """
        if not is_positive(beta1) and not is_positive(beta2):
            raise InputError("Expected positive float values for variable beta1 and beta2. Got %s and %s." % (str(beta1),str(beta2)))
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)

        if not is_positive(epsilon):
            raise InputError("Expected positive float value for variable epsilon. Got %s" % str(epsilon))
        self.epsilon = float(epsilon)

        if not is_positive(rho):
            raise InputError("Expected positive float value for variable rho. Got %s" % str(rho))
        self.rho = float(rho)

        if not is_positive(initial_accumulator_value) and not is_positive(initial_gradient_squared_accumulator_value):
            raise InputError("Expected positive float value for accumulator values. Got %s and %s" %
                             (str(initial_accumulator_value), str(initial_gradient_squared_accumulator_value)))
        self.initial_accumulator_value = float(initial_accumulator_value)
        self.initial_gradient_squared_accumulator_value = float(initial_gradient_squared_accumulator_value)

        if not is_positive_or_zero(l1_regularization_strength) and not is_positive_or_zero(l2_regularization_strength):
            raise InputError("Expected positive or zero float value for regularisation variables. Got %s and %s" %
                             (str(l1_regularization_strength), str(l2_regularization_strength)))
        self.l1_regularization_strength = float(l1_regularization_strength)
        self.l2_regularization_strength = float(l2_regularization_strength)

    def _set_optimiser_type(self, optimiser):
        """
        This function sets which numerical optimisation algorithm will be used for training.

        :param optimiser: Optimiser
        :type optimiser: string or tf class
        :return: tf optimiser to use
        :rtype: tf class
        """
        self.AdagradDA = False
        if optimiser in ['AdamOptimizer', tf.train.AdamOptimizer]:
            optimiser_type = tf.train.AdamOptimizer
        elif optimiser in ['AdadeltaOptimizer', tf.train.AdadeltaOptimizer]:
            optimiser_type = tf.train.AdadeltaOptimizer
        elif optimiser in ['AdagradOptimizer', tf.train.AdagradOptimizer]:
            optimiser_type = tf.train.AdagradOptimizer
        elif optimiser in ['AdagradDAOptimizer', tf.train.AdagradDAOptimizer]:
            optimiser_type = tf.train.AdagradDAOptimizer
            self.AdagradDA = True
        elif optimiser in ['GradientDescentOptimizer', tf.train.GradientDescentOptimizer]:
            optimiser_type = tf.train.GradientDescentOptimizer
        else:
            raise InputError("Unknown optimiser. Got %s" % str(optimiser))

        return optimiser_type

    def _set_optimiser(self):
        """
        This function instantiates an object from the optimiser class that has been selected by the user. It also sets
        the parameters for the optimiser.

        :return: Optimiser with set parameters
        :rtype: object of tf optimiser class
        """
        self.AdagradDA = False
        if self.optimiser in ['AdamOptimizer', tf.train.AdamOptimizer]:
            optimiser_obj = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2,
                                                    epsilon=self.epsilon)
        elif self.optimiser in ['AdadeltaOptimizer', tf.train.AdadeltaOptimizer]:
             optimiser_obj = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate, rho=self.rho, epsilon=self.epsilon)
        elif self.optimiser in ['AdagradOptimizer', tf.train.AdagradOptimizer]:
             optimiser_obj = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                       initial_accumulator_value=self.initial_accumulator_value)
        elif self.optimiser in ['AdagradDAOptimizer', tf.train.AdagradDAOptimizer]:
            self.global_step = tf.placeholder(dtype=tf.int64)
            optimiser_obj = tf.train.AdagradDAOptimizer(learning_rate=self.learning_rate, global_step=self.global_step,
                                                         initial_gradient_squared_accumulator_value=self.initial_gradient_squared_accumulator_value,
                                                         l1_regularization_strength=self.l1_regularization_strength,
                                                         l2_regularization_strength=self.l2_regularization_strength)
            self.AdagradDA = True
        elif self.optimiser in ['GradientDescentOptimizer', tf.train.GradientDescentOptimizer]:
            optimiser_obj = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        else:
            raise InputError("Unknown optimiser class. Got %s" % str(self.optimiser))

        return optimiser_obj

    def _set_scoring_function(self, scoring_function):
        """
        This function sets which scoring metrics to use when scoring the results.

        :param scoring_function: name of the scoring function to use
        :type scoring_function: string
        :return: None
        """
        if not is_string(scoring_function):
            raise InputError("Expected a string for variable 'scoring_function'. Got %s" % str(scoring_function))
        if scoring_function.lower() not in ['mae', 'rmse', 'r2']:
            raise InputError("Available scoring functions are 'mae', 'rmse', 'r2'. Got %s" % str(scoring_function))

        self.scoring_function = scoring_function

    def _set_hidden_layers_sizes(self, hidden_layer_sizes):
        """
        This function sets the number of hidden layers and the number of neurons in each hidden layer. The length of the
        tuple tells the number of hidden layers (n_hidden_layers) while each element of the tuple specifies the number
        of hidden neurons in that layer.

        :param hidden_layer_sizes: number of hidden layers and hidden neurons.
        :type hidden_layer_sizes: tuple of length n_hidden_layer
        :return: None
        """
        try:
            iterator = iter(hidden_layer_sizes)
        except TypeError:
            raise InputError("'hidden_layer_sizes' must be a tuple of positive integers. Got a non-iterable object.")

        if None in hidden_layer_sizes:
            raise InputError("'hidden_layer_sizes' must be a tuple of positive integers. Got None elements")
        if not is_positive_integer_array(hidden_layer_sizes):
            raise InputError("'hidden_layer_sizes' must be a tuple of positive integers")

        self.hidden_layer_sizes = np.asarray(hidden_layer_sizes, dtype = int)

    def _set_tensorboard(self, tensorboard, store_frequency, tensorboard_subdir):
        """
        This function prepares all the things needed to use tensorboard when training the estimator.

        :param tensorboard: whether to use tensorboard or not
        :type tensorboard: boolean
        :param store_frequency: Every how many steps to save data to tensorboard
        :type store_frequency: int
        :param tensorboard_subdir: directory where to save the tensorboard data
        :type tensorboard_subdir: string
        :return: None
        """

        if not is_bool(tensorboard):
            raise InputError("Expected boolean value for variable tensorboard. Got %s" % str(tensorboard))
        self.tensorboard = bool(tensorboard)

        if not self.tensorboard:
            return

        if not is_string(tensorboard_subdir):
            raise InputError('Expected string value for variable tensorboard_subdir. Got %s' % str(tensorboard_subdir))

        # TensorBoardLogger will handle all tensorboard related things
        self.tensorboard_logger_training = TensorBoardLogger(tensorboard_subdir + '/training')
        self.tensorboard_subdir_training = tensorboard_subdir + '/training'

        self.tensorboard_logger_representation = TensorBoardLogger(tensorboard_subdir + '/representation')
        self.tensorboard_subdir_representation = tensorboard_subdir + '/representation'

        if not is_positive_integer(store_frequency):
            raise InputError("Expected positive integer value for variable store_frequency. Got %s" % str(store_frequency))

        if store_frequency > self.iterations:
            print("Only storing final iteration for tensorboard")
            self.tensorboard_logger_training.set_store_frequency(self.iterations)
        else:
            self.tensorboard_logger_training.set_store_frequency(store_frequency)

    def _init_weight(self, n1, n2, name):
        """
        This function generates a the matrix of weights to go from layer l to the next layer l+1. It initialises the
        weights from a truncated normal distribution where the standard deviation is 1/sqrt(n2) and the mean is zero.

        :param n1: size of the layer l+1
        :type n1: int
        :param n2: size of the layer l
        :type n2: int
        :param name: name to give to the tensor of weights to make tensorboard clear
        :type name: string

        :return: weights to go from layer l to layer l+1
        :rtype: tf tensor of shape (n1, n2)
        """

        w = tf.Variable(tf.truncated_normal([n1,n2], stddev = 1.0 / np.sqrt(n2), dtype = self.tf_dtype),
                dtype = self.tf_dtype, name = name)

        return w

    def _init_bias(self, n, name):
        """
        This function initialises the biases to go from layer l to layer l+1.

        :param n: size of the layer l+1
        :type n: int
        :param name: name to give to the tensor of biases to make tensorboard clear
        :type name: string
        :return: biases
        :rtype: tf tensor of shape (n, 1)
        """

        b = tf.Variable(tf.zeros([n], dtype = self.tf_dtype), name=name, dtype = self.tf_dtype)

        return b

    def _generate_weights(self, n_out):
        """
        Generates the weights and the biases, by looking at the size of the hidden layers,
        the number of features in the representation and the number of outputs. The weights are initialised from
        a zero centered normal distribution with precision :math:`\\tau = a_{m}`, where :math:`a_{m}` is the number
        of incoming connections to a neuron. Weights larger than two standard deviations from the mean is
        redrawn.

        :param n_out: Number of outputs
        :type n_out: integer
        :return: tuple of weights and biases, each being of length (n_hidden_layers + 1)
        :rtype: tuple
        """

        weights = []
        biases = []

        # Weights from input layer to first hidden layer
        weights.append(self._init_weight(self.hidden_layer_sizes[0], self.n_features, 'weight_in'))
        biases.append(self._init_bias(self.hidden_layer_sizes[0], 'bias_in'))

        # Weights from one hidden layer to the next
        for i in range(1, self.hidden_layer_sizes.size):
            weights.append(self._init_weight(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1], 'weight_hidden_%d' %i))
            biases.append(self._init_bias(self.hidden_layer_sizes[i], 'bias_hidden_%d' % i))

        # Weights from last hidden layer to output layer
        weights.append(self._init_weight(n_out, self.hidden_layer_sizes[-1], 'weight_out'))
        biases.append(self._init_bias(n_out, 'bias_out'))

        return weights, biases

    def _l2_loss(self, weights):
        """
        Creates the expression for L2-regularisation on the weights

        :param weights: tensorflow tensors representing the weights
        :type weights: list of tf tensors
        :return: tensorflow scalar representing the regularisation contribution to the cost function
        :rtype: tf.float32
        """

        reg_term = tf.zeros([], name="l2_loss")

        for i in range(self.hidden_layer_sizes.size):
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

        for i in range(self.hidden_layer_sizes.size):
            reg_term += tf.reduce_sum(tf.abs(weights[i]))

        return self.l1_reg * reg_term

    def _get_batch_size(self):
        """
        Determines the actual batch size. If set to auto, the batch size will be set to 100.
        If the batch size is larger than the number of samples, it is truncated and a warning
        is printed.

        Furthermore the returned batch size will be slightly modified from the user input if
        the last batch would be tiny compared to the rest.

        :return: Batch size
        :rtype: int
        """

        if self.batch_size == 'auto':
            batch_size = min(100, self.n_samples)
        else:
            if self.batch_size > self.n_samples:
                print("Warning: batch_size larger than sample size. It is going to be clipped")
                return min(self.n_samples, self.batch_size)
            else:
                batch_size = self.batch_size

        # see if the batch size can be modified slightly to make sure the last batch is similar in size
        # to the rest of the batches
        # This is always less that the requested batch size, so no memory issues should arise
        better_batch_size = ceil(self.n_samples, ceil(self.n_samples, batch_size))

        return better_batch_size

    def _set_slatm_parameters(self, params):
        """
        This function sets the parameters for the slatm representation.
        :param params: dictionary
        :return: None
        """

        self.slatm_parameters = {'slatm_sigma1': 0.05, 'slatm_sigma2': 0.05, 'slatm_dgrid1': 0.03, 'slatm_dgrid2': 0.03,
                                 'slatm_rcut': 4.8, 'slatm_rpower': 6, 'slatm_alchemy': False}

        if not is_none(params):
            for key, value in params.items():
                if key in self.slatm_parameters:
                    self.slatm_parameters[key] = value

            self._check_slatm_values()

    def _set_acsf_parameters(self, params):
        """
        This function sets the parameters for the acsf representation.
        :param params: dictionary
        :return: None
        """

        self.acsf_parameters = {'radial_cutoff': 10.0, 'angular_cutoff': 10.0, 'radial_rs': (0.0, 0.1, 0.2),
                                'angular_rs': (0.0, 0.1, 0.2), 'theta_s': (3.0, 2.0), 'zeta': 3.0, 'eta': 2.0}

        if not is_none(params):
            for key, value in params.items():
                if key in self.acsf_parameters:
                    self.acsf_parameters[key] = value

            self._check_acsf_values()

    def score(self, x, y=None, dy=None, classes=None):
        """
        This function calls the appropriate function to score the model. One needs to pass a representation and some
        properties to it or alternatively if the compounds/representations and the properties are stored in the class one
        can pass indices.

        :param x: either the representations or the indices to the representations
        :type x: either a numpy array of shape (n_samples, n_features) or (n_samples, n_atoms, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: either the gradients of the properties or none
        :type dy: either a numpy array of shape (n_samples, n_atoms, 3) or None
        :param classes: either the classes to do the NN decomposition or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None

        :return: score
        :rtype: float
        """
        return self._score(x, y, dy, classes)

    def _score(self, x, y=None, dy=None, classes=None):
        """
        This function calls the appropriate function to score the model. One needs to pass a representation and some
        properties to it or alternatively if the compounds/representations and the properties are stored in the class one
        can pass indices.

        :param x: either the representations or the indices to the representations
        :type x: either a numpy array of shape (n_samples, n_features) or (n_samples, n_atoms, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: either the gradients of the properties or none
        :type dy: either a numpy array of shape (n_samples, n_atoms, 3) or None
        :param classes: either the classes to do the NN decomposition or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None

        :return: score
        :rtype: float
        """
        if self.scoring_function == 'mae':
            return self._score_mae(x, y, dy, classes)
        if self.scoring_function == 'rmse':
            return self._score_rmse(x, y, dy, classes)
        if self.scoring_function == 'r2':
            return self._score_r2(x, y, dy, classes)

    def generate_compounds(self, filenames):
        """
        Creates QML compounds. Needs to be called before fitting.

        :param filenames: path of xyz-files
        :type filenames: list
        """

        # Check that the number of properties match the number of compounds if the properties have already been set
        if is_none(self.properties):
            pass
        else:
            if self.properties.size == len(filenames):
                pass
            else:
                raise InputError("Number of properties (%d) does not match number of compounds (%d)"
                                 % (self.properties.size, len(filenames)))

        self.compounds = np.empty(len(filenames), dtype=object)
        for i, filename in enumerate(filenames):
            self.compounds[i] = Compound(filename)

    def generate_representation(self, xyz=None, classes=None):
        """
        This function can generate representations either from the data contained in the compounds or from xyz data passed
        through the argument. If the Compounds have already being set and xyz data is given, it complains.

        :param xyz: cartesian coordinates
        :type xyz: numpy array of shape (n_samples, n_atoms, 3)
        :param classes: The classes to do the atomic decomposition of the networks (most commonly nuclear charges)
        :type classes: numpy array of shape (n_samples, n_atoms)
        :return: None
        """

        if is_none(self.compounds) and is_none(xyz) and is_none(classes):
            raise InputError("QML compounds need to be created in advance or Cartesian coordinates need to be passed in "
                             "order to generate the representation.")

        if not is_none(self.representation):
            raise InputError("The representations have already been set!")

        if is_none(self.compounds):

            self.representation, self.classes = self._generate_representations_from_data(xyz, classes)

        elif is_none(xyz):
            # Make representations from compounds

            self.representation, self.classes = self._generate_representations_from_compounds()
        else:
            raise InputError("Compounds have already been set but new xyz data is being passed.")

    def set_properties(self, properties):
        """
        Set properties. Needed to be called before fitting.

        :param y: array of properties of size (nsamples,)
        :type y: array
        """
        if is_none(properties):
            raise InputError("Properties cannot be set to none.")
        else:
            if is_numeric_array(properties) and np.asarray(properties).ndim == 1:
                self.properties = np.asarray(properties)
            else:
                raise InputError(
                    'Variable "properties" expected to be array like of dimension 1. Got %s' % str(properties))

    def set_representations(self, representations):
        """
        This function takes representations as input and stores them inside the class.

        :param representations: global or local representations
        :type representations: numpy array of shape (n_samples, n_features) or (n_samples, n_atoms, n_features)
        """

        if not is_none(self.representation):
            raise InputError("The representations have already been set!")

        if is_none(representations):
            raise InputError("Descriptor cannot be set to none.")
        else:
            if is_numeric_array(representations):
                self._set_representation(representations)
            else:
                raise InputError('Variable "representation" expected to be array like.')

    def set_gradients(self, gradients):
        """
        This function enables to set the gradient information.

        :param gradients: The gradients of the properties with respect to the input. For example, forces.
        :type gradients: numpy array (for example, numpy array of shape (n_samples, n_atoms, 3))
        :return: None
        """

        if is_none(gradients):
            raise InputError("Gradients cannot be set to none.")
        else:
            if is_numeric_array(gradients):
                self.gradients = np.asarray(gradients)
            else:
                raise InputError('Variable "gradients" expected to be array like.')

    # TODO move to ARMP? MRMP does not neet this function
    def set_classes(self, classes):
        """
        This function stores the classes to be used during training for local networks.

        :param classes: what class does each atom belong to.
        :type classes: numpy array of shape (n_samples, n_atoms) of ints
        :return: None
        """
        if is_none(classes):
            raise InputError("Classes cannot be set to none.")
        else:
            if is_positive_integer_array(classes):
                self.classes = np.asarray(classes)
            else:
                raise InputError('Variable "gradients" expected to be array like of positive integers.')

    def fit(self, x, y=None, dy=None, classes=None):
        """
        This function calls the specific fit method of the child classes.

        :param x: either the representations or the indices to the representations
        :type x: either a numpy array of shape (n_samples, n_features) or (n_samples, n_atoms, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: either the gradients of the properties or none
        :type dy: either a numpy array of shape (n_samples, n_atoms, 3) or None
        :param classes: either the classes to do the NN decomposition or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None

        :return: None
        """

        return self._fit(x, y, dy, classes)

    def _check_slatm_values(self):
        """
        This function checks that the parameters passed to slatm make sense.
        :return: None
        """
        if not is_positive(self.slatm_parameters['slatm_sigma1']):
            raise InputError("Expected positive float for variable 'slatm_sigma1'. Got %s." % str(self.slatm_parameters['slatm_sigma1']))

        if not is_positive(self.slatm_parameters['slatm_sigma2']):
            raise InputError("Expected positive float for variable 'slatm_sigma2'. Got %s." % str(self.slatm_parameters['slatm_sigma2']))

        if not is_positive(self.slatm_parameters['slatm_dgrid1']):
            raise InputError("Expected positive float for variable 'slatm_dgrid1'. Got %s." % str(self.slatm_parameters['slatm_dgrid1']))

        if not is_positive(self.slatm_parameters['slatm_dgrid2']):
            raise InputError("Expected positive float for variable 'slatm_dgrid2'. Got %s." % str(self.slatm_parameters['slatm_dgrid2']))

        if not is_positive(self.slatm_parameters['slatm_rcut']):
            raise InputError("Expected positive float for variable 'slatm_rcut'. Got %s." % str(self.slatm_parameters['slatm_rcut']))

        if not is_non_zero_integer(self.slatm_parameters['slatm_rpower']):
            raise InputError("Expected non-zero integer for variable 'slatm_rpower'. Got %s." % str(self.slatm_parameters['slatm_rpower']))

        if not is_bool(self.slatm_parameters['slatm_alchemy']):
            raise InputError("Expected boolean value for variable 'slatm_alchemy'. Got %s." % str(self.slatm_parameters['slatm_alchemy']))

    def _check_acsf_values(self):
        """
        This function checks that the user input parameters to acsf make sense.
        :return: None
        """

        if not is_positive(self.acsf_parameters['radial_cutoff']):
            raise InputError("Expected positive float for variable 'radial_cutoff'. Got %s." % str(self.acsf_parameters['radial_cutoff']))

        if not is_positive(self.acsf_parameters['angular_cutoff']):
            raise InputError("Expected positive float for variable 'angular_cutoff'. Got %s." % str(self.acsf_parameters['angular_cutoff']))

        if not is_numeric_array(self.acsf_parameters['radial_rs']):
            raise InputError("Expecting an array like radial_rs. Got %s." % (self.acsf_parameters['radial_rs']) )
        if not len(self.acsf_parameters['radial_rs'])>0:
            raise InputError("No radial_rs values were given." )

        if not is_numeric_array(self.acsf_parameters['angular_rs']):
            raise InputError("Expecting an array like angular_rs. Got %s." % (self.acsf_parameters['angular_rs']) )
        if not len(self.acsf_parameters['angular_rs'])>0:
            raise InputError("No angular_rs values were given." )

        if not is_numeric_array(self.acsf_parameters['theta_s']):
            raise InputError("Expecting an array like theta_s. Got %s." % (self.acsf_parameters['theta_s']) )
        if not len(self.acsf_parameters['theta_s'])>0:
            raise InputError("No theta_s values were given. " )

        if is_numeric_array(self.acsf_parameters['eta']):
            raise InputError("Expecting a scalar value for eta. Got %s." % (self.acsf_parameters['eta']))

        if is_numeric_array(self.acsf_parameters['zeta']):
            raise InputError("Expecting a scalar value for zeta. Got %s." % (self.acsf_parameters['zeta']))

    def _get_msize(self, pad = 0):
        """
        Gets the maximum number of atoms in a single molecule. To support larger molecules
        an optional padding can be added by the ``pad`` variable.

        :param pad: Add an integer padding to the returned dictionary
        :type pad: integer

        :return: largest molecule with respect to number of atoms.
        :rtype: integer

        """

        if self.compounds.size == 0:
            raise RuntimeError("QML compounds have not been generated")
        if not is_positive_integer_or_zero(pad):
            raise InputError("Expected variable 'pad' to be a positive integer or zero. Got %s" % str(pad))

        nmax = max(mol.natoms for mol in self.compounds)

        return nmax + pad

    def _get_asize(self, pad = 0):
        """
        Gets the maximum occurrences of each element in a single molecule. To support larger molecules
        an optional padding can be added by the ``pad`` variable.

        :param pad: Add an integer padding to the returned dictionary
        :type pad: integer

        :return: dictionary of the maximum number of occurences of each element in a single molecule.
        :rtype: dictionary

        """

        if self.compounds.size == 0:
            raise RuntimeError("QML compounds have not been generated")
        if not is_positive_integer_or_zero(pad):
            raise InputError("Expected variable 'pad' to be a positive integer or zero. Got %s" % str(pad))

        asize = {}

        for mol in self.compounds:
            for key, value in mol.natypes.items():
                if key not in asize:
                    asize[key] = value + pad
                    continue
                asize[key] = max(asize[key], value + pad)

        return asize

    def _get_slatm_mbtypes(self, arr):
        """
        This function takes an array containing all the classes that are present in a data set and returns a list of all
        the unique classes present, all the possible pairs and triplets of classes.

        :param arr: classes for each atom in a data set
        :type arr: numpy array of shape (n_samples, n_atoms)
        :return: unique single, pair and triplets of classes
        :rtype: list of lists
        """

        return qml_rep.get_slatm_mbtypes(arr)

    def _get_xyz_from_compounds(self, indices):
        """
        This function takes some indices and returns the xyz of the compounds corresponding to those indices.

        :param indices: indices of the compounds to use for training
        :type indices: numpy array of ints of shape (n_samples, )
        :return: the xyz of the specified compounds
        :rtype: numpy array of shape (n_samples, n_atoms, 3)
        """

        xyzs = []
        zs = []
        max_n_atoms = 0

        for compound in self.compounds[indices]:
            xyzs.append(compound.coordinates)
            zs.append(compound.nuclear_charges)
            if len(compound.nuclear_charges) > max_n_atoms:
                max_n_atoms = len(compound.nuclear_charges)

        # Padding so that all the samples have the same shape
        n_samples = len(zs)
        for i in range(n_samples):
            current_n_atoms = len(zs[i])
            missing_n_atoms = max_n_atoms - current_n_atoms
            xyz_padding = np.zeros((missing_n_atoms, 3))
            xyzs[i] = np.concatenate((xyzs[i], xyz_padding))

        xyzs = np.asarray(xyzs, dtype=np.float32)

        return xyzs

    def _get_properties(self, indices):
        """
        This returns the properties that have been set through QML.

        :param indices: The indices of the properties to return
        :type indices: numpy array of ints of shape (n_samples, )
        :return: the properties of the compounds specified
        :rtype: numpy array of shape (n_samples, 1)
        """

        return np.atleast_2d(self.properties[indices]).T

    def _get_classes(self, indices):
        """
        This returns the classes that have been set through QML.

        :param indices: The indices of the properties to return
        :type indices: numpy array of ints of shape (n_samples, )
        :return: classes of the compounds specified
        :rtype: numpy array of shape (n_samples, n_atoms)
        """

        zs = []
        max_n_atoms = 0

        for compound in self.compounds[indices]:
            zs.append(compound.nuclear_charges)
            if len(compound.nuclear_charges) > max_n_atoms:
                max_n_atoms = len(compound.nuclear_charges)

        # Padding so that all the samples have the same shape
        n_samples = len(zs)
        for i in range(n_samples):
            current_n_atoms = len(zs[i])
            missing_n_atoms = max_n_atoms - current_n_atoms
            zs_padding = np.zeros(missing_n_atoms)
            zs[i] = np.concatenate((zs[i], zs_padding))

        return np.asarray(zs, dtype=np.float32)

    def predict(self, x, classes=None):
        """
        This function calls the predict function for either ARMP or MRMP.

        :param x: representation or indices
        :type x: numpy array of shape (n_samples, n_features) or (n_samples, n_atoms, n_features) or an array of ints
        :param classes: the classes to use for atomic decomposition
        :type classes: numpy array of shape (n_sample, n_atoms)


        :return: predictions of the molecular properties.
        :rtype: numpy array of shape (n_samples,)
        """
        predictions = self._predict(x, classes)

        if predictions.ndim > 1 and predictions.shape[1] == 1:
            return predictions.ravel()
        else:
            return predictions

### --------------------- ** Molecular representation - molecular properties network ** --------------------------------

class MRMP(_NN):
    """
    Neural network for either
    1) predicting global properties, such as energies, using molecular representations, or
    2) predicting local properties, such as chemical shieldings, using atomic representations.
    """

    def __init__(self, hidden_layer_sizes = (5,), l1_reg = 0.0, l2_reg = 0.0001, batch_size = 'auto', learning_rate = 0.001,
                 iterations = 500, tensorboard = False, store_frequency = 200, tf_dtype = tf.float32, scoring_function = 'mae',
                 activation_function = tf.sigmoid, optimiser = tf.train.AdamOptimizer, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-08,
                 rho = 0.95, initial_accumulator_value = 0.1, initial_gradient_squared_accumulator_value = 0.1,
                 l1_regularization_strength = 0.0, l2_regularization_strength = 0.0,
                 tensorboard_subdir = os.getcwd() + '/tensorboard', representation='unsorted_coulomb_matrix', representation_params=None):
        """
        Descriptors is used as input to a single or multi layered feed-forward neural network with a single output.
        This class inherits from the _NN class and all inputs not unique to the NN class is passed to the _NN
        parent.

        """

        super(MRMP, self).__init__(hidden_layer_sizes, l1_reg, l2_reg, batch_size, learning_rate,
                 iterations, tensorboard, store_frequency, tf_dtype, scoring_function,
                 activation_function, optimiser, beta1, beta2, epsilon,
                 rho, initial_accumulator_value, initial_gradient_squared_accumulator_value,
                 l1_regularization_strength,l2_regularization_strength, tensorboard_subdir)

        self._initialise_representation(representation, representation_params)

    def _initialise_representation(self, representation, parameters):
        """
        This function sets the representation and the parameters of the representation.

        :param representation: the name of the representation
        :type representation: string
        :param parameters: all the parameters of the representation.
        :type parameters: dictionary
        :return: None
        """

        if not is_string(representation):
            raise InputError("Expected string for variable 'representation'. Got %s" % str(representation))
        if representation.lower() not in ['sorted_coulomb_matrix', 'unsorted_coulomb_matrix', 'bag_of_bonds', 'slatm']:
            raise InputError("Unknown representation %s" % representation)
        self.representation_name = representation.lower()

        if not is_none(parameters):
            if not type(parameters) is dict:
                raise InputError("The representation parameters passed should be either None or a dictionary.")

        if self.representation_name == 'slatm':

            self._set_slatm_parameters(parameters)

        else:

            if not is_none(parameters):
                raise InputError("The representation %s does not take any additional parameters." % (self.representation_name))

    def _set_representation(self, representation):
        """
        This function takes representations as input and stores them inside the class.

        :param representations: global representations
        :type representations: numpy array of shape (n_samples, n_features)
        return: None
        """

        if len(representation.shape) != 2:
            raise InputError("The representation should have a shape (n_samples, n_features). Got %s" % (str(representation.shape)))

        self.representation = representation

    def _generate_representations_from_data(self, xyz, classes):
        """
        This function makes the representation from xyz data and nuclear charges.

        :param xyz: cartesian coordinates
        :type xyz: numpy array of shape (n_samples, n_atoms, 3)
        :param classes: classes for atomic decomposition
        :type classes: None
        :return: numpy array of shape (n_samples, n_features) and None
        """
        # TODO implement
        raise InputError("Not implemented yet. Use compounds.")

    def _generate_representations_from_compounds(self):
        """
        This function generates the representations from the compounds.

        :return: the representation and None (in the ARMP class this would be the classes for atomic decomposition)
        :rtype: numpy array of shape (n_samples, n_features) and None
        """

        if is_none(self.compounds):
            raise InputError("This should never happen.")

        n_samples = len(self.compounds)

        if self.representation_name == 'unsorted_coulomb_matrix':

            nmax = self._get_msize()
            representation_size = (nmax*(nmax+1))//2
            x = np.empty((n_samples, representation_size), dtype=float)
            for i, mol in enumerate(self.compounds):
                mol.generate_coulomb_matrix(size = nmax, sorting = "unsorted")
                x[i] = mol.representation

        elif self.representation_name == 'sorted_coulomb_matrix':

            nmax = self._get_msize()
            representation_size = (nmax*(nmax+1))//2
            x = np.empty((n_samples, representation_size), dtype=float)
            for i, mol in enumerate(self.compounds):
                mol.generate_coulomb_matrix(size = nmax, sorting = "row-norm")
                x[i] = mol.representation

        elif self.representation_name == "bag_of_bonds":
            asize = self._get_asize()
            x = np.empty(n_samples, dtype=object)
            for i, mol in enumerate(self.compounds):
                mol.generate_bob(asize = asize)
                x[i] = mol.representation
            x = np.asarray(list(x), dtype=float)

        elif self.representation_name == "slatm":
            mbtypes = self._get_slatm_mbtypes([mol.nuclear_charges for mol in self.compounds])
            x = np.empty(n_samples, dtype=object)
            for i, mol in enumerate(self.compounds):
                mol.generate_slatm(mbtypes, local = False, sigmas = [self.slatm_parameters['slatm_sigma1'],
                                                                     self.slatm_parameters['slatm_sigma2']],
                        dgrids = [self.slatm_parameters['slatm_dgrid1'], self.slatm_parameters['slatm_dgrid2']],
                                   rcut = self.slatm_parameters['slatm_rcut'],
                                   alchemy = self.slatm_parameters['slatm_alchemy'],
                        rpower = self.slatm_parameters['slatm_rpower'])
                x[i] = mol.representation
            x = np.asarray(list(x), dtype=float)

        else:

            raise InputError("This should never happen. Unrecognised representation. Got %s." % str(self.representation_name))

        return x, None

    #TODO upgrade so that this uses tf.Dataset like the ARMP class
    def _fit(self, x, y, dy, classes):
        """
        This function fits a NON atomic decomposed network to the data.

        :param x: either the representations or the indices to the data points to use
        :type x: either a numpy array of shape (n_samples, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: None
        :type dy: None
        :param classes: None
        :type classes: None

        :return: None
        """

        x_approved, y_approved, dy_approved, classes_approved = self._check_inputs(x, y, dy, classes)

        # Useful quantities
        self.n_features = x_approved.shape[1]
        self.n_samples = x_approved.shape[0]

        # Set the batch size
        batch_size = self._get_batch_size()

        tf.reset_default_graph()

        # Initial set up of the NN
        with tf.name_scope("Data"):
            tf_x = tf.placeholder(self.tf_dtype, [None, self.n_features], name="Descriptors")
            tf_y = tf.placeholder(self.tf_dtype, [None, 1], name="Properties")

        # Either initialise the weights and biases or restart training from wherever it was stopped
        with tf.name_scope("Weights"):
            weights, biases = self._generate_weights(n_out = 1)

            # Log weights for tensorboard
            if self.tensorboard:
                self.tensorboard_logger_training.write_weight_histogram(weights)

        with tf.name_scope("Model"):
            y_pred = self._model(tf_x, weights, biases)

        with tf.name_scope("Cost_func"):
            cost = self._cost(y_pred, tf_y, weights)

        if self.tensorboard:
            cost_summary = tf.summary.scalar('cost', cost)

        # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        optimiser = self._set_optimiser()
        optimisation_op = optimiser.minimize(cost)
        # Initialisation of the variables
        init = tf.global_variables_initializer()

        if self.tensorboard:
            self.tensorboard_logger_training.initialise()

        # This is the total number of batches in which the training set is divided
        n_batches = ceil(self.n_samples, batch_size)

        self.session = tf.Session()

        # Running the graph
        if self.tensorboard:
            self.tensorboard_logger_training.set_summary_writer(self.session)

        self.session.run(init)

        indices = np.arange(0,self.n_samples, 1)

        for i in range(self.iterations):
            # This will be used to calculate the average cost per iteration
            avg_cost = 0
            # Learning over the batches of data
            for j in range(n_batches):
                batch_x = x_approved[indices][j * batch_size:(j + 1) * batch_size]
                batch_y = y_approved[indices][j * batch_size:(j+1) * batch_size]
                if self.AdagradDA:
                    feed_dict = {tf_x: batch_x, tf_y: batch_y, self.global_step:i}
                    opt, c = self.session.run([optimisation_op, cost], feed_dict=feed_dict)
                else:
                    feed_dict = {tf_x: batch_x, tf_y: batch_y}
                    opt, c = self.session.run([optimisation_op, cost], feed_dict=feed_dict)
                avg_cost += c * batch_x.shape[0] / x_approved.shape[0]

                if self.tensorboard:
                    if i % self.tensorboard_logger_training.store_frequency == 0:
                        self.tensorboard_logger_training.write_summary(self.session, feed_dict, i, j)

            self.training_cost.append(avg_cost)

            # Shuffle the dataset at each iteration
            np.random.shuffle(indices)

    def _model(self, x, weights, biases):
        """
        Constructs the molecular neural network.

        :param x: representation
        :type x: tf.placeholder of shape (n_samples, n_features)
        :param weights: Weights used in the network.
        :type weights: list of tf.Variables of length hidden_layer_sizes.size + 1
        :param biases: Biases used in the network.
        :type biases: list of tf.Variables of length hidden_layer_sizes.size + 1
        :return: Output
        :rtype: tf.Variable of size (n_samples, n_targets)
        """

        # Calculate the activation of the first hidden layer
        z = tf.add(tf.matmul(x, tf.transpose(weights[0])), biases[0])
        h = self.activation_function(z)

        # Calculate the activation of the remaining hidden layers
        for i in range(self.hidden_layer_sizes.size - 1):
            z = tf.add(tf.matmul(h, tf.transpose(weights[i + 1])), biases[i + 1])
            h = self.activation_function(z)

        # Calculating the output of the last layer
        z = tf.add(tf.matmul(h, tf.transpose(weights[-1])), biases[-1], name="output")

        return z

    def _score_r2(self, x, y=None, dy=None, classes=None):
        """
        Calculate the coefficient of determination (R^2).
        Larger values corresponds to a better prediction.

        :param x: either the representations or the indices to the representations
        :type x: either a numpy array of shape (n_samples, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: None
        :type dy: None
        :param classes: None
        :type classes: None

        :return: R^2
        :rtype: float
        """

        x_approved, y_approved, dy_approved, classes_approved = self._check_inputs(x, y, dy, classes)

        y_pred = self.predict(x_approved)
        r2 = r2_score(y_approved, y_pred, sample_weight = None)
        return r2

    def _score_mae(self, x, y=None, dy=None, classes=None):
        """
        Calculate the mean absolute error.
        Smaller values corresponds to a better prediction.

        :param x: either the representations or the indices to the representations
        :type x: either a numpy array of shape (n_samples, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: None
        :type dy: None
        :param classes: None
        :type classes: None

        :return: Mean absolute error
        :rtype: float

        """

        x_approved, y_approved, dy_approved, classes_approved = self._check_inputs(x, y, dy, classes)

        y_pred = self.predict(x_approved)
        mae = (-1.0)*mean_absolute_error(y_approved, y_pred, sample_weight = None)
        print("Warning! The mae is multiplied by -1 so that it can be minimised in Osprey!")
        return mae

    def _score_rmse(self, x, y=None, dy=None, classes=None):
        """
        Calculate the root mean squared error.
        Smaller values corresponds to a better prediction.

        :param x: either the representations or the indices to the representations
        :type x: either a numpy array of shape (n_samples, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: None
        :type dy: None
        :param classes: None
        :type classes: None

        :return: Mean absolute error
        :rtype: float

        """

        x_approved, y_approved, dy_approved, classes_approved = self._check_inputs(x, y, dy, classes)

        y_pred = self.predict(x_approved)
        rmse = np.sqrt(mean_squared_error(y_approved, y_pred, sample_weight = None))
        return rmse

    def _check_inputs(self, x, y, dy, classes):
        """
        This function checks whether x contains indices or data. If it contains indices, the data is extracted by the
        appropriate compound objects. Otherwise it checks what data is passed through the arguments.

        :param x: indices or data
        :type x: numpy array of ints of shape (n_samples,) or floats of shape (n_samples, n_atoms, 3)
        :param y: None or energies
        :type y: None or numpy array of floats of shape (n_samples,)
        :param dy: None or forces
        :type dy: None or floats of shape (n_samples, n_atoms, 3)
        :param classes: None or different NN classes available
        :type classes: None or numpy array of ints of shape (n_samples, n_atoms)

        :return: the approved x, y dy and classes
        :rtype: numpy array of shape (n_samples, n_features), (n_samples, 1), None, None
        """

        if not is_array_like(x):
            raise InputError("x should be an array either containing indices or data.")

        if not is_none(dy) and not is_none(classes):
            raise InputError("MRMP estimator cannot predict gradients and do atomic decomposition.")

        # Check if x is made up of indices or data
        if is_positive_integer_or_zero_array(x):

            if is_none(self.representation):
                if is_none(self.compounds):
                    raise InputError("No representations or QML compounds have been set yet.")
                else:
                    self.representation, _ = self._generate_representations_from_compounds()
            if is_none(self.properties):
                raise InputError("The properties need to be set in advance.")

            approved_x = self.representation[x]
            approved_y = self._get_properties(x)
            approved_dy = None
            approved_classes = None

            check_sizes(approved_x, approved_y, approved_dy, approved_classes)

        else:

            if is_none(y):
                raise InputError("y cannot be of None type.")

            approved_x = check_global_representation(x)
            approved_y = check_y(y)
            approved_dy = None
            approved_classes = None

            check_sizes(approved_x, approved_y, approved_dy, approved_classes)

        return approved_x, approved_y, approved_dy, approved_classes

    def _check_predict_input(self, x, classes):
        """
        This function checks whether x contains indices or data. If it contains indices, the data is extracted by the
        appropriate compound objects. Otherwise it checks what data is passed through the arguments.

        :param x: indices or data
        :type x: numpy array of ints of shape (n_samples,) or floats of shape (n_samples, n_features)
        :param classes: None
        :type classes: None

        :return: the approved x and classes
        :rtype: numpy array of shape (n_samples, n_features), None
        """

        if not is_array_like(x):
            raise InputError("x should be an array either containing indices or data.")

        if not is_none(classes):
            raise InputError("MRMP estimator cannot do atomic decomposition.")

        # Check if x is made up of indices or data
        if is_positive_integer_or_zero_array(x):

            if is_none(self.representation):
                if is_none(self.compounds):
                    raise InputError("No representations or QML compounds have been set yet.")
                else:
                    self.representation, _ = self._generate_representations_from_compounds()
            if is_none(self.properties):
                raise InputError("The properties need to be set in advance.")

            approved_x = self.representation[x]
            approved_classes = None

        else:

            approved_x = check_global_representation(x)
            approved_classes = None

        return approved_x, approved_classes

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

    def _predict(self, x, classes):
        """
        This function checks whether x contains indices or data. If it contains indices, the data is extracted by the
        appropriate compound objects. Otherwise it checks what data is passed through the arguments. Then, the data is
        used as input to the trained network to predict some properties.

        :param x: indices or data
        :type x: numpy array of ints of shape (n_samples,) or floats of shape (n_samples, n_features)
        :param classes: None
        :type classes: None

        :return: the predicted properties
        :rtype: numpy array of shape (n_samples,)
        """

        approved_x, approved_classes = self._check_predict_input(x, classes)

        if self.session == None:
            raise InputError("Model needs to be fit before predictions can be made.")

        graph = tf.get_default_graph()

        with graph.as_default():
            tf_x = graph.get_tensor_by_name("Data/Descriptors:0")
            model = graph.get_tensor_by_name("Model/output:0")
            y_pred = self.session.run(model, feed_dict = {tf_x : approved_x})
            return y_pred

    # TODO these need to be checked if they still work
    def save_nn(self, save_dir="./saved_model"):
        """
        This function saves the model to be used for later prediction.

        :param save_dir: name of the directory to create to save the model
        :type save_dir: string
        :return: None
        """
        if self.session == None:
            raise InputError("Model needs to be fit before predictions can be made.")

        if not os.path.exists(save_dir):
            pass
        else:
            ii = 1
            while True:
                new_save_dir = save_dir + "_" + str(ii)
                if not os.path.exists(new_save_dir):
                    save_dir = new_save_dir
                    break

        graph = tf.get_default_graph()

        with graph.as_default():
            tf_x = graph.get_tensor_by_name("Data/Descriptors:0")
            model = graph.get_tensor_by_name("Model/output:0")

        tf.saved_model.simple_save(self.session, export_dir=save_dir,
                                   inputs={"Data/Descriptors:0": tf_x},
                                   outputs={"Model/output:0": model})

    def load_nn(self, save_dir="saved_model"):
        """
        This function reloads a model for predictions.
        :param save_dir: the name of the directory where the model is saved.
        :type save_dir: string
        :return: None
        """

        self.session = tf.Session(graph=tf.get_default_graph())
        tf.saved_model.loader.load(self.session, [tf.saved_model.tag_constants.SERVING], save_dir)

### --------------------- ** Atomic representation - molecular properties network ** -----------------------------------

class ARMP(_NN):
    """
    The ``ARMP`` class is used to build neural networks that take as an input atomic representations of molecules and
    output molecular properties such as the energies.
    """

    def __init__(self, hidden_layer_sizes = (5,), l1_reg = 0.0, l2_reg = 0.0001, batch_size = 'auto', learning_rate = 0.001,
                 iterations = 500, tensorboard = False, store_frequency = 200, tf_dtype = tf.float32, scoring_function = 'mae',
                 activation_function = tf.sigmoid, optimiser = tf.train.AdamOptimizer, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-08,
                 rho = 0.95, initial_accumulator_value = 0.1, initial_gradient_squared_accumulator_value = 0.1,
                 l1_regularization_strength = 0.0, l2_regularization_strength = 0.0,
                 tensorboard_subdir = os.getcwd() + '/tensorboard', representation='acsf', representation_params=None):
        """
        To see what parameters are required, look at the description of the _NN class init.
        This class inherits from the _NN class and all inputs not unique to the ARMP class are passed to the _NN
        parent.
        """

        super(ARMP, self).__init__(hidden_layer_sizes, l1_reg, l2_reg, batch_size, learning_rate,
                 iterations, tensorboard, store_frequency, tf_dtype, scoring_function,
                 activation_function, optimiser, beta1, beta2, epsilon,
                 rho, initial_accumulator_value, initial_gradient_squared_accumulator_value,
                 l1_regularization_strength,l2_regularization_strength, tensorboard_subdir)

        self._initialise_representation(representation, representation_params)

    def _initialise_representation(self, representation, parameters):
        """
        This function sets the representation and the parameters of the representation.

        :param representation: the name of the representation
        :type representation: string
        :param parameters: all the parameters of the representation.
        :type parameters: dictionary
        :return: None
        """

        if not is_string(representation):
            raise InputError("Expected string for variable 'representation'. Got %s" % str(representation))
        if representation.lower() not in ['slatm', 'acsf']:
            raise InputError("Unknown representation %s" % representation)
        self.representation_name = representation.lower()

        if not is_none(parameters):
            if not type(parameters) is dict:
                raise InputError("The representation parameters passed should be either None or a dictionary.")
            self._check_representation_parameters(parameters)

        if self.representation_name == 'slatm':

            self._set_slatm_parameters(parameters)

        elif self.representation_name == 'acsf':

            self._set_acsf_parameters(parameters)

        else:

            if not is_none(parameters):
                raise InputError("The representation %s does not take any additional parameters." % (self.representation_name))

    def _set_representation(self, representation):

        if len(representation.shape) != 3:
            raise InputError(
                "The representation should have a shape (n_samples, n_atoms, n_features). Got %s" % (str(representation.shape)))

        self.representation = representation

    def _generate_representations_from_data(self, xyz, classes):
        """
        This function generates the representations from xyz data

        :param xyz: the cartesian coordinates
        :type xyz: numpy array of shape (n_samples, n_atoms, 3)
        :param classes: classes to use for atomic decomposition
        :type classes: numpy array of shape (n_samples, n_atoms)
        :return: representations and classes
        :rtype: numpy arrays of shape (n_samples, n_atoms, n_features) and (n_samples, n_atoms)
        """

        if is_none(classes):
            raise InputError("The classes need to be provided for the ARMP estimator.")
        else:
            if len(classes.shape) > 2 or np.all(xyz.shape[:2] != classes.shape):
                raise InputError("Classes should be a 2D array with shape matching the first 2 dimensions of the xyz data"
                                 ". Got shape %s" % (str(classes.shape)))

        representation = None

        if self.representation_name == 'slatm':
            # TODO implement
            raise InputError("Slatm from data has not been implemented yet. Use Compounds.")

        elif self.representation_name == 'acsf':

            representation = self._generate_acsf_from_data(xyz, classes)

        return representation, classes

    # TODO modify so that it uses the new map function
    def _generate_acsf_from_data(self, xyz, classes):
        """
        This function generates the acsf from the cartesian coordinates and the classes.

        :param xyz: cartesian coordinates
        :type xyz: numpy array of shape (n_samples, n_atoms, 3)
        :param classes: the classes to use for atomic decomposition
        :type classes: numpy array of shape (n_samples, n_atoms)
        :return: representation acsf
        :rtype: numpy array of shape (n_samples, n_atoms, n_features)
        """
        mbtypes = qml_rep.get_slatm_mbtypes([classes[i] for i in range(classes.shape[0])])

        elements = []
        element_pairs = []

        # Splitting the one and two body interactions in mbtypes
        for item in mbtypes:
            if len(item) == 1:
                elements.append(item[0])
            if len(item) == 2:
                element_pairs.append(list(item))
            if len(item) == 3:
                break

        # Need the element pairs in descending order for TF
        for item in element_pairs:
            item.reverse()

        if self.tensorboard:
            run_metadata = tf.RunMetadata()
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

        n_samples = xyz.shape[0]
        max_n_atoms = xyz.shape[1]

        # Turning the quantities into tensors
        with tf.name_scope("Inputs"):
            zs_tf = tf.placeholder(shape=[n_samples, max_n_atoms], dtype=tf.int32, name="zs")
            xyz_tf = tf.placeholder(shape=[n_samples, max_n_atoms, 3], dtype=tf.float32, name="xyz")

            dataset = tf.data.Dataset.from_tensor_slices((xyz_tf, zs_tf))
            dataset = dataset.batch(20)
            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
            batch_xyz, batch_zs = iterator.get_next()

        representation = generate_parkhill_acsf(xyzs=batch_xyz, Zs=batch_zs, elements=elements, element_pairs=element_pairs,
                                            radial_cutoff=self.acsf_parameters['radial_cutoff'],
                                            angular_cutoff=self.acsf_parameters['angular_cutoff'],
                                            radial_rs=self.acsf_parameters['radial_rs'],
                                            angular_rs=self.acsf_parameters['angular_rs'],
                                            theta_s=self.acsf_parameters['theta_s'], eta=self.acsf_parameters['eta'],
                                            zeta=self.acsf_parameters['zeta'])

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.make_initializer(dataset), feed_dict={xyz_tf: xyz, zs_tf: classes})

        representation_slices = []

        if self.tensorboard:
            summary_writer = tf.summary.FileWriter(logdir="tensorboard", graph=sess.graph)

            batch_counter = 0
            while True:
                try:
                    representation_np = sess.run(representation, options=options, run_metadata=run_metadata)
                    summary_writer.add_run_metadata(run_metadata=run_metadata, tag="batch %s" % batch_counter,
                                                    global_step=None)
                    representation_slices.append(representation_np)
                    batch_counter += 1
                except tf.errors.OutOfRangeError:
                    print("Generated all the representations.")
                    break
        else:
            batch_counter = 0
            while True:
                try:
                    representation_np = sess.run(representation)
                    representation_slices.append(representation_np)
                    batch_counter += 1
                except tf.errors.OutOfRangeError:
                    print("Generated all the representations.")
                    break

        representation_conc = np.concatenate(representation_slices, axis=0)
        print("The representation has shape %s." % (str(representation_conc.shape)))

        sess.close()

        return representation_conc

    def _generate_representations_from_compounds(self):
        """
        This function generates the representations from the compounds.
        :return: the representations and the classes
        :rtype: numpy array of shape (n_samples, n_atoms, n_features) and (n_samples, n_atoms)
        """

        if is_none(self.compounds):
            raise InputError("QML compounds needs to be created in advance")

        if self.representation_name == 'slatm':

            representations, classes = self._generate_slatm_from_compounds()

        elif self.representation_name == 'acsf':

            representations, classes = self._generate_acsf_from_compounds()

        else:
            raise InputError("This should never happen, unrecognised representation %s." % (self.representation_name))

        return representations, classes

    def _generate_acsf_from_compounds(self):
        """
        This function generates the atom centred symmetry functions.

        :return: representation acsf and classes
        :rtype: numpy array of shape (n_samples, n_atoms, n_features) and (n_samples, n_atoms)
        """

        # Obtaining the total elements and the element pairs
        mbtypes = qml_rep.get_slatm_mbtypes([mol.nuclear_charges for mol in self.compounds])

        elements = []
        element_pairs = []

        # Splitting the one and two body interactions in mbtypes
        for item in mbtypes:
            if len(item) == 1:
                elements.append(item[0])
            if len(item) == 2:
                element_pairs.append(list(item))
            if len(item) == 3:
                break

        # Need the element pairs in descending order for TF
        for item in element_pairs:
            item.reverse()

        # Obtaining the xyz and the nuclear charges
        xyzs = []
        zs = []
        max_n_atoms = 0

        for compound in self.compounds:
            xyzs.append(compound.coordinates)
            zs.append(compound.nuclear_charges)
            if len(compound.nuclear_charges) > max_n_atoms:
                max_n_atoms = len(compound.nuclear_charges)

        # Padding so that all the samples have the same shape
        n_samples = len(zs)
        for i in range(n_samples):
            current_n_atoms = len(zs[i])
            missing_n_atoms = max_n_atoms - current_n_atoms
            zs_padding = np.zeros(missing_n_atoms)
            zs[i] = np.concatenate((zs[i], zs_padding))
            xyz_padding = np.zeros((missing_n_atoms, 3))
            xyzs[i] = np.concatenate((xyzs[i], xyz_padding))

        zs = np.asarray(zs, dtype=np.int32)
        xyzs = np.asarray(xyzs, dtype=np.float32)

        if self.tensorboard:
            self.tensorboard_logger_representation.initialise()
            # run_metadata = tf.RunMetadata()
            # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

        # Turning the quantities into tensors
        with tf.name_scope("Inputs"):
            zs_tf = tf.placeholder(shape=[n_samples, max_n_atoms], dtype=tf.int32, name="zs")
            xyz_tf = tf.placeholder(shape=[n_samples, max_n_atoms, 3], dtype=tf.float32, name="xyz")

            dataset = tf.data.Dataset.from_tensor_slices((xyz_tf, zs_tf))
            dataset = dataset.batch(20)
            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
            batch_xyz, batch_zs = iterator.get_next()

        representations = generate_parkhill_acsf(xyzs=batch_xyz, Zs=batch_zs, elements=elements, element_pairs=element_pairs,
                                            radial_cutoff=self.acsf_parameters['radial_cutoff'],
                                            angular_cutoff=self.acsf_parameters['angular_cutoff'],
                                            radial_rs=self.acsf_parameters['radial_rs'],
                                            angular_rs=self.acsf_parameters['angular_rs'],
                                            theta_s=self.acsf_parameters['theta_s'], eta=self.acsf_parameters['eta'],
                                            zeta=self.acsf_parameters['zeta'])

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.make_initializer(dataset), feed_dict={xyz_tf: xyzs, zs_tf: zs})

        representations_slices = []

        if self.tensorboard:
            self.tensorboard_logger_representation.set_summary_writer(sess)

            batch_counter = 0
            while True:
                try:
                    representations_np = sess.run(representations, options=self.tensorboard_logger_representation.options,
                                             run_metadata=self.tensorboard_logger_representation.run_metadata)
                    self.tensorboard_logger_representation.write_metadata(batch_counter)

                    representations_slices.append(representations_np)
                    batch_counter += 1
                except tf.errors.OutOfRangeError:
                    break
        else:
            batch_counter = 0
            while True:
                try:
                    representations_np = sess.run(representations)
                    representations_slices.append(representations_np)
                    batch_counter += 1
                except tf.errors.OutOfRangeError:
                    break

        representation_conc = np.concatenate(representations_slices, axis=0)

        sess.close()

        return representation_conc, zs

    def _generate_slatm_from_compounds(self):
        """
        This function generates the slatm using the data in the compounds.

        :return: representation slatm and the classes
        :rtype: numpy array of shape (n_samples, n_atoms, n_features) and (n_samples, n_atoms)
        """
        mbtypes = qml_rep.get_slatm_mbtypes([mol.nuclear_charges for mol in self.compounds])
        list_representations = []
        max_n_atoms = 0

        # Generating the representation in the shape that ARMP requires it
        for compound in self.compounds:
            compound.generate_slatm(mbtypes, local=True, sigmas=[self.slatm_parameters['slatm_sigma1'],
                                                                  self.slatm_parameters['slatm_sigma2']],
                                    dgrids=[self.slatm_parameters['slatm_dgrid1'],
                                            self.slatm_parameters['slatm_dgrid2']],
                                    rcut=self.slatm_parameters['slatm_rcut'],
                                    alchemy=self.slatm_parameters['slatm_alchemy'],
                                    rpower=self.slatm_parameters['slatm_rpower'])
            representation = compound.representation
            if max_n_atoms < representation.shape[0]:
                max_n_atoms = representation.shape[0]
            list_representations.append(representation)

        # Padding the representations of the molecules that have fewer atoms
        n_samples = len(list_representations)
        n_features = list_representations[0].shape[1]
        padded_representations = np.zeros((n_samples, max_n_atoms, n_features))
        for i, item in enumerate(list_representations):
            padded_representations[i, :item.shape[0], :] = item

        # Generating zs in the shape that ARMP requires it
        zs = np.zeros((n_samples, max_n_atoms))
        for i, mol in enumerate(self.compounds):
            zs[i, :mol.nuclear_charges.shape[0]] = mol.nuclear_charges

        return padded_representations, zs

    def _atomic_model(self, x, hidden_layer_sizes, weights, biases):
        """
        Constructs the atomic part of the network. It calculates the output for all atoms as if they all were the same
        element.

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
        h = self.activation_function(z)

        # Calculate the activation of the remaining hidden layers
        for i in range(hidden_layer_sizes.size - 1):
            z = tf.add(tf.tensordot(h, tf.transpose(weights[i + 1]), axes=1), biases[i + 1])
            h = self.activation_function(z)

        # Calculating the output of the last layer
        z = tf.add(tf.tensordot(h, tf.transpose(weights[-1]), axes=1), biases[-1])

        z_squeezed = tf.squeeze(z, axis=[-1])

        return z_squeezed

    def _model(self, x, zs, element_weights, element_biases):
        """
        This generates the molecular model by combining all the outputs from the atomic networks.

        :param x: Atomic representation
        :type x: tf tensor of shape (n_samples, n_atoms, n_features)
        :param zs: Nuclear charges of the systems
        :type zs: tf tensor of shape (n_samples, n_atoms)
        :param element_weights: Element specific weights
        :type element_weights: list of tf.Variables of length hidden_layer_sizes.size + 1
        :param element_biases: Element specific biases
        :type element_biases: list of tf.Variables of length hidden_layer_sizes.size + 1
        :return: Predicted properties for all samples
        :rtype: tf tensor of shape (n_samples, 1)
        """

        atomic_energies = tf.zeros_like(zs)

        for i in range(self.elements.shape[0]):

            # Calculating the output for every atom in all data as if they were all of the same element
            atomic_energies_all = self._atomic_model(x, self.hidden_layer_sizes, element_weights[self.elements[i]],
                                                 element_biases[self.elements[i]])  # (n_samples, n_atoms)

            # Figuring out which atomic energies correspond to the current element.
            current_element = tf.expand_dims(tf.constant(self.elements[i], dtype=tf.int32), axis=0)
            where_element = tf.equal(tf.cast(zs, dtype=tf.int32), current_element)  # (n_samples, n_atoms)

            # Extracting the energies corresponding to the right element
            element_energies = tf.where(where_element, atomic_energies_all, tf.zeros_like(zs))

            # Adding the energies of the current element to the final atomic energies tensor
            atomic_energies = tf.add(atomic_energies, element_energies)

        # Summing the energies of all the atoms
        total_energies = tf.reduce_sum(atomic_energies, axis=-1, name="output", keepdims=True)

        return total_energies

    def _cost(self, y_pred, y, weights_dict):
        """
        This function calculates the cost function during the training of the neural network.

        :param y_pred: the neural network predictions
        :type y_pred: tensors of shape (n_samples, 1)
        :param y: the truth values
        :type y: tensors of shape (n_samples, 1)
        :param weights_dict: the dictionary containing all of the weights
        :type weights_dict: dictionary where the key is a nuclear charge and the value is a list of tensors of length hidden_layer_sizes.size + 1
        :return: value of the cost function
        :rtype: tf.Variable of size (1,)
        """

        err =  tf.square(tf.subtract(y, y_pred))
        cost_function = tf.reduce_mean(err, name="loss")

        if self.l2_reg > 0:
            l2_loss = 0
            for element in weights_dict:
                l2_loss += self._l2_loss(weights_dict[element])
            cost_function += l2_loss
        if self.l1_reg > 0:
            l1_loss = 0
            for element in weights_dict:
                l1_loss += self._l1_loss(weights_dict[element])
            cost_function += l1_loss

        return cost_function

    def _check_inputs(self, x, y, dy, classes):
        """
        This function checks that all the needed input data is available.

        :param x: either the representations or the indices to the data points to use
        :type x: either a numpy array of shape (n_samples, n_atoms, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: None
        :type dy: None
        :param classes: classes to use for the atomic decomposition
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None

        :return: the approved x, y dy and classes
        :rtype: numpy array of shape (n_samples, n_atoms, n_features), (n_samples, 1), None, (n_samples, n_atoms)
        """

        if not is_array_like(x):
            raise InputError("x should be an array either containing indices or data.")

        if not is_none(dy):
            raise InputError("ARMP estimator cannot be used to predict gradients. Use ARMP_G estimator.")

        # Check if x is made up of indices or data
        if is_positive_integer_or_zero_array(x):

            if is_none(self.representation):

                if is_none(self.compounds):
                    raise InputError("No representations or QML compounds have been set yet.")
                else:
                    self.representation, self.classes = self._generate_representations_from_compounds()

            if is_none(self.properties):
                raise InputError("The properties need to be set in advance.")
            if is_none(self.classes):
                raise InputError("The classes need to be set in advance.")

            approved_x = self.representation[x]
            approved_y = self._get_properties(x)
            approved_dy = None
            approved_classes = self.classes[x]

            check_sizes(approved_x, approved_y, approved_dy, approved_classes)

        else:

            if is_none(y):
                raise InputError("y cannot be of None type.")
            if is_none(classes):
                raise InputError("ARMP estimator needs the classes to do atomic decomposition.")

            approved_x = check_local_representation(x)
            approved_y = check_y(y)
            approved_dy = None
            approved_classes = check_classes(classes)

            check_sizes(approved_x, approved_y, approved_dy, approved_classes)

        return approved_x, approved_y, approved_dy, approved_classes

    def _check_predict_input(self, x, classes):
        """
        This function checks whether x contains indices or data. If it contains indices, the data is extracted by the
        appropriate compound objects. Otherwise it checks what data is passed through the arguments.

        :param x: indices or data
        :type x: numpy array of ints of shape (n_samples,) or floats of shape (n_samples, n_atoms, n_features)
        :param classes: classes to use for the atomic decomposition or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None

        :return: the approved representation and classes
        :rtype: numpy array of shape (n_samples, n_atoms, n_features), (n_samples, n_atoms)
        """

        if not is_array_like(x):
            raise InputError("x should be an array either containing indices or data.")

        # Check if x is made up of indices or data
        if is_positive_integer_or_zero_array(x):

            if is_none(self.representation):
                if is_none(self.compounds):
                    raise InputError("No representations or QML compounds have been set yet.")
                else:
                    self.representation, self.classes = self._generate_representations_from_compounds()
            if is_none(self.properties):
                raise InputError("The properties need to be set in advance.")

            approved_x = self.representation[x]
            approved_classes = self.classes[x]

            check_sizes(x=approved_x, classes=approved_classes)

        else:

            if is_none(classes):
                raise InputError("ARMP estimator needs the classes to do atomic decomposition.")

            approved_x = check_local_representation(x)
            approved_classes = check_classes(classes)

            check_sizes(x=approved_x, classes=approved_classes)

        return approved_x, approved_classes

    def _check_representation_parameters(self, parameters):
        """
        This function checks that the dictionary passed that contains parameters of the representation contains the right
        parameters.

        :param parameters: all the parameters of the representation.
        :type parameters: dictionary
        :return: None
        """

        if self.representation_name == "slatm":

            slatm_parameters = {'slatm_sigma1': 0.05, 'slatm_sigma2': 0.05, 'slatm_dgrid1': 0.03, 'slatm_dgrid2': 0.03,
                                'slatm_rcut': 4.8, 'slatm_rpower': 6, 'slatm_alchemy': False}

            for key, value in parameters.items():
                try:
                    slatm_parameters[key]
                except Exception:
                    raise InputError("Unrecognised parameter for slatm representation: %s" % (key))

        elif self.representation_name == "acsf":

            acsf_parameters = {'radial_cutoff': 10.0, 'angular_cutoff': 10.0, 'radial_rs': (0.0, 0.1, 0.2),
                                    'angular_rs': (0.0, 0.1, 0.2), 'theta_s': (3.0, 2.0), 'zeta': 3.0, 'eta': 2.0}

            for key, value in parameters.items():
                try:
                    acsf_parameters[key]
                except Exception:
                    raise InputError("Unrecognised parameter for acsf representation: %s" % (key))

    def _find_elements(self, zs):
        """
        This function finds the unique atomic numbers in Zs and returns them in a list.

        :param zs: nuclear charges
        :type zs: numpy array of floats of shape (n_samples, n_atoms)
        :return: unique nuclear charges
        :rtype: numpy array of floats of shape (n_elements,)
        """

        # Obtaining the unique atomic numbers (but still includes the dummy atoms)
        elements = np.unique(zs)

        # Removing the dummy
        return np.trim_zeros(elements)

    def _fit(self, x, y, dy, classes):
        """
        This function fits an atomic decomposed network to the data.

        :param x: either the representations or the indices to the data points to use
        :type x: either a numpy array of shape (n_samples, n_atoms, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: None
        :type dy: None
        :param classes: classes to use for the atomic decomposition or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None

        :return: None
        """

        x_approved, y_approved, dy_approved, classes_approved = self._check_inputs(x, y, dy, classes)

        # Obtaining the array of unique elements in all samples
        self.elements = self._find_elements(classes_approved)

        if self.tensorboard:
            self.tensorboard_logger_training.initialise()

        # Useful quantities
        self.n_samples = x_approved.shape[0]
        self.n_atoms = x_approved.shape[1]
        self.n_features = x_approved.shape[2]

        # Set the batch size
        batch_size = self._get_batch_size()

        # This is the total number of batches in which the training set is divided
        n_batches = ceil(self.n_samples, batch_size)

        tf.reset_default_graph()

        # Initial set up of the NN
        with tf.name_scope("Data"):
            x_ph = tf.placeholder(dtype=self.tf_dtype, shape=[None, self.n_atoms, self.n_features])
            zs_ph = tf.placeholder(dtype=self.tf_dtype, shape=[None, self.n_atoms])
            y_ph = tf.placeholder(dtype=self.tf_dtype, shape=[None, 1])

            dataset = tf.data.Dataset.from_tensor_slices((x_ph, zs_ph, y_ph))
            dataset = dataset.shuffle(buffer_size=self.n_samples)
            dataset = dataset.batch(batch_size)
            # batched_dataset = dataset.prefetch(buffer_size=batch_size)

            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
            tf_x, tf_zs, tf_y = iterator.get_next()
            tf_x = tf.identity(tf_x, name="Descriptors")
            tf_zs = tf.identity(tf_zs, name="Atomic-numbers")
            tf_y = tf.identity(tf_y, name="Properties")

        # Creating dictionaries of the weights and biases for each element
        element_weights = {}
        element_biases = {}

        with tf.name_scope("Weights"):
            for i in range(self.elements.shape[0]):
                weights, biases = self._generate_weights(n_out=1)
                element_weights[self.elements[i]] = weights
                element_biases[self.elements[i]] = biases

                # Log weights for tensorboard
                if self.tensorboard:
                    self.tensorboard_logger_training.write_weight_histogram(weights)

        with tf.name_scope("Model"):
            molecular_energies = self._model(tf_x, tf_zs, element_weights, element_biases)

        with tf.name_scope("Cost_func"):
            cost = self._cost(molecular_energies, tf_y, element_weights)

        if self.tensorboard:
            cost_summary = self.tensorboard_logger_training.write_cost_summary(cost)

        optimiser = self._set_optimiser()
        optimisation_op = optimiser.minimize(cost)

        # Initialisation of the variables
        init = tf.global_variables_initializer()
        iterator_init = iterator.make_initializer(dataset)

        self.session = tf.Session()

        # Running the graph
        if self.tensorboard:
            self.tensorboard_logger_training.set_summary_writer(self.session)

        self.session.run(init)
        self.session.run(iterator_init, feed_dict={x_ph:x_approved, zs_ph:classes_approved, y_ph:y_approved})

        for i in range(self.iterations):

            self.session.run(iterator_init, feed_dict={x_ph: x_approved, zs_ph: classes_approved, y_ph: y_approved})
            avg_cost = 0

            for j in range(n_batches):
                if self.tensorboard:
                    opt, c = self.session.run([optimisation_op, cost], options=self.tensorboard_logger_training.options,
                             run_metadata=self.tensorboard_logger_training.run_metadata)
                else:
                    opt, c = self.session.run([optimisation_op, cost])

                avg_cost += c

            # This seems to run the iterator.get_next() op, which gives problems with end of sequence
            # Hence why I re-initialise the iterator
            self.session.run(iterator_init, feed_dict={x_ph: x_approved, zs_ph: classes_approved, y_ph: y_approved})
            if self.tensorboard:
                if i % self.tensorboard_logger_training.store_frequency == 0:
                    self.tensorboard_logger_training.write_summary(self.session, i)

            self.training_cost.append(avg_cost/n_batches)

    def _predict(self, x, classes):
        """
        This function checks whether x contains indices or data. If it contains indices, the data is extracted by the
        appropriate compound objects. Otherwise it checks what data is passed through the arguments. Then, the data is
        used as input to the trained network to predict some properties.

        :param x: indices or data
        :type x: numpy array of ints of shape (n_samples,) or of floats of shape (n_samples, n_atoms, n_features)
        :param classes: classes to use for the atomic decomposition or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None

        :return: the predicted properties
        :rtype: numpy array of shape (n_samples,)
        """

        approved_x, approved_classes = self._check_predict_input(x, classes)

        if self.session == None:
            raise InputError("Model needs to be fit before predictions can be made.")

        graph = tf.get_default_graph()

        with graph.as_default():
            tf_x = graph.get_tensor_by_name("Data/Descriptors:0")
            tf_zs = graph.get_tensor_by_name("Data/Atomic-numbers:0")
            model = graph.get_tensor_by_name("Model/output:0")
            y_pred = self.session.run(model, feed_dict={tf_x: approved_x, tf_zs:approved_classes})

        return y_pred

    def _score_r2(self, x, y=None, dy=None, classes=None):
        """
        Calculate the coefficient of determination (R^2).
        Larger values corresponds to a better prediction.

        :param x: either the representations or the indices to the representations
        :type x: either a numpy array of shape (n_samples, n_atoms, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: None
        :type dy: None
        :param classes: either the classes or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None

        :return: R^2
        :rtype: float
        """

        x_approved, y_approved, dy_approved, classes_approved = self._check_inputs(x, y, dy, classes)

        y_pred = self.predict(x_approved, classes_approved)
        r2 = r2_score(y_approved, y_pred, sample_weight = None)
        return r2

    def _score_mae(self, x, y=None, dy=None, classes=None):
        """
        Calculate the mean absolute error.
        Smaller values corresponds to a better prediction.

        :param x: either the representations or the indices to the representations
        :type x: either a numpy array of shape (n_samples, n_atoms, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: None
        :type dy: None
        :param classes: either the classes or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None

        :param sample_weight: Weights of the samples. None indicates that that each sample has the same weight.
        :type sample_weight: array of shape (n_samples,)

        :return: Mean absolute error
        :rtype: float

        """

        x_approved, y_approved, dy_approved, classes_approved = self._check_inputs(x, y, dy, classes)

        y_pred = self.predict(x_approved, classes_approved)
        mae = (-1.0) * mean_absolute_error(y_approved, y_pred, sample_weight=None)
        print("Warning! The mae is multiplied by -1 so that it can be minimised in Osprey!")
        return mae

    def _score_rmse(self, x, y=None, dy=None, classes=None):
        """
        Calculate the root mean squared error.
        Smaller values corresponds to a better prediction.

        :param x: either the representations or the indices to the representations
        :type x: either a numpy array of shape (n_samples, n_atoms, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: None
        :type dy: None
        :param classes: either the classes or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None

        :return: Mean absolute error
        :rtype: float

        """

        x_approved, y_approved, dy_approved, classes_approved = self._check_inputs(x, y, dy, classes)

        y_pred = self.predict(x_approved, classes_approved)
        rmse = np.sqrt(mean_squared_error(y_approved, y_pred, sample_weight = None))
        return rmse

    def save_nn(self, save_dir="saved_model"):
        """
        This function saves the trained model to be used for later prediction.

        :param save_dir: directory in which to save the model
        :type save_dir: string
        :return: None
        """

        if self.session == None:
            raise InputError("Model needs to be fit before predictions can be made.")

        graph = tf.get_default_graph()

        with graph.as_default():
            tf_x = graph.get_tensor_by_name("Data/Descriptors:0")
            tf_zs = graph.get_tensor_by_name("Data/Atomic-numbers:0")
            model = graph.get_tensor_by_name("Model/output:0")

        tf.saved_model.simple_save(self.session, export_dir=save_dir,
                                   inputs={"Data/Descriptors:0": tf_x, "Data/Atomic-numbers:0": tf_zs},
                                   outputs={"Model/output:0": model})

    def load_nn(self, save_dir="saved_model"):
        """
        This function reloads a model for predictions.

        :param save_dir: the name of the directory where the model is saved.
        :type save_dir: string
        :return: None
        """

        self.session = tf.Session(graph=tf.get_default_graph())
        tf.saved_model.loader.load(self.session, [tf.saved_model.tag_constants.SERVING], save_dir)
