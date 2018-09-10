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
Tests directly related to the class _NN and it's children.

"""
import tensorflow as tf
import numpy as np

# TODO relative imports
from qml.aglaia.aglaia import MRMP
from qml.utils import InputError


# ------------ ** All functions to test the inputs to the classes ** ---------------

def hidden_layer_sizes(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(hidden_layer_sizes = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(hidden_layer_sizes = [4,5])
    C(hidden_layer_sizes = (4,5))
    C(hidden_layer_sizes = [4.0])

    # This should be caught
    catch([])
    catch([0,4])
    catch([4.2])
    catch(["x"])
    catch([None])
    catch(None)
    catch(4)
    catch([0])

def l1_reg(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(l1_reg = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(l1_reg = 0.1)
    C(l1_reg = 0.0)

    # This should be caught
    catch(-0.1)
    catch("x")
    catch(None)
    catch([0])

def l2_reg(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(l2_reg = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(l2_reg = 0.1)
    C(l2_reg = 0.0)

    # This should be caught
    catch(-0.1)
    catch("x")
    catch(None)
    catch([0])

def batch_size(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(batch_size = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(batch_size = 2)
    C(batch_size = 2.0)
    C(batch_size = "auto")

    # This should be caught
    catch(1)
    catch(-2)
    catch("x")
    catch(4.2)
    catch(None)

def learning_rate(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(learning_rate = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(learning_rate = 0.1)

    # This should be caught
    catch(0.0)
    catch(-0.1)
    catch("x")
    catch(None)

def iterations(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(iterations = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(iterations = 1)
    C(iterations = 1.0)

    # This should be caught
    catch(-2)
    catch("x")
    catch(4.2)
    catch(None)

def tf_dtype(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(tf_dtype = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(tf_dtype = "64")
    C(tf_dtype = 64)
    C(tf_dtype = "float64")
    C(tf_dtype = tf.float64)
    C(tf_dtype = "32")
    C(tf_dtype = 32)
    C(tf_dtype = "float32")
    C(tf_dtype = tf.float32)
    C(tf_dtype = "16")
    C(tf_dtype = 16)
    C(tf_dtype = "float16")
    C(tf_dtype = tf.float16)

    # This should be caught
    catch(8)
    catch("x")
    catch(None)

def hl1(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(hl1 = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(hl1 = 1)
    C(hl1 = 1.0)

    # This should be caught
    catch(0)
    catch("x")
    catch(4.2)
    catch(None)
    catch(-1)

def hl2(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(hl2 = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(hl2 = 1)
    C(hl2 = 1.0)
    C(hl2 = 0)

    # This should be caught
    catch("x")
    catch(4.2)
    catch(None)
    catch(-1)

def hl3(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(hl2 = 2, hl3 = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(hl2 = 2, hl3 = 1)
    C(hl2 = 2, hl3 = 1.0)
    C(hl2 = 2, hl3 = 0)

    # This should be caught
    catch("x")
    catch(4.2)
    catch(None)
    catch(-1)

def representation(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(representation = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(representation = "unsorted_couLomb_matrix")
    C(representation = "sorted_couLomb_matrix")
    C(representation = "bag_of_bOnds")
    C(representation = "slAtm")

    # This should be caught
    catch("none")
    catch(4.2)
    catch(None)
    catch(-1)

def scoringfunction(C):
    """
    This function checks that the function _set_scoring_function accepts only mae, rmsd and r2 as scoring functions.
    """

    def catch(s):
        try:
            C(scoring_function = s)
            raise Exception
        except InputError:
            pass

    accepted_inputs = ['mae', 'rmse', 'r2']
    unaccepted_inputs = [0, "none", True, None]

    # This should not raise an exception
    for item in accepted_inputs:
        C(scoring_function=item)

    # This should be caught
    for item in unaccepted_inputs:
        catch(item)

def test_input():
    # Additional test that inheritance is ok

    C = MRMP

    hidden_layer_sizes(C)
    l1_reg(C)
    l2_reg(C)
    batch_size(C)
    learning_rate(C)
    iterations(C)
    tf_dtype(C)
    scoringfunction(C)

# --------------------- ** tests for regularisation terms ** -----------------

def test_l2_loss():
    """
    This tests the evaluation of the l2 regularisation term on the weights of the neural net.
    :return: None
    """

    # Some example weights
    weights = [tf.constant([2.0, 4.0], dtype=tf.float32)]

    # Creating object with known l2_reg parameter
    obj = MRMP(l2_reg=0.1)
    expected_result = [2.0]

    # Evaluating l2 term
    l2_loss_tf = obj._l2_loss(weights=weights)
    sess = tf.Session()
    l2_loss = sess.run(l2_loss_tf)

    # Testing
    assert np.isclose(l2_loss, expected_result)

def test_l1_loss():
    """
    This tests the evaluation of the l1 regularisation term on the weights of the neural net.
    :return: None
    """

    # Some example weights
    weights = [tf.constant([2.0, 4.0], dtype=tf.float32)]

    # Creating object with known l1_reg parameter
    obj = MRMP(l1_reg=0.1)
    expected_result = [0.6]

    # Evaluating l1 term
    l1_loss_tf = obj._l1_loss(weights=weights)
    sess = tf.Session()
    l1_loss = sess.run(l1_loss_tf)

    # Testing
    assert np.isclose(l1_loss, expected_result)

def test_get_batch_size():
    """
    This tests the get_batch_size function
    :return:
    """

    example_data = [200, 50, 50]
    possible_cases = ["auto", 100, 20]
    expected_batch_sizes = [100, 50, 17]

    actual_batch_sizes = []
    for i, case in enumerate(possible_cases):
        obj = MRMP(batch_size=case)
        obj.n_samples = example_data[i]
        actual_batch = obj._get_batch_size()
        actual_batch_sizes.append(actual_batch)

    for i in range(len(expected_batch_sizes)):
        assert actual_batch_sizes[i] == expected_batch_sizes[i]

def test_fit1():
    """This tests that the neural net can overfit a cubic function."""

    x = np.linspace(-2.0, 2.0, 200)
    X = np.reshape(x, (len(x), 1))
    y = x ** 3

    estimator = MRMP(hidden_layer_sizes=(5, 5, 5), learning_rate=0.01, iterations=35000)
    estimator.fit(X, y)

    x_test = np.linspace(-1.5, 1.5, 15)
    X_test = np.reshape(x_test, (len(x_test), 1))
    y_test = x_test ** 3
    y_pred = estimator.predict(X_test)

    y_pred_row = np.reshape(y_pred, (y_pred.shape[0],))
    np.testing.assert_array_almost_equal(y_test, y_pred_row, decimal=1)


if __name__ == "__main__":
    test_input()
    test_l2_loss()
    test_l1_loss()
    test_get_batch_size()
    test_fit1()
