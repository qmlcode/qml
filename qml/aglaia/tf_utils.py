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
Various routines related to tensorflow

"""

import os
import numpy as np
import tensorflow as tf
from qml.utils import ceil

class TensorBoardLogger(object):
    """
    Helper class for tensorboard functionality

    """

    def __init__(self, path):
        self.path = path
        self.store_frequency = None

    def initialise(self):
        # Create tensorboard directory
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.options = tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE,
                                     output_partition_graphs=True)
        self.run_metadata = tf.RunMetadata()

    def set_store_frequency(self, freq):
        self.store_frequency = freq

    def set_summary_writer(self, sess):
        self.summary_writer = tf.summary.FileWriter(logdir=self.path, graph=sess.graph)

    def write_summary(self, sess, iteration):

        self.merged_summary = tf.summary.merge_all()
        summary = sess.run(self.merged_summary)
        self.summary_writer.add_summary(summary, iteration)
        self.summary_writer.add_run_metadata(self.run_metadata, 'iteration %d' % (iteration))

    def write_metadata(self, step):
        self.summary_writer.add_run_metadata(self.run_metadata, 'batch %d' % (step))

    def write_weight_histogram(self, weights):
        tf.summary.histogram("weights_in", weights[0])
        for i in range(len(weights) - 1):
            tf.summary.histogram("weights_hidden_%d" % i, weights[i + 1])
        tf.summary.histogram("weights_out", weights[-1])

    def write_cost_summary(self, cost):
        tf.summary.scalar('cost', cost)

def generate_weights(n_in, n_out, hl):

    weights = []
    biases = []

    # Support for linear models
    if len(hl) == 0:
        # Weights from input layer to first hidden layer
        w = tf.Variable(tf.truncated_normal([n_out, n_in], stddev = 1.0, dtype = tf.float32),
                    dtype = tf.float32, name = "weights_in")
        b = tf.Variable(tf.zeros([n_out], dtype = tf.float32), name="bias_in", dtype = tf.float32)

        weights.append(w)
        biases.append(b)

        return weights, biases

    # Weights from input layer to first hidden layer
    w = tf.Variable(tf.truncated_normal([hl[0], n_in], stddev = 1.0 / np.sqrt(hl[0]), dtype = tf.float32),
                dtype = tf.float32, name = "weights_in")
    b = tf.Variable(tf.zeros([hl[0]], dtype = tf.float32), name="bias_in", dtype = tf.float32)

    weights.append(w)
    biases.append(b)

    # Weights from one hidden layer to the next
    for i in range(1, len(hl)):
        w = tf.Variable(tf.truncated_normal([hl[i], hl[i-1]], stddev=1.0 / np.sqrt(hl[i-1]), dtype=tf.float32),
                        dtype=tf.float32, name="weights_hidden_%d" % i)
        b = tf.Variable(tf.zeros([hl[i]], dtype=tf.float32), name="bias_hidden_%d" % i, dtype=tf.float32)

        weights.append(w)
        biases.append(b)


    # Weights from last hidden layer to output layer
    w = tf.Variable(tf.truncated_normal([n_out, hl[-1]],
                                        stddev=1.0 / np.sqrt(hl[-1]), dtype=tf.float32),
                    dtype=tf.float32, name="weights_out")
    b = tf.Variable(tf.zeros([n_out], dtype=tf.float32), name="bias_out", dtype=tf.float32)

    weights.append(w)
    biases.append(b)

    return weights, biases

