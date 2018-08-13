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
import tensorflow as tf

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
