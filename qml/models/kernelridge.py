# MIT License
#
# Copyright (c) 2017 Anders S. Christensen
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

from time import time

import numpy as np

import ase

from ..ml.representations import generate_coulomb_matrix
from ..ml.kernels import gaussian_kernel
from ..ml.math import cho_solve
from .mlmodel import MLModel

class GenericKRR(MLModel):

    def __init__(self, targets, representation="coulomb-matrix", 
            kernel="gaussian", sigma=4000.0, llambda=1e-10):

        super(GenericKRR,self).__init__(targets)

        self.representation = representation
        self.kernel = kernel
        self.sigma = sigma
        self.llambda = llambda

    def _train(self, data):

        start = time()
        X = []

        (properties, compounds) = data.get_properties()

        for row in compounds.select():
            # for key in row:
            #    print('{0:22}: {1}'.format(key, row[key])) 
            X.append(generate_coulomb_matrix(row["numbers"],
                row["positions"]))
       
        self.Y = np.array(properties)
        self.X = np.array(X)
        self.K = gaussian_kernel(self.X, self.X, self.sigma)
        self.K[np.diag_indices_from(self.K)] += self.llambda
        self.alpha = cho_solve(self.K, self.Y) 
       
        Yss = np.dot(self.K, self.alpha)
        mae_training_error = np.mean(np.abs(Yss - self.Y))
        rmse_training_error = np.sqrt(np.mean(np.square(Yss - self.Y)))

        result = {
            "MAETrainingError": mae_training_error,
            "RMSETrainingError": rmse_training_error,
            "ElapsedTime": time() - start,
            }

        return result

    def _predict(self, data):
        
        Xs = []

        (properties, compounds) = data.get_properties()

        for row in compounds.select():
            # for key in row:
            #    print('{0:22}: {1}'.format(key, row[key])) 
            Xs.append(generate_coulomb_matrix(row["numbers"],
                row["positions"]))
       
        Xs = np.array(Xs)
        Ks = gaussian_kernel(Xs, self.X, self.sigma)

        return np.dot(Ks, self.alpha)


    def _restore(self, path, load_kernel=False):
        self.Y = np.load(path + "/Y.npy")
        self.X = np.load(path + "/X.npy")
        self.alpha = np.load(path + "/alpha.npy")
        if load_kernel:
            self.K = np.load(path + "/K.npy")


    def _save(self, path, save_kernel=False):
        np.save(path + "/X.npy", self.X)
        np.save(path + "/Y.npy", self.Y)
        np.save(path + "/alpha.npy", self.alpha)

        if save_kernel:
            np.save(path + "/K.npy")

        

