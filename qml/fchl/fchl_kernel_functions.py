# MIT License
#
# Copyright (c) 2017 Anders Steen Christensen and Felix A. Faber
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

from __future__ import division
from __future__ import print_function

import numpy as np
from copy import copy

import scipy
from scipy.special import binom
from scipy.misc import factorial

def get_gaussian_parameters(tags):

    if tags is None:
        tags = {"sigma": [2.5],}

    parameters = np.array(tags["sigma"])

    for i in range(len(parameters)):
        parameters[i] = -0.5 / (parameters[i])**2

    np.resize(parameters, (1,len(tags["sigma"])))

    n_kernels = len(tags["sigma"])

    return 1, parameters, n_kernels


def get_linear_parameters(tags):

    if tags is None:
        tags = {"c": [0.0],}


    parameters = np.array(tags["c"])

    np.resize(parameters, (1,len(tags["c"])))
    
    n_kernels = len(tags["c"])

    return 2, parameters, n_kernels


def get_polynomial_parameters(tags):
    
    if tags is None:
        tags = {
            "alpha": [1.0],
            "c": [0.0],
            "d": [1.0]
        }

    parameters = np.array([
                tags["alpha"],
                tags["c"],
                tags["d"]
            ]).T
    assert len(tags["alpha"]) == len(tags["c"])
    assert len(tags["alpha"]) == len(tags["d"])

    n_kernels = len(tags["alpha"])
    return 3, parameters, n_kernels


def get_sigmoid_parameters(tags):
    
    if tags is None:
        tags = {
            "alpha": [1.0],
            "c": [0.0],

        }

    parameters = np.array([
                tags["alpha"],
                tags["c"],
            ]).T
    assert len(tags["alpha"]) == len(tags["c"])
    n_kernels = len(tags["alpha"])

    return 4, parameters, n_kernels


def get_multiquadratic_parameters(tags):
    
    if tags is None:
        tags = {
            "c": [0.0],

        }

    parameters = np.array([
                tags["c"],
            ]).T
    
    np.resize(parameters, (1,len(tags["c"])))
    n_kernels = len(tags["c"])
    
    return 5, parameters, n_kernels


def get_inverse_multiquadratic_parameters(tags):
    
    if tags is None:
        tags = {
            "c": [0.0],

        }

    parameters = np.array([
                tags["c"],
            ]).T
    
    np.resize(parameters, (1,len(tags["c"])))
    n_kernels = len(tags["c"])
    
    return 6, parameters, n_kernels


def get_bessel_parameters(tags):
    
    if tags is None:
        tags = {
            "sigma": [1.0],
            "v": [1.0],
            "n": [1.0]
        }

    parameters = np.array([
                tags["sigma"],
                tags["v"],
                tags["n"]
            ]).T
    assert len(tags["sigma"]) == len(tags["v"])
    assert len(tags["sigma"]) == len(tags["n"])

    n_kernels = len(tags["sigma"])

    return 7, parameters, n_kernels

def get_l2_parameters(tags):

    if tags is None:
        tags = {
            "alpha": [1.0],
            "c": [0.0],

        }

    parameters = np.array([
                tags["alpha"],
                tags["c"],
            ]).T
    assert len(tags["alpha"]) == len(tags["c"])
    n_kernels = len(tags["alpha"])

    return 8, parameters, n_kernels


def get_matern_parameters(tags):

    if tags is None:
        tags = {
            "sigma": [10.0],
            "n": [2.0],

        }

    assert len(tags["sigma"]) == len(tags["n"])
    n_kernels = len(tags["sigma"])

    n_max = int(max(tags["n"])) + 1

    parameters = np.zeros((2+n_max, n_kernels))

    for i in range(n_kernels):
        parameters[0,i] = tags["sigma"][i]
        parameters[1,i] = tags["n"][i]

        n = int(tags["n"][i])
        for k in range(0, n+1):
            parameters[2+k,i] = float(factorial(n + k)  * binom(n, k))/ factorial(2*n)

    parameters = parameters.T

    return 9, parameters, n_kernels


def get_cauchy_parameters(tags):
    
    if tags is None:
        tags = {
            "sigma": [1.0],

        }

    parameters = np.array([
                tags["sigma"],
            ]).T
    
    np.resize(parameters, (1,len(tags["sigma"])))
    n_kernels = len(tags["sigma"])

    return 10, parameters, n_kernels

def get_polynomial2_parameters(tags):
    
    if tags is None:
        tags = {
            "coeff": [[1.0, 1.0, 1.0]],
        }

    parameters = np.zeros((10,len(tags["coeff"])))

    for i, c in enumerate(tags["coeff"]):
        for j, v in enumerate(c):
            parameters[j,i] = v

    n_kernels = len(tags["coeff"])
    parameters = parameters.T
    return 11, parameters, n_kernels

def get_kernel_parameters(name, tags):

    parameters = None
    idx = 1
    n_kernels = 1
   
    if name == "gaussian":
        idx, parameters, n_kernels = get_gaussian_parameters(tags)

    elif name == "linear":
        idx, parameters, n_kernels = get_linear_parameters(tags)

    elif name == "polynomial":
        idx, parameters, n_kernels = get_polynomial_parameters(tags)

    elif name == "sigmoid":
        idx, parameters, n_kernels = get_sigmoid_parameters(tags)
    
    elif name == "multiquadratic":
        idx, parameters, n_kernels = get_multiquadratic_parameters(tags)
    
    elif name == "inverse-multiquadratic":
        idx, parameters, n_kernels = get_inverse_multiquadratic_parameters(tags)
    
    elif name == "bessel":
        idx, parameters, n_kernels = get_bessel_parameters(tags)
    
    elif name == "l2":
        idx, parameters, n_kernels = get_l2_parameters(tags)

    elif name == "matern":
        idx, parameters, n_kernels = get_matern_parameters(tags)

    elif name == "cauchy":
        idx, parameters, n_kernels = get_cauchy_parameters(tags)
    
    elif name == "polynomial2":
        idx, parameters, n_kernels = get_polynomial2_parameters(tags)

    else:

        print("QML ERROR: Unsupported kernel specification,", name)
        exit()

    return idx, parameters, n_kernels
