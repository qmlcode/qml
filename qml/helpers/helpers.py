# MIT License
#
# Copyright (c) 2017-2019 Anders Steen Christensen, Jakub Wagner
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

import numpy as np

def get_BoB_groups(asize, sort=True):
    """
    Get starting and ending indices of bags in Bags of Bonds representation.

    :param asize: Atomtypes and their maximal numbers in the representation
    :type asize: dictionary
    :param sort: Whether to sort indices as usually automatically done
    :type sort: bool
    """
    if sort:
        asize = {k: asize[k] for k in sorted(asize, key=asize.get)}
    n = 0
    low_indices = {}
    high_indices = {}
    for i, (key1, value1) in enumerate(asize.items()):
        for j, (key2, value2) in enumerate(asize.items()):
            if j == i:   # Comparing same-atoms bonds like C-C
                new_key = key1 + key2
                low_indices[new_key] = n
                n += int(value1 * (value1+1) / 2)
                high_indices[new_key] = n
            elif j >= i:   # Comparing different-atoms bonds like C-H
                new_key = key1 + key2
                low_indices[new_key] = n
                n += int(value1 * value2)
                high_indices[new_key] = n      
    return low_indices, high_indices

def compose_BoB_sigma_vector(sigmas_for_bags, low_indices, high_indices):
    """
    Create a vector of per-feature kernel widths.

    In BoB features are grouped by bond types, so a vector of per-group kernel
    width would suffice for the computation, however having a per-feature
    vector is easier for improving computations with Fortran.

    :param sigmas_for_bags: Kernel widths for different bond types
    :type sigmas_for_bags: dictionary
    :param low_indices: Starting indices for different bond types
    :type low_indices: dictionary
    :param high_indices: End indices for different bond types
    :type high_indices: dictionary
    :return: A vector of per-feature kernel widths
    :rtype: numpy array
    
    """
    length = high_indices[list(sigmas_for_bags.keys())[-1]]
    sigmas = np.zeros(length)
    for group in sigmas_for_bags:
        sigmas[low_indices[group]:high_indices[group]] = sigmas_for_bags[group]
    return sigmas
