# MIT License
#
# Copyright (c) 2018 Anders Steen Christensen
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

from __future__ import print_function

import os

import qml
import numpy as np

def compare_lists(a, b):
    for pair in zip(a,b):
        if pair[0] != pair[1]:
            return False
    return True

def test_compound():

    test_dir = os.path.dirname(os.path.realpath(__file__))
    c = qml.Compound(xyz=test_dir + "/data/compound_test.xyz")
    
    ref_atomtypes = ['C', 'Cl', 'Br', 'H', 'H']
    ref_charges = [ 6, 17, 35,  1 , 1]

    assert compare_lists(ref_atomtypes, c.atomtypes), "Failed parsing atomtypes"
    assert compare_lists(ref_charges, c.nuclear_charges), "Failed parsing nuclear_charges"
   
    # Test extended xyz
    c2 = qml.Compound(xyz=test_dir + "/data/compound_test.exyz")
    
    ref_atomtypes = ['C', 'Cl', 'Br', 'H', 'H']
    ref_charges = [ 6, 17, 35,  1 , 1]

    assert compare_lists(ref_atomtypes, c.atomtypes), "Failed parsing atomtypes"
    assert compare_lists(ref_charges, c.nuclear_charges), "Failed parsing nuclear_charges"

if __name__ == "__main__":

    test_compound()
