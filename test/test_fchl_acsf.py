# MIT License
#
# Copyright (c) 2019 Anders S. Christensen
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
This file contains tests for the atom centred symmetry function module.
"""
from __future__ import print_function
import os
from copy import deepcopy
import numpy as np
np.set_printoptions(linewidth=666, edgeitems=10)
import qml
from qml import Compound
from qml.representations import generate_fchl_acsf

REP_PARAMS = dict()
REP_PARAMS["elements"] = [1, 6, 7]


def get_acsf_numgrad(mol, dx=1e-5):

    true_coords = deepcopy(mol.coordinates)

    true_rep = generate_fchl_acsf(mol.nuclear_charges, mol.coordinates, 
        gradients=False, **REP_PARAMS)

    gradient = np.zeros((3, mol.natoms, true_rep.shape[0], true_rep.shape[1]))

    for n, coord in enumerate(true_coords):
        for xyz, x in enumerate(coord):

            temp_coords = deepcopy(true_coords)
            temp_coords[n,xyz] = x + 2.0 *dx

            (rep, grad) = generate_fchl_acsf(mol.nuclear_charges, temp_coords, gradients=True, **REP_PARAMS)
            gradient[xyz, n] -= rep
            
            temp_coords[n,xyz] = x + dx
            (rep, grad) = generate_fchl_acsf(mol.nuclear_charges, temp_coords, gradients=True, **REP_PARAMS)
            gradient[xyz, n] += 8.0 * rep
            
            temp_coords[n,xyz] = x - dx
            (rep, grad) = generate_fchl_acsf(mol.nuclear_charges, temp_coords, gradients=True, **REP_PARAMS)
            gradient[xyz, n] -= 8.0 * rep
            
            temp_coords[n,xyz] = x - 2.0 *dx
            (rep, grad) = generate_fchl_acsf(mol.nuclear_charges, temp_coords, gradients=True, **REP_PARAMS)
            gradient[xyz, n] += rep

    gradient /= (12 * dx)

    gradient = np.swapaxes(gradient, 0, 1 )
    gradient = np.swapaxes(gradient, 2, 0 )
    gradient = np.swapaxes(gradient, 3, 1)

    return gradient

    
def test_fchl_acsf():
    
    test_dir = os.path.dirname(os.path.realpath(__file__))

    mol = Compound(xyz=test_dir+"/qm7/0101.xyz")

    (repa, anal_grad) = generate_fchl_acsf(mol.nuclear_charges, mol.coordinates, 
        gradients=True, **REP_PARAMS)
    
    repb = generate_fchl_acsf(mol.nuclear_charges, mol.coordinates, 
        gradients=False, **REP_PARAMS)

    assert np.allclose(repa, repb), "Error in FCHL-ACSF representation implementation"

    num_grad = get_acsf_numgrad(mol)

    assert np.allclose(anal_grad, num_grad), "Error in FCHL-ACSF gradient implementation"


if __name__ == "__main__":

    test_fchl_acsf()

