#!/usr/bin/env python3
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
This file contains tests for the atom centred symmetry function module.
"""
from __future__ import print_function

import sys
sys.path.insert(0, "/home/andersx/dev/qml/andersx/fchl_acsf/build/lib.linux-x86_64-3.5")

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import os

from copy import deepcopy

import numpy as np

def ff(x):
    return  "%f" % x

# np.set_printoptions(linewidth=999, suppress=True, formatter ={"float": ff})
np.set_printoptions(linewidth=999, suppress=True, edgeitems=1000)
# np.set_printoptions(linewidth=999)

import qml
# from qml.representations import generate_fchl_acsf
from fchl_acsf import generate_fchl_acsf

REP_PARAMS = dict()
REP_PARAMS["elements"] = [1, 6, 7]
REP_PARAMS["nRs2"] = 5
REP_PARAMS["nRs3"] = 3
REP_PARAMS["nTs"]  = 2
REP_PARAMS["eta2"] = 2.0
REP_PARAMS["eta3"] = 1.0
REP_PARAMS["zeta"] = 4.0
REP_PARAMS["rcut"] = 8.0
REP_PARAMS["acut"] = 5.0
REP_PARAMS["three_body_weight"] = 3.0
REP_PARAMS["three_body_decay"]  = 2.0
REP_PARAMS["two_body_decay"]    = 4.0

DX = 1e-5


def get_mols(files):
    
    path = os.path.dirname(os.path.realpath(__file__))

    return [qml.data.Compound(xyz=path + "/" + f) for f in files]


def get_acsf(mol):

    return generate_fchl_acsf(mol.nuclear_charges, mol.coordinates, gradients=True, **REP_PARAMS)[0]


def get_acsf_numgrad(mol, dx=DX):

    true_coords = deepcopy(mol.coordinates)

    true_rep = get_acsf(mol)

    gradient = np.zeros((3, mol.natoms, true_rep.shape[0], true_rep.shape[1]))
    # print(gradient.shape)

    for n, coord in enumerate(true_coords):
        for xyz, x in enumerate(coord):

            temp_coords = deepcopy(true_coords)

            temp_coords[n,xyz] = x + 2.0 *dx
            # print(temp_coords - true_coords)

            rep = generate_fchl_acsf(mol.nuclear_charges, temp_coords, gradients=True, **REP_PARAMS)[0]
            gradient[xyz, n] -= rep
            
            temp_coords[n,xyz] = x + dx
            rep = generate_fchl_acsf(mol.nuclear_charges, temp_coords, gradients=True, **REP_PARAMS)[0]
            gradient[xyz, n] += 8.0 * rep
            
            temp_coords[n,xyz] = x - dx
            rep = generate_fchl_acsf(mol.nuclear_charges, temp_coords, gradients=True, **REP_PARAMS)[0]
            gradient[xyz, n] -= 8.0 * rep
            
            temp_coords[n,xyz] = x - 2.0 *dx
            rep = generate_fchl_acsf(mol.nuclear_charges, temp_coords, gradients=True, **REP_PARAMS)[0]
            gradient[xyz, n] += rep


    gradient /= (12 * dx)

    # print(generate_acsf(mol.nuclear_charges, mol.coordinates, **REP_PARAMS))

    return gradient

    






if __name__ == "__main__":
    
    files = [
            # "ammonia.xyz",
            # "water.xyz",
            # "qm7/0101.xyz",
            "qm7/0102.xyz",
            "qm7/0103.xyz",
            "qm7/0104.xyz",
            "qm7/0105.xyz",
            "qm7/0106.xyz",
            "qm7/0107.xyz",
            "qm7/0108.xyz",
            "qm7/0109.xyz",
            "qm7/0110.xyz"
            ]

    mols = get_mols(files) 
    reps = np.array([get_acsf(mol) for mol in mols])

    num_grad = get_acsf_numgrad(mols[0])[:,:,:,:]

    #print(mols[0].generate_acsf(gradients = True, **REP_PARAMS))

    (repa, grad) = generate_fchl_acsf(mols[0].nuclear_charges, mols[0].coordinates, 
        gradients=True, **REP_PARAMS)
    
    repb = generate_fchl_acsf(mols[0].nuclear_charges, mols[0].coordinates, 
        gradients=False, **REP_PARAMS)

    print(repb.shape)
    print("Reference representation:")
    print(repa[:,10:])
    print("Analytical gradient implementation representation:")
    print(repb[:,10:])
    print("Error representation:")
    print((repa - repb)[:,10:])


    print(grad.shape)
    grad = np.swapaxes(grad, 3, 1)
    grad = np.swapaxes(grad, 2, 0 )
    grad = np.swapaxes(grad, 0, 1 )
    
    # anal_grad = grad[:,:,:,:-4]
    anal_grad = grad[:,:,:,:]


    # print(num_grad.shape)
    # print(anal_grad.shape)
    # exit()
    # print(grad)
    # print(grad.shape)

    # print(anal_grad - num_grad)
    diff = anal_grad - num_grad
    print(num_grad.shape)
    print(anal_grad.shape)

    print("Numerical Gradient:")
    print(num_grad[:,0,:,10:])
    print("Analytical Gradient:")
    print(anal_grad[:,0,:,10:])
    # print(diff)
    print("Error Ratio Gradient:")
    print(anal_grad[:,0,:,10:]/num_grad[:,0,:,10:])
    print("Difference analytical-numerical gradient:")
    print(anal_grad[:,:,:,10:]-num_grad[:,:,:,10:])
    # print(np.amax(np.abs(diff)))
    Rs2 = np.linspace(0, 8.0, 1+5)[1:]
    # print(Rs2)
