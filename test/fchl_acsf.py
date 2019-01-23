# MIT License
#
# Copyright (c) 2017-2018 Anders Steen Christensen, Lars Andersen Bratholm, Bing Huang
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

import numpy as np
# from qml.data.alchemy import NUCLEAR_CHARGE
from ffchl_acsf import fgenerate_fchl_acsf, fgenerate_fchl_acsf_and_gradients


def generate_fchl_acsf(nuclear_charges, coordinates, elements = [1,6,7,8,16], 
        nRs2 = 3, nRs3 = 3, nTs = 2, eta2 = 1,
        eta3 = 1, zeta = 1, rcut = 8, acut = 5, 
        two_body_decay=4.0, three_body_decay=2.0, three_body_weight=0.2,
        gradients = False):
    """
    Generate the variant of atom-centered symmetry functions used in https://doi.org/10.1039/C7SC04934J

    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param coordinates: Input coordinates
    :type coordinates: numpy array
    :param elements: list of unique nuclear charges (atom types)
    :type elements: numpy array
    :param nRs2: Number of gaussian basis functions in the two-body terms
    :type nRs2: integer
    :param nRs3: Number of gaussian basis functions in the three-body radial part
    :type nRs3: integer
    :param nTs: Number of basis functions in the three-body angular part
    :type nTs: integer
    :param eta2: Precision in the gaussian basis functions in the two-body terms
    :type eta2: float
    :param eta3: Precision in the gaussian basis functions in the three-body radial part
    :type eta3: float
    :param zeta: Precision parameter of basis functions in the three-body angular part
    :type zeta: float
    :param rcut: Cut-off radius of the two-body terms
    :type rcut: float
    :param acut: Cut-off radius of the three-body terms
    :type acut: float
    :param gradients: To return gradients or not
    :type gradients: boolean
    :return: Atom-centered symmetry functions representation
    :rtype: numpy array
    """

    Rs2 = np.linspace(0, rcut, 1+nRs2)[1:]
    Rs3 = np.linspace(0, acut, 1+nRs3)[1:]
    # print(Rs2)
    # print(Rs3)
    # exit()
    Ts = np.linspace(0, np.pi, nTs)
    n_elements = len(elements)
    natoms = len(coordinates)

    descr_size = n_elements * nRs2 + (n_elements * (n_elements + 1)) // 2 * nRs3*nTs

    if gradients:
        return fgenerate_fchl_acsf_and_gradients(coordinates, nuclear_charges, elements, Rs2, Rs3,
                Ts, eta2, eta3, zeta, rcut, acut, natoms, descr_size,
                two_body_decay, three_body_decay, three_body_weight)
    else:
        return fgenerate_fchl_acsf(coordinates, nuclear_charges, elements, Rs2, Rs3, 
                Ts, eta2, eta3, zeta, rcut, acut, natoms, descr_size,  
                two_body_decay, three_body_decay, three_body_weight)
