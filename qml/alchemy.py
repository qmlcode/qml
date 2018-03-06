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

PTP = {\
         1  :[1,1] ,2:  [1,8]#Row1

        ,3  :[2,1] ,4:  [2,2]#Row2\
        ,5  :[2,3] ,6:  [2,4] ,7  :[2,5] ,8  :[2,6] ,9  :[2,7] ,10 :[2,8]\

        ,11 :[3,1] ,12: [3,2]#Row3\
        ,13 :[3,3] ,14: [3,4] ,15 :[3,5] ,16 :[3,6] ,17 :[3,7] ,18 :[3,8]\

        ,19 :[4,1] ,20: [4,2]#Row4\
        ,31 :[4,3] ,32: [4,4] ,33 :[4,5] ,34 :[4,6] ,35 :[4,7] ,36 :[4,8]\
        ,21 :[4,9] ,22: [4,10],23 :[4,11],24 :[4,12],25 :[4,13],26 :[4,14],27 :[4,15],28 :[4,16],29 :[4,17],30 :[4,18]\

        ,37 :[5,1] ,38: [5,2]#Row5\
        ,49 :[5,3] ,50: [5,4] ,51 :[5,5] ,52 :[5,6] ,53 :[5,7] ,54 :[5,8]\
        ,39 :[5,9] ,40: [5,10],41 :[5,11],42 :[5,12],43 :[5,13],44 :[5,14],45 :[5,15],46 :[5,16],47 :[5,17],48 :[5,18]\

        ,55 :[6,1] ,56: [6,2]#Row6\
        ,81 :[6,3] ,82: [6,4] ,83 :[6,5] ,84 :[6,6] ,85 :[6,7] ,86 :[6,8]
               ,72: [6,10],73 :[6,11],74 :[6,12],75 :[6,13],76 :[6,14],77 :[6,15],78 :[6,16],79 :[6,17],80 :[6,18]\
        ,57 :[6,19],58: [6,20],59 :[6,21],60 :[6,22],61 :[6,23],62 :[6,24],63 :[6,25],64 :[6,26],65 :[6,27],66 :[6,28],67 :[6,29],68 :[6,30],69 :[6,31],70 :[6,32],71 :[6,33]\

        ,87 :[7,1] ,88: [7,2]#Row7\
        ,113:[7,3] ,114:[7,4] ,115:[7,5] ,116:[7,6] ,117:[7,7] ,118:[7,8]\
               ,104:[7,10],105:[7,11],106:[7,12],107:[7,13],108:[7,14],109:[7,15],110:[7,16],111:[7,17],112:[7,18]\
        ,89 :[7,19],90: [7,20],91 :[7,21],92 :[7,22],93 :[7,23],94 :[7,24],95 :[7,25],96 :[7,26],97 :[7,27],98 :[7,28],99 :[7,29],100:[7,30],101:[7,31],101:[7,32],102:[7,14],103:[7,33]}

QtNm = {
    #Row1
    1  :[1,0,0,1./2.]
    ,2:  [1,0,0,-1./2.]


    #Row2
    ,3  :[2,0,0,1./2.]
    ,4: [2,0,0,-1./2.]

    ,5  :[2,-1,1,1./2.],   6: [2,0,1,1./2.]  , 7  : [2,1,1,1./2.]
    ,8  : [2,-1,1,-1./2.] ,9: [2,0,1,-1./2.] ,10 :[2,1,1,-1./2.]


    #Row3
    ,11 :[3,0,0,1./2.]
    ,12: [3,0,0,-1./2.]

    ,13 :[3,-1,1,1./2.] ,  14: [3,0,1,1./2.] ,   15 :[3,1,1,1./2.]
    ,16 :[3,-1,1,-1./2.]  ,17 :[3,0,1,-1./2.]  ,18 :[3,1,1,-1./2.]


    #Row3
    ,19 :[4,0,0,1./2.]
    ,20: [4,0,0,-1./2.]

    ,31 :[4,-1,2,1./2.] , 32: [4,0,1,1./2.]  , 33 :[4,1,1,1./2.]
    ,34 :[4,-1,1,-1./2.] ,35 :[4,0,1,-1./2.] ,36 :[4,1,1,-1./2.]

    ,21 :[4,-2,2,1./2.],  22:[4,-1,2,1./2.],  23 :[4,0,2,1./2.], 24 :[4,1,2,1./2.], 25 :[4,2,2,1./2.]
    ,26 :[4,-2,2,-1./2.], 27:[4,-1,2,-1./2.], 28 :[4,0,2,-1./2.],29 :[4,1,2,-1./2.],30 :[4,2,2,-1./2.]


    #Row5
    ,37 :[5,0,0,1./2.]
    ,38: [5,0,0,-1./2.]

    ,49 :[5,-1,1,1./2.] ,  50: [5,0,1,1./2.]  ,  51 :[5,1,1,1./2.]
    ,52 :[5,-1,1,-1./2.]  ,53 :[5,0,1,-1./2.]  ,54 :[5,1,1,-1./2.]


    ,39 :[5,-2,2,1./2.], 40:[5,-1,2,1./2.],   41 :[5,0,2,1./2.], 42 :[5,1,2,1./2.], 43 :[5,2,2,1./2.]
    ,44 :[5,-2,2,-1./2.],45 :[5,-1,2,-1./2.],46 :[5,0,2,-1./2.],47 :[5,1,2,-1./2.],48 :[5,2,2,-1./2.]


    #Row6
    ,55 :[6,0,0,1./2.]
    ,56: [6,0,0,-1./2.]

    ,81 :[6,-1,1,1./2.] ,82: [6,0,1,1./2.] ,83: [6,1,1,1./2.]
    ,84 :[6,-1,1,-1./2.] ,85 :[6,0,1,-1./2.] ,86 :[6,1,1,-1./2.]

    ,71 :[6,-2,2,1./2.], 72: [6,-1,2,1./2.],  73 :[6,0,2,1./2.], 74 :[6,1,2,1./2.], 75 :[6,2,2,1./2.]
    ,76 :[6,-2,2,-1./2.],77 :[6,-1,2,-1./2.],78 :[6,0,2,-1./2.],79 :[6,1,2,-1./2.],80 :[6,2,2,-1./2.]

    ,57 :[6,-3,3,1./2.], 58: [6,-2,3,1./2.],  59 :[6,-1,3,1./2.], 60 :[6,0,3,1./2.],  61 :[6,1,3,1./2.], 62 :[6,2,3,1./2.], 63 :[6,3,3,1./2.]
    ,64 :[6,-3,3,-1./2.],65 :[6,-2,3,-1./2.],66 :[6,-1,3,-1./2.],67 :[6,0,3,-1./2.],68 :[6,1,3,-1./2.],69 :[6,2,3,-1./2.],70 :[6,3,3,-1./2.]


    #Row7
    ,87 :[7,0,0,1./2.]
    ,88: [7,0,0,-1./2.]

    ,113:[7,-1,1,1./2.] , 114:[7,0,1,1./2.] ,  115:[7,1,1,1./2.]
    ,116:[7,-1,1,-1./2.] ,117:[7,0,1,-1./2.] ,118:[7,1,1,-1./2.]

    ,103:[7,-2,2,1./2.], 104:[7,-1,2,1./2.],  105:[7,0,2,1./2.], 106:[7,1,2,1./2.],  107:[7,2,2,1./2.]
    ,108:[7,-2,2,-1./2.],109:[7,-1,2,-1./2.],110:[7,0,2,-1./2.],111:[7,1,2,-1./2.],112:[7,2,2,-1./2.]

    ,89 :[7,-3,3,1./2.], 90: [7,-2,3,1./2.],  91 :[7,-1,3,1./2.], 92 :[7,0,3,1./2.],  93 :[7,1,3,1./2.],  94 :[7,2,3,1./2.],  95 :[7,3,3,1./2.]
    ,96 :[7,-3,3,-1./2.],97 :[7,-2,3,-1./2.],98 :[7,-1,3,-1./2.],99 :[7,0,3,-1./2.],100:[7,1,3,-1./2.],101:[7,2,3,-1./2.],102:[7,3,3,-1./2.]}







def QNum_distance(a,b, n_width, m_width, l_width, s_width):
    """ Calculate stochiometric distance
        a -- nuclear charge of element a
        b -- nuclear charge of element b
        r_width -- sigma in row-direction
        c_width -- sigma in column direction
    """

    na = QtNm[int(a)][0]
    nb = QtNm[int(b)][0]

    ma = QtNm[int(a)][1]
    mb = QtNm[int(b)][1]

    la = QtNm[int(a)][2]
    lb = QtNm[int(b)][2]

    sa = QtNm[int(a)][3]
    sb = QtNm[int(b)][3]

    return  np.exp(-(na - nb)**2/(4 * n_width**2)
                   -(ma - mb)**2/(4 * m_width**2)
                   -(la - lb)**2/(4 * l_width**2)
                   -(sa - sb)**2/(4 * s_width**2))

def gen_QNum_distances(emax=100, n_width = 0.001, m_width = 0.001, l_width = 0.001, s_width = 0.001):
    """ Generate stochiometric ditance matrix
        emax -- Largest element
        r_width -- sigma in row-direction
        c_width -- sigma in column direction
    """

    pd = np.zeros((emax,emax))

    for i in range(emax):
        for j in range(emax):

            pd[i,j] = QNum_distance(i+1, j+1, n_width, m_width, l_width, s_width)

    return pd

def periodic_distance(a, b, r_width, c_width):
    """ Calculate stochiometric distance

        a -- nuclear charge of element a
        b -- nuclear charge of element b
        r_width -- sigma in row-direction
        c_width -- sigma in column direction
    """

    ra = PTP[int(a)][0]
    rb = PTP[int(b)][0]
    ca = PTP[int(a)][1]
    cb = PTP[int(b)][1]

    # return (r_width**2 * c_width**2) / ((r_width**2 + (ra - rb)**2) * (c_width**2 + (ca - cb)**2))

    return np.exp(-(ra - rb)**2/(4 * r_width**2)-(ca - cb)**2/(4 * c_width**2))


def gen_pd(emax=100, r_width=0.001, c_width=0.001):
    """ Generate stochiometric ditance matrix

        emax -- Largest element
        r_width -- sigma in row-direction
        c_width -- sigma in column direction
    """

    pd = np.zeros((emax,emax))

    for i in range(emax):
        for j in range(emax):

            pd[i,j] = periodic_distance(i+1, j+1, r_width, c_width)

    return pd


def gen_custom(e_vec, emax=100):
    """ Generate stochiometric ditance matrix
        emax -- Largest element
        r_width -- sigma in row-direction
        c_width -- sigma in column direction
    """

    def check_if_unique(iterator):
        return len(set(iterator)) == 1

    num_dims = []

    for k,v in e_vec.items():
        assert isinstance(k,int), 'Error! Keys need to be int'
        num_dims.append(len(v))

    assert check_if_unique(num_dims), 'Error! Unequal number of dimensions'


    tmp = np.zeros((emax,num_dims[0]))

    for k,v in e_vec.items():
        tmp[k,:] = copy(v)
    pd = np.dot(tmp,tmp.T)

    return pd

def get_alchemy(alchemy, emax=100, r_width=0.001, c_width=0.001, elemental_vectors={}, \
                n_width = 0.001, m_width = 0.001, l_width = 0.001, s_width = 0.001):


    if (type(alchemy) == np.ndarray):

        doalchemy = True
        return doalchemy, alchemy

    elif (alchemy == "off"):

        pd = np.eye(emax)
        doalchemy = False

        return doalchemy, pd

    elif (alchemy == "periodic-table"):

        pd = gen_pd(emax=emax, r_width=r_width, c_width=c_width)
        doalchemy = True

        return doalchemy, pd


    elif (alchemy == "quantum-numbers"):
        pd = gen_QNum_distances(emax=emax, n_width = n_width, m_width = m_width,
                                          l_width = l_width, s_width = s_width)
        doalchemy = True

        return doalchemy, pd

    elif (alchemy == "custom"):

        pd = gen_custom(elemental_vectors,emax)
        doalchemy = True


        return doalchemy, pd

    else:

        print("QML ERROR: Unknown alchemical method specified:", alchemy)
        exit(1)
