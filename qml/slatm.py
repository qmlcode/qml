# MIT License
#
# Copyright (c) 2016 Bing Huang and Anders S. Christensen
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

from time import time
import scipy.spatial.distance as ssd
import itertools as itl
import numpy as np

from .fslatm import fget_sbot
from .fslatm import fget_sbop

def get_pbc(obj, d0 = 3.6):
    """
    automatically tell if an compound object is periodic or not

    :param d0: the threshhold value to tell if two cells are adjacent
    :type d0: float
    """

    pbc = []

    zs, ps, c = obj
    na = len(zs); idxs = np.arange(na)

    for i in range(3):
        psx = ps[:,i]; xmin = min(psx)
        idxs_i = idxs[ psx == xmin ]
        ps1 = ps[idxs_i[0]] + c[i]
        if np.min( ssd.cdist([ps1,], ps)[0] ) < d0:
            pbc.append( '1' )
        else:
            pbc.append( '0' )

    return ''.join(pbc)

def update_m(obj, ia, rcut=9.0, pbc=None):
    """
    retrieve local structure around atom `ia
    for periodic systems (or very large system)
    """

    zs, coords, c = obj
    v1, v2, v3 = c
    vs = ssd.norm(c, axis=0)

    nns = []; ns = []
    for i,vi in enumerate(vs):
        n1_doulbe = rcut/li
        n1 = int(n1_doulbe)
        if n1 - n1_doulbe == 0:
            n1s = range(-n1, n1+1) if pbc[i] else [0,]
        elif n1 == 0:
            n1s = [-1,0,1] if pbc[i] else [0,]
        else:
            n1s = range(-n1-1, n1+2) if pbc[i] else [0,]

        nns.append(n1s)

    n1s,n2s,n3s = nns

    n123s_ = np.array( list( itl.product(n1s,n2s,n3s) ) )
    n123s = []
    for n123 in n123s_:
        n123u = list(n123)
        if n123u != [0,0,0]: n123s.append(n123u)

    nau = len(n123s)
    n123s = np.array(n123s, np.float)

    na = len(zs)
    ai = m[ia]; cia = coords[ia]
    if na == 1:
        ds = np.array([[0.]])
    else:
        ds = ssd.squareform( ssd.pdist(coords) )

    zs_u = []; coords_u = []
    zs_u.append( zs[ia] ); coords_u.append( coords[ia] )
    for i in range(na) :
        di = ds[i,ia]
        if (di > 0) and (di <= rcut):
            zs_u.append(zs[i]); coords_u.append(coords[ia])

# add new coords by translation
            ts = np.zeros((nau,3))
            for iau in range(nau):
                ts[iau] = np.dot(n123s[iau],c)

            coords_iu = coords[i] + ts #np.dot(n123s, c)
            dsi = ssd.norm(coords_iu - cia, axis=1);
            filt = np.logical_and(dsi > 0, dsi <= rcut); nx = filt.sum()
            zs_u += [zs[i],]*nx
            coords_u += [ list( coords_iu[filt,:] ), ]

    obj_u = [zs_u, coords_u]

    return obj_u


def get_boa(z1, zs_):
    return z1*np.array( [(zs_ == z1).sum(), ])
    #return -0.5*z1**2.4*np.array( [(zs_ == z1).sum(), ])

def get_sbop(mbtype, obj, iloc=False, ia=None, normalize=True, sigma=0.05, \
             rcut=4.8, dgrid=0.03, ipot=True, pbc='000', rpower=6):
    """
    two-body terms

    :param obj: molecule object, consisting of two parts: [ zs, coords ]
    :type obj: list
    """

    z1, z2 = mbtype
    zs, coords, c = obj

    if iloc:
        assert ia != None, '#ERROR: plz specify `za and `ia '

    if pbc != '000':
        if rcut < 9.0: raise '#ERROR: rcut too small for systems with pbc'
        assert iloc, '#ERROR: for periodic system, plz use atomic rpst'
        zs, coords = update_m(obj, ia, rcut=rcut, pbc=pbc)

        # after update of `m, the query atom `ia will become the first atom
        ia = 0

    # if iloc:
    #     dsu = []; icnt = 0
    #     for j1,j2 in ias12:
    #         if ia == j1 or ia == j2:
    #             dsu.append( ds[j1,j2] )
    #         icnt += 1
    # else:
    #     dsu = [ ds[i,j] for (i,j) in ias12 ]

    # bop potential distribution
    r0 = 0.1
    nx = (rcut - r0)/dgrid + 1
    xs = np.linspace(r0, rcut, nx)

    # coeff = 1/np.sqrt(2*sigma**2*np.pi) if normalize else 1.0

    ys = fget_sbop(coords, zs, z1, z2, rcut, nx, dgrid, sigma, rpower)

    return ys

def get_sbot(mbtype, obj, iloc=False, ia=None, normalize=True, sigma=0.05, label=None, \
             rcut=4.8, dgrid=0.0262, ipot=True, pbc='000'):

    """
    sigma -- standard deviation of gaussian distribution centered on a specific angle
            defaults to 0.05 (rad), approximately 3 degree
    dgrid    -- step of angle grid
            defaults to 0.0262 (rad), approximately 1.5 degree
    """

    z1, z2, z3 = mbtype
    zs, coords, c = obj

    if iloc:
        assert ia != None, '#ERROR: plz specify `za and `ia '

    if pbc != '000':
        assert iloc, '#ERROR: for periodic system, plz use atomic rpst'
        zs, coords = update_m(obj, ia, rcut=rcut, pbc=pbc)

        # after update of `m, the query atom `ia will become the first atom
        ia = 0

    # # print "python lentas", len(tas)
    # if iloc:
    #     tas_u = []
    #     for tas_i in tas:
    #         if ia == tas_i[1]:
    #             tas_u.append( tas_i )
    #     tas = tas_u

    # # for a normalized gaussian distribution, u should multiply this coeff
    # coeff = 1/np.sqrt(2*sigma**2*np.pi) if normalize else 1.0


    d2r = np.pi/180 # degree to rad
    a0 = -20.0*d2r
    a1 = np.pi + 20.0*d2r
    nx = int((a1-a0)/dgrid) + 1

    ys = fget_sbot(coords, zs, z1, z2, z3, rcut, nx, dgrid, sigma)


    return ys
