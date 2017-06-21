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

import ase
import scipy.spatial.distance as ssd
import itertools as itl
import numpy as np

from .data import NUCLEAR_CHARGE

def get_pbc(m, d0 = 3.6):

    pbc = []

    c = m.cell
    ps = m.positions
    na = len(m); idxs = np.arange(na)

    for ii in range(3):
        psx = ps[:,ii]; xmin = min(psx)
        idxs_i = idxs[ psx == xmin ]
        ps1 = ps[idxs_i[0]] + c[ii]
        if np.min( ssd.cdist([ps1,], ps)[0] ) < d0:
            pbc.append( '1' )
        else:
            pbc.append( '0' )

    return ''.join(pbc)

def update_m(m, ia, rcut=9.0, pbc=None):
    # """
    # retrieve local structure around atom `ia
    # for periodic systems (or very large system)
    # """

    c = m.cell
    v1, v2, v3 = c
    ls = ssd.norm(c, axis=0)

    nns = []; ns = []
    for i,li in enumerate(ls):
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
    #print ' -- n123s = ', n123s

    coords = m.positions; zs = m.numbers; ai = m[ia]; cia = coords[ia]
    na = len(m)
    if na == 1:
        ds = np.array([[0.]])
    else:
        ds = ssd.squareform( ssd.pdist(coords) )

    idxs0 = []

    mu = ase.Atoms([], cell=c); mu.append( ai ); idxs0.append( ia )
    for i in range(na) :
        di = ds[i,ia]
        if di <= rcut:
            if di > 0:
                mu.append( m[i] ); idxs0.append( i )

            # add new coords by translation
            ts = np.zeros((nau,3))
            for iau in range(nau):
                ts[iau] = np.dot(n123s[iau],c)

            coords_iu = coords[i] + ts
            dsi = ssd.norm( coords_iu - cia, axis=1);
            filt = np.logical_and(dsi > 0, dsi <= rcut); nx = filt.sum()
            mii = ase.Atoms([zs[i],]*nx, coords_iu[filt,:])
            for aii in mii: mu.append( aii ); idxs0.append( i )

    return mu, idxs0


def get_boa(z1, zs_):
    return z1*np.array( [(zs_ == z1).sum(), ])

def get_sbop(mbtype, m, zsm, iloc=False, ia=None, normalize=True, sigma=0.05, \
             rcut=4.8, dgrid=0.03, ipot=True, pbc='000', rpower=6):
    # """
    # two-body terms
    # """

    z1, z2 = mbtype

    if iloc:
        assert ia != None, '#ERROR: plz specify `za and `ia '

    if pbc != '000':
        if rcut < 9.0: raise '#ERROR: rcut too small for systems with pbc'
        assert iloc, '#ERROR: for periodic system, plz use atomic rpst'
        m, idxs0 = update_m(m, ia, rcut=rcut, pbc=pbc)
        zsmu = [ zsm[i] for i in idxs0 ]; zsm = zsmu

        # after update of `m, the query atom `ia will become the first atom
        ia = 0

    na = len(m)
    coords = m.positions
    ds = ssd.squareform( ssd.pdist(coords) )

    ias = np.arange(na)
    ias1 = ias[zsm == z1]
    ias2 = ias[zsm == z2]

    if z1 == z2:
        ias12 = list( itl.combinations(ias1,2) )
    else:
        ias12 = itl.product(ias1,ias2)

    if iloc:
        dsu = []; icnt = 0
        for j1,j2 in ias12:
            if ia == j1 or ia == j2:
                dsu.append( ds[j1,j2] )
            icnt += 1
    else:
        dsu = [ ds[i,j] for (i,j) in ias12 ]

    dsu = np.array(dsu)

    #print ' -- (d_min, d_max) = (%.3f, %.3f)'%(np.min(ds), np.max(ds))

    # bop potential distribution
    r0 = 0.1
    nx = (rcut - r0)/dgrid + 1
    xs = np.linspace(r0, rcut, nx)
    ys0 = np.zeros(xs.shape)

    # update dsu by exluding d > 6.0
    nr = dsu.shape[0]
    if nr > 0:
        dsu = dsu[ dsu <= rcut ]
        nr = len(dsu)

    #print ' -- dsu = ', dsu

    coeff = 1/np.sqrt(2*sigma**2*np.pi) if normalize else 1.0
    #print ' -- now calculating 2-body terms...'
    if ipot:
        # get distribution of 2-body potentials
        # unit of x: Angstrom
        c0 = (z1%1000)*(z2%1000)*coeff
        ys = ys0
        for i in range(nr):
            ys += ( c0/(xs**rpower) )*np.exp( -0.5*((xs-dsu[i])/sigma)**2 )
        ys *= dgrid
    else:
        # print distribution of distances
        c0 = coeff
        ys = ys0
        for i in range(nr):
            ys += c0*np.exp( -0.5*((xs-dsu[i])/sigma)**2 )

    return xs, ys

def vang(u,v):
    cost = np.dot(u,v)/(np.linalg.norm(u) * np.linalg.norm(v))
    # sometimes, cost might be 1.00000000002, then np.arccos(cost)
    # does not exist!
    u = cost if abs(cost) <= 1 else 1.0
    return np.arccos( u )

def cvang(u,v):
    return np.dot(u,v)/np.sqrt(np.dot(u,u)*np.dot(v,v))

def get_sbot(mbtype, m, zsm, iloc=False, ia=None, normalize=True, sigma=0.05, label=None, \
             rcut=4.8, dgrid=0.0262, ipot=True, pbc='000'):
    # """
    # sigma -- standard deviation of gaussian distribution centered on a specific angle
    #         defaults to 0.05 (rad), approximately 3 degree
    # dgrid    -- step of angle grid
    #         defaults to 0.0262 (rad), approximately 1.5 degree
    # """

    z1, z2, z3 = mbtype

    if iloc:
        assert ia != None, '#ERROR: plz specify `za and `ia '

    if pbc != '000':
        assert iloc, '#ERROR: for periodic system, plz use atomic rpst'
        m, idxs0 = update_m(m, ia, rcut=rcut, pbc=pbc)
        zsm = [ zsm[i] for i in idxs0 ]

        # after update of `m, the query atom `ia will become the first atom
        ia = 0

    na = len(m)
    coords = m.positions
    ds = ssd.squareform( ssd.pdist(coords) )
    #get_date(' ds matrix calc done ')

    ias = np.arange(na)
    ias1 = ias[zsm == z1]; n1 = len(ias1)
    ias2 = ias[zsm == z2]; n2 = len(ias2)
    ias3 = ias[zsm == z3]; n3 = len(ias3)
    tas = []

    for ia1 in ias1:
        ias2u = ias2[ np.logical_and( ds[ia1,ias2] > 0, ds[ia1,ias2] <= rcut ) ]
        for ia2 in ias2u:
            filt1 = np.logical_and( ds[ia1,ias3] > 0, ds[ia1,ias3] <= rcut )
            filt2 = np.logical_and( ds[ia2,ias3] > 0, ds[ia2,ias3] <= rcut )
            ias3u = ias3[ np.logical_and(filt1, filt2) ]
            for ia3 in ias3u:
                tasi = [ia1,ia2,ia3]
                iok1 = (tasi not in tas)
                iok2 = (tasi[::-1] not in tas)
                if iok1 and iok2:
                    tas.append( tasi )

    if iloc:
        tas_u = []
        for tas_i in tas:
            if ia == tas_i[1]:
                tas_u.append( tas_i )
        tas = tas_u

    d2r = np.pi/180 # degree to rad
    a0 = -20.0*d2r; a1 = np.pi + 20.0*d2r
    nx = int((a1-a0)/dgrid) + 1
    xs = np.linspace(a0, a1, nx)
    ys0 = np.zeros(nx, np.float)
    nt = len(tas)

    # u actually have considered the same 3-body term for
    # three times, so rescale it
    prefactor = 1.0/3

    # for a normalized gaussian distribution, u should multiply this coeff
    coeff = 1/np.sqrt(2*sigma**2*np.pi) if normalize else 1.0

    tidxs = np.array(tas, np.int)
    if ipot:
        # get distribution of 3-body potentials
        # unit of x: Angstrom
        c0 = prefactor*(z1%1000)*(z2%1000)*(z3%1000)*coeff
        ys = ys0
        for it in range(nt):
            i,j,k = tas[it]
            # angle spanned by i <-- j --> k, i.e., vector ji and jk
            u = coords[i]-coords[j]; v = coords[k] - coords[j]
            ang = vang( u, v ) # ang_j
            #print ' -- (i,j,k) = (%d,%d,%d),  ang = %.2f'%(i,j,k, ang)
            cak = cvang( coords[j]-coords[k], coords[i]-coords[k] ) # cos(ang_k)
            cai = cvang( coords[k]-coords[i], coords[j]-coords[i] ) # cos(ang_i)
            ys += c0*( (1.0 + 1.0*np.cos(xs)*cak*cai)/(ds[i,j]*ds[i,k]*ds[j,k])**3 )*\
                                ( np.exp(-(xs-ang)**2/(2*sigma**2)) )
        ys *= dgrid
    else:
        # print distribution of angles (unit: degree)
        sigma = sigma/d2r
        xs = xs/d2r
        c0 = 1

        ys = ys0
        for it in range(nt):
            i,j,k = tas[it]
            # angle spanned by i <-- j --> k, i.e., vector ji and jk
            ang = vang( coords[i]-coords[j], coords[k]-coords[j] )/d2r
            ys += c0*np.exp( -(xs-ang)**2/(2*sigma**2) )

    return xs, ys
