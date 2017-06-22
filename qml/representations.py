# MIT License
#
# Copyright (c) 2017 Anders Steen Christensen, Lars A. Bratholm and Bing Huang
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
import itertools as itl

from .frepresentations import fgenerate_coulomb_matrix
from .frepresentations import fgenerate_unsorted_coulomb_matrix
from .frepresentations import fgenerate_local_coulomb_matrix
from .frepresentations import fgenerate_atomic_coulomb_matrix
from .frepresentations import fgenerate_eigenvalue_coulomb_matrix
from .frepresentations import fgenerate_bob

from .data import NUCLEAR_CHARGE

from .slatm import get_boa
from .slatm import get_sbop
from .slatm import get_sbot

def generate_coulomb_matrix(nuclear_charges, coordinates, size = 23, sorting = "row-norm"):
    """ Generates a sorted molecular coulomb, sort either by ``"row-norm"`` or ``"unsorted"``.
    ``size=`` denotes the max number of atoms in the molecule (thus the size of the resulting square matrix.
    The resulting matrix is the upper triangle put into the form of a 1D-vector.

    :param coordinates: Input coordinates.
    :type coordinates: numpy array
    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param size: Max number of atoms in representation.
    :type size: integer
    :param sorting: Matrix sorting scheme, "row-norm" or "unsorted".
    :type sorting: string
    :return: 1D Coulomb matrix representation
    :rtype: numpy array
    """

    if (sorting == "row-norm"):
        return fgenerate_coulomb_matrix(nuclear_charges, \
            coordinates, size)

    elif (sorting == "unsorted"):
        return fgenerate_unsorted_coulomb_matrix(nuclear_charges, \
            coordinates, size)

    else:
        print("ERROR: Unknown sorting scheme requested")
        raise SystemExit

def generate_atomic_coulomb_matrix(nuclear_charges, coordinates, size = 23, sorting = "distance"):
    """ Generates a list of sorted Coulomb matrices, sorted either by ``"row-norm"`` or ``"distance"``, the latter refers to sorting by distance to each query atom.
    ``size=`` denotes the max number of atoms in the molecule (thus the size of the resulting square matrix.
    The resulting matrix is the upper triangle put into the form of a 1D-vector.

    :param coordinates: Input coordinates.
    :type coordinates: numpy array
    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param size: Max number of atoms in representation.
    :type size: integer
    :param sorting: Matrix sorting scheme, "row-norm" or "distance".
    :type sorting: string
    :return: List of 1D Coulomb matrix representations.
    :rtype: numpy array
    """

    if (sorting == "row-norm"):
        return fgenerate_local_coulomb_matrix(nuclear_charges,
            coordinates, nuclear_charges.size, size)

    elif (sorting == "distance"):
        return fgenerate_atomic_coulomb_matrix(nuclear_charges,
            coordinates, nuclear_charges.size, size)

    else:
        print("ERROR: Unknown sorting scheme requested")
        raise SystemExit

def generate_eigenvalue_coulomb_matrix(nuclear_charges, coordinates, size = 23):
    """ Generates the eigenvalue-Coulomb matrix representation.
    ``size=`` denotes the max number of atoms in the molecule (thus the size of the resulting square matrix.
    The resulting matrix is in the form of a 1D-vector.

    :param coordinates: Input coordinates.
    :type coordinates: numpy array
    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param size: Max number of atoms in representation.
    :type size: integer
    :return: 1D representation.
    :rtype: numpy array
    """
    return fgenerate_eigenvalue_coulomb_matrix(nuclear_charges,
        coordinates, size)

def generate_bob(nuclear_charges, coordinates, atomtypes, asize = {"O":3, "C":7, "N":3, "H":16, "S":1}):
    """ Generates a bag-of-bonds (BOB) representation of the molecule. ``size=`` denotes the max number of atoms in the molecule (thus relates to the size of the resulting matrix.)
    ``asize=`` is the maximum number of atoms of each type (necessary to generate bags of minimal sizes).
    The resulting matrix is the BOB representation put into the form of a 1D-vector.

    :param coordinates: Input coordinates.
    :type coordinates: numpy array
    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param size: Max number of atoms in representation.
    :type size: integer
    :param asize: Max number of each element type.
    :type asize: dict
    :return: 1D BOB representation.
    :rtype: numpy array
    """

    n = 0
    atoms = sorted(asize, key=asize.get)
    nmax = [asize[key] for key in atoms]
    # print(atoms,nmax)
    ids = np.zeros(len(nmax), dtype=int)
    for i, (key, value) in enumerate(zip(atoms,nmax)):
        n += value * (1+value)
        ids[i] = NUCLEAR_CHARGE[key]
        for j in range(i):
            v = nmax[j]
            n += 2 * value * v
    n /= 2

    return fgenerate_bob(nuclear_charges, coordinates, nuclear_charges, ids, nmax, n)


def get_slatm_mbtypes(nuclear_charges, pbc='000'):
    """
    Get the list of minimal types of many-body terms in a dataset. This resulting list
    is necessary as input in the ``generate_slatm_representation()`` function.

    :param nuclear_charges: A list of the nuclear charges for each compound in the dataset.
    :type nuclear_charges: list of numpy arrays
    :param pbc: periodic boundary condition along x,y,z direction, defaulted to '000', i.e., molecule
    :type pbc: string
    :return: A list containing the types of many-body terms.
    :rtype: list
    """

    zs = nuclear_charges

    nm = len(zs)
    zsmax = set()
    nas = []
    zs_ravel = []
    for zsi in zs:
        na = len(zsi); nas.append(na)
        zsil = list(zsi); zs_ravel += zsil
        zsmax.update( zsil )

    zsmax = np.array( list(zsmax) )
    nass = []
    for i in range(nm):
        zsi = np.array(zs[i],np.int)
        nass.append( [ (zi == zsi).sum() for zi in zsmax ] )

    nzmax = np.max(np.array(nass), axis=0)
    nzmax_u = []
    if pbc != '000':
        # the PBC will introduce new many-body terms, so set
        # nzmax to 3 if it's less than 3
        for nzi in nzmax:
            if nzi <= 2:
                nzi = 3
            nzmax_u.append(nzi)
        nzmax = nzmax_u

    boas = [ [zi,] for zi in zsmax ]
    bops = [ [zi,zi] for zi in zsmax ] + list( itl.combinations(zsmax,2) )

    bots = []
    for i in zsmax:
        for bop in bops:
            j,k = bop
            tas = [ [i,j,k], [i,k,j], [j,i,k] ]
            for tasi in tas:
                if (tasi not in bots) and (tasi[::-1] not in bots):
                    nzsi = [ (zj == tasi).sum() for zj in zsmax ]
                    if np.all(nzsi <= nzmax):
                        bots.append( tasi )
    mbtypes = boas + bops + bots

    return mbtypes #, np.array(zs_ravel), np.array(nas)


def generate_slatm(coordinates, nuclear_charges, mbtypes,
        unit_cell=None, local=False, sigmas=[0.05,0.05], dgrids=[0.03,0.03],
        rcut=4.8, alchemy=False, pbc='000', rpower=6):
    """
    Generate Spectrum of London and Axillrod-Teller-Muto potential (SLATM) representation.
    Both global (``local=False``) and local (``local=True``) SLATM are available.

    A version that works for periodic boundary conditions will be released soon.

    NOTE: You will need to run the ``get_slatm_mbtypes()`` function to get the ``mbtypes`` input (or generate it manually).

    :param coordinates: Input coordinates
    :type coordinates: numpy array
    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param mbtypes: Many-body types for the whole dataset, including 1-, 2- and 3-body types. Could be obtained by calling ``get_slatm_mbtypes()``.
    :type mbtypes: list
    :param local: Generate a local representation. Defaulted to False (i.e., global representation); otherwise, atomic version.
    :type local: bool
    :param sigmas: Controlling the width of Gaussian smearing function for 2- and 3-body parts, defaulted to [0.05,0.05], usually these do not need to be adjusted.
    :type sigmas: list
    :param dgrids: The interval between two sampled internuclear distances and angles, defaulted to [0.03,0.03], no need for change, compromised for speed and accuracy.
    :type dgrids: list
    :param rcut: Cut-off radius, defaulted to 4.8 Angstrom.
    :type rcut: float
    :param alchemy: Swith to use the alchemy version of SLATM. (default=False)
    :type alchemy: bool
    :param pbc: defaulted to '000', meaning it's a molecule; the three digits in the string corresponds to x,y,z direction
    :type pbc: string
    :param rpower: The power of R in 2-body potential, defaulted to London potential (=6).
    :type rpower: float
    :return: 1D SLATM representation
    :rtype: numpy array
    """

    c = unit_cell
    iprt=False
    if c is None:
        c = np.array([[1,0,0],[0,1,0],[0,0,1]])

    if pbc != '000':
        # print(' -- handling systems with periodic boundary condition')
        assert c != None, 'ERROR: Please specify unit cell for SLATM'
        # =======================================================================
        # PBC may introduce new many-body terms, so at the stage of get statistics
        # info from db, we've already considered this point by letting maximal number
        # of nuclear charges being 3.
        # =======================================================================

    zs = nuclear_charges
    na = len(zs)
    coords = coordinates
    obj = [ zs, coords, c ]

    iloc = local

    if iloc:
        mbs = []
        X2Ns = []
        for ia in range(na):
            # if iprt: print '               -- ia = ', ia + 1
            n1 = 0; n2 = 0; n3 = 0
            mbs_ia = np.zeros(0)
            icount = 0
            for mbtype in mbtypes:
                if len(mbtype) == 1:
                    mbsi = get_boa(mbtype[0], np.array([zs[ia],]))
                    #print ' -- mbsi = ', mbsi
                    if alchemy:
                        n1 = 1
                        n1_0 = mbs_ia.shape[0]
                        if n1_0 == 0:
                            mbs_ia = np.concatenate( (mbs_ia, mbsi), axis=0 )
                        elif n1_0 == 1:
                            mbs_ia += mbsi
                        else:
                            raise '#ERROR'
                    else:
                        n1 += len(mbsi)
                        mbs_ia = np.concatenate( (mbs_ia, mbsi), axis=0 )
                elif len(mbtype) == 2:
                    #print ' 001, pbc = ', pbc
                    mbsi = get_sbop(mbtype, obj, iloc=iloc, ia=ia, \
                                    sigma=sigmas[0], dgrid=dgrids[0], rcut=rcut, \
                                    pbc=pbc, rpower=rpower)[1]
                    mbsi *= 0.5 # only for the two-body parts, local rpst
                    #print ' 002'
                    if alchemy:
                        n2 = len(mbsi)
                        n2_0 = mbs_ia.shape[0]
                        if n2_0 == n1:
                            mbs_ia = np.concatenate( (mbs_ia, mbsi), axis=0 )
                        elif n2_0 == n1 + n2:
                            t = mbs_ia[n1:n1+n2] + mbsi
                            mbs_ia[n1:n1+n2] = t
                        else:
                            raise '#ERROR'
                    else:
                        n2 += len(mbsi)
                        mbs_ia = np.concatenate( (mbs_ia, mbsi), axis=0 )
                else: # len(mbtype) == 3:
                    mbsi = get_sbot(mbtype, obj, iloc=iloc, ia=ia, \
                                    sigma=sigmas[1], dgrid=dgrids[1], rcut=rcut, pbc=pbc)[1]
                    if alchemy:
                        n3 = len(mbsi)
                        n3_0 = mbs_ia.shape[0]
                        if n3_0 == n1 + n2:
                            mbs_ia = np.concatenate( (mbs_ia, mbsi), axis=0 )
                        elif n3_0 == n1 + n2 + n3:
                            t = mbs_ia[n1+n2:n1+n2+n3] + mbsi
                            mbs_ia[n1+n2:n1+n2+n3] = t
                        else:
                            raise '#ERROR'
                    else:
                        n3 += len(mbsi)
                        mbs_ia = np.concatenate( (mbs_ia, mbsi), axis=0 )

            mbs.append( mbs_ia )
            X2N = [n1,n2,n3];
            if X2N not in X2Ns:
                X2Ns.append(X2N)
        assert len(X2Ns) == 1, '#ERROR: multiple `X2N ???'
    else:
        n1 = 0; n2 = 0; n3 = 0
        mbs = np.zeros(0)
        for mbtype in mbtypes:
            if len(mbtype) == 1:
                mbsi = get_boa(mbtype[0], zs)
                if alchemy:
                    n1 = 1
                    n1_0 = mbs.shape[0]
                    if n1_0 == 0:
                        mbs = np.concatenate( (mbs, [sum(mbsi)] ), axis=0 )
                    elif n1_0 == 1:
                        mbs += sum(mbsi )
                    else:
                        raise '#ERROR'
                else:
                    n1 += len(mbsi)
                    mbs = np.concatenate( (mbs, mbsi), axis=0 )
            elif len(mbtype) == 2:
                mbsi = get_sbop(mbtype, obj, sigma=sigmas[0], \
                                dgrid=dgrids[0], rcut=rcut, rpower=rpower)[1]

                if alchemy:
                    n2 = len(mbsi)
                    n2_0 = mbs.shape[0]
                    if n2_0 == n1:
                        mbs = np.concatenate( (mbs, mbsi), axis=0 )
                    elif n2_0 == n1 + n2:
                        t = mbs[n1:n1+n2] + mbsi
                        mbs[n1:n1+n2] = t
                    else:
                        raise '#ERROR'
                else:
                    n2 += len(mbsi)
                    mbs = np.concatenate( (mbs, mbsi), axis=0 )
            else: # len(mbtype) == 3:
                mbsi = get_sbot(mbtype, obj, sigma=sigmas[1], \
                        dgrid=dgrids[1], rcut=rcut)[1]
                if alchemy:
                    n3 = len(mbsi)
                    n3_0 = mbs.shape[0]
                    if n3_0 == n1 + n2:
                        mbs = np.concatenate( (mbs, mbsi), axis=0 )
                    elif n3_0 == n1 + n2 + n3:
                        t = mbs[n1+n2:n1+n2+n3] + mbsi
                        mbs[n1+n2:n1+n2+n3] = t
                    else:
                        raise '#ERROR'
                else:
                    n3 += len(mbsi)
                    mbs = np.concatenate( (mbs, mbsi), axis=0 )

    return mbs

