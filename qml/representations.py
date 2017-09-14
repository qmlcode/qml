# MIT License
#
# Copyright (c) 2017 Anders Steen Christensen, Lars Andersen Bratholm and Bing Huang
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

from .data import NUCLEAR_CHARGE

from .slatm import get_boa
from .slatm import get_sbop
from .slatm import get_sbot

def vector_to_matrix(v):
    """ Converts a representation from 1D vector to 2D square matrix.
    :param v: 1D input representation.
    :type v: numpy array 
    :return: Square matrix representation.
    :rtype: numpy array 
    """

    if not (np.sqrt(8*v.shape[0]+1) == int(np.sqrt(8*v.shape[0]+1))):
        print("ERROR: Can not make a square matrix.")
        exit(1)

    n = v.shape[0]
    l = (-1 + int(np.sqrt(8*n+1)))//2
    M = np.empty((l,l))

    index = 0
    for i in range(l):
        for j in range(l):
            if j > i:
                continue

            M[i,j] = v[index]
            M[j,i] = M[i,j]

            index += 1
    return M

def generate_coulomb_matrix(nuclear_charges, coordinates, size = 23, sorting = "row-norm"):
    """ Creates a Coulomb Matrix representation of a molecule.
        Sorting of the elements can either be done by ``sorting="row-norm"`` or ``sorting="unsorted"``.
        A matrix :math:`M` is constructed with elements

        .. math::

            M_{ij} =
              \\begin{cases}
                 \\tfrac{1}{2} Z_{i}^{2.4} & \\text{if } i = j \\\\
                 \\frac{Z_{i}Z_{j}}{\\| {\\bf R}_{i} - {\\bf R}_{j}\\|}       & \\text{if } i \\neq j
              \\end{cases},

        where :math:`i` and :math:`j` are atom indices, :math:`Z` is nuclear charge and
        :math:`\\bf R` is the coordinate in euclidean space.
        If ``sorting = 'row-norm'``, the atom indices are reordered such that

            :math:`\\sum_j M_{1j}^2 \\geq \\sum_j M_{2j}^2 \\geq ... \\geq \\sum_j M_{nj}^2`

        The upper triangular of M, including the diagonal, is concatenated to a 1D
        vector representation.

        If ``sorting = 'unsorted``, the elements are sorted in the same order as the input coordinates
        and nuclear charges.

        The representation is calculated using an OpenMP parallel Fortran routine.

        :param nuclear_charges: Nuclear charges of the atoms in the molecule
        :type nuclear_charges: numpy array
        :param coordinates: 3D Coordinates of the atoms in the molecule
        :type coordinates: numpy array
        :param size: The size of the largest molecule supported by the representation
        :type size: integer
        :param sorting: How the atom indices are sorted ('row-norm', 'unsorted')
        :type sorting: string

        :return: 1D representation - shape (size(size+1)/2,)
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

def generate_atomic_coulomb_matrix(nuclear_charges, coordinates, size = 23, sorting = "distance",
            central_cutoff = 1e6, central_decay = -1, interaction_cutoff = 1e6, interaction_decay = -1,
            indices = None):
    """ Creates a Coulomb Matrix representation of the local environment of a central atom.
        For each central atom :math:`k`, a matrix :math:`M` is constructed with elements

        .. math::

            M_{ij}(k) =
              \\begin{cases}
                 \\tfrac{1}{2} Z_{i}^{2.4} \\cdot f_{ik}^2 & \\text{if } i = j \\\\
                 \\frac{Z_{i}Z_{j}}{\\| {\\bf R}_{i} - {\\bf R}_{j}\\|} \\cdot f_{ik}f_{jk}f_{ij} & \\text{if } i \\neq j
              \\end{cases},

        where :math:`i`, :math:`j` and :math:`k` are atom indices, :math:`Z` is nuclear charge and
        :math:`\\bf R` is the coordinate in euclidean space.

        :math:`f_{ij}` is a function that masks long range effects:

        .. math::

            f_{ij} =
              \\begin{cases}
                 1 & \\text{if } \\|{\\bf R}_{i} - {\\bf R}_{j} \\| \\leq r - \Delta r \\\\
                 \\tfrac{1}{2} \\big(1 + \\cos\\big(\\pi \\tfrac{\\|{\\bf R}_{i} - {\\bf R}_{j} \\|
                    - r + \Delta r}{\Delta r} \\big)\\big)     
                    & \\text{if } r - \Delta r < \\|{\\bf R}_{i} - {\\bf R}_{j} \\| \\leq r - \Delta r \\\\
                 0 & \\text{if } \\|{\\bf R}_{i} - {\\bf R}_{j} \\| > r
              \\end{cases},

        where the parameters ``central_cutoff`` and ``central_decay`` corresponds to the variables
        :math:`r` and :math:`\Delta r` respectively for interactions involving the central atom,
        and ``interaction_cutoff`` and ``interaction_decay`` corresponds to the variables
        :math:`r` and :math:`\Delta r` respectively for interactions not involving the central atom.

        if ``sorting = 'row-norm'``, the atom indices are ordered such that

            :math:`\\sum_j M_{1j}(k)^2 \\geq \\sum_j M_{2j}(k)^2 \\geq ... \\geq \\sum_j M_{nj}(k)^2`

        if ``sorting = 'distance'``, the atom indices are ordered such that

        .. math::

            \\|{\\bf R}_{1} - {\\bf R}_{k}\\| \\leq \\|{\\bf R}_{2} - {\\bf R}_{k}\\|
                \\leq ... \\leq \\|{\\bf R}_{n} - {\\bf R}_{k}\\|

        The upper triangular of M, including the diagonal, is concatenated to a 1D
        vector representation.

        The representation can be calculated for a subset by either specifying
        ``indices = [0,1,...]``, where :math:`[0,1,...]` are the requested atom indices,
        or by specifying ``indices = 'C'`` to only calculate central carbon atoms.

        The representation is calculated using an OpenMP parallel Fortran routine.

        :param nuclear_charges: Nuclear charges of the atoms in the molecule
        :type nuclear_charges: numpy array
        :param coordinates: 3D Coordinates of the atoms in the molecule
        :type coordinates: numpy array
        :param size: The size of the largest molecule supported by the representation
        :type size: integer
        :param sorting: How the atom indices are sorted ('row-norm', 'distance')
        :type sorting: string
        :param central_cutoff: The distance from the central atom, where the coulomb interaction
            element will be zero
        :type central_cutoff: float
        :param central_decay: The distance over which the the coulomb interaction decays from full to none
        :type central_decay: float
        :param interaction_cutoff: The distance between two non-central atom, where the coulomb interaction
            element will be zero
        :type interaction_cutoff: float
        :param interaction_decay: The distance over which the the coulomb interaction decays from full to none
        :type interaction_decay: float
        :param indices: Subset indices or atomtype
        :type indices: Nonetype/array/string


        :return: nD representation - shape (:math:`N_{atoms}`, size(size+1)/2)
        :rtype: numpy array
    """


    if indices == None:
        nindices = len(nuclear_charges)
        indices = np.arange(1,1+nindices, 1, dtype = int)
    elif type("") == type(indices):
        if indices in NUCLEAR_CHARGE:
            indices = np.where(nuclear_charges == NUCLEAR_CHARGE[indices])[0] + 1
            nindices = indices.size
            if nindices == 0:
                return np.zeros((0,0))

        else:
            print("ERROR: Unknown value %s given for 'indices' variable" % indices)
            raise SystemExit
    else:
        indices = np.asarray(indices, dtype = int) + 1
        nindices = indices.size


    if (sorting == "row-norm"):
        return fgenerate_local_coulomb_matrix(indices, nindices, nuclear_charges,
            coordinates, nuclear_charges.size, size,
            central_cutoff, central_decay, interaction_cutoff, interaction_decay)

    elif (sorting == "distance"):
        return fgenerate_atomic_coulomb_matrix(indices, nindices, nuclear_charges,
            coordinates, nuclear_charges.size, size, 
            central_cutoff, central_decay, interaction_cutoff, interaction_decay)

    else:
        print("ERROR: Unknown sorting scheme requested")
        raise SystemExit

def generate_eigenvalue_coulomb_matrix(nuclear_charges, coordinates, size = 23):
    """ Creates an eigenvalue Coulomb Matrix representation of a molecule.
        A matrix :math:`M` is constructed with elements

        .. math::

            M_{ij} =
              \\begin{cases}
                 \\tfrac{1}{2} Z_{i}^{2.4} & \\text{if } i = j \\\\
                 \\frac{Z_{i}Z_{j}}{\\| {\\bf R}_{i} - {\\bf R}_{j}\\|}       & \\text{if } i \\neq j
              \\end{cases},

        where :math:`i` and :math:`j` are atom indices, :math:`Z` is nuclear charge and
        :math:`\\bf R` is the coordinate in euclidean space.
        The molecular representation of the molecule is then the sorted eigenvalues of M.
        The representation is calculated using an OpenMP parallel Fortran routine.

        :param nuclear_charges: Nuclear charges of the atoms in the molecule
        :type nuclear_charges: numpy array
        :param coordinates: 3D Coordinates of the atoms in the molecule
        :type coordinates: numpy array
        :param size: The size of the largest molecule supported by the representation
        :type size: integer

        :return: 1D representation - shape (size, )
        :rtype: numpy array
    """
    return fgenerate_eigenvalue_coulomb_matrix(nuclear_charges,
        coordinates, size)

def generate_bob(nuclear_charges, coordinates, atomtypes, size=23, asize = {"O":3, "C":7, "N":3, "H":16, "S":1}):
    """ Creates a Bag of Bonds (BOB) representation of a molecule.
        The representation expands on the coulomb matrix representation.
        For each element a bag (vector) is constructed for self interactions
        (e.g. ('C', 'H', 'O')).
        For each element pair a bag is constructed for interatomic interactions
        (e.g. ('CC', 'CH', 'CO', 'HH', 'HO', 'OO')), sorted by value.
        The self interaction of element :math:`I` is given by

            :math:`\\tfrac{1}{2} Z_{I}^{2.4}`,

        with :math:`Z_{i}` being the nuclear charge of element :math:`i`
        The interaction between atom :math:`i` of element :math:`I` and 
        atom :math:`j` of element :math:`J` is given by

            :math:`\\frac{Z_{I}Z_{J}}{\\| {\\bf R}_{i} - {\\bf R}_{j}\\|}`

        with :math:`R_{i}` being the euclidean coordinate of atom :math:`i`.
        The sorted bags are concatenated to an 1D vector representation.
        The representation is calculated using an OpenMP parallel Fortran routine.

        :param nuclear_charges: Nuclear charges of the atoms in the molecule
        :type nuclear_charges: numpy array
        :param coordinates: 3D Coordinates of the atoms in the molecule
        :type coordinates: numpy array
        :param size: The maximum number of atoms in the representation
        :type size: integer
        :param asize: The maximum number of atoms of each element type supported by the representation
        :type asize: dictionary

        :return: 1D representation
        :rtype: numpy array
    """
    natoms = len(nuclear_charges)

    coulomb_matrix = fgenerate_unsorted_coulomb_matrix(nuclear_charges, coordinates, size)

    coulomb_matrix = vector_to_matrix(coulomb_matrix)
    descriptor = []
    atomtypes = np.asarray(atomtypes)
    for atom1, size1 in sorted(asize.items()):
        pos1 = np.where(atomtypes == atom1)[0]
        feature_vector = np.zeros(size1)
        feature_vector[:pos1.size] = np.diag(coulomb_matrix)[pos1]
        feature_vector.sort()
        descriptor.append(feature_vector[:])
        for atom2, size2 in sorted(asize.items()):
            if atom1 > atom2:
                continue
            if atom1 == atom2:
                size = size1*(size1-1)//2
                feature_vector = np.zeros(size)
                sub_matrix = coulomb_matrix[np.ix_(pos1,pos1)]
                feature_vector[:pos1.size*(pos1.size-1)//2] = sub_matrix[np.triu_indices(pos1.size, 1)]
                feature_vector.sort()
                descriptor.append(feature_vector[:])
            else:
                pos2 = np.where(atomtypes == atom2)[0]
                feature_vector = np.zeros(size1*size2)
                feature_vector[:pos1.size*pos2.size] = coulomb_matrix[np.ix_(pos1,pos2)].ravel()
                feature_vector.sort()
                descriptor.append(feature_vector[:])

    return np.concatenate(descriptor)

    n = 0
    atoms = sorted(asize, key=asize.get)
    nmax = [asize[key] for key in atoms]
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
                                    pbc=pbc, rpower=rpower)
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
                                    sigma=sigmas[1], dgrid=dgrids[1], rcut=rcut, pbc=pbc)

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
                                dgrid=dgrids[0], rcut=rcut, rpower=rpower)


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
                        dgrid=dgrids[1], rcut=rcut)

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
