# MIT License
#
# Copyright (c) 2016 Anders Steen Christensen
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

NUCLEAR_CHARGE = dict()
NUCLEAR_CHARGE["H"] = 1.0
NUCLEAR_CHARGE["C"] = 6.0
NUCLEAR_CHARGE["N"] = 7.0
NUCLEAR_CHARGE["O"] = 8.0
NUCLEAR_CHARGE["F"] = 9.0
NUCLEAR_CHARGE["Si"] = 14.0
NUCLEAR_CHARGE["P"] = 15.0
NUCLEAR_CHARGE["S"] = 16.0
NUCLEAR_CHARGE["Ge"] = 32.0

# from frepresentations import fgenerate_coulomb_matrix
# import frepresentations

def triangular_to_vector(M, uplo="U"):

    if not (M.shape[0] == M.shape[1]):
        print "ERROR: Not a square matrix."
        exit(1)

    n = M.shape[0]
    l = (n + 1) * n // 2 # Explicit integer division
    v = np.empty((l))

    index = 0
    for i in range(n):
        for j in range(n):
            
            if j > i:
                continue

            v[index] = M[i, j]

            index += 1

    return v


def generate_coulomb_matrix(atomtypes, coordinates, size=23):

    # Generate row norms for sorting

    row_norms = []

    for i, atomtype_i in enumerate(atomtypes):

        row_norm = 0.0

        for j, atomtype_j in enumerate(atomtypes):

            if i == j:
                row_norm += 0.5 * NUCLEAR_CHARGE[atomtype_i] ** 2.4

            else:
                row_norm += NUCLEAR_CHARGE[atomtype_i] * NUCLEAR_CHARGE[atomtype_j] \
                            / np.linalg.norm(coordinates[i] - coordinates[j])

        row_norms.append((row_norm, i))

    # Sort by row norms

    row_norms.sort(reverse=True)

    sorted_atomtypes = []
    sorted_coordinates = []

    for row_norm in row_norms:
        i = row_norm[1]

        sorted_atomtypes.append(atomtypes[i])
        sorted_coordinates.append(coordinates[i])

    # Fill out 

    Mij = np.zeros((size, size))
    for i, atomtype_i in enumerate(sorted_atomtypes):
        for j, atomtype_j in enumerate(sorted_atomtypes):
            if i == j:
                Mij[i, j] = 0.5 * NUCLEAR_CHARGE[atomtype_i] ** 2.4

            elif j > i:
                continue

            else:
                Mij[i, j] = NUCLEAR_CHARGE[atomtype_i] * NUCLEAR_CHARGE[atomtype_j] \
                            / np.linalg.norm(sorted_coordinates[i] - sorted_coordinates[j])

    # return Mij.flatten(size**2)
    return triangular_to_vector(Mij)



def generate_local_coulomb_matrix(atomtypes, coordinates, size=23, calc="all"):

    # Generate row norms for sorting

    row_norms = []

    for i, atomtype_i in enumerate(atomtypes):

        row_norm = 0.0

        for j, atomtype_j in enumerate(atomtypes):

            if i == j:
                row_norm += 0.5 * NUCLEAR_CHARGE[atomtype_i] ** 2.4

            else:
                row_norm += NUCLEAR_CHARGE[atomtype_i] * NUCLEAR_CHARGE[atomtype_j] \
                            / np.linalg.norm(coordinates[i] - coordinates[j])

        row_norms.append((row_norm, i))

    # Sort by row norms

    row_norms.sort(reverse=True)

    Mijs = []

    if calc == "all":


        for index in range(len(atomtypes)):

            sorted_atomtypes = []
            sorted_coordinates = []

            sorted_atomtypes.append(atomtypes[index])
            sorted_coordinates.append(coordinates[index])

            for row_norm in row_norms:

                i = row_norm[1]

                if i == index:
                    continue

                sorted_atomtypes.append(atomtypes[i])
                sorted_coordinates.append(coordinates[i])

            Mij = np.zeros((size, size))
            for i, atomtype_i in enumerate(sorted_atomtypes):
                for j, atomtype_j in enumerate(sorted_atomtypes):
                    if i == j:
                        Mij[i, j] = 0.5 * NUCLEAR_CHARGE[atomtype_i] ** 2.4

                    elif j > i:
                        continue

                    else:
                        Mij[i, j] = NUCLEAR_CHARGE[atomtype_i] * NUCLEAR_CHARGE[atomtype_j] \
                                    / np.linalg.norm(sorted_coordinates[i] - sorted_coordinates[j])

            # Mijs.append(Mij.flatten(size**2))
            Mijs.append(triangular_to_vector(Mij))

        return Mijs

    else:

        index = calc 

        sorted_atomtypes = []
        sorted_coordinates = []

        sorted_atomtypes.append(atomtypes[index])
        sorted_coordinates.append(coordinates[index])

        for row_norm in row_norms[:index] + row_norms[index+1:]:

            i = row_norm[1]

            sorted_atomtypes.append(atomtypes[i])
            sorted_coordinates.append(coordinates[i])

        Mij = np.zeros((size, size))
        for i, atomtype_i in enumerate(sorted_atomtypes):
            for j, atomtype_j in enumerate(sorted_atomtypes):
                if i == j:
                    Mij[i, j] = 0.5 * NUCLEAR_CHARGE[atomtype_i] ** 2.4

                elif j > i:
                    continue

                else:
                    Mij[i, j] = NUCLEAR_CHARGE[atomtype_i] * NUCLEAR_CHARGE[atomtype_j] \
                                / np.linalg.norm(sorted_coordinates[i] - sorted_coordinates[j])
    
        # return Mij.flatten(size**2)
        return triangular_to_vector[Mij]




def generate_atomic_coulomb_matrix(atomtypes, coordinates, size=23, calc="all"):

    # Generate an local coloumb matrix sorted by distance to query atom

    Mijs = []
    for i, atomtype_i in enumerate(atomtypes):

        Mij = np.zeros((size, size))
        distances = []

        for j, atomtype_j in enumerate(atomtypes):

            distance = np.linalg.norm(coordinates[i] - coordinates[j])

            distances.append((distance, j))

        distances.sort()
        print distances

        for m, (dist_j, l) in enumerate(distances):
            for n, (dist_k, k) in enumerate(distances):
                if m == n:
                    Mij[m, n] = 0.5 * NUCLEAR_CHARGE[atomtypes[l]] ** 2.4
                else:
                    Mij[m, n] = NUCLEAR_CHARGE[atomtypes[k]] * NUCLEAR_CHARGE[atomtypes[l]] \
                            / np.linalg.norm(coordinates[k] - coordinates[l])

        Mijs.append(triangular_to_vector(Mij))

    return Mijs
