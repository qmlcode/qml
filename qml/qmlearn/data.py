
from __future__ import print_function

import glob
import numpy as np
from ..utils import NUCLEAR_CHARGE
import copy


class Data(object):
    """
    Temporary data class which should be replaced at some point by the ASE-interface.
    This could in principle also be replaced by a dictionary

    """

    def __init__(self, filenames=None, property_type = "energy"):
        """
        :param filenames: list of filenames or a string to be read by glob. e.g. 'dir/*.xyz'
        :type filenames: list or string
        :param property_type: What kind of property will be predicted ('energy')
        :type property_type: string
        """

        self.property_type = property_type

        self._set_ncompounds(0)
        self.coordinates = None
        self.nuclear_charges = None
        self.natoms = None
        self.energies = None

        if isinstance(filenames, str):
            filenames = sorted(glob.glob(filenames))
        if isinstance(filenames, list):
            self._parse_xyz_files(filenames)
        # Overwritten in various parts of a standard prediction pipeline
        # so don't use these within the class
        #self._has_transformed_labels
        #self._representations
        #self._kernel
        #self._indices
        #self._representation_type
        #self._representation_short_name
        #self._representation_cutoff
        #self._representation_alchemy

    def _set_ncompounds(self, n):
        self.ncompounds = n
        # Hack for sklearn CV
        self.shape = (n,)

    def take(self, i, axis=None):
        """
        Hack for sklearn CV
        """
        other = copy.copy(self)
        other._indices = i
        return other

    # Hack for sklearn CV
    def __getitem__(self, i):
        return i

    # Hack for sklearn CV but also convenience
    def __len__(self):
        if hasattr(self, '_indices'):
            return len(self._indices)
        return self.ncompounds

    # Hack for sklearn CV but also convenience
    def __eq__(self, other):
        """
        Overrides the == operator.
        """

        if type(self) != type(other):
            return False

        self_vars = vars(self)
        other_vars = vars(other)

        if len(self_vars) != len(other_vars):
            return False

        for key, val in self_vars.items():
            if val is not other_vars[key]:
                return False

        return True

    # Hack for sklearn CV but also convenience
    def __ne__(self, other):
        """
        Overrides the != operator (unnecessary in Python 3)
        """
        return not self.__eq__(other)

    def set_energies(self, energies):
        self.energies = energies

    def _parse_xyz_files(self, filenames):
        """
        Parse a list of xyz files.
        """

        self._set_ncompounds(len(filenames))
        self.coordinates = np.empty(self.ncompounds, dtype=object)
        self.nuclear_charges = np.empty(self.ncompounds, dtype=object)
        self.natoms = np.empty(self.ncompounds, dtype = int)

        for i, filename in enumerate(filenames):
            with open(filename, "r") as f:
                lines = f.readlines()

            natoms = int(lines[0])
            self.natoms[i] = natoms
            self.nuclear_charges[i] = np.empty(natoms, dtype=int)
            self.coordinates[i] = np.empty((natoms, 3), dtype=float)

            for j, line in enumerate(lines[2:natoms+2]):
                tokens = line.split()

                if len(tokens) < 4:
                    break

                self.nuclear_charges[i][j] = NUCLEAR_CHARGE[tokens[0]]
                self.coordinates[i][j] = np.asarray(tokens[1:4], dtype=float)

        # Try to convert dtype to int/float in cases where you have the
        # same molecule, just different conformers

        try:
            self.nuclear_charges = np.asarray([self.nuclear_charges[i] for i in range(self.ncompounds)], 
                    dtype=int)
            self.coordinates = np.asarray([self.coordinates[i] for i in range(self.ncompounds)],
                    dtype=float)
        except ValueError:
            pass
