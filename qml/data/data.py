
from __future__ import print_function

import glob
import numpy as np
from ..utils.alchemy import NUCLEAR_CHARGE

class Data(object):
    """
    Temporary data class which should be replaced at some point by the ASE-interface.
    This could in principle also be replaced by a dictionary

    """

    def __init__(self, filenames=None):#, property_type = "energy"):
        #"""
        #:param property_type: What kind of property will be predicted ('energy')
        #:type property_type: string
        #"""

        #self.property_type = property_type

        self.ncompounds = 0
        self.coordinates = None
        self.nuclear_charges = None
        #self.energies = None
        #self.properties = None
        self.natoms = None

        if isinstance(filenames, str):
            filenames = sorted(glob.glob(filenames))
        if isinstance(filenames, list):
            self._parse_xyz_files(filenames)

    def set_energies(self, energies):
        self.energies = energies

    def _parse_xyz_files(self, filenames):
        """
        Parse a list of xyz files.
        """

        self.ncompounds = len(filenames)
        self.coordinates = np.empty(self.ncompounds, dtype=object)
        self.nuclear_charges = np.empty(self.ncompounds, dtype=object)
        self.natoms = np.empty(self.ncompounds, dtype = int)

        #if self.property_type == "energy":
        #    self.energies = np.empty(self.ncompounds, dtype=float)

        for i, filename in enumerate(filenames):
            with open(filename, "r") as f:
                lines = f.readlines()

            natoms = int(lines[0])
            self.natoms[i] = natoms
            self.nuclear_charges[i] = np.empty(natoms, dtype=int)
            self.coordinates[i] = np.empty((natoms, 3), dtype=float)

            #if self.property_type == "energy":
            #    self.energies[i] = float(lines[1])

            for j, line in enumerate(lines[2:natoms+2]):
                tokens = line.split()

                if len(tokens) < 4:
                    break

                self.nuclear_charges[i][j] = NUCLEAR_CHARGE[tokens[0]]
                self.coordinates[i][j] = np.asarray(tokens[1:4], dtype=float)
