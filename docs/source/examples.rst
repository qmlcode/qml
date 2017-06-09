Examples
--------

Generating representations using the ``Compound`` class:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example demonstrates how to generate a representation via
the ``qml.Compound`` class.

.. code:: python

    from qml import Compound

    # Read in an xyz or cif file.
    water = Compound(xyz="water.xyz")

    # Generate a molecular coulomb matrices sorted by row norm.
    water.generate_coulomb_matrix(size=5, sort="row-norm")

    print water.coulomb_matrix

Generating representations via the ``qml.representations`` module:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from qml.representations import *

    # Dummy atomtypes and coordinates
    atomtypes = ["O", "H", H"]
    coordinates = np.array([1.464, 0.707, 1.056],
                           [0.878, 1.218, 0.498],
                           [2.319, 1.126, 0.952])

    # Generate a molecular coulomb matrices sorted by row norm.
    cm1 = generate_coulomb_matrix(atomtypes, coordinates,
                                    size=5, sort="row-norm")
    print cm1

    # Generate all atomic coulomb matrices sorted by distance to
    # query atom.
    cm2 = generate_atomic_coulomb_matrix(atomtypes, coordinates,
                                    size=5, sort="distance")
    print cm2
