Examples
--------

Generating representations using the ``Compound`` class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example demonstrates how to generate a representation via
the ``qml.Compound`` class.

.. code:: python

    # Read in an xyz or cif file.
    water = Compound(xyz="water.xyz")

    # Generate a molecular coulomb matrices sorted by row norm.
    water.generate_coulomb_matrix(size=5, sorting="row-norm")

    print(water.representation)

Might print the following representation:

.. code:: 
    
    [ 73.51669472   8.3593106    0.5          8.35237809   0.66066557   0.5
       0.           0.           0.           0.           0.           0.           0.
       0.           0.        ]

Generating representations via the ``qml.representations`` module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    import numpy as np
    from qml.representations import *

    # Dummy coordinates for a water molecule
    coordinates = np.array([[1.464, 0.707, 1.056],
                            [0.878, 1.218, 0.498],
                            [2.319, 1.126, 0.952]])

    # Oxygen, Hydrogen, Hydrogen
    nuclear_charges = np.array([8, 1, 1])

    # Generate a molecular coulomb matrices sorted by row norm.
    cm1 = generate_coulomb_matrix(nuclear_charges, coordinates,
                                    size=5, sorting="row-norm")
    print(cm1)


The resulting Coulomb-matrix for water:

.. code:: 
    
    [ 73.51669472   8.3593106    0.5          8.35237809   0.66066557   0.5
       0.           0.           0.           0.           0.           0.           0.
       0.           0.        ]



.. code:: python

    # Generate all atomic coulomb matrices sorted by distance to
    # query atom.
    cm2 = generate_atomic_coulomb_matrix(atomtypes, coordinates,
                                    size=5, sort="distance")
    print cm2

.. code:: 

    [[ 73.51669472   8.3593106    0.5          8.35237809   0.66066557   0.5
        0.           0.           0.           0.           0.           0.
        0.           0.           0.        ]
     [  0.5          8.3593106   73.51669472   0.66066557   8.35237809   0.5
        0.           0.           0.           0.           0.           0.
        0.           0.           0.        ]
     [  0.5          8.35237809  73.51669472   0.66066557   8.3593106    0.5
        0.           0.           0.           0.           0.           0.
        0.           0.           0.        ]]


Calculating a Gaussian kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The input for most of the kernels in QML is a numpy array, where the first dimension is the number of representations, and the second dimension is the size of each representation. An brief example is presented here, where ``compounds`` is a list of ``Compound()`` objects:

.. code:: python
    
    import numpy as np
    from qml.kernels import gaussian_kernel

    # Generate a numpy-array of the representation
    X = np.array([c.representation for c in compounds])

    # Kernel-width
    sigma = 100.0

    # Calculate the kernel-matrix
    K = gaussian_kernel(X, X, sigma)


Calculating a Gaussian kernel using a local representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to calculate the kernel matrix using an explicit, local representation is via the wrappers module. Note that here the sigmas is a list of sigmas, and the result is a kernel for each sigma. The following examples currently work with the atomic coulomb matrix representation and the local SLATM representation:

.. code:: python

    import numpy as np
    from qml.kernels import get_local_kernels_gaussian

    # Assume the QM7 dataset is loaded into a list of Compound()
    for compound in qm7:

        # Generate the desired representation for each compound
        compound.generate_atomic_coulomb_matrix(size=23, sort="row-norm")

    # Make a big array with all the atomic representations
    X = np.concatenate([mol.representation for mol in qm7])

    # Make an array with the number of atoms in each compound
    N = np.array([mol.natoms for mol in qm7])

    # List of kernel-widths
    sigmas = [50.0, 100.0, 200.0]

    # Calculate the kernel-matrix
    K = get_local_kernels_gaussian(X, X, N, N, sigmas)

    print(K.shape)

.. code:: 

    (3, 7101, 7101)

Note that ``mol.representation`` is just a 1D numpy array.


Generating the SLATM representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Spectrum of London and Axillrod-Teller-Muto potential (SLATM) representation requires additional input to reduce the size of the representation.
This input (the types of many-body terms) is generate via the ``get_slatm_mbtypes()`` function. The function takes a list of the nuclear charges for each molecule in the dataset as input. E.g.:


.. code:: python

    from qml.representations import get_slatm_mbtypes

    # Assume 'qm7' is a list of Compound() objects.
    mbtypes = get_slatm_mbtypes([mol.nuclear_charges for compound in qm7])

    # Assume the QM7 dataset is loaded into a list of Compound()
    for compound in qm7:

        # Generate the desired representation for each compound
        compound.generate_slatm(mbtypes, local=True)

The ``local`` keyword in this example specifies that a local representation is produced. Alternatively the SLATM representation can be generate via the ``qml.representations`` module:
    
.. code:: python

    from qml.representations import generate_slatm

    # Dummy coordinates
    coordinates = ... 

    # Dummy nuclear charges
    nuclear_charges = ...

    # Dummy mbtypes
    mbtypes = get_slatm_mbtypes( ... )

    # Generate one representation
    rep = generate_slatm(coordinates, nuclear_charges, mbtypes)

Here ``coordinates`` is an Nx3 numpy array, and ``nuclear_charges`` is simply a list of charges.

Generating the FCHL representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The FCHL representation does not have an explicit representation in the form of a vector, and the kernel elements must be calculated analytically in a separate kernel function.
The syntax is analogous to the explicit representations (e.g. Coulomb matrix, BoB, SLATM, etc), but is handled by kernels from the separate ``qml.fchl`` module.

The code below show three ways to create the input representations for the FHCL kernel functions.

First using the ``Compound`` class:

.. code:: python

    # Assume the dataset is loaded into a list of Compound()
    for compound in mols:

        # Generate the desired representation for each compound, cut off in angstrom
        compound.generate_fchl_representation(size=23, cut_off=10.0)
    
    # Make Numpy array of the representation, which can be parsed to the kernel
    X = np.array([c.representation for c in mols])
    

The dimensions of the array should be ``(number_molecules, size, 5, size)``, where ``size`` is the
size keyword used when generating the representations. 

In addition to using the ``Compound`` class to generate the representations, FCHL representations can also be generated via the ``qml.fchl.generate_fchl_representation()`` function, using similar notation to the functions in the ``qml.representations.*`` functions.


.. code:: python

    from qml.fchl import generate_representation 

    # Dummy coordinates for a water molecule
    coordinates = np.array([[1.464, 0.707, 1.056],
                            [0.878, 1.218, 0.498],
                            [2.319, 1.126, 0.952]])

    # Oxygen, Hydrogen, Hydrogen
    nuclear_charges = np.array([8, 1, 1])

    rep = generate_representation(coordinates, nuclear_charges)

To create the representation for a crystal, the notation is as follows:


.. code:: python

    from qml.fchl import generate_representation 

    # Dummy fractional coordinates
    fractional_coordinates = np.array(
            [[ 0.        ,  0.        ,  0.        ],
             [ 0.75000042,  0.50000027,  0.25000015],
             [ 0.15115386,  0.81961403,  0.33154037],
             [ 0.51192691,  0.18038651,  0.3315404 ],
             [ 0.08154025,  0.31961376,  0.40115401],
             [ 0.66846017,  0.81961403,  0.48807366],
             [ 0.08154025,  0.68038678,  0.76192703],
             [ 0.66846021,  0.18038651,  0.84884672],
             [ 0.23807355,  0.31961376,  0.91846033],
             [ 0.59884657,  0.68038678,  0.91846033],
             [ 0.50000031,  0.        ,  0.50000031],
             [ 0.25000015,  0.50000027,  0.75000042]]
        )

    # Dummy nuclear charges
    nuclear_charges = np.array(
            [58, 58, 8, 8, 8, 8, 8, 8, 8, 8, 23, 23]
        )

    # Dummy unit cell
    unit_cell = np.array(
            [[ 3.699168,  3.699168, -3.255938],
             [ 3.699168, -3.699168,  3.255938],
             [-3.699168, -3.699168, -3.255938]]
        )

    # Generate the representation
    rep = generate_representation(fractional_coordinates, nuclear_charges, 
            cell=unit_cell, neighbors=100, cut_distance=7.0)


The neighbors keyword is the max number of atoms with the cutoff-distance

Generating the FCHL kernel 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example demonstrates how to calculate the local FCHL kernel elements between FCHL representations. ``X1`` and ``X2`` are numpy arrays with the shape ``(number_compounds,max_size, 5,neighbors)``, as generated in one of the previous examples. You MUST use the same, or larger, cut-off distance to generate the representation, as to calculate the kernel.


.. code:: python

    from qml.fchl import get_local_kernels

    # You can get kernels for multiple kernel-widths
    sigmas = [2.5, 5.0, 10.0]

    # Calculate the kernel-matrices for each sigma
    K = get_local_kernels(X1, X2, sigmas, cut_distance=10.0)

    print(K.shape)


As output you will get a kernel for each kernel-width.

.. code:: 

    (3, 100, 200)


In case ``X1`` and ``X2`` are identical, K will be symmetrical. This is handled by a separate function with exploits this symmetry (thus being twice as fast).

.. code:: python
    
    from qml.fchl import get_local_symmetric_kernels

    # You can get kernels for multiple kernel-widths
    sigmas = [2.5, 5.0, 10.0]

    # Calculate the kernel-matrices for each sigma
    K = get_local_kernels(X1, sigmas, cut_distance=10.0)

    print(K.shape)


.. code:: 

    (3, 100, 100)

In addition to the local kernel, the FCHL module also provides kernels for atomic properties (e.g. chemical shifts, partial charges, etc). These have the name "atomic", rather than "local".

.. code:: python

    from qml.fchl import get_atomic_kernels
    from qml.fchl import get_atomic_symmetric_kernels

The only difference between the local and atomic kernels is the shape of the input.
Since the atomic kernel outputs kernels with atomic resolution, the atomic input has the shape ``(number_atoms, 5, size)``.
