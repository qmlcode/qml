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
    from qml.wrappers import get_atomic_kernels_gaussian

    # Assume the QM7 dataset is loaded into a list of Compound()
    for compound in qm7:

        # Generate the desired representation for each compound
        compound.generate_atomic_coulomb_matrix(size=23, sort="row-norm")

    # List of kernel-widths
    sigmas = [50.0, 100.0, 200.0]

    # Calculate the kernel-matrix
    K = get_atomic_kernels_gaussian(qm7, qm7, sigmas)

    print(K.shape)

.. code:: 

    (3, 7101, 7101)


Generating the SLATM representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Spectrum of London and Axillrod-Teller-Muto potential (SLATM) representation requires additional input to reduce the size of the representation.
This input (the types of many-body terms) is generate via the ``get_slatm_mbtypes()`` function. The function takes a list of the nuclearges for each molecule in the dataset as input. E.g.:


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

Generating the ARAD representation and kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ARAD representation does not have an explicit representation in the form of a vector, and the L2-distance must be calculated analytically in a separate kernel function.
The syntax is analogous to the explicit representations (e.g. Coulomb matrix, BoB, SLATM, etc), but is handled by kernels from the separate ``qml.arad`` module.

.. code:: python

    from qml.arad import get_local_kernels_arad

    # Assume the QM7 dataset is loaded into a list of Compound()
    for compound in qm7:

        # Generate the desired representation for each compound
        compound.generate_arad_representation(size=23)
    
    # Make Numpy array of the representation
    X = np.array([c.representation for c in qm7])
    
    # List of kernel-widths
    sigmas = [50.0, 100.0, 200.0]

    # Calculate the kernel-matrices for each sigma
    K = get_local_kernels_arad(X, X, sigmas)

    print(K.shape)

.. code:: 

    (3, 7101, 7101)

The dimensions of the input should be ``(number_molecules, size, 5, size)``, where ``size`` is the
size keyword used when generating the representations. 

In addition to using the ``Compound`` class to generate the representations, ARAD representations can also be generated via the ``qml.arad.generate_arad_representation()`` function, using similar notation to the functions in the ``qml.representations.*`` functions.

In case the two datasets used to calculate the kernel matrix are identical 
- resulting in a symmetric kernel matrix - it is possible use a faster kernel,
since only half of the kernel elements must be calculated explicitly:

.. code:: python

    from qml.arad import get_local_symmetric_kernels_arad

    # Calculate the kernel-matrices for each sigma
    K = get_local_symmetric_kernels_arad(X, sigmas)

    print(K.shape)

.. code:: 

    (3, 7101, 7101)

In addition to the local kernel, the ARAD module also provides kernels for atomic properties (e.g. chemical shifts, partial charges, etc). These have the name "atomic", rather than "local".

.. code:: python

    from qml.arad import get_atomic_kernels_arad
    from qml.arad import get_atomic_symmetric_kernels_arad

The only difference between the local and atomic kernels is the shape of the input.
Since the atomic kernel outputs kernels with atomic resolution, the atomic input has the shape ``(number_atoms, 5, size)``.
