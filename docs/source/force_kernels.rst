FCHL19 and force kernels
------------------------

This is a tutorial that shows the FCHL19 representation and corresponding kernels. See the paper (not yet on arxiv) for details on the representation.


Representation
~~~~~~~~~~~~~~

Generating the representation
"""""""""""""""""""""""""""""

.. code:: python

    import qml
    from qml.representations import generate_fchl_acsf

    # Dummy coordinates for a water molecule
    coordinates = np.array([[1.464, 0.707, 1.056],
                            [0.878, 1.218, 0.498],
                            [2.319, 1.126, 0.952]])

    # Oxygen, Hydrogen, Hydrogen
    nuclear_charges = np.array([8, 1, 1])

    # Generate representations for the atoms in the water molecule
    rep = generate_fchl_acsf(nuclear_charges, coordinates)

    print(rep)
    print(rep.shape)

Should print the following output:

.. code:: python

    [[1.16160253e+00 2.11737093e+00 1.42984457e+00 ... 0.00000000e+00
      0.00000000e+00 0.00000000e+00]
     [4.73880430e-04 8.12967740e-02 3.00106360e-01 ... 0.00000000e+00
      0.00000000e+00 0.00000000e+00]
     [4.73880430e-04 8.12967740e-02 3.00106360e-01 ... 0.00000000e+00
      0.00000000e+00 0.00000000e+00]]
    (3, 720)

The representation requires you to specify a list of the possible elements that are in your training/test set. The default is :math:`[1,6,7,8,16]`. The representation increases in in the order of :math:`\mathcal{O}(n^2)` with the number of elements in the list, so it is beneficial to only include the minimal set. In the above example, the vector for each of the three atoms is of length 720. The size can be reduced by only including the elements 1 and 8 (hydrogen and oxygen).

.. code:: python
    
    # Generate a minimal representations for the atoms in the water molecule
    rep = generate_fchl_acsf(nuclear_charges, coordinates, elements=[1,8])

    print(rep)
    print(rep.shape)

Should print the following output:

.. code:: python

    [[1.16160253e+00 2.11737093e+00 1.42984457e+00 ... 0.00000000e+00
      0.00000000e+00 0.00000000e+00]
     [4.73880430e-04 8.12967740e-02 3.00106360e-01 ... 0.00000000e+00
      0.00000000e+00 0.00000000e+00]
     [4.73880430e-04 8.12967740e-02 3.00106360e-01 ... 0.00000000e+00
      0.00000000e+00 0.00000000e+00]]
    (3, 168)

The size of the representation is reduced to 168.

Additional parameters
"""""""""""""""""""""

The representation contains a number of parameters, which can be optimized in order to improve the accuracy of the resulting machine learning model.
The default parameters are optimized for forces on a set of small molecules (see arXiv paper), and will work well for most cases.
Additionally, we also found a different set if parameters to work well for energy learning.

.. code:: python

    # Energy-optimized kwargs
    kwargs_energy = {
        'nRs2': 22, 
        'nRs3': 17, 
        'eta2': 0.41, 
        'eta3': 0.97, 
        'three_body_weight': 45.83, 
        'three_body_decay': 2.39,     
        'two_body_decay': 2.39,
     }

    
    # Generate energy-optimized representation for water 
    rep = generate_fchl_acsf(nuclear_charges, coordinates, elements=[1,8], **kwargs_energy)

Kernels
~~~~~~~

The kernel functions implemented for the representation is a Gaussian function which only compares atomic environments of atoms of the same element type, that is:

    :math:`k(\mathbf{q}_I,\mathbf{q}_{J}^{*}) = \delta_{Z_I Z_{J}^{*}}  \exp\left(-\frac{\| \mathbf{q}_I - \mathbf{q}_{J}^{*} \|^2_2}{2\sigma^2}\right)`

where :math:`Z_i` and :math:`Z_j` are the nuclear charges of the atoms :math:`i` and :math:`j`.

The ``QML.kernels`` module contains functions to generate kernel functions a number of machine learning approaches as detailed below.
In all cases, the resulting kernels are simply matrices in numpy's ``ndarray`` format.


Kernel Ridge Regression
"""""""""""""""""""""""


Regression model of some property, :math:`U`, for some system, this could correspond to e.g. the atomization energy of a molecule:

    :math:`\boldsymbol{\alpha} = (\mathbf{K} + \lambda \mathbf{I})^{-1} \mathbf{U}`


.. code:: python

    # Generate representations
    reps = np.array([generate_fchl_acsf(mol.nuclear_charges, mol.coordinates, pad=23) for mol in mols]

    # Generate lists of nuclear charges    
    charges = [mol.nuclear_charges for mol in mols] 

    # Energies for each molecule
    energies = np.array([mol.energy for mol in mols])

    # Divide in training and test representations
    X  = reps[:100]
    Xs = reps[100:]
    
    # Divide in training and test nuclear charges 
    Q  = nuclear_charges[:100]
    Qs = nuclear_charges[100:]
    
    # Divide in training and test energies 
    U  = energies[:100]
    Us = energies[100:]


The training kernel is symmetrical and can be calculated faster by using the dedicated kernel function that only calculates the upper triangle.
The test kernel is not symmetrical, and the ordering is first representations/charges for the basis functions (usually the same as the training set), and secondly the query representations/charges.
Additionally, the functions take a kernel width (sigma) as argument.

A minimal kernel ridge regression program is as follows:

.. code:: python

    from qml.kernels import get_local_kernel
    from qml.kernels import get_local_symmetric_kernel
    from qml.math import cho_solve

    # Example kernel width
    sigma = 20.0

    # Generate training and test kernel
    K_training = get_local_symmetric_kernel(X, Q, sigma)
    K_test = get_local_kernel(X, Xs, Q, Qs, sigma)

    # Solve the regression using lambda=1e-9
    alphas = cho_solve(K_training, U, l2reg=1e-9)

    # Make predictions using the test kernel
    U_test = np.dot(K_test, alphas)



Response Operators
""""""""""""""""""

QML with response operators expand the learned properties in a basis of kernel functions centered on the atoms of the training set.
For example, the equation that simultaneously describes the energy (F) and forces (F) of a system is:

.. math::

    \begin{bmatrix}
        \mathbf{U} \\
        \mathbf{F} 
    \end{bmatrix} = \begin{bmatrix}
        \mathbf{K}^{a,u*} \\
        \mathbf{K}^{a,g*} 
    \end{bmatrix} \alpha

Here, the subscripts relate to the two dimensions of the kernel matrices. :math:`a` denotes the first dimension consists of the kernel functions centered on the atomic environments, and :math:`u` and :math:`g` denotes that the second dimension are the zeroth and first derivative of the kernel, respectively.

In addition to the list of representations, nuclear charges and energies in the above example, we also need to generate arrays containgin the derivatives of the representations as well as the derivatives of the energy (the forces).


.. code:: python

    # Forces for each molecule
    forces = [mol.properties for mol in mols]

    # Generate derivatives of representations
    dreps = np.array([generate_fchl_acsf(mol.nuclear_charges, mol.coordinates, gradient=True, pad=23)[1] for mol in mols]

    # Divide in training and test derivatives
    dX  = dreps[:100]
    dXs = dreps[100:]

    # Divide in training and test forces (and flatten to 1-D)
    F  = np.concatenate(forces[100:]).flatten()
    Fs = np.concatenate(forces[:100]).flatten()


A minimal program to train and predict energies and forces with response operator kernels:

.. code:: python

    from qml.kernels import get_atomic_local_gradient_kernel
    from qml.kernels import get_atomic_local_kernel
    from qml.math import svd_solve

    # Example kernel width
    sigma = 20.0

    # Generate training and test kernel for energies
    Ke  = get_atomic_local_kernel(X, X,  Q, Q,  SIGMA)
    Kes = get_atomic_local_kernel(X, Xs, Q, Qs, SIGMA)

    # Generate training and test kernel for forces - note that only 
    # one set of derivatives is required
    Kf  = get_atomic_local_gradient_kernel(X, X,  dX,  Q, Q,  SIGMA)
    Kfs = get_atomic_local_gradient_kernel(X, Xs, dXs, Q, Qs, SIGMA)

    # Concatenate energy and force kernels for the training set
    C = np.concatenate(Ke, Kf)

    # Concatenate matching energy and force labels 
    Y = np.concatenate(U, F)

    # Solve the regression ignoring singular values smaller than 1e-9
    alphas = svd_solve(C, Y, rcond=1e-9)

    # Make energy predictions using the test energy kernel
    U_test = np.dot(Kes, alphas)

    # Make force predictions using the test force kernel
    F_test = np.dot(Kfs, alphas)


Gaussian Process Regression
"""""""""""""""""""""""""""

Gaussian process regression with derivatives works similarly to the response operators in the above example.
The only difference is that the basis set consists of kernels placed on the molecules in the training set (rather than the atoms), as well as placed on their derivatives.
The resulting kernel is roughly 3x larger, and requires evaluation of the second derivative of the kernel, but is also usually more accurate, especially for small training set sizes.

.. math::

    \begin{bmatrix}
        \mathbf{U} \\
        \mathbf{F} 
    \end{bmatrix} = \begin{bmatrix}
       \mathbf{K}^{u,u*} && \mathbf{K}^{u,g*} \\
       \mathbf{K}^{g,u*} && \mathbf{K}^{g,g*} 
    \end{bmatrix} \alpha


A minimal program to train and predict energies and forces with Gaussian proces regression:

.. code:: python

    from qml.kernels import get_symmetric_gp_kernel
    from qml.kernels import get_gp_kernel
    from qml.math import cho_solve

    # Example kernel width
    sigma = 20.0

    # Generate training and test kernel for energies
    K_train = get_symmetric_gp_kernel(X, dX, Q, SIGMA)
    K_test = get_gp_kernel(X, Xs, dX, dXs, Q, Qs, SIGMA)

    # Concatenate matching energy and force labels 
    Y = np.concatenate(U, F)

    # Solve alpha coefficients
    alphas = cho_solve(K_train, Y, l2reg=1e-9)

    # Make predictions using the test energy kernel
    Y_test = np.dot(K_test, alphas)

    # Get test energies and forces from the prediction
    U_test = Y_test[:100]
    F_test = Y_test[100:]




