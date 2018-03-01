Installing QML
---------------

Installing prerequisite modules (for most Linux systems):

.. code:: bash

    sudo apt-get install python-pip gfortran libblas-dev liblapack-dev git

These should already be installed on most systems. The Intel compilers
and MKL math-libraries are supported as well (see section 1.3).

Installing via ``pip``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install the stable version from the Python Package
Index (PyPI) is using the official, built-in Python package manager,
``pip``:

.. code:: bash

    pip install qml --user -U

To use the Intel compiler, together with the MKL math library:


In addition to the stable version available from the official PyPI repository, you can install the most recent stable development snapshot directly from GitHub:

.. code:: bash

    pip install git+https://github.com/qmlcode/qml@develop --user -U

Use ``pip2 install ...`` or ``pip3 install ...`` to get the Python2 or
Python3 versions explicitly. QML supports both flavors.

To uninstall simply use ``pip`` again.

.. code:: bash

    pip uninstall qml

Installing via with Intel compilers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have Intel compilers installed (2016 and newer), you can compile
QML with Ifort/MKL from PyPI using the following options:

.. code:: bash

    pip install qml --user -U  --global-option="build" --global-option="--compiler=intelem" --global-option="--fcompiler=intelem"

Or alternatively the `develop` branch from GitHub:

.. code:: bash

    pip install git+https://github.com/qmlcode/qml@develop --user -U --global-option="build" --global-option="--compiler=intelem" --global-option="--fcompiler=intelem"


Note on Apple/Mac support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installing QML requires a Fortran compiler. On MacOS you can install it
using ``brew``:

.. code:: bash

    # Update brew
    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

    # Install GCC
    brew install gcc

Note the Fortran compiler from brew (gfortran) unfortunately does not support OpenMP.
Therefore parallelism via OpenMP is disabled as default for MacOS systems.

Additionally, we found that some users have multiple version of the ``as`` assembler - this might happen if you have GCC from e.g. brew and macports at the same time. Look for the following error:


  ``FATAL:/opt/local/bin/../libexec/as/x86_64/as: I don't understand 'm' flag!``

If you experience this problems the setting the following path might fix the problem:

.. code:: bash

    export PATH=/usr/bin:$PATH


Report Bugs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Please report any bugs by opening an issue on GitHub: https://github.com/qmlcode/qml/issues
