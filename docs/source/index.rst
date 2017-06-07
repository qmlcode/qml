.. QML documentation master file, created by
   sphinx-quickstart on Sun Jun  4 14:41:04 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|Build Status|

QML: A Python Toolkit for Quantum Machine Learning
==================================================

QML is a Python2/3-compatible toolkit for representation learning of
properties of molecules and solids. QML is not a high-level framework
where you can do ``model.train()``, but supplies the building blocks to
carry out efficient and accurate machine learning on chemical compounds.
As such, the goal is to provide usable and efficient implementations of
concepts such as representations and kernels, which can then be
implemented in other machine learning codes.

Current list of contributors:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Anders S. Christensen (University of Basel)
-  Felix Faber (University of Basel)
-  Bing Huang (University of Basel)
-  Lars A. Bratholm (University of Copenhagen)
-  O. Anatole von Lilienfeld (University of Basel)

1) Citing QML:
--------------

Until the preprint is available from arXiv, please cite this GitHub
repository as:

::

    AS Christensen, F Faber, B Huang, LA Bratholm, OA von Lilienfeld (2017) "QML: A Python Toolkit for Quantum Machine Learning" https://github.com/qmlcode/qml


2) Installation
---------------

Please go to the QML github repository at https://github.com/qmlcode/qml/ 



.. toctree::
   :maxdepth: 2
   :caption: GETTING STARTED:
   :name: index

   installation
   examples


.. toctree::
   :maxdepth: 2
   :caption: SOURCE DOCUMENTATION:
   :name: qml

   qml

3) License:
-----------

QML is freely available under the terms of the MIT license.

.. |Build Status| image:: https://travis-ci.org/qmlcode/qml.svg?branch=master
   :target: https://travis-ci.org/qmlcode/qml
