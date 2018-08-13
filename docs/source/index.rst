.. QML documentation master file, created by
   sphinx-quickstart on Sun Jun  4 14:41:04 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|Build Status| |pypi| |doi| |Beta|

QML: A Python Toolkit for Quantum Machine Learning
==================================================

QML is a Python2/3-compatible toolkit for representation learning of
properties of molecules and solids. QML is not a high-level framework
where you can do ``model.train()``, but supplies the building blocks to
carry out efficient and accurate machine learning on chemical compounds.
As such, the goal is to provide usable and efficient implementations of
concepts such as representations and kernels. 


Current list of contributors:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Anders S. Christensen (University of Basel)
-  Felix A. Faber (University of Basel)
-  Bing Huang (University of Basel)
-  Lars A. Bratholm (University of Copenhagen)
-  Alexandre Tkatchenko (University of Luxembourg)
-  Klaus-Robert Muller (Technische Universität Berlin/Korea University)
-  O. Anatole von Lilienfeld (University of Basel)

Code development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The QML code is developed through our GitHub repository: 

https://github.com/qmlcode/qml
------------------------------

Please add you code to QML by forking and making pull-requests to the "develop" branch.
Every now and then develop branch is pushed to the "master" branch and automatically deployed to PyPI, where the latest stable version is hosted.

See the "Installing QML" page for up-to-date installation instructions.


Citing QML:
--------------

Until the preprint is available from arXiv, please cite use of QML as:

::

    AS Christensen, FA Faber, B Huang, LA Bratholm, A Tkatchenko, KR Muller, OA von Lilienfeld (2017) "QML: A Python Toolkit for Quantum Machine Learning" https://github.com/qmlcode/qml


For citation of the individual procedures of QML, please see the "Citing use of QML" section.

.. toctree::
   :maxdepth: 2
   :caption: GETTING STARTED:
   :name: index

   installation
   citation
   tutorial
   examples


.. toctree::
   :maxdepth: 2
   :caption: SOURCE DOCUMENTATION:
   :name: qml

   qml

License:
-----------

QML is freely available under the terms of the MIT license.

.. |Build Status| image:: https://travis-ci.org/qmlcode/qml.svg?branch=master
   :target: https://travis-ci.org/qmlcode/qml

.. |doi| image:: https://zenodo.org/badge/89045103.svg
   :target: https://zenodo.org/badge/latestdoi/89045103

.. |Beta| image:: http://i.imgur.com/5fMAeek.jpg
   :width: 90 px
   :target: https://github.com/qmlcode/qml

.. |Pypi| image:: https://badge.fury.io/py/qml.svg
    :target: https://badge.fury.io/py/qml
