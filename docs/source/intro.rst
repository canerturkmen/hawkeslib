Introduction
================

.. image:: https://travis-ci.org/canerturkmen/hawkeslib.svg?branch=master
   :target: https://travis-ci.org/canerturkmen/hawkeslib
.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT

About hawkeslib
----------------

hawkeslib started with the ambition of presenting easy-to-use, well maintained
implementations of plain-vanilla Hawkes (self-exciting) processes [1]_ [2]_, a form of evolutionary
temporal point processes that is increasingly put to use in a variety of domains.

Some features of the library are

* Fast. most algorithms are implemented in Cython, cutting away most of the Python
  overhead for likelihood computations and parameter estimation algorithms that require
  *scanning* the data.
* Easy-to-use. Models implement a familiar, common interface.
* Good for beginners. The library implements a variety of *plain vanilla* self-exciting
  processes such as univariate and multivariate Hawkes processes with exponential delay
  densities, Poisson processes, and related Bayesian inference machinery.

In the future, we hope to add several extended models.

Installation
----------------

Cython (>=0.28) and numpy (>=1.14) must be installed prior to the installation.

.. code-block:: bash

   $ pip install -U Cython numpy
   $ pip install hawkeslib

Currently, the library only supports python 2.7.

Getting Started
-----------------

The ``examples/`` folder contains Jupyter notebooks that demonstrate basic use cases,
in addition to the tutorial and example provided in these docs.


**References**

.. [1] Hawkes, Alan G. "Point spectra of some mutually exciting point processes." Journal of the Royal
   Statistical Society. Series B (Methodological) (1971): 438-443.
.. [2] Bacry, Emmanuel, Iacopo Mastromatteo, and Jean-François Muzy. "Hawkes processes in finance."
   Market Microstructure and Liquidity 1.01 (2015): 1550005.