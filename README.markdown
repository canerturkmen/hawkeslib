[![deprecated](http://badges.github.io/stability-badges/dist/deprecated.svg)](http://github.com/badges/stability-badges)

⚠️ Due to time constraints I am no longer able to develop / support this library. For more stable options on Hawkes processes, check out `tick` or `pyhawkes`.

Thanks!

--- 

# Welcome to `hawkeslib`

[![Build Status](https://travis-ci.org/canerturkmen/hawkeslib.svg?branch=master)](https://travis-ci.org/canerturkmen/hawkeslib)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/hawkeslib/badge/?version=latest)](https://hawkeslib.readthedocs.io/en/latest/?badge=latest)
[![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)](https://www.python.org/downloads/release/python-2715/)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

`hawkeslib` started with the ambition of having a simple Python implementation
of *plain-vanilla* Hawkes (or *self-exciting* processes), i.e. those
with factorized triggering kernels with exponential decay functions.

The [docs](http://hawkeslib.rtfd.io/) contain tutorials, examples and a detailed API reference.
For other examples, see the `examples/` folder.

The following models are available:

- Univariate Hawkes Process (with exponential delay)
- Bayesian Univariate Hawkes Process (with exponential delay)
- Poisson Process
- 'Bayesian' Poisson process

Bayesian variants implement methods for sampling from the posterior as well as calculating
marginal likelihood (e.g. for Bayesian model comparison).

## Installation

`Cython` (>=0.28) and `numpy` (>=1.14) and `scipy` must be installed prior to the installation as
they are required for the build.

```
$ pip install -U Cython numpy scipy
$ pip install hawkeslib
```
