# Welcome to `hawkeslib`

[![Build Status](https://travis-ci.org/canerturkmen/hawkeslib.svg?branch=master)](https://travis-ci.org/canerturkmen/hawkeslib)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/hawkeslib/badge/?version=latest)](https://hawkeslib.readthedocs.io/en/latest/?badge=latest)

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

```
$ pip install hawkeslib
```
