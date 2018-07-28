# Welcome to `fasthawkes`

[![Build Status](https://travis-ci.org/canerturkmen/fasthawkes.svg?branch=master)](https://travis-ci.org/canerturkmen/fasthawkes)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`fasthawkes` started with the ambition of having a go-to implementation of plain-vanilla Hawkes processes (HP), i.e. those
with factorized triggering kernels with exponential decay functions.

Currently, the following models are available with a unified interface:

- Univariate Hawkes Process (with exponential delay)
- Bayesian Univariate Hawkes Process (with exponential delay)
- Poisson Process
- 'Bayesian' Poisson process

Bayesian variants implement methods for sampling from the posterior as well as calculating marginal likelihood (e.g. for
Bayesian model comparison).

For an example, see the examples folder.

## Installation

```
$ python setup.py build_ext --inplace
```
