# Welcome to `fasthawkes`

[![Build Status](https://travis-ci.org/canerturkmen/fasthawkes.svg?branch=master)](https://travis-ci.org/canerturkmen/fasthawkes)


`fasthawkes` started with the ambition of having a go-to implementation of plain-vanilla Hawkes processes (HP), i.e. those
with factorized triggering kernels with exponential decay functions, in addition to *simpler* variants that allow for
more expressive models with little more in the way of computational cost. One such example is the **Beta Delay HP**.

<div class="alert alert-danger">
<b>Warning: </b> The entire library is currently being refactored, and is not stable. This warning will be removed and
instructions added when it's available for use.
</div>


## Installation

```
$ python setup.py build_ext --inplace
```
