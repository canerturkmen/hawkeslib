API Reference
=============================

Univariate Hawkes Processes
---------------------------

.. autoclass:: hawkeslib.UnivariateExpHawkesProcess
    :members:

.. autoclass:: hawkeslib.BayesianUVExpHawkesProcess
    :members:

    .. automethod:: __init__

Multivariate Hawkes Processes
-----------------------------

*Multivariate* Hawkes processes are those in which event occurrences assume discrete marks
from a finite set of cardinality K. Analogously, we can think of K distinct Hawkes processes
running, that not only *self-excite*, but also excite other processes (i.e. are *mutually exciting*).

.. autoclass:: hawkeslib.MultivariateExpHawkesProcess
    :members:

Poisson Processes
---------------------------

For sake of completeness and comparability, we provide temporal Poisson processes
(and a Bayesian variant) implementing the :class:`PointProcess` interface, just like
Hawkes processes.

The same functionality as for Hawkes; such as computing log likelihoods (or posterior potentials),
maximum likelihood (or MAP) estimates of parameters, and posterior sampling are implemented. Note that due to
the well-known *complete randomness* property of Poisson processes, and also the use of a conjugate prior
for the Bayesian case, these methods are implemented just in a few lines of code.

.. autoclass:: hawkeslib.PoissonProcess
    :members:

.. autoclass:: hawkeslib.BayesianPoissonProcess
    :members:

    .. automethod:: __init__
