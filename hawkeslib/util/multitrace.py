import numpy as np


class MultiTrace:
    """
    A MultiTrace object, that emulates the MultiTrace object of `pymc3`, that
    encapsulates sequentially drawn samples (such as from an MCMC routine).
    """

    def __init__(self, names, *args):
        """
        Initialize the MultiTrace object.
        :param list names: names of the variables
        :param np.ndarray args: variables, each represented as a single numpy.array
        """

        assert len(names) == len(args), \
            "Variable names and arrays should be the same number"

        assert all(map(lambda x: isinstance(x, np.ndarray), args))

        self.data_dict = dict()

        for i, n in enumerate(names):
            self.data_dict[n] = args[i]

    def __getitem__(self, in_slice):
        if isinstance(in_slice, str):
            return self.data_dict[in_slice]
        elif isinstance(in_slice, int) or isinstance(in_slice, slice):
            return MultiTrace(
                list(self.data_dict.keys()),
                *[x[in_slice] for i, x in self.data_dict.items()]
            )
        else:
            raise ValueError("__getitem__ not supported for type %s" % type(in_slice))

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __delitem__(self, key):
        raise NotImplementedError()
