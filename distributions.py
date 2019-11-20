from numpy.random import zipf, randint as uniform

__all__ = ['zipf', 'uniform', 'constant']


def constant(*, k: int) -> int:
    return k
