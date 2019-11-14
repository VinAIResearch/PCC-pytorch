"""Common functions for the MDP package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



def bound(x, low, high):
    """Truncates data between low and high boundaries."""
    return min(max(x, low), high)


def wrap(x, low, high):
    """Wraps data between low and high boundaries."""
    diff = high - low
    while x > high:
        x -= diff
    while x < low:
        x += diff
    return x


class StateIndex(object):
    """Flexible way to index states in the CartPole Domain.

    This class enumerates the different indices used when indexing the state.
    e.g. ``s[StateIndex.THETA]`` is guaranteed to return the angle state.
    """
    THETA, THETA_DOT = 0, 1
    X, X_DOT = 2, 3
    PEND_ACTION = 2
    CARTPOLE_ACTION = 4