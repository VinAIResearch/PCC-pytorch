"""Common functions for the MDP package."""

from __future__ import absolute_import, division, print_function


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

    THETA_1, THETA_2, THETA_3 = 0, 2, 4
    THETA_1_DOT, THETA_2_DOT, THETA_3_DOT = 1, 3, 5
    TORQUE_3_1, TORQUE_3_2, TORQUE_3_3 = 6, 7, 8
