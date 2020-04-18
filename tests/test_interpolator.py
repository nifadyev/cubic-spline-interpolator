import pytest

import src.interpolator as intr
import numpy as np


def function(argument):
    """Find function's result with specified argument."""
    return argument * np.sin(argument) / (1 + argument**2)

class TestCubicSplineInterpolator():

    def test_get_interpolation_error(self):
        epsilon = (10.0 + 10.0) / 30000
        interpolator = intr.CubicSplineInterpolator(-10.0, 10.0, epsilon, 10, function)

        assert interpolator.get_interpolation_error() == 0.11197798190992969