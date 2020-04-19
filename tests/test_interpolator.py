import src.interpolator as intr
import numpy as np


def test_get_interpolation_error():
    """Check if interpolation error is calculated as expected."""
    epsilon = (10.0 + 10.0) / 30000
    interpolator = intr.CubicSplineInterpolator(
        -10.0, 10.0, epsilon, 10, lambda arg: arg * np.sin(arg) / (1 + arg**2))

    assert interpolator.get_interpolation_error() == 0.11197798190992969
