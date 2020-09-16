"""Demonstration of CubicSplineInterpolator usage."""

from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
import click

from src.interpolator import CubicSplineInterpolator


def function(argument: float) -> float:
    """Find function's result with specified argument."""
    return argument * np.sin(argument) / (1 + argument**2)


@click.command()
@click.option('--range_start', type=float, default=-10.0, help='Left range boundary')
@click.option('--range_end', type=float, default=10.0, help='Right range boundary')
@click.option('--range_length', type=int, default=30000, help='Number of values in range')
@click.option('--intervals', type=int, default=10, help='Number of intervals')
@click.option(
    '--compare_default/--do_not_compare', default=False,
    help='Compare custom spline with default scipy spline'
)
def main(
        range_start: float,
        range_end: float,
        range_length: int,
        intervals: int,
        compare_default: bool
) -> None:
    """Build spline and interpolate it's values.

    Set range boundaries for CubicSplineInterpolator to interpolate spline in specified intervals.
    """
    epsilon = (range_end - range_start) / range_length
    spline = CubicSplineInterpolator(range_start, range_end, epsilon, intervals, function)
    arguments = spline.args
    interpolated_results = spline.results

    function_results = tuple(function(i) for i in arguments)

    spline.print_calculations()

    plt.figure('Interpolator')
    plt.title('Function: x * sin(x) / (1 + x * x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.plot(arguments, function_results, label='function')
    plt.plot(arguments, interpolated_results, label='spline')

    if compare_default:
        scipy_spline = CubicSpline(arguments, interpolated_results)
        plt.figure('Scipy spline interpolator')
        plt.plot(arguments, function_results, label='function')
        plt.plot(arguments, scipy_spline(arguments), label="S")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
