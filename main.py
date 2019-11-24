"""Demonstration of CubicSplineInterpolator usage."""

from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
import click

from src.interpolator import CubicSplineInterpolator


def function(argument):
    """Find function's result with specified argument."""
    return argument * np.sin(argument) / (1 + argument**2)

@click.command()
@click.option(
    '--range_start', type=float, default=-10.0, help='Left range boundary')
@click.option(
    '--range_end', type=float, default=10.0, help='Right range boundary')
@click.option(
    '--step', type=int, default=10, help='Step between values in range')
@click.option(
    '--compare_default/--do_not_compare', default=False,
    help='Compare custom spline with default scipy spline'
)
def main(range_start, range_end, step, compare_default):
    spline = CubicSplineInterpolator(range_start, range_end, 0.0, step, function)
    x = spline.args
    y = spline.results

    def_y = [function(i) for i in x]

    plt.figure('Interpolator')
    plt.title('Cubic spline interpolation with tridiagonal matrix algorithm')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.plot(x, def_y, label='function')
    plt.plot(x, y, label='spline')
    plt.legend()
    plt.show()



# @click.command()
# @click.option(
#     '--range_start', type=float, default=0.0, help='Left range boundary')
# @click.option(
#     '--range_end', type=float, default=10.0, help='Right range boundary')
# @click.option(
#     '--step', type=float, default=0.25, help='Step between values in range')
# @click.option(
#     '--compare_default/--do_not_compare', default=False,
#     help='Compare custom spline with default scipy spline'
# )
# def main(range_start, range_end, step, compare_default):
#     """Build spline and interpolate it's values.

#     Set function and it's function_arguments and use CubicSplineInterpolator
#     to interpolate spline in specified function_arguments.
#     """
#     function_arguments = np.linspace(
#         range_start, range_end, (range_end - range_start) / 0.25)
#     spline_arguments = np.linspace(
#         range_start, range_end, (range_end - range_start) / step)
#     results = [function(arg) for arg in function_arguments]

#     spline = CubicSplineInterpolator(
#         function_arguments, results, steps)
#     # spline = CubicSplineInterpolator(
#     #     function_arguments, results, spline_arguments)

#     print('Function: x * sin(x) / (1 + x * x)\n')
#     spline.print_calculations(function)

#     plt.figure('Interpolator')
#     plt.title('Cubic spline interpolation with tridiagonal matrix algorithm')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.grid(True)
#     plt.plot(function_arguments, results, label='function')
#     plt.plot(spline_arguments, spline.interpolated_data, label='spline')

#     if compare_default:
#         scipy_spline = CubicSpline(function_arguments, results)
#         plt.figure('Scipy spline interpolator')
#         # plt.plot(function_arguments, results, 'o', label='data')
#         plt.plot(function_arguments, results, label='function')
#         plt.plot(spline_arguments, scipy_spline(spline_arguments), label="S")
#         plt.plot(
#             spline_arguments, scipy_spline(spline_arguments, 1), label="S'")
#         # plt.plot(
#         #     spline_arguments, scipy_spline(spline_arguments, 2), label="S''")
#         # plt.plot(
#             # spline_arguments, scipy_spline(spline_arguments, 3), label="S'''")

#     plt.legend()
#     plt.show()


if __name__ == '__main__':
    main()
