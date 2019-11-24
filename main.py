"""Demonstration of CubicSplineInterpolator usage."""

from dataclasses import dataclass
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
import click

from src.interpolator import CubicSplineInterpolator

@dataclass
class Spline:
    """Store coefficients and function arguments."""

    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    x: float = 0.0

def function(argument):
    """Find function's result with specified argument."""
    return np.cos(argument**2/4)
    # return argument * np.sin(argument) / (1 + argument**2)

def do_everything(lhs, rhs, steps, function):
    # x_range = np.linspace(lhs, rhs, (rhs-lhs) / steps) 
    x_range = [lhs, rhs]
    l = lhs
    r = rhs
    h = (r - l) / (steps - 1)
    

    splines = []
    f = []
    y_max = -np.Infinity
    y_min = np.Infinity

    for i in range(steps):
        value = l + i * h
        result = function(value)

        f.append(result)
        splines.append(
            Spline(
                a=result,
                b=0.0,
                c=0.0,
                d=0.0,
                x=value,
            )
        )

    alpha, beta = [0.0], [0.0]
    splines[0].c = 0.0
    splines[steps - 1].c = 0.0

    for i in range(1, steps - 1):
        alpha.append(-1 / (4 + alpha[i-1]))
        beta.append(
            1 / (4 + alpha[i - 1]) * (6 / (h * h)\
            * (f[i + 1] - 2 * f[i] + f[i - 1]) - beta[i - 1])
        )
    for spline in splines:
        print(spline.x)

    for i in range(steps-2, 0, -1):
        splines[i].c = alpha[i] * splines[i+1].c + beta[i]

    for i in range(steps-1, 0, -1):
        splines[i].d = (splines[i].c - splines[i - 1].c) / h
        splines[i].b = h / 2 * splines[i].c - h**2 / 6 * splines[i].d + (f[i] - f[i-1]) / h

    dots = []
    epsilon = (x_range[1] - x_range[0]) / 30000

    for i in range(1, len(splines)):
        x = splines[i - 1].x
        while x <= splines[i].x:
            dx = x - splines[i].x
            current_value = splines[i].a + splines[i].b * dx + splines[i].c / 2 * dx**2 + splines[i].d / 6 * dx**3
            dots.append([x, current_value])
            y_max = max(y_max, current_value)
            y_min = min(y_min, current_value)
            x += epsilon
        # x = splines[i].x

    x_diff = x_range[1] - x_range[0]
    y_diff = y_max - y_min

    return dots

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
    dots = do_everything(range_start, range_end, step, function)

    x = [i[0] for i in dots]
    def_y = [function(i) for i in x]
    print(len(x))
    y = [i[1] for i in dots]

    plt.figure('Interpolator')
    plt.title('Cubic spline interpolation with tridiagonal matrix algorithm')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    # plt.plot(function_arguments, results, label='function')
    plt.plot(x, y, label='spline')
    plt.plot(x, def_y, label='fucntion')
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
