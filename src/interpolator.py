"""Build cubic spline and interpolate data."""

from dataclasses import dataclass
from itertools import islice
from bisect import bisect_left

import numpy as np


# TODO: Add type hints


@dataclass
class Spline:
    """Store coefficients and function arguments."""

    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    x: float = 0.0


class CubicSplineInterpolator():
    """Build cubic spline and interpolate it's values."""

    def __init__(self, left_boundary, right_boundary, epsilon, steps, function):
        """Initialize class instance with values.

        Args:
            function_arguments: ascending function arguments.
            results: function results.
            spline_arguments: ascending spline arguments.

        """
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.epsilon = epsilon
        self.steps = steps
        self.function = function

        self.args, self.results = self.do_everything()

    def do_everything(self):
        x_range = [self.left_boundary, self.right_boundary]
        l = self.left_boundary
        r = self.right_boundary
        h = (r - l) / (self.steps - 1)

        splines = []
        f = []
        y_max = -np.Infinity
        y_min = np.Infinity

        for i in range(self.steps):
            value = l + i * h
            result = self.function(value)

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
        splines[self.steps - 1].c = 0.0

        for i in range(1, self.steps - 1):
            alpha.append(-1 / (4 + alpha[i-1]))
            beta.append(
                1 / (4 + alpha[i - 1]) * (6 / (h * h)
                * (f[i + 1] - 2 * f[i] + f[i - 1]) - beta[i - 1])
            )

        for i in range(self.steps-2, 0, -1):
            splines[i].c = alpha[i] * splines[i+1].c + beta[i]

        for i in range(self.steps-1, 0, -1):
            splines[i].d = (splines[i].c - splines[i - 1].c) / h
            splines[i].b = (
                h / 2 * splines[i].c - h**2 / 6 * splines[i].d
                + (f[i] - f[i-1]) / h
            )

        args = []
        results = []
        epsilon = (x_range[1] - x_range[0]) / 30000

        for i in range(1, len(splines)):
            x = splines[i - 1].x
            while x <= splines[i].x:
                dx = x - splines[i].x
                current_value = splines[i].a + splines[i].b * dx + splines[i].c / 2 * dx**2 + splines[i].d / 6 * dx**3
                # dots.append([x, current_value])
                args.append(x)
                results.append(current_value)
                y_max = max(y_max, current_value)
                y_min = min(y_min, current_value)
                x += epsilon
            # x = splines[i].x

        x_diff = x_range[1] - x_range[0]
        y_diff = y_max - y_min

        print('Function arguments and results:\n')
        self.print_args_and_results(args, results)

        print('\nCoefficients on each step:\n')
        self.print_coefficients(splines)

        print(f'\nInterpolation error: {y_diff:.5f}')

        return args, results
        # return dots


    def build(self):
        """Build cubic spline.

        For finding coefficients tridiagonal matrix algorithm is used.

        Returns:
            splines: list of `Spline` dataclass instances.

        """
        splines = [
            Spline(
                a=self.results[i],
                b=0.0,
                c=0.0,
                d=0.0,
                x=self.function_arguments[i],
            )
            for i in range(self.lines)
        ]

        # Required condition
        splines[0].c = splines[self.lines-1].c = 0.0

        # Calculate `c` coefficients
        self.solve_equations_system(splines)

        # Use backward sweep to simplify calculation process
        for i in range(self.lines - 1, 0, -1):
            # Interval between nearby values
            delta = splines[i].x - splines[i - 1].x
            splines[i].d = (splines[i].c - splines[i - 1].c) / delta
            splines[i].b = (
                delta * (2.0 * splines[i].c + splines[i - 1].c) / 6.0
                + (self.results[i] - self.results[i - 1]) / delta
            )

        return splines

    def solve_equations_system(self, splines):
        """Solve system of equations using tridiagonal matrix algorithm.

        Args:
            splines: equation system with unknown `c` variables.

        """
        alpha = np.zeros(self.lines - 1)
        beta = np.zeros(self.lines - 1)

        # Forward sweep - modify coefficients
        for i in range(1, self.lines - 1):
            current_delta = splines[i].x - splines[i - 1].x
            next_delta = splines[i + 1].x - splines[i].x
            F = 6.0 * (
                (self.results[i + 1] - self.results[i]) / next_delta
                - (self.results[i] - self.results[i - 1]) / current_delta
            )
            divider = current_delta * (alpha[i - 1] + 2.0 * (1 + next_delta))

            alpha[i] = - next_delta / divider
            beta[i] = (F - current_delta * beta[i - 1]) / divider

        # Backward sweep - produce the solution
        for i in range(self.lines - 2, 0, -1):
            splines[i].c = alpha[i] * splines[i + 1].c + beta[i]

    def interpolate(self, value):
        """Calculate interpolated value.

        Args:
            value: argument to interpolate.

        """
        # Use binary search to find closest value from `function_arguments`
        index = bisect_left(self.function_arguments, value)
        spline = self.splines[index]
        delta = value - spline.x

        return (
            spline.a
            + spline.b * delta
            + spline.c * delta**2 / 2.0
            + spline.d * delta**3 / 6.0
        )

    def print_calculations(self):
        """Print results of various calculations.

        They were calculated during building spline and interpolating data.
        """
        print('Function arguments and results:\n')
        self.print_args_and_results(self.function_arguments, self.results)

        print('\nSpline arguments and interpolated values:\n')
        self.print_args_and_results(
            self.spline_arguments, self.interpolated_data)

        print('\nCoefficients on each step:\n')
        self.print_coefficients()

        error = self.get_interpolation_error(function)
        print(f'\nInterpolation error: {error:.5f}')

    @staticmethod
    def print_args_and_results(args, results):
        """Pretty print arguments and function results."""
        args_slice = islice(args, 20) if len(args) > 20 else args
        results_slice = islice(results, 20) if len(results) > 20 else results
        # Show only first 20 values and results
        values = " | ".join(f'{value:6.3f}' for value in args_slice)
        solutions = " | ".join(f'{result:6.3f}' for result in results_slice)
        vertical_line = '-' * (len(values) + len(' x | '))

        print(f' x | {values}')
        print(vertical_line)
        print(f' y | {solutions}')

    def print_coefficients(self, splines):
        """Pretty print spline coefficients on each step."""
        table_header = 'Step|    x    |    a    |    b    |    c    |    d    '
        splines_slice = islice(splines, 10) if len(splines) > 10\
            else splines[:-1]

        print(table_header)
        print('-' * len(table_header))

        for step_number, spline in enumerate(splines_slice, start=1):
            # Format coefficients to 7 chars string with 3 digits after float
            print(
                f' {step_number:2} | {spline.x:7.3f} | {spline.a:7.3f}'
                f' | {spline.b:7.3f} | {spline.c:7.3f} | {spline.d:7.3f}'
            )

    def get_interpolation_error(self, function):
        """Max diff between function result and interpolated value."""
        return max(
            function(arg) - next(iter(self.interpolated_data))
            for arg in self.spline_arguments
        )
