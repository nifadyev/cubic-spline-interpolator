"""Build cubic spline and interpolate data."""

from dataclasses import dataclass
from itertools import islice

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

    def __init__(
        self, left_boundary, right_boundary, epsilon, intervals, function):
        """Initialize class instance with values.

        Args:
            function_arguments: ascending function arguments.
            results: function results.
            spline_arguments: ascending spline arguments.

        """
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.epsilon = epsilon
        self.intervals = intervals
        self.function = function

        self.splines = self.build()
        self.args, self.results = self.interpolate()

    def build(self):
        """Build cubic splines.

        For finding coefficients tridiagonal matrix algorithm is used.

        Returns:
            splines: list of `Spline` dataclass instances.

        """
        step = (self.right_boundary - self.left_boundary) / (self.intervals - 1)

        splines = []
        function_results = []

        for i in range(self.intervals):
            value = self.left_boundary + i * step
            result = self.function(value)

            function_results.append(result)
            splines.append(
                Spline(
                    a=result,
                    b=0.0,
                    c=0.0,
                    d=0.0,
                    x=value,
                )
            )

        # Required conditions
        splines[0].c = splines[self.intervals - 1].c = 0.0

        self.solve_equations_system(splines, step, function_results)

        # Use backward sweep to simplify calculation process
        for i in range(self.intervals-1, 0, -1):
            splines[i].d = (splines[i].c - splines[i - 1].c) / step
            splines[i].b = (
                step / 2 * splines[i].c
                - step**2 / 6 * splines[i].d
                + (function_results[i] - function_results[i-1]) / step
            )

        return splines

    def solve_equations_system(self, splines, step, function_results):
        """Solve system of equations using tridiagonal matrix algorithm.

        Args:
            splines: equation system with unknown `c` variables.

        """
        alpha = np.zeros(self.intervals - 1)
        beta = np.zeros(self.intervals - 1)

        # Forward sweep - modify coefficients
        for i in range(1, self.intervals - 1):
            alpha[i] = (-1 / (4 + alpha[i-1]))
            beta[i] = (
                1 / (4 + alpha[i-1])
                * (6 / step**2
                   * (function_results[i + 1] - 2 * function_results[i]
                      + function_results[i - 1])
                   - beta[i - 1])
            )

        # Backward sweep - produce the solution
        for i in range(self.intervals-2, 0, -1):
            splines[i].c = alpha[i] * splines[i+1].c + beta[i]

    def interpolate(self):
        """Calculate interpolated value.

        Args:
            value: argument to interpolate.

        """
        args, results = [], []

        for i in range(1, self.intervals):
            current_argument = self.splines[i - 1].x
            while current_argument <= self.splines[i].x:
                current_interval = current_argument - self.splines[i].x
                current_result = (
                    self.splines[i].a
                    + self.splines[i].b * current_interval
                    + self.splines[i].c / 2 * current_interval**2
                    + self.splines[i].d / 6 * current_interval**3
                )

                args.append(current_argument)
                results.append(current_result)
                current_argument += self.epsilon

        return args, results

    def print_calculations(self):
        """Print results of various calculations.

        They were calculated during building spline and interpolating data.
        """
        print('Function arguments and results:\n')
        function_results = [self.function(arg) for arg in self.args]
        self.print_args_and_results(self.args, function_results)

        print('\nSpline arguments and interpolated values:\n')
        self.print_args_and_results(self.args, self.results)

        print('\nCoefficients on each step:\n')
        self.print_coefficients()

        error = self.get_interpolation_error()
        print(f'\nInterpolation error: {error:.5f}')

    @staticmethod
    def print_args_and_results(args, results):
        """Pretty print arguments and results.

        Results can be produced by function or by interpolation.
        """
        # Show only first 15 values and results
        args_slice = islice(args, 15)
        results_slice = islice(results, 15)
        values = " | ".join(f'{value:7.3f}' for value in args_slice)
        solutions = " | ".join(f'{result:7.3f}' for result in results_slice)
        vertical_line = '-' * (len(values) + len(' x | '))

        print(f' x | {values}')
        print(vertical_line)
        print(f' y | {solutions}')

    def print_coefficients(self):
        """Pretty print spline coefficients on each step."""
        table_header = 'Step|    x    |    a    |    b    |    c    |    d    '
        splines_slice = islice(self.splines, 10)

        print(table_header)
        print('-' * len(table_header))

        for step_number, spline in enumerate(splines_slice, start=1):
            # Format coefficients to 7 chars string with 3 digits after float
            print(
                f' {step_number:2} | {spline.x:7.3f} | {spline.a:7.3f}'
                f' | {spline.b:7.3f} | {spline.c:7.3f} | {spline.d:7.3f}'
            )

    def get_interpolation_error(self):
        """Max diff between function result and interpolated value."""
        res = iter(self.results)
        return max(
            self.function(arg) - next(res)
            for arg in self.args
        )
