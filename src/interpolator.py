"""
"""
from dataclasses import dataclass
from itertools import islice
from bisect import bisect_left
import numpy as np


# TODO: добавить точность (погрешность) max((f(x_i) - s(x_i)))
# TODO: добавить type hints

@dataclass
class SplineTuple:
    """Dataclass for storing coefficients and function arguments."""

    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    x: float = 0.0


class CubicSplineInterpolator():
    """Build cubic spline and interpolate it's values."""

    def __init__(self, x, y):
        """Initialize class instance with values.

        x -- ascending values.
        y -- results of function in value x.
        """
        self.x = x
        self.y = y
        # Matrix dimension is lines * 5 (coefficient's number)
        self.lines = len(x)

        self.splines = self.build()

    def build(self):
        """Build cubic spline.

        For finding coefficients tridiagonal matrix algorithm is used.
        """
        splines = [
            SplineTuple(
                a=self.y[i],
                b=0.0,
                c=0.0,
                d=0.0,
                x=self.x[i],
            )
            for i in range(self.lines)
        ]

        # Required condition
        # ? Any other condition
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
                + (self.y[i] - self.y[i - 1]) / delta
            )

        return splines

    def solve_equations_system(self, splines):
        """Solve system of equations using tridiagonal matrix algorithm (or Thomas algorithm).

        splines -- equation system with unknown `c` variables.
        """
        alpha = np.zeros(self.lines - 1)
        beta = np.zeros(self.lines - 1)

        # Forward sweep - modify coefficients
        for i in range(1, self.lines - 1):
            current_delta = splines[i].x - splines[i - 1].x
            next_delta = splines[i + 1].x - splines[i].x
            F = 6.0 * (
                (self.y[i + 1] - self.y[i]) / next_delta
                - (self.y[i] - self.y[i - 1]) / current_delta
            )
            divider = current_delta * (alpha[i - 1] + 2.0 * (1 + next_delta))

            alpha[i] = - next_delta / divider
            beta[i] = (F - current_delta * beta[i - 1]) / divider

        # Backward sweep - produce the solution
        for i in range(self.lines - 2, 0, -1):
            splines[i].c = alpha[i] * splines[i + 1].c + beta[i]

    def interpolate(self, sequence):
        """Calculate interpolated value for each item from specified array.

        sequence -- values, usually from the same interval as origin values.

        Yields -- interpolated value in specified point.
        Args:
        ----
            n: the number to get the square root of.
        Returns:
        --------
            the square root of n.
        Raises:
        -------
            TypeError: if n is not a number.
            ValueError: if n is negative.

        """
        for x in sequence:
            # Use binary search to find closest value from `self.x`
            index = bisect_left(self.x, x)
            spline = self.splines[index]
            delta = x - spline.x

            yield (
                spline.a
                + spline.b * delta
                + spline.c * delta**2 / 2.0
                + spline.d * delta**3 / 6.0
            )

    def print_calculations(self, new_values, interpolated_results):
        print('Function arguments and results:\n')
        self.print_args_and_results(self.x, self.y)

        print('\nNew arguments and interpolated results:\n')
        self.print_args_and_results(new_values, interpolated_results)

        print('\nCoefficients on each step:\n')
        self.print_coefficients(new_values)

    def print_args_and_results(self, args, solutions):
        """Pretty print `x` values and results of function is each `x`."""
        # Show only first 20 values and results
        values = " | ".join(f'{value:6.3f}' for value in islice(args, 20))
        results = " | ".join(f'{result:6.3f}' for result in islice(solutions, 20))
        vertical_line = '-' * (len(values) + len(' x | '))

        print(f' x | {values}')
        print(vertical_line)
        print(f' y | {results}')

    def print_coefficients(self, x):
        """Pretty print spline coefficients on each step."""
        table_header = 'Step|    x    |    a    |    b    |    c    |    d    '

        print(table_header)
        print('-' * len(table_header))

        for step_number, spline in enumerate(islice(self.splines, 10), start=1):
            # Format coeffs to max 7 chars length and 3 digits after float
            print(
                f' {step_number:2} | {next(x):7.3f} | {spline.a:7.3f} |'
                f' {spline.b:7.3f} | {spline.c:7.3f} | {spline.d:7.3f}'
            )
