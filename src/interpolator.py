"""
"""
from dataclasses import dataclass
from bisect import bisect_left
import numpy as np

# TODO: добавить вывод промежуточной инфы
# TODO: добавить задание шага между х
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
