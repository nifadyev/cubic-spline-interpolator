"""
"""
from dataclasses import dataclass
from bisect import bisect, bisect_left
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


# Cubic spline interpolation using scipy module
# from scipy.interpolate import CubicSpline
# x = np.arange(10)
# y = np.sin(x)
# cs = CubicSpline(x, y)
# xs = np.arange(-0.5, 9.6, 0.1)
# plt.figure(figsize=(6.5, 4))
# plt.plot(x, y, 'o', label='data')
# plt.plot(xs, np.sin(xs), label='true')
# plt.plot(xs, cs(xs), label="S")
# plt.plot(xs, cs(xs, 1), label="S'")
# plt.plot(xs, cs(xs, 2), label="S''")
# plt.plot(xs, cs(xs, 3), label="S'''")
# plt.xlim(-0.5, 9.5)
# plt.legend(loc='lower left', ncol=2)
# plt.show()

# https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.interpolate.CubicSpline.html
# https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html

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

        x -- values.
        y -- results of function in value x.
        """
        # x - узлы сетки, должны быть упорядочены по возрастанию, кратные узлы запрещены
        # y - значения функции в узлах сетки
        self.x = x
        self.y = y
        # Matrix dimension is lines * 5 (coefficient's number)
        self.lines = len(x)

        self.splines = self.build()
        # self.interpolate()

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
        # Step is constant
        # TODO: Rename step to knot
        step = splines[1].x - splines[0].x

        # Required condition
        # ? Any other condition
        splines[0].c = splines[self.lines-1].c = 0.0

        # Calculate `c` coefficients
        self.solve_equations_system(splines, self.lines, step, self.y)

        # Use backward sweep to simplify calculation process
        for i in range(self.lines - 1, 0, -1):
            hi = splines[i].x - splines[i - 1].x
            splines[i].d = (splines[i].c - splines[i - 1].c) / hi
            splines[i].b = (
                hi * (2.0 * splines[i].c + splines[i - 1].c) / 6.0
                + (y[i] - y[i - 1]) / hi
            )

        return splines

    # TODO: Rename new_x
    def interpolate(self, new_x):
        """Calculate results of interpolated function."""
        # length = len(new_x)
        # interpolated_values = []
        for x in new_x:
            # if x <= self.splines[0].x:  # Если x меньше точки сетки x[0] - пользуемся первым эл-тов массива
            #     index = 0
            # # Если x больше точки сетки x[n - 1] - пользуемся последним эл-том массива
            # elif x >= self.splines[self.lines - 1].x:
            # # elif x >= self.splines[-2].x:
            #     index = self.lines - 1
            # # index = bisect(self.x, x)
            # else:
            #     i = 0
            #     j = self.lines - 1
            #     while i + 1 < j:
            #         k = i + (j - i) // 2
            #         if x <= self.splines[k].x:
            #             j = k
            #         else:
            #             i = k
            #     index = j
            index = bisect_left(self.x, x)
            # if index == 49:
            #     index = 44
            print(f'wrong index {index}')
            spline = self.splines[index]
            delta = x - spline.x

            yield (
                spline.a
                + spline.b * delta
                + spline.c * delta**2 / 2.0
                + spline.d * delta**3 / 6.0
            )

    # TODO: add same func but for non constant step
    @staticmethod
    def solve_equations_system(splines, lines, step, y):
        """Solve system of equations using tridiagonal matrix algorithm (or Thomas algorithm)."""
        alpha = np.zeros(lines - 1)
        beta = np.zeros(lines - 1)

        # Forward sweep - modify coefficients
        # for i in range(1, lines - 1):
        #     F = 6.0 * (y[i + 1] - 2.0 * y[i] + y[i - 1]) / step
        #     z = (step * alpha[i - 1] + 4.0 * step)
        #     alpha[i] = - step / z
        #     beta[i] = (F - step * beta[i - 1]) / z

        for i in range(1, lines - 1):
            hi = splines[i].x - splines[i - 1].x
            hi1 = splines[i + 1].x - splines[i].x
            A = hi
            C = 2.0 * (hi + hi1)
            B = hi1
            F = 6.0 * ((y[i + 1] - y[i]) / hi1 - (y[i] - y[i - 1]) / hi)
            z = (A * alpha[i - 1] + C)
            alpha[i] = -B / z
            beta[i] = (F - A * beta[i - 1]) / z

        # Backward sweep - produce the solution
        for i in range(lines - 2, 0, -1):
            splines[i].c = alpha[i] * splines[i + 1].c + beta[i]


# Построение сплайна
# x - узлы сетки, должны быть упорядочены по возрастанию, кратные узлы запрещены
# y - значения функции в узлах сетки
# n - количество узлов сетки
def BuildSpline(x, y, n):
    # Инициализация массива сплайнов
    splines = [SplineTuple() for _ in range(n)]
    for i in range(n):
        splines[i].x = x[i]
        splines[i].a = y[i]

    # Required condition
    splines[0].c = splines[n - 1].c = 0.0

    # Решение СЛАУ относительно коэффициентов сплайнов c[i] методом прогонки для трехдиагональных матриц
    # Вычисление прогоночных коэффициентов - прямой ход метода прогонки
    # alpha = [0.0 for _ in range(0, n - 1)]
    # beta = [0.0 for _ in range(0, n - 1)]
    alpha = np.zeros(n - 1)
    beta = np.zeros(n - 1)

    for i in range(1, n - 1):
        hi = x[i] - x[i - 1]
        hi1 = x[i + 1] - x[i]
        A = hi
        C = 2.0 * (hi + hi1)
        B = hi1
        F = 6.0 * ((y[i + 1] - y[i]) / hi1 - (y[i] - y[i - 1]) / hi)
        z = (A * alpha[i - 1] + C)
        alpha[i] = -B / z
        beta[i] = (F - A * beta[i - 1]) / z

    # Нахождение решения - обратный ход метода прогонки
    for i in range(n - 2, 0, -1):
        splines[i].c = alpha[i] * splines[i + 1].c + beta[i]

    # По известным коэффициентам c[i] находим значения b[i] и d[i]
    for i in range(n - 1, 0, -1):
        hi = x[i] - x[i - 1]
        splines[i].d = (splines[i].c - splines[i - 1].c) / hi
        splines[i].b = (
            hi * (2.0 * splines[i].c + splines[i - 1].c) / 6.0
            + (y[i] - y[i - 1]) / hi
        )
    return splines


# Вычисление значения интерполированной функции в произвольной точке
def Interpolate(splines, x):
    if not splines:
        return None  # Если сплайны ещё не построены - возвращаем NaN

    n = len(splines)
    s = SplineTuple()

    if x <= splines[0].x:  # Если x меньше точки сетки x[0] - пользуемся первым эл-тов массива
        s = splines[0]
        print(f'valid index 0')
    # Если x больше точки сетки x[n - 1] - пользуемся последним эл-том массива
    elif x >= splines[n - 1].x:
        s = splines[n - 1]
        print(f'valid index {n-1}')
    else:  # Иначе x лежит между граничными точками сетки - производим бинарный поиск нужного эл-та массива
        i = 0
        j = n - 1
        while i + 1 < j:
            k = i + (j - i) // 2
            if x <= splines[k].x:
                j = k
            else:
                i = k
        print(f'valid index {j}')
        s = splines[j]

    dx = x - s.x
    # Вычисляем значение сплайна в заданной точке по схеме Горнера (в принципе, "умный" компилятор применил бы схему Горнера сам, но ведь не все так умны, как кажутся)
    return s.a + (s.b + (s.c / 2.0 + s.d * dx / 6.0) * dx) * dx


# x = [1, 3, 7, 9]
# y = [5, 6, 7, 8]
# new_x = 5


# plt.scatter(x, y)
# plt.plot(x, y)
# plt.scatter(new_x, Interpolate(spline, new_x))
# plt.show()


if __name__ == '__main__':

    x = np.linspace(0, 1, 50)
    # print(x)
    y = x * np.sin(x) / (1 + x**2)
    # plt.scatter(x, y)
    plt.plot(x, y, label='function')

    spline = BuildSpline(x, y, len(x))
    x_new = np.linspace(0, 1, 10)
    right = [Interpolate(spline, x_i) for x_i in x_new]
    plt.plot(x_new, right, label='cubic')

    custom_spline = CubicSplineInterpolator(x, y)
    interp = list(custom_spline.interpolate(x_new))
    # interp = [Interpolate(custom_spline.splines, x_i) for x_i in x_new]
    count = 0
    for i in range(len(x_new)):
        if interp[i] != right[i]:
            print(f'{right[i]=} {interp[i]=}')
            count += 1

    # for i in range(len(x)):
    #     if spline[i].x != custom_spline.splines[i].x:
    #         print('fsdg')
    print(f'{len(x_new)=} {count=}')
    for i in range(len(x)):
        if spline[i].a != custom_spline.splines[i].a:
            print('a')
        elif spline[i].b != custom_spline.splines[i].b:
            print('b')
        elif spline[i].c != custom_spline.splines[i].c:
            print('c')
        elif spline[i].d != custom_spline.splines[i].d:
            print('d')
    plt.plot(x_new, interp, label='custom')
    # plt.plot(x, y, label='func')
    plt.legend()

    # # x = np.arange(10)
    # # y = np.sin(x)
    # cs = CubicSpline(x, y)
    # # xs = np.arange(-0.5, 9.6, 0.1)
    # # plt.figure(figsize=(6.5, 4))
    # plt.figure()
    # # plt.plot(x, y, 'o', label='data')
    # plt.plot(x, x * np.sin(x) / (1 + x**2), label='true')
    # plt.plot(x_new, cs(x_new), label="S")
    # # plt.plot(x_new, cs(x_new, 1), label="S'")
    # # plt.plot(x_new, cs(x_new, 2), label="S''")
    # # plt.plot(x_new, cs(x_new, 3), label="S'''")
    # # plt.xlim(-0.5, 9.5)
    # plt.legend(loc='lower left', ncol=2)
    plt.show()
