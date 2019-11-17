import math
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

# ! Use dataclass
# class SplineTuple:
#     def __init__(self, a, b, c, d, x):
#         self.a = a
#         self.b = b
#         self.c = c
#         self.d = d
#         self.x = x

class CubeSplineInterpolator():

    @staticmethod
    def f(x):
        """ Basic function to get values in specified interval."""
        if not -1 <= x <= 1:
            print('x should be more or equal than -1 and less or equal than 1')
            return

        # sign = -1 if -1 <= x <= 0 else 1
        sign = -1 if -1 <= x <= 0.5 else 1
        return sign * x**3 + 3 * x**2

        # return math.cos(x**2 / 7.0)

    @staticmethod
    def df(x):
        """ First function derivative."""

        sign = -1 if -1 <= x <= 0.5 else 1
        return sign * 3 * x**2 + 6 * x

    @staticmethod
    def d2f(x):
        """ Second function derivative."""

        sign = -1 if -1 <= x <= 0.5 else 1
        return sign * 6 * x + 6

    # @staticmethod
    # def S(x):
    #     """Spline."""
    #     return


# Структура, описывающая сплайн на каждом сегменте сетки


class SplineTuple:
    def __init__(self, a, b, c, d, x):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.x = x

# Построение сплайна
# x - узлы сетки, должны быть упорядочены по возрастанию, кратные узлы запрещены
# y - значения функции в узлах сетки
# n - количество узлов сетки


def BuildSpline(x, y, n):
    # Инициализация массива сплайнов
    splines = [SplineTuple(0, 0, 0, 0, 0) for _ in range(0, n)]
    for i in range(0, n):
        splines[i].x = x[i]
        splines[i].a = y[i]

    splines[0].c = splines[n - 1].c = 0.0

    # Решение СЛАУ относительно коэффициентов сплайнов c[i] методом прогонки для трехдиагональных матриц
    # Вычисление прогоночных коэффициентов - прямой ход метода прогонки
    alpha = [0.0 for _ in range(0, n - 1)]
    beta = [0.0 for _ in range(0, n - 1)]

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
        splines[i].b = hi * (2.0 * splines[i].c +
                             splines[i - 1].c) / 6.0 + (y[i] - y[i - 1]) / hi
    return splines


# Вычисление значения интерполированной функции в произвольной точке
def Interpolate(splines, x):
    if not splines:
        return None  # Если сплайны ещё не построены - возвращаем NaN

    n = len(splines)
    s = SplineTuple(0, 0, 0, 0, 0)

    if x <= splines[0].x:  # Если x меньше точки сетки x[0] - пользуемся первым эл-тов массива
        s = splines[0]
    # Если x больше точки сетки x[n - 1] - пользуемся последним эл-том массива
    elif x >= splines[n - 1].x:
        s = splines[n - 1]
    else:  # Иначе x лежит между граничными точками сетки - производим бинарный поиск нужного эл-та массива
        i = 0
        j = n - 1
        while i + 1 < j:
            k = i + (j - i) // 2
            if x <= splines[k].x:
                j = k
            else:
                i = k
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

    x = np.linspace(0, 10, 50)
    # print(x)
    y = np.sin(x+10)
    # plt.scatter(x, y)
    plt.plot(x, y, label='function')

    spline = BuildSpline(x, y, len(x))
    x_new = np.linspace(0, 10, 10)
    plt.plot(x_new, [Interpolate(spline, x_i) for x_i in x_new], label='cubic')
    # plt.plot(x, y, label='func')
    plt.legend()

    # x = np.arange(10)
    # y = np.sin(x)
    cs = CubicSpline(x, y)
    # xs = np.arange(-0.5, 9.6, 0.1)
    # plt.figure(figsize=(6.5, 4))
    plt.figure()
    # plt.plot(x, y, 'o', label='data')
    plt.plot(x, np.sin(x+10), label='true')
    plt.plot(x_new, cs(x_new), label="S")
    # plt.plot(x_new, cs(x_new, 1), label="S'")
    # plt.plot(x_new, cs(x_new, 2), label="S''")
    # plt.plot(x_new, cs(x_new, 3), label="S'''")
    # plt.xlim(-0.5, 9.5)
    plt.legend(loc='lower left', ncol=2)
    plt.show()
