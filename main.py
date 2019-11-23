"""
"""
from src.interpolator import CubicSplineInterpolator
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

if __name__ == '__main__':
    x = np.linspace(0, 10, 200)
    y = x * np.sin(x) / (1 + x**2)
    plt.plot(x, y, label='function')

    x_new = np.linspace(0, 10, 500)

    custom_spline = CubicSplineInterpolator(x, y)
    interp = list(custom_spline.interpolate(x_new))
    print('Function: x * sin(x) / (1 + x * x)\n')
    # custom_spline.print_args_and_results()
    # custom_spline.print_interpolated_results(iter(x_new), interp)
    # custom_spline.print_coefficients(iter(x_new))
    custom_spline.print_calculations(iter(x_new), interp)
    # print(f'{len(x_new)=} {count=}')

    plt.plot(x_new, interp, label='custom')
    # plt.plot(x, y, label='func')
    # plt.xlim(-0.5, 9.5)
    # plt.ylim(-0.5, 2.0)
    plt.legend()

    cs = CubicSpline(x, y)
    # plt.figure(figsize=(6.5, 4))
    plt.figure()
    # plt.plot(x, y, 'o', label='data')
    plt.plot(x, x * np.sin(x) / (1 + x**2), label='true')
    plt.plot(x_new, cs(x_new), label="S")
    # plt.plot(x_new, cs(x_new, 1), label="S'")
    # plt.plot(x_new, cs(x_new, 2), label="S''")
    # plt.plot(x_new, cs(x_new, 3), label="S'''")
    plt.xlim(-0.5, 9.5)
    plt.legend(loc='lower left', ncol=2)
    plt.show()
