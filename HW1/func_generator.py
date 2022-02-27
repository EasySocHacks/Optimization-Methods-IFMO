import numpy as np


def func_single(prev_func, i, constant=1):
    return lambda coord: prev_func(coord) + coord[i] * constant


def func_double(prev_func, i, j, constant=1):
    return lambda coord: prev_func(coord) + coord[i] * coord[j] * constant


def func_zero():
    return lambda coord: 0


def func_scale_result(prev_func, constant=1):
    return lambda coord: prev_func(coord) * constant


def generate_function(dims, inner_constant_scale=100, outer_constant_scale=1, print_debug=False):
    f = func_zero()
    if print_debug:
        print("0", end="")
    # 1 for const -- not used
    # n for x_1
    # (1 + n) * n / 2  -- for x_i * x_j, i>=j
    # n + (1 + n) / 2 * n = (n + 3) * n / 2

    for i in range((dims + 3) * dims):
        if np.random.random() >= 0.5:
            rand_const = inner_constant_scale * np.random.uniform(-1, 1)
            if i < dims:
                if print_debug:
                    print(" + \n{:.2f}x{}".format(rand_const, i), end="")
                f = func_single(f, i, rand_const)
            else:
                i_ind = (i - dims) // dims
                j_ind = (i - dims) % dims
                if i_ind <= j_ind:
                    if print_debug:
                        print(" + \n{:.2f}(x{}*x{})".format(rand_const, i_ind, j_ind), end="")
                    f = func_double(f, i_ind, j_ind, rand_const)
    if print_debug:
        print()
    return func_scale_result(f, outer_constant_scale)
