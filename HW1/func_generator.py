import numpy as np


def func_single(prev_func, i, constant=1):
    """
    Wraps function 'prev_func' with adding to its invocation result the value of 'i'-th point's coordinate
     value scaled by 'constant'
    :param prev_func: function that computes some polynomial's value at given point
    :param i: the number of point's coordinate to be used
    :param constant: scale coefficient
    :return: the new function that like f(coordinate)='prev_func'(coordinate)+'constant'*coordinate['i']
    """
    return lambda coord: prev_func(coord) + coord[i] * constant


def func_double(prev_func, i, j, constant=1):
    """
    Wraps function 'prev_func' with adding to its invocation result the value of 'i'-th point's coordinate
      value multiplied by 'j'-th point's coordinate value and scaled by 'constant'
    :param prev_func: function that computes some polynomial's value at given point
    :param i: the number of point's first coordinate to be used
    :param j: the number of point's second coordinate to be used
    :param constant: scale coefficient
    :return: the new function that like f(coordinate)='prev_func'(coordinate)+'constant'*coordinate['i']*coordinate['j']
    """
    return lambda coord: prev_func(coord) + coord[i] * coord[j] * constant


def func_zero():
    """
    Returns function that always return zero for any given point
    :return: f(coordinate)=0
    """
    return lambda coord: 0


def func_scale_result(prev_func, constant=1):
    """
    Wraps given function 'prev_func' by scaling its invocation result by scaling coefficient 'constant'
    :param prev_func: function that computes some polynomial's value at given point
    :param constant: scale coefficient
    :return: the new function that like f(coordinate)='prev_func'(coordinate)*'constant'
    """
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
