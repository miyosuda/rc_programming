import symengine as sm
import itertools
import math


def mv_series(function_expression,
              variable_list, 
              evaluation_point, 
              degree):
    """ 
    Multivariate version of sympy series() function.
    https://stackoverflow.com/questions/22857162/multivariate-taylor-approximation-in-sympy
    """
    
    n_var = len(variable_list)
    
    # dict of variables and their evaluation_point coordinates, ready for the subs() method
    point_coordinates = {i : j for i, j in zip(variable_list, evaluation_point) }
    
    # list with exponents of the partial derivatives
    deriv_orders = list(itertools.product(range(degree + 1), repeat=n_var))
    
    # Discarding some higher-order terms
    deriv_orders = [deriv_orders[i] for i in range(len(deriv_orders)) if sum(deriv_orders[i]) <= degree]
    n_terms = len(deriv_orders)
    
    # same as before, but now ready for the diff() method
    deriv_orders_as_input = [list(sum(list(zip(variable_list, deriv_orders[i])), ())) for i in range(n_terms)]
    
    polynomial = 0
    for i in range(n_terms):
        partial_derivatives_at_point = function_expression.diff(*deriv_orders_as_input[i]).subs(point_coordinates)
        # e.g. df/(dx*dy**2) evaluated at (x0,y0)
        
        denominator = math.prod([math.factorial(j) for j in deriv_orders[i]])
        # e.g. (1! * 2!)
        
        distances_powered = math.prod([(sm.Matrix(variable_list) - sm.Matrix(evaluation_point))[j] ** deriv_orders[i][j] for j in range(n_var)])  
        # e.g. (x-x0)*(y-y0)**2
        
        polynomial += partial_derivatives_at_point / denominator * distances_powered
    return polynomial

    
def compare_degrees(degrees0, degrees1):
    def find_first_hit_pos(degrees):
        for i,degree in enumerate(degrees):
            if degree != 0:
                return i
        return -1

    def compare(a, b):
        if a == b: 
            return 0
        if a > b:
            return 1
        else:
            return -1
    
    var_size = len(degrees0) // 2

    xs0 = degrees0[:var_size]
    xs1 = degrees1[:var_size]    
    xdots0 = degrees0[var_size:]    
    xdots1 = degrees1[var_size:]
    
    x_max_degree0 = max(xs0)
    x_max_degree1 = max(xs1)
    xdot_max_degree0 = max(xdots0)
    xdot_max_degree1 = max(xdots1)
    
    x_sum_degree0 = sum(xs0)
    x_sum_degree1 = sum(xs1)    
    xdot_sum_degree0 = sum(xdots0)
    xdot_sum_degree1 = sum(xdots1)
    
    x_first_hit0 = find_first_hit_pos(xs0)
    x_first_hit1 = find_first_hit_pos(xs1)
    xdot_first_hit0 = find_first_hit_pos(xdots0)
    xdot_first_hit1 = find_first_hit_pos(xdots1)
    
    ret = compare(xdot_sum_degree0, xdot_sum_degree1)
    if ret != 0:
        return ret
    
    ret = compare(x_sum_degree0, x_sum_degree1)
    if ret != 0:
        return ret    
    
    ret = compare(x_first_hit0, x_first_hit1)
    if ret != 0:
        return ret
        
    ret = compare(x_max_degree0, x_max_degree1)
    if ret != 0:
        return -ret
        
    ret = compare(xdot_sum_degree0, xdot_sum_degree1)
    if ret != 0:
        return ret    
    
    ret = compare(xdot_first_hit0, xdot_first_hit1)
    if ret != 0:
        return ret    
        
    ret = compare(xdot_max_degree0, xdot_max_degree1)
    if ret != 0:
        return -ret
        
    return 0


def get_term_name(degrees):
    if sum(degrees) == 0:
        return '1'
    
    ret = ''
    for i,degree in enumerate(degrees):
        if degree > 0:
            if i < len(degrees) // 2:
                ret += f'x{i}'
            else:
                ret += f'xd{i-len(degrees)//2}'
            if degree >= 2:
                ret += f'^{degree}'
    return ret


def get_poly_dict(p, var_list):
    """ 
    When making poly symengine expression, error occurs with var list arguemnt poly(expr, var_list),
    so we made workaround to reorder coeffs.
    """
    def reoder_key(key, indices):
        ret = [0] * len(var_list)
        for degree,index in zip(key, indices):
            ret[index] = degree
        return tuple(ret)
    
    poly_vars = p.args[1:]
    indices = []
    for poly_var in poly_vars:
        index = var_list.index(poly_var)
        indices.append(index)
    d = p.as_dict()
    new_d = {}
    for key,value in d.items():
        new_key = reoder_key(key, indices)
        new_d[new_key] = value
    return new_d


import numpy as np

def get_target_coeffs(target_coeff_dicts, degrees_list):
    target_coeffs = []

    for coeff_dict in target_coeff_dicts:
        coeffs = []
        for degrees in degrees_list:
            if degrees in coeff_dict:
                coeffs.append(coeff_dict[degrees])
            else:
                coeffs.append(0)            
        target_coeffs.append(coeffs)

    target_coeffs = np.array(target_coeffs, dtype=np.float64)
    return target_coeffs


import joblib
import sympy as smp
from functools import cmp_to_key


def process_poly(expr,
                 x_xdot_list, 
                 x_xdot_zero, 
                 series_degree):
    r_ex = mv_series(expr, 
                     x_xdot_list, 
                     x_xdot_zero, 
                     series_degree)                 
    # Using sympy instead of symengine for poly().
    p_c = smp.poly(r_ex)
    return get_poly_dict(p_c, x_xdot_list)


def calc_coeffs(expr,
                x_xdot_list,
                x_xdot_zero_list,
                series_degree,
                verbose=0):
    r_dim = len(expr)
    
    coeff_dicts = joblib.Parallel(
        n_jobs=-1,
        verbose=verbose)( [joblib.delayed(process_poly)(
            expr[r_index, 0],
            x_xdot_list,
            x_xdot_zero_list,
            series_degree) for r_index in range(r_dim)])

    degrees_sets = set()

    for coeff_dict in coeff_dicts:
        for degrees in coeff_dict.keys():
            degrees_sets.add(degrees)

    degrees_list = sorted(list(degrees_sets),
                          key=cmp_to_key(compare_degrees))

    all_coeffs = []
    for coeff_dict in coeff_dicts:
        coeffs = []
        for degrees in degrees_list:
            if degrees in coeff_dict:
                coeffs.append(coeff_dict[degrees])
            else:
                coeffs.append(0)            
        all_coeffs.append(coeffs)
        
    all_coeffs = np.array(all_coeffs, dtype=np.float64)
    return all_coeffs, degrees_list
