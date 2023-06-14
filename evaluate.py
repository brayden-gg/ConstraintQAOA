from helpers import *
import numpy as np

def get_f_min_max(cost_fn, n):
    f_max = -np.inf
    f_min = np.inf
    best_z = None
    best_ind = None
    for i in range(2 ** n):
        z = int2bits(i, n)
        val = cost_fn(z)
        if val > f_max:
            f_max = val
            best_z = z
            best_ind = i

        if val < f_min:
            f_min = val
    
    return f_max, f_min, best_z, best_ind


def approx_ratio(f_obs, f_max, f_min):
    return (f_obs - f_min)/(f_max - f_min)

def prob_of_f_max(probs, cost_fn, f_max, n):
    total_prob = 0
    for i, prob in enumerate(probs):
        z = int2bits(i, n)
        val = cost_fn(z)
        if val == f_max:
            total_prob += prob

    return total_prob