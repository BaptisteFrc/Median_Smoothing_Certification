import scipy.stats
import numpy as np
from random import random


def good_gaussian(sigma : float):
    """
    Ensures that the Gaussian is well-valued in Rd.
    """
    def inner(x):
        d = len(x)
        return scipy.stats.multivariate_normal(np.zeros(d), sigma).rvs()
    return inner


def q_p(n : int, p : float):
    """
    We do not take the average of two values. Here we choose to consider the lower index.
    """
    return min(n-1, max(0, int((n+1)*p)-1))


def exp(sample : list):
    """
    Returns the mean of the sample.
    """
    res = 0
    for el in sample:
        res += el
    return res/len(sample)

# the four following functions compute the variables described in the report

def q_lower(n : int, p : float, alpha : float, epsilon : float, sigma :float):
    p_l = scipy.stats.norm.cdf(scipy.stats.norm.ppf(p, 0, 1)-epsilon/sigma, 0, 1)
    ql = max(0, int(n - 1 - scipy.stats.binom.ppf(alpha, n, 1 - p_l)))
    return ql


def q_upper( n : int, p : float, alpha : float, epsilon : float, sigma :float):
    p_u = scipy.stats.norm.cdf(scipy.stats.norm.ppf(p, 0, 1)+epsilon/sigma, 0, 1)
    qu = min(n-1, int(scipy.stats.binom.ppf(alpha, n, p_u)))
    return qu


def p_minus(n : int, p : float, alpha : float, precision :float):
    a = 0
    b = 1
    while b-a > precision:
        m = (a+b)/2
        if scipy.stats.binom.cdf(q_p(n, p), n, m) > alpha:
            a = m
        else:
            b = m
    return a


def p_plus(n : int, p : float, alpha : float, precision :float):
    a = 0
    b = 1
    while b-a > precision:
        m = (a+b)/2
        if scipy.stats.binom.cdf(n-1-q_p(n, p), n, 1-m) > alpha:
            b = m
        else:
            a = m
    return b


def attack_set(x : list, epsilon : float, n_attack : list):
    """
    Prepare a random attack_set.
    """
    l_attack = []
    d = len(x)
    for i in range(n_attack):
        attack = [random()-0.5 for _ in range(d)]
        attack = np.array(attack)*epsilon/norm_2(attack)
        l_attack.append(attack)
    return l_attack


def norm_2(x : list):
    res = 0
    if type(x)==np.float64 :
        return abs(x)
    for el in x :
        res += el**2
    return np.sqrt(res)


def very_good_gaussian(*args):
    """
    Ensures that the Gaussian is well-valued in Rd. Enables a more complex variance matrix.
    """
    def inner(x):
        d = len(x)
        sigma=np.diag(args)
        return scipy.stats.multivariate_normal(np.zeros(d), sigma).rvs()
    return inner
