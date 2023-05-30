from Smoothing_v3.smoothing import *
import matplotlib.pyplot as plt
import sys
from pprint import pprint
from models_neural_network.regression_model import load_model, NN_to_function
from adversarial_attacks.attack_FGSM import attack_1



def graph_diff(f : callable, n : int, G : callable, p : float) ->None:
    """
    only works for d=1
    """

    l_x = np.linspace(2, 5, 1000)

    smoothed_f = smoothing(f, n, G, p)
    exp_f = smoothing_exp(f, n, G)

    l_f = [f([x]) for x in l_x]
    l_smoothed = [smoothed_f([x]) for x in l_x]
    l_exp = [exp_f([x]) for x in l_x]

    plt.plot(l_x, l_f, label='f')
    plt.plot(l_x, l_smoothed, label='f_p')
    plt.plot(l_x, l_exp, label='f_exp')

    plt.legend()

    plt.show()


# graph_diff(lambda x: abs(np.sin(x)), 10, good_gaussian(0.1), 0.5)


def graph_and_bounds(f : callable, n : int, sigma : float, p : float, alpha : float, epsilon : float) ->None:
    smoothed_f = smoothing_and_bounds(f, n, sigma, p, alpha, epsilon)

    l_x = np.linspace(2, 5, 1000)

    l_f = [f([x]) for x in l_x]
    l_smoothed = [smoothed_f([x])[1] for x in l_x]
    l_lower = [smoothed_f([x])[0] for x in l_x]
    l_upper = [smoothed_f([x])[2] for x in l_x]

    plt.plot(l_x, l_f, label='f')
    plt.plot(l_x, l_smoothed, label='smoothed_f')
    plt.plot(l_x, l_lower, label='f_l')
    plt.plot(l_x, l_upper, label='f_u')

    plt.legend()

    plt.show()


# graph_and_bounds(np.sin, 100, 0.1, 0.5, 0.99, 0.1)

def graph_and_bounds_exp(f : callable, n : int, sigma : float, l : float, u : float, alpha : float, epsilon : float):
    smoothed_f = smoothing_and_bounds_exp(f, n, sigma, l, u, epsilon, alpha)

    l_x = np.linspace(-10, 10, 1000)

    l_f = [f([x]) for x in l_x]
    l_smoothed = [smoothed_f([x])[1] for x in l_x]
    l_lower = [smoothed_f([x])[0] for x in l_x]
    l_upper = [smoothed_f([x])[2] for x in l_x]

    plt.plot(l_x, l_f, label='f')
    plt.plot(l_x, l_smoothed, label='smoothed_f')
    plt.plot(l_x, l_lower, label='f_l')
    plt.plot(l_x, l_upper, label='f_u')

    plt.legend()

    plt.show()


# graph_and_bounds_exp(np.sin, 10, 1, -1, 1, 0.99, 0.1)


def max_graph(f : callable, n : int, sigma : float, p : float, alpha : float, epsilon : float, precision : float):
    smoothed_f = max_bound(f, n, sigma, p, alpha, epsilon, precision)

    l_x = np.linspace(2, 5, 50)

    l_f = [f([x]) for x in l_x]
    l_smoothed = [smoothed_f([x])[2] for x in l_x]
    l_lower = [smoothed_f([x])[1] for x in l_x]
    l_upper = [smoothed_f([x])[3] for x in l_x]
    l_lmax = [smoothed_f([x])[0] for x in l_x]
    l_umax = [smoothed_f([x])[4] for x in l_x]

    plt.plot(l_x, l_f, label='f')
    plt.plot(l_x, l_smoothed, label='smoothed_f')
    plt.plot(l_x, l_lower, label='f_l')
    plt.plot(l_x, l_upper, label='f_u')
    plt.plot(l_x, l_lmax, label='f_lmax')
    plt.plot(l_x, l_umax, label='f_umax')

    plt.legend()

    plt.show()


# max_graph(lambda x: abs(np.sin(x)), 10, 1, 0.5, 0.99, 0.1, 0.001)


def max_graph_exp(f : callable, n : int, sigma : float, l : float, u : float, alpha : float, epsilon : float):
    smoothed_f = max_bound_exp(f, n, sigma, l, u, epsilon, alpha)

    l_x = np.linspace(2, 5, 2)

    l_f = [f([x]) for x in l_x]
    l_smoothed = [smoothed_f([x])[2] for x in l_x]
    l_lower = [smoothed_f([x])[1] for x in l_x]
    l_upper = [smoothed_f([x])[3] for x in l_x]
    l_lmax = [smoothed_f([x])[0] for x in l_x]
    l_umax = [smoothed_f([x])[4] for x in l_x]

    plt.plot(l_x, l_f, label='f')
    plt.plot(l_x, l_smoothed, label='smoothed_f')
    plt.plot(l_x, l_lower, label='f_l')
    plt.plot(l_x, l_upper, label='f_u')
    plt.plot(l_x, l_lmax, label='f_lmax')
    plt.plot(l_x, l_umax, label='f_umax')

    plt.legend()

    plt.show()


# max_graph_exp(lambda x: abs(np.sin(x)), 1000, 1, 0, 1, 0.9, 0.1)

# max_graph(Rd_to_R(NN_to_function(load_model()), 4), 3, 1, 0.5, 0.99, 0.1, 0.001)


def out_of_bound(f : callable, n : int, sigma : float, x : list, p : float, alpha : float, epsilon : float, precision : float, n_attack : list):
    """
    simulates attacks to see if the proportion of tries out of bound is close to the expected value.
    """
    res = [0]*5
    l_attack = attack_set(x, epsilon, n_attack)
    l_x = [x+attack for attack in l_attack]
    smoothed_f = max_bound(f, n, sigma, p, alpha, epsilon, precision)
    lower = smoothed_f(x)[1]
    upper = smoothed_f(x)[3]
    lmax = smoothed_f(x)[0]
    umax = smoothed_f(x)[4]
    for x_with_noise in l_x:
        answer = smoothed_f(x_with_noise)[2]
        if answer < lmax:
            res[0] += 1
        elif lmax <= answer < lower:
            res[1] += 1
        elif lower <= answer < upper:
            res[2] += 1
        elif upper <= answer < umax:
            res[3] += 1
        elif umax <= answer:
            res[4] += 1
    print(np.array(res)/len(l_attack))

# out_of_bound(NN_to_function(load_model()), 100, 1, [17.76, 42.42, 1009.09, 66.26], 0.5, 0.5, 1, 0.001, 100)


def out_of_bound_same_attack(f : callable, n : int, sigma : float, x : list, p : float, alpha : float, epsilon : float, precision : float, n_attack : list, attack : list):
    """
    simulates attacks to see if the proportion of tries out of bound is close to the expected value.
    """
    res = [0]*5
    smoothed_f = max_bound(f, n, sigma, p, alpha, epsilon, precision)
    lower = smoothed_f(x)[1]
    upper = smoothed_f(x)[3]
    lmax = smoothed_f(x)[0]
    umax = smoothed_f(x)[4]
    x_with_noise = x+np.array(attack[0])
    for _ in range(n_attack):
        answer = smoothed_f(x_with_noise)[2]
        if answer < lmax:
            res[0] += 1
        elif lmax <= answer < lower:
            res[1] += 1
        elif lower <= answer < upper:
            res[2] += 1
        elif upper <= answer < umax:
            res[3] += 1
        elif umax <= answer:
            res[4] += 1
    print(np.array(res)/n_attack)


"""Tests"""

# test = NN_to_function(load_model())
# test_smoothed = smoothing_and_bounds(test, 100, 1, 0.5, 0.9, 1)
# print(test_smoothed([17.76, 42.42, 1009.09, 66.26]),
#       test([17.76, 42.42, 1009.09, 66.26]))

# out_of_bound_same_attack(NN_to_function(load_model()), 100, 1, [17.76, 42.42, 1009.09, 66.26], 0.5, 0.99, 1, 0.001, 100, attack_1(load_model(), [[[17.76, 42.42, 1009.09, 66.26], [468.27]]], 1))















##suite

from scipy.integrate import nquad
from Smoothing_v3.utils import norm_2
from numpy import array, exp, log
import time

def sensitivity_at_x(f, x, G_attack, N) :
    res=0
    for _ in range(N) :
        attack=G_attack(x)
        res+=abs(f(x+attack)-f(x))/norm_2(attack)
    return res/N

# print(sensitivity_at_x(lambda x: x[0], [0], good_gaussian(1), 100))

def sensitivity_at_x_rel(f, x, G_attack, N, fmax, fmin) :
    return sensitivity_at_x(f, x, G_attack, N)/(fmax-fmin)

def sensitivity(f, N, G_attack, M, G_entree, x_moyen) :
    res=0
    for _ in range (M) :
        res+=sensitivity_at_x(f, x_moyen+G_entree(x_moyen), G_attack, N)
    return res/M

def sensitivity_rel(f, N, G_attack, M, G_entree, x_moyen, fmax, fmin) :
    return sensitivity(f, N, G_attack, M, G_entree, x_moyen)/(fmax-fmin)

def robustness(f, N, G_attack, M, G_entree, x_moyen) :
    return 1/sensitivity(f, N, G_attack, M, G_entree, x_moyen)

def approx_robustness(f, x_moyen, G_attack, N) :
    return 1/sensitivity_at_x(f, x_moyen, G_attack, N)

# print(robustness(lambda x: x[0]+x[1]**2, 1000, good_gaussian(1), 1000, good_gaussian(10), [0,0]))

def robustness_rel(f, N, G_attack, M, G_entree, x_moyen, fmax, fmin) :
    return robustness(f, N, G_attack, M, G_entree, x_moyen)/(fmax-fmin)

##compare robustesse

def compare_robustesse(f, N, G_attack, M, G_entree, x_moyen) :
    print(robustness(f, N, G_attack, M, G_entree, x_moyen), approx_robustness(f, x_moyen, G_attack, N))

##impact sigma

def compare_sigma(f, n, sigma1, sigma2, p, N, G_attack, M, G_entree, x_moyen) :
    return robustness(f, N, G_attack, M, G_entree, x_moyen), robustness(smoothing(f, n, good_gaussian(sigma1), p), N, G_attack, M, G_entree, x_moyen), robustness(smoothing(f, n, good_gaussian(sigma2), p), N, G_attack, M, G_entree, x_moyen)

# print(compare_sigma(lambda x: x[0]+x[1]**2, 100, 1, 10, 0.5, 100, good_gaussian(1), 100, good_gaussian(10), [0,0]))

##graph en plus

def graph_en_plus(f, n, sigma, p, alpha, epsilon, precision, X):
    smoothed_f = max_bound(f, n, sigma, p, alpha, epsilon, precision)

    l_x = range(10)

    l_f = [f([x]) for x in X]
    l_smoothed = [smoothed_f([x])[2] for x in X]
    l_lower = [smoothed_f([x])[1] for x in X]
    l_upper = [smoothed_f([x])[3] for x in X]
    l_lmax = [smoothed_f([x])[0] for x in X]
    l_umax = [smoothed_f([x])[4] for x in X]

    # plt.plot(l_x, l_f, label='f')
    plt.plot(l_x, l_smoothed, label='smoothed_f')
    plt.plot(l_x, l_lower, label='f_l')
    plt.plot(l_x, l_upper, label='f_u')
    # plt.plot(l_x, l_lmax, label='f_lmax')
    # plt.plot(l_x, l_umax, label='f_umax')

    plt.legend()

    plt.show()

# graph_en_plus(NN_to_function(load_model()), 100, 1, 0.5, 0.9, 1, 0.001, [[20, 50, 1020, 50]]*10)
