from Smoothing_v3.smoothing import *
import matplotlib.pyplot as plt
import sys
from pprint import pprint
from models_neural_network.regression_model import load_model, NN_to_function
from adversarial_attacks.attack_FGSM import attack_1



def graph_diff(f, n, G, p):
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


def graph_and_bounds(f, n, sigma, p, alpha, epsilon):
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

def graph_and_bounds_exp(f, n, sigma, l, u, alpha, epsilon):
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


def max_graph(f, n, sigma, p, alpha, epsilon, precision):
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


def max_graph_exp(f, n, sigma, l, u, alpha, epsilon):
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


def out_of_bound(f, n, sigma, x, p, alpha, epsilon, precision, n_attack):
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
    return np.array(res)/len(l_attack)

# print(out_of_bound(NN_to_function(load_model()),
#       100, 1, [17.76, 42.42, 1009.09, 66.26], 0.5, 0.5, 1, 0.001, 100))


def out_of_bound_same_attack(f, n, sigma, x, p, alpha, epsilon, precision, n_attack, attack):
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
    return np.array(res)/n_attack


"""Tests"""

# test = NN_to_function(load_model())
# test_smoothed = smoothing_and_bounds(test, 100, 1, 0.5, 0.9, 1)
# print(test_smoothed([17.76, 42.42, 1009.09, 66.26]),
#       test([17.76, 42.42, 1009.09, 66.26]))

# print(out_of_bound_same_attack(NN_to_function(load_model()),
#       100, 1, [17.76, 42.42, 1009.09, 66.26], 0.5, 0.99, 1, 0.001, 100, attack_1(load_model(), [[[17.76, 42.42, 1009.09, 66.26], [468.27]]], 1)))
