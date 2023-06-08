#quelques erreurs de placements des variables encore

from smoothing.smoothing import *
import matplotlib.pyplot as plt
from models_neural_network.regression_model import load_model, NN_to_function
from models_neural_network.regression_model_v3 import load_model_v2, NN_to_function_v2
from adversarial_attacks.attack_FGSM import attack_2



def graph_diff(f : callable, n : int, G : callable, p : float):
    """
    only works for d=1
    """

    l_x = np.linspace(2, 5, 100)

    smoothed_f = smoothing(f, n, G, p)
    exp_f = smoothing_exp(f, n, G)

    l_f = [f([x]) for x in l_x]
    l_smoothed = [smoothed_f([x]) for x in l_x]
    # l_exp = [exp_f([x]) for x in l_x]

    plt.plot(l_x, l_f, label='y=abs(sin(x))')
    plt.plot(l_x, l_smoothed, label='smoothed version')
    # plt.plot(l_x, l_exp, label='f_exp')

    plt.legend()
    plt.title('smoothing of f(x)=abs(sin(x)) for p=0.5, n=1000 and sigma=0.1')

    plt.show()


# graph_diff(lambda x: abs(np.sin(x)), 1000, good_gaussian(0.1), 0.5)


def graph_and_bounds(f : callable, n : int, sigma : float, p : float, alpha : float, epsilon : float):
    smoothed_f = smoothing_and_bounds(f, n, sigma, p, alpha, epsilon)

    l_x = np.linspace(2, 5, 1000)

    l_f = [f([x]) for x in l_x]
    l_smoothed = [smoothed_f([x])[1] for x in l_x]
    l_lower = [smoothed_f([x])[0] for x in l_x]
    l_upper = [smoothed_f([x])[2] for x in l_x]

    plt.plot(l_x, l_f, label='f(x)=abs(sin(x))')
    plt.plot(l_x, l_smoothed, label='smoothed version of f k_p')
    plt.plot(l_x, l_lower, label='lower bound k_p-')
    plt.plot(l_x, l_upper, label='upper bound k_p+')

    plt.legend()
    plt.title('smoothing and bounds of f(x)=abs(sin(x)) for p=0.5, n=100, sigma=0.01, alpha=0.99 and epsilon=0.01')

    plt.show()

# graph_and_bounds(lambda x: abs(np.sin(x)), 1000, 0.01, 0.5, 0.99, 0.01)

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

    l_x = np.linspace(2, 5, 1000)

    l_f = [f([x]) for x in l_x]
    l_smoothed = [smoothed_f([x])[2] for x in l_x]
    l_lower = [smoothed_f([x])[1] for x in l_x]
    l_upper = [smoothed_f([x])[3] for x in l_x]
    l_lmax = [smoothed_f([x])[0] for x in l_x]
    l_umax = [smoothed_f([x])[4] for x in l_x]


    plt.plot(l_x, l_f, label='f(x)=abs(sin(x))')
    plt.plot(l_x, l_smoothed, label='smoothed version of f k_p')
    plt.plot(l_x, l_lower, label='lower bound k_p-')
    plt.plot(l_x, l_upper, label='upper bound k_p+')
    plt.plot(l_x, l_lmax, label='k_l-')
    plt.plot(l_x, l_umax, label='k_u+')

    plt.legend()
    plt.title('smoothing and new bounds of f(x)=abs(sin(x)) for p=0.5, n=1000, sigma=0.1, alpha=0.99 and epsilon=0.01')

    plt.show()


# max_graph(lambda x: abs(np.sin(x)), 1000, 0.1, 0.5, 0.99, 0.01, 0.001)


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


# out_of_bound(NN_to_function_v2(load_model_v2()), 1000, 5, [17.76, 42.42, 1009.09, 66.26], 0.5, 0.1, 1, 0.001, 100)
# out_of_bound(lambda x: abs(np.sin(x[0])), 1000, 1, [2], 0.5, 0.5, 0.1, 0.001, 1000)


def out_of_bound_same_attack(f : callable, n : int, sigma : float, x : list, p : float, alpha : float, epsilon : float, precision : float, n_attack : list, attack : list):
    """
    simulates attacks to see if the proportion of tries out of bound is close to the expected value.
    """
    attack=[a[0] for a in attack]
    attack=np.array(attack)*epsilon/norm_2(attack)
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

# l1 = [4.216, 2.79, 4.07, 3.206, 3.314, 3.464, 2.41, 1.044, 1.008,
#       1.334, 2.542, 2.402, 2.294, 2.076, 0.222, 2.28, 2.94, 2.292,
#       3.65, 1.962, 1.754, 1.744, 2.11, 2.818, 3.388]
# out_of_bound_same_attack(NN_to_function_v2(load_model_v2()), 100, 0.01, l1[:24], 0.5, 0.515, 0.001, 0.001, 100, attack_2(load_model_v2(), l1, 0.01)[0][0])












##suite

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
        res+=np.log(sensitivity_at_x(f, x_moyen+G_entree(x_moyen), G_attack, N))
    return np.exp(res/M)

def sensitivity_rel(f, N, G_attack, M, G_entree, x_moyen, fmax, fmin) :
    return sensitivity(f, N, G_attack, M, G_entree, x_moyen)/(fmax-fmin)

def robustness(f, N, G_attack, M, G_entree, x_moyen) :
    return 1/sensitivity(f, N, G_attack, M, G_entree, x_moyen)

def approx_robustness(f, x_moyen, G_attack, N) :
    return 1/sensitivity_at_x(f, x_moyen, G_attack, N)

# print(robustness(lambda x: abs(np.sin(x)), 100, good_gaussian(0.01), 1000, good_gaussian(0.5), [2]))
# print(robustness(NN_to_function_v2(load_model_v2()), 100, good_gaussian(0.01), 100, good_gaussian(0.5), [2 for _ in range(24)]))
# print(robustness(smoothing_exp(NN_to_function_v2(load_model_v2()), 100, good_gaussian(0.1)), 100, good_gaussian(0.01), 10, good_gaussian(0.5), [2 for _ in range(24)]))

def robustness_rel(f, N, G_attack, M, G_entree, x_moyen, fmax, fmin) :
    return robustness(f, N, G_attack, M, G_entree, x_moyen)/(fmax-fmin)

##compare robustesse

def compare_robustesse(f, N, G_attack, M, G_entree, x_moyen) :
    print(robustness(f, N, G_attack, M, G_entree, x_moyen), approx_robustness(f, x_moyen, G_attack, N))

##impact sigma

def compare_sigma(f, n, sigma1, sigma2, p, N, G_attack, M, G_entree, x_moyen) :
    return robustness(f, N, G_attack, M, G_entree, x_moyen), robustness(smoothing(f, n, good_gaussian(sigma1), p), N, G_attack, M, G_entree, x_moyen), robustness(smoothing(f, n, good_gaussian(sigma2), p), N, G_attack, M, G_entree, x_moyen)


# print(compare_sigma(NN_to_function_v2(load_model_v2()), 100, 0.1, 1, 0.5, 100, good_gaussian(0.01), 10, good_gaussian(0.5), [2 for _ in range(24)]))
print(compare_sigma(lambda x : np.sin(x), 100, 0.1, 1, 0.5, 100, good_gaussian(0.01), 10, good_gaussian(0.5), [0]))

##graph en plus

def graph_precision(f, n, sigma, p, alpha, epsilon, precision, X):
    smoothed_f = max_bound(f, n, sigma, p, alpha, epsilon, precision)

    l_x = range(len(X))

    l_f = [f([x]) for x in X]
    l_smoothed = [smoothed_f([x])[2] for x in X]
    l_lower = [smoothed_f([x])[1] for x in X]
    l_upper = [smoothed_f([x])[3] for x in X]
    l_lmax = [smoothed_f([x])[0] for x in X]
    l_umax = [smoothed_f([x])[4] for x in X]

    plt.plot(l_x, l_f, label='the RNN respons')
    plt.plot(l_x, l_smoothed, label='smoothed version of f k_p')
    plt.plot(l_x, l_lower, label='lower bound k_p-')
    plt.plot(l_x, l_upper, label='upper bound k_p+')

    plt.title('smoothing and bounds of our RNN for p=0.5, n=1000, sigma=0.1, alpha=0.9 and epsilon=0.01')
    # plt.plot(l_x, l_lmax, label='f_lmax')
    # plt.plot(l_x, l_umax, label='f_umax')

    plt.legend()

    plt.show()

# point=[1,1,1,1,2,2,2,2,1,1,1,2,2,2,2,3,3,3,3,3,3,2,2,2]
# graph_precision(NN_to_function_v2(load_model_v2()), 1000, 0.1, 0.5, 0.9, 0.01, 0.001, [point]*10)


def compare_p_and_exp(f, n, sigma, p, alpha, epsilon, precision, l, u, X) :
    f_p = max_bound(f, n, sigma, p, alpha, epsilon, precision)
    f_exp = max_bound_exp(f, n, sigma, l, u, epsilon, alpha)
    l_x = range(len(X))

    l_p_1 = [f_p([x])[3]-f_p([x])[1] for x in X]
    l_p_2 = [f_p([x])[4]-f_p([x])[0] for x in X]
    l_exp_1=[f_exp([x])[3]-f_exp([x])[1] for x in X]
    l_exp_2=[f_exp([x])[4]-f_exp([x])[0] for x in X]

    # plt.plot(l_x, l_p_1, label='small bounds p')
    # plt.plot(l_x, l_p_2, label='big bounds p')
    plt.plot(l_x, l_exp_1, label='small bounds exp')
    # plt.plot(l_x, l_exp_2, label='big bounds exp')

    plt.legend()

    plt.show()


# compare_p_and_exp(NN_to_function_v2(load_model_v2()), 1000, 1, 0.5, 0.9, 0.1, 0.001, 0, 9, [[20, 50, 1020, 50]]*10)


# l1 = [4.216, 2.79, 4.07, 3.206, 3.314, 3.464, 2.41, 1.044, 1.008,
#       1.334, 2.542, 2.402, 2.294, 2.076, 0.222, 2.28, 2.94, 2.292,
#       3.65, 1.962, 1.754, 1.744, 2.11, 2.818, 3.388]

# l2 = [3.328, 4.078, 2.388, 3.576, 2.69, 1.116, 0.278, 0.296, 0.21,
#       0.21, 0.378, 0.396, 0.214, 0.684, 2.128, 3.678, 1.352, 1.254,
#       1.734, 1.48, 0.408, 1.888, 1.68, 1.946, 2.786]

# test = NN_to_function_v2(load_model_v2())
# print(test([1,2,3,4,5,6,7,8,9,10,11,12]*2))