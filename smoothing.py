import scipy.stats as stat
from numpy import sort
import numpy as np
from math import *


def Monte_Carlo(f, alpha, n, p, X, sigma=1, moy=0):
    '''évaluer en X la fonction f grâce à un lissage gaussien de paramètre sigma et moy'''
    experience = []
    length = len(X)
    for i in range(n):
        Y = f(X+stat.norm.rvs(moy, sigma*np.eyes(length)))
        experience.append(Y)

    experience = sort.experience()

    for _ in range(len(experience)):
        if stat.binom.cdf(experience[_], n, p) >= alpha:
            U = experience(_)  # on a trouvé K_q_u

    for _ in range(len(experience)):
        if 1-stat.binom.cdf(experience[len(experience)-1-_], n, p) <= alpha:
            L = experience[len(experience)-1-_]  # K_q_L

    return L, U


# h[X] permet donc d'évaluer la fonction lissée en X

def lissage(f, alpha=0.1, n, p):
    '''renvoie une fonction qui correspond aux bornes de h_p'''
    h = {}  # la création d'un dictionnaire permet de s'assurer que pour une valeur donnée, f renvoie toujours la même chose

    def f_lissee(X):
        '''renvoie un couple de borne'''
        if X not in h:
            h[X] = Monte_Carlo(f, alpha, n, p, X)
        return h[X]
    return f_lissee


f = lissage(sin)
