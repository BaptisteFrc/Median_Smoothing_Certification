'''
axes de dvp :
ce n'est pas fait dans le papier mais on pourrait ajouter un coefficient multiplicatif à la gaussienne pour qu'elle soit plus adaptée à l'entrée
travail sur les bornes
'''

import scipy.stats
import pylab as pl


def lissage(f, n, G, p):
    '''
    Prend une fonction f de Rd dans R et retourne sa fonction lissée.
    Necessite de choisir :
    Le nombre d'itération du tirage aléatoire du bruit n,
    La variable aléatoire du bruit (ex: gaussienne centrée réduite) G,
    La méthode de choix du tirage retenu (ex médiane). Si on se limite à des quantils alors p
    '''
    def f_lissee(x):
        '''
        x est un element de Rd
        '''
        experience = []
        for _ in range(n):
            x_bruite = x+bruit(G)
            experience.append(f(x_bruite))
        return choix(p, experience)
    return f_lissee


def lissage_esp(f, n, G):
    '''
    Prend une fonction f de Rd dans R et retourne sa fonction lissée.
    Necessite de choisir :
    Le nombre d'itération du tirage aléatoire du bruit n,
    La variable aléatoire du bruit (ex: gaussienne centrée réduite) G,
    La méthode de choix du tirage retenu (ex médiane). Si on se limite à des quantils alors p
    '''
    def f_lissee(x):
        '''
        x est un element de Rd
        '''
        experience = []
        for _ in range(n):
            x_bruite = x+bruit(G)
            experience.append(f(x_bruite))
        return choix_esp(experience)
    return f_lissee


def bruit(G):
    '''
    pour le moment ne fonctionne qu'avec les fonctions de scipy
    '''
    return G.rvs()


def choix(p, experience):
    '''
    retourne le quantils adéquat
    '''
    experience.sort()
    i = int(p*(len(experience)+1)//1)
    if p*(len(experience)+1) != i:
        return (experience[i-1]+experience[i])/2
    else:
        return experience[i-1]


def choix_esp(experience):
    '''
    retourne l'esperance
    '''
    res = 0
    for el in experience:
        res += el
    return res/len(experience)

# lissee = lissage(lambda x: x[0], 100,
#                  scipy.stats.multivariate_normal(0, 1), 0.1)
# print(lissee([10]*10))


def courbe_diff(f, n, G, p):
    l_x = pl.linspace(-10, 10, 1000)
    l_f = [f(x) for x in l_x]
    f_lissee = lissage(f, n, G, p)
    f_esp = lissage_esp(f, n, G)
    l_lissee = [f_lissee(x) for x in l_x]
    l_esp = [f_esp(x) for x in l_x]
    pl.plot(l_x, l_f, label='f')
    pl.plot(l_x, l_lissee, label='f_p')
    pl.plot(l_x, l_esp, label='f_esp')
    pl.legend()
    pl.show()


# le résultat est sympa si vous avez le temps de faire tourner.
courbe_diff(pl.sin, 100, scipy.stats.multivariate_normal(0, 1), 0.5)
