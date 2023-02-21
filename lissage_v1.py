'''
axes de dvp :

- travail sur les bornes

- comment on choisit sigma (de la gaussienne) ?
Si on manipule des pixels ou des températures en entrée, les echelles de variations sont différentes donc ça mérite pas le même sigma.
'''

import scipy.stats
import pylab as pl


def lissage(f, n, G, p):
    '''
    Prend une fonction f de Rd dans R et retourne sa fonction lissée.
    Necessite de choisir :
    Le nombre d'itération du tirage aléatoire du bruit n,
    La variable aléatoire du bruit (ex: gaussienne centrée réduite) G,
    La méthode de choix du tirage retenu (ex médiane). Si on se limite à des quantils alors p.
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
    La méthode de choix du tirage retenu : ici l'espérance.
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
    retourne le quantils adéquat.
    Le résultat retourné peut être une moyenne de deux résultats atteignables alors que le papier préconise l'inverse
    (pour être sûr que le résultat de f_lissée ait du sens dans le cas où f ne prendrait qu'un nombre fini de valeurs)
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
# courbe_diff(pl.sin, 100, scipy.stats.multivariate_normal(0, 1), 0.5)


def borne_en_x(f, n, G, p, x):
    '''
    plus difficile que pour l'espérance car cette fois on ne sait pas calculer les bornes.
    '''
    return


def borne_en_x_esp(f, sigma, x, u, l, delta):
    '''
    pour avoir les bornes du papier, il faut normaliser f et donc que celle-ci soit bornée.
    la formule ne fonctionne qu'avec une gaussienne donc pas besoin de G mais seulement de sigma.
    necessite de connaitre la borne sur les attaques delta (pour l'instant j'ai mis 1 au hasard)
    '''
    return l+(u-l)*phi(sigma)((eta(sigma, f, u, l)(x)-delta)/sigma), l+(u-l)*phi(sigma)((eta(sigma, f, u, l)(x)+delta)/sigma)


def phi(sigma):
    '''
    retourne la cdf de la gaussienne centrée.
    '''
    def inner_phi(x):
        return scipy.stats.norm.cdf(x, 0, sigma)

    return inner_phi


def phi_moins_1(sigma):
    '''
    retourne la reciproque de la cdf de la gaussienne centrée.
    '''
    def inner_phi_moins_1(p):
        return scipy.stats.norm.ppf(p, 0, sigma)

    return inner_phi_moins_1


def eta(sigma, f, u, l):
    '''
    il doit y avoir une erreur dans le papier. J'espère que c'est f à la place de g.
    '''
    def inner_eta(x):
        return sigma*phi_moins_1(sigma)((f(x)-l)/(u-l))

    return inner_eta


# print(borne_en_x_esp(pl.sin, 1, 1, 1, -1, 1))

def courbe_et_borne_esp(f, n, sigma, u, l, delta):
    G = scipy.stats.norm(0, sigma)
    l_x = pl.linspace(-10, 10, 1000)
    l_f = [f(x) for x in l_x]
    f_esp = lissage_esp(f, n, G)
    l_esp = [f_esp(x) for x in l_x]
    l_inf = []
    l_sup = []
    for x in l_x:
        a, b = borne_en_x_esp(f, sigma, x, u, l, delta)
        l_inf.append(a)
        l_sup.append(b)
    pl.plot(l_x, l_f, label='f')
    pl.plot(l_x, l_esp, label='f_esp')
    pl.plot(l_x, l_inf, label='f_inf')
    pl.plot(l_x, l_sup, label='f_sup')
    pl.legend()
    pl.show()


courbe_et_borne_esp(pl.sin, 100, 1, 1, -1, 1)
'''
résultat bizarre ici f_esp n'est pas toujours compris entre les deux bornes.
peut être que c'est bien g dans la formule d'eta et pas f...
mais alors ca veut dire que les bornes ne peuvent pas être calculées exactement puisque g est une approximation de l'esperance.
'''
