'''
axes de dvp :
- travail sur les bornes dans le cas quantile
- comment on choisit sigma (de la gaussienne) ? Si on manipule des pixels ou des températures en entrée, les echelles de variations sont différentes donc ça mérite pas le même sigma.

remarques :
- quand on travail en dimension 2 ou plus, il faut bien préciser l'ensemble de départ aux gaussiennes multivariables de scipy
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
    tirages = []
    for _ in range(n):
        tirages.append(bruit(G))
    '''
    Tous les calculs seront faits à partir de ce même échantillon.
    Cela permet notamment d'obtenir le même résultat quand on recalcule f_lissee à un même point.
    '''

    def f_lissee(x):
        '''
        x est un element de Rd
        '''
        experience = []
        for tirage in tirages:
            x_bruite = x+tirage
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

    tirages = []
    for _ in range(n):
        tirages.append(bruit(G))

    def f_lissee(x):
        experience = []
        for tirage in tirages:
            x_bruite = x+tirage
            experience.append(f(x_bruite))
        return choix_esp(experience)

    return f_lissee


def bruit(G):
    '''
    Pour le moment ne fonctionne qu'avec les fonctions de scipy.
    '''
    return G.rvs()


def choix(p, experience):
    '''
    Le résultat retourné peut être une moyenne de deux résultats atteignables alors que le papier préconise l'inverse
    (pour être sûr que le résultat de f_lissée ait du sens dans le cas où f ne prendrait qu'un nombre fini de valeurs).
    '''
    experience.sort()
    i = int(p*(len(experience)+1)//1)
    if p*(len(experience)+1) != i:
        return (experience[i-1]+experience[i])/2
    else:
        return experience[i-1]


def choix_esp(experience):
    '''
    Retourne l'espérance d'experience.
    '''
    res = 0
    for el in experience:
        res += el
    return res/len(experience)


def courbe_diff(f, n, G, p):

    l_x = pl.linspace(-10, 10, 1000)

    f_lissee = lissage(f, n, G, p)
    f_esp = lissage_esp(f, n, G)

    l_f = [f(x) for x in l_x]
    l_lissee = [f_lissee(x) for x in l_x]
    l_esp = [f_esp(x) for x in l_x]

    pl.plot(l_x, l_f, label='f')
    pl.plot(l_x, l_lissee, label='f_p')
    pl.plot(l_x, l_esp, label='f_esp')

    pl.legend()

    pl.show()


# courbe_diff(pl.sin, 1000, scipy.stats.multivariate_normal(0, 1), 0.5)


def borne_en_x(f, n, G, p, x):
    '''
    Le coeur du papier.
    '''
    return


def borne_en_x_esp(n, sigma, u, l, delta, alpha, phi_sigma, eta_x, x):
    '''
    J'ai essayé de coder ça de sorte à éviter de recalculer les mêmes choses plusieurs fois mais au final c'est vraiment moche.
    '''
    '''
    Pour avoir les bornes du papier, il faut normaliser f et donc que celle-ci soit bornée dans [u,l].
    La formule ne fonctionne qu'avec une gaussienne centrée donc pas besoin de G mais seulement de sigma.
    Necessite de connaitre la borne sur les attaques delta (pour l'instant j'ai mis 0.1 au hasard pour le cas 1D).
    alplha est la confiance qu'on veut avoir en la borne (0.999 par exemple).
    n sert au calcul de f_lissee et aussi à la qualité de la borne car plus n est grand plus on est confiant.
    L'expression de securite découle de la loi faible des grands nombres.
    '''
    securite = (u-l)/(2*pl.sqrt(n*(1-alpha)))
    return l+(u-l)*phi_sigma((eta_x(x)-delta-securite)/sigma), l+(u-l)*phi_sigma((eta_x(x)+delta+securite)/sigma)


def phi(sigma):
    '''
    Retourne la cdf de la gaussienne centrée.
    '''
    def inner_phi(x):
        return scipy.stats.norm.cdf(x, 0, sigma)

    return inner_phi


def phi_moins_1(sigma):
    '''
    Retourne la reciproque de la cdf de la gaussienne centrée.
    '''
    def inner_phi_moins_1(p):
        return scipy.stats.norm.ppf(p, 0, sigma)

    return inner_phi_moins_1


def eta(f_esp, sigma, u, l, phi_moins_1_sigma):
    '''
    Le même eta que celui du papier.
    '''

    def inner_eta(x):
        return sigma*phi_moins_1_sigma((f_esp(x)-l)/(u-l))

    return inner_eta


def courbe_et_borne_esp(f, n, sigma, u, l, delta, alpha):

    G = scipy.stats.norm(0, sigma)
    l_x = pl.linspace(-10, 10, 1000)

    f_esp = lissage_esp(f, n, G)

    l_f = [f(x) for x in l_x]
    l_esp = [f_esp(x) for x in l_x]

    phi_sigma = phi(sigma)
    phi_moins_1_sigma = phi_moins_1(sigma)
    eta_x = eta(f_esp, sigma, u, l, phi_moins_1_sigma)
    l_inf = []
    l_sup = []
    for x in l_x:
        a, b = borne_en_x_esp(n, sigma, u, l, delta,
                              alpha, phi_sigma, eta_x, x)
        l_inf.append(a)
        l_sup.append(b)

    pl.plot(l_x, l_f, label='f')
    pl.plot(l_x, l_esp, label='f_esp')
    pl.plot(l_x, l_inf, label='f_inf')
    pl.plot(l_x, l_sup, label='f_sup')

    pl.legend()

    pl.show()


courbe_et_borne_esp(pl.sin, 1000, 1, 1, -1, 0.1, 0.99)
