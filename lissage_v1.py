'''
axes de dvp :
- comment on choisit sigma (de la gaussienne) ? Si on manipule des pixels ou des températures en entrée, les echelles de variations sont différentes donc ça mérite pas le même sigma.

à faire :
- clareté du code
- en anglais

remarques : epsilon=0 donne l'incertitude sur la valeur de f_lissee.
'''

import scipy.stats
import pylab as pl


def bonne_gaussienne(sigma, moy=0):
    '''
    Fait en sorte que la gaussienne soit bien à valeur dans Rd.
    '''
    def inner(x):
        d = len(x)
        return scipy.stats.multivariate_normal(moy*pl.ones(d), sigma*pl.identity(d)).rvs()
    return inner


def lissage(f, n, G, p):
    '''
    Prend une fonction f de Rd dans R et retourne sa fonction lissée.
    Necessite de choisir :
    Le nombre d'itération du tirage aléatoire du bruit n,
    La variable aléatoire du bruit (ex: gaussienne centrée réduite) G,
    La méthode de choix du tirage retenu (ex médiane). Si on se limite à des quantils alors p.
    '''

    tirage_a_faire = True
    h = {}
    tirages = None

    '''
    Tous les calculs seront faits à partir du même échantillon.
    Cela permet notamment d'obtenir le même résultat quand on recalcule f_lissee à un même point.
    '''

    def f_lissee(x):
        '''
        x est un element de Rd
        '''

        nonlocal tirage_a_faire
        nonlocal h
        nonlocal tirages

        if tirage_a_faire:
            tirages = []
            for _ in range(n):
                tirages.append(G(x))
            tirage_a_faire = False

        if tuple(x) not in h:
            experience = []
            for tirage in tirages:
                x_bruite = x+tirage
                experience.append(float(f(x_bruite)))
            experience.sort()

            h[tuple(x)] = choix(p, experience)

        return h[tuple(x)]

    return f_lissee


def lissage_esp(f, n, G):
    '''
    Prend une fonction f de Rd dans R et retourne sa fonction lissée.
    Necessite de choisir :
    Le nombre d'itération du tirage aléatoire du bruit n,
    La variable aléatoire du bruit (ex: gaussienne centrée réduite) G,
    La méthode de choix du tirage retenu : ici l'espérance.
    '''

    tirage_a_faire = True
    g = {}
    tirages = None

    def f_lissee(x):
        '''
        x est un element de Rd
        '''

        nonlocal tirage_a_faire
        nonlocal g
        nonlocal tirages

        if tirage_a_faire:
            tirages = []
            for _ in range(n):

                tirages.append(G(x))
            tirage_a_faire = False

        if tuple(x) not in g:
            experience = []
            for tirage in tirages:
                x_bruite = x+tirage
                experience.append(float(f(x_bruite)))

            g[tuple(x)] = choix_esp(experience)

        return g[tuple(x)]

    return f_lissee


def choix(p, experience):
    '''
    Le résultat retourné peut être une moyenne de deux résultats atteignables alors que le papier préconise l'inverse
    (pour être sûr que le résultat de f_lissée ait du sens dans le cas où f ne prendrait qu'un nombre fini de valeurs).
    La liste en entrée doit déja être triée.
    '''
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
    l_lissee = [f_lissee([x]) for x in l_x]
    l_esp = [f_esp([x]) for x in l_x]

    pl.plot(l_x, l_f, label='f')
    pl.plot(l_x, l_lissee, label='f_p')
    pl.plot(l_x, l_esp, label='f_esp')

    pl.legend()

    pl.show()


# courbe_diff(pl.sin, 10, bonne_gaussienne(2), 0.5)


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


def lissage_et_bornes_esp(f, n, sigma, u, l, delta, alpha):
    '''
    Pour avoir les bornes du papier, il faut normaliser f et donc que celle-ci soit bornée dans [u,l].
    La formule ne fonctionne qu'avec une gaussienne centrée donc pas besoin de G mais seulement de sigma.
    Necessite de connaitre la borne sur les attaques delta (pour l'instant j'ai mis 0.1 au hasard pour le cas 1D).
    alplha est la confiance qu'on veut avoir en la borne (0.999 par exemple).
    n sert au calcul de f_lissee et aussi à la qualité de la borne car plus n est grand plus on est confiant.
    L'expression de securite découle de la loi faible des grands nombres.
    '''

    G = bonne_gaussienne(sigma)

    tirage_a_faire = True
    g = {}
    tirages = None
    securite = (u-l)/(2*pl.sqrt(n*(1-alpha)))

    phi_sigma = phi(sigma)
    phi_moins_1_sigma = phi_moins_1(sigma)

    def f_lissee(x):
        '''
        x est un element de Rd
        '''

        nonlocal tirage_a_faire
        nonlocal g
        nonlocal tirages

        if tirage_a_faire:
            tirages = []
            for _ in range(n):

                tirages.append(G(x))
            tirage_a_faire = False

        if tuple(x) not in g:
            experience = []
            for tirage in tirages:
                x_bruite = x+tirage
                experience.append(float(f(x_bruite)))

            f_esp = choix_esp(experience)
            g[tuple(x)] = l+(u-l)*phi_sigma((sigma*phi_moins_1_sigma((f_esp-l)/(u-l))-delta-securite) /
                                            sigma), f_esp, l+(u-l)*phi_sigma((sigma*phi_moins_1_sigma((f_esp-l)/(u-l))+delta+securite)/sigma)

        return g[tuple(x)]

    return f_lissee


def lissage_et_bornes(f, n, sigma, p, alpha, epsilon):
    '''
    Prend une fonction f de Rd dans R et retourne sa fonction lissée.
    Necessite de choisir :
    Le nombre d'itération du tirage aléatoire du bruit n,
    La variable aléatoire du bruit (ex: gaussienne centrée réduite) G,
    La méthode de choix du tirage retenu (ex médiane). Si on se limite à des quantils alors p.
    '''

    G = bonne_gaussienne(sigma)
    tirage_a_faire = True
    h = {}
    tirages = None

    '''
    Tous les calculs seront faits à partir du même échantillon.
    Cela permet notamment d'obtenir le même résultat quand on recalcule f_lissee à un même point.
    '''

    def f_lissee(x):
        '''
        x est un element de Rd
        '''

        nonlocal tirage_a_faire
        nonlocal h
        nonlocal tirages

        if tirage_a_faire:
            tirages = []
            for _ in range(n):
                tirages.append(G(x))
            tirage_a_faire = False

        if tuple(x) not in h:
            experience = []
            for tirage in tirages:
                x_bruite = x+tirage
                experience.append(float(f(x_bruite)))
                experience.sort()

            h[tuple(x)] = low(p, experience, alpha, epsilon, sigma), choix(
                p, experience), up(p, experience, alpha, epsilon, sigma)

        return h[tuple(x)]

    return f_lissee


'''pas terrible le recalcul de phi'''


def low(p, experience, alpha, epsilon, sigma):
    p_low = phi(sigma)(phi_moins_1(sigma)(p)-epsilon/sigma)
    n = len(experience)
    ql = max(0, int(n - scipy.stats.binom.ppf(alpha, n, 1 - p_low) - 2))
    return experience[ql]


def up(p, experience, alpha, epsilon, sigma):
    n = len(experience)
    p_up = phi(sigma)(phi_moins_1(sigma)(p)+epsilon/sigma)
    qu = min(n-1, int(scipy.stats.binom.ppf(alpha, n, p_up)))
    return experience[qu]


def courbes_et_bornes(f, n, sigma, p, alpha, epsilon):

    f_lissee = lissage_et_bornes(f, n, sigma, p, alpha, epsilon)

    l_x = pl.linspace(-10, 10, 1000)

    l_f = [f([x]) for x in l_x]
    l_lissee = [f_lissee([x])[1] for x in l_x]
    l_inf = [f_lissee([x])[0] for x in l_x]
    l_sup = [f_lissee([x])[2] for x in l_x]

    pl.plot(l_x, l_f, label='f')
    pl.plot(l_x, l_lissee, label='f_lissee')
    pl.plot(l_x, l_inf, label='f_inf')
    pl.plot(l_x, l_sup, label='f_sup')

    pl.legend()

    pl.show()


# courbes_et_bornes(pl.sin, 1000, 1, 0.5, 0.99, 0.1)


def courbes_et_bornes_esp(f, n, sigma, u, l, alpha, epsilon):

    f_lissee = lissage_et_bornes_esp(f, n, sigma, u, l, epsilon, alpha)

    l_x = pl.linspace(-10, 10, 1000)

    l_f = [f([x]) for x in l_x]
    l_lissee = [f_lissee([x])[1] for x in l_x]
    l_inf = [f_lissee([x])[0] for x in l_x]
    l_sup = [f_lissee([x])[2] for x in l_x]

    pl.plot(l_x, l_f, label='f')
    pl.plot(l_x, l_lissee, label='f_lissee')
    pl.plot(l_x, l_inf, label='f_inf')
    pl.plot(l_x, l_sup, label='f_sup')

    pl.legend()

    pl.show()


# courbes_et_bornes_esp(pl.sin, 1000, 1, -1, 1, 0.99, 0.1)
