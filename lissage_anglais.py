'''
axes de dvp :
- comment on choisit sigma (de la gaussienne) ? Si on manipule des pixels ou des températures en entrée, les echelles de variations sont différentes donc ça mérite pas le même sigma.

à faire :
- clareté du code
- en anglais

remarques : epsilon=0 donne l'incertitude sur la valeur de f_lissee.
et justement les résultats d'incertitudes sur les bornes fonctionnent avec la véritable fonction lissée or la médiane telle qu'on l'a fait n'est qu'une approximation...
'''

from regression_model import test
import scipy.stats
import pylab as pl


def good_gaussian(sigma, moy=0):
    '''
    Fait en sorte que la gaussienne soit bien à valeur dans Rd.
    Make sure that the gaussian function's output is in Rd.
    '''
    def inner(x):
        d = len(x)
        return scipy.stats.multivariate_normal(moy*pl.ones(d), sigma*pl.identity(d)).rvs()
    return inner


def smoothing(f, n, G, p):
    """Returns the smoothed function of the input

    Args:
        f (function): Rd -> R
        n (int): number of iterations for the random draw for the noise
        G (function): random variable for the noise
        p (float): depends on the method of draw chosen, here quantiles

    Returns:
        function: f_smoothed
    """
    '''
    Prend une fonction f de Rd dans R et retourne sa fonction lissée.
    Necessite de choisir :
    Le nombre d'itération du tirage aléatoire du bruit n,
    La variable aléatoire du bruit (ex: gaussienne centrée réduite) G,
    La méthode de choix du tirage retenu (ex médiane). Si on se limite à des quantils alors p.
    '''

    draw_to_do = True
    h = {}
    draws = None
    qp = q_p(p, n)

    '''
    Tous les calculs seront faits à partir du même échantillon.
    Cela permet notamment d'obtenir le même résultat quand on recalcule f_lissee à un même point.
    '''
    '''
    All the calculations are made with the same sample.
    It notably allows to obtain the same result when the f_smoothed(x) is evaluated once more at a given x.
    '''

    def f_smoothed(x):
        '''
        x belongs to Rd
        '''

        nonlocal draw_to_do
        nonlocal h
        nonlocal draws

        if draw_to_do:
            draws = []
            for _ in range(n):
                draws.append(G(x))
            draw_to_do = False

        if tuple(x) not in h:
            experiment = []
            for draw in draws:
                x_with_noise = x+draw
                experiment.append(float(f(x_with_noise)))
            experiment.sort()

            h[tuple(x)] = experiment[qp]

        return h[tuple(x)]

    return f_smoothed


def smoothing_exp(f, n, G):
    """

    Args:
        f (function)): _description_
        n (int): number of iterations for the random draw of the noise
        G (function): random variable of the noise (e.g. standard normal distribution)

    Returns:
        function: smoothed version of f
    """
    '''
    Prend une fonction f de Rd dans R et retourne sa fonction lissée.
    Necessite de choisir :
    Le nombre d'itération du tirage aléatoire du bruit n,
    La variable aléatoire du bruit (ex: gaussienne centrée réduite) G,
    La méthode de choix du tirage retenu : ici l'espérance.
    '''

    draw_to_do = True
    g = {}
    draws = None

    def f_smoothed(x):
        '''
        x belongs to Rd
        '''

        nonlocal draw_to_do
        nonlocal g
        nonlocal draws

        if draw_to_do:
            draws = []
            for _ in range(n):
                draws.append(G(x))
            draw_to_do = False

        if tuple(x) not in g:
            experiment = []
            for draw in draws:
                x_with_noise = x+draw
                experiment.append(float(f(x_with_noise)))

            g[tuple(x)] = choice_exp(experiment)

        return g[tuple(x)]

    return f_smoothed


def q_p(p, n):
    '''
    we do not take the mean of two values. Here, we chose to consider the inferior index.
    on ne prend pas la moyenne de deux valeurs. ici on a pris l'indice inférieur.
    '''
    return min(n-1, max(0, int((n+1)*p)-1))


def choice_exp(experiment):
    '''
    Input : experiment
    Output : the expected value of the experiment
    '''
    res = 0
    for el in experiment:
        res += el
    return res/len(experiment)


def graph_diff(f, n, G, p):
    '''
    Only works for d=1
    '''

    l_x = pl.linspace(-10, 10, 1000)

    f_smoothed = smoothing(f, n, G, p)
    f_exp = smoothing_exp(f, n, G)

    l_f = [f([x]) for x in l_x]
    l_smoothed = [f_smoothed([x]) for x in l_x]
    l_exp = [f_exp([x]) for x in l_x]

    pl.plot(l_x, l_f, label='f')
    pl.plot(l_x, l_smoothed, label='f_p')
    pl.plot(l_x, l_exp, label='f_exp')

    pl.legend()

    pl.show()


# graph_diff(pl.sin, 10, bonne_gaussienne(2), 0.5)


def phi(sigma, moy=0):
    '''
    Returns the cdf (Cumulative Distribution Function) of the standard normal distribution 
    '''
    def inner_phi(x):
        return scipy.stats.norm.cdf(x, moy, sigma)

    return inner_phi


def phi_moins_1(sigma, moy=0):
    '''
    Returns the inverse function of the cdf of the centered Gaussian distribution.
    '''
    def inner_phi_moins_1(p):
        return scipy.stats.norm.ppf(p, moy, sigma)

    return inner_phi_moins_1


def smoothing_and_bounds_exp(f, n, sigma, u, l, epsilon, alpha):
    """In order to obtain the bounds of the article, one must normalize f and thus f has to be bounded in [u,l].
    The formula only works with a centered Gaussian distribution so there is no need for G but only sigma.
    It is necessary to know the bound epsilon of the attack (for now, the value 0.1 has been randomly chosen 
    for dimension1).

    Args:
        f (function): _description_
        n (int): number of iterations for the random draw of the noise
        sigma (float): standard deviation of the noise, has an impact on the quality of the bound, 
            the bigger the more trustworthy 
        u (_type_): _description_
        l (_type_): _description_
        epsilon (float): bound of the attack
        alpha (float): confidence rate of the bounds obtained for the output of the function

    Returns:
        function: f_smoothed
    """
    '''
    Pour avoir les bornes du papier, il faut normaliser f et donc que celle-ci soit bornée dans [u,l].
    La formule ne fonctionne qu'avec une gaussienne centrée donc pas besoin de G mais seulement de sigma.
    Necessite de connaitre la borne sur les attaques delta (pour l'instant j'ai mis 0.1 au hasard pour le cas 1D).
    alplha est la confiance qu'on veut avoir en la borne (0.999 par exemple).
    n sert au calcul de f_lissee et aussi à la qualité de la borne car plus n est grand plus on est confiant.
    L'expression de securite découle de la loi faible des grands nombres.  
    '''

    G = good_gaussian(sigma)

    draw_to_do = True
    g = {}
    draws = None
    securite = (u-l)/(2*pl.sqrt(n*(1-alpha)))

    phi_sigma = phi(sigma)
    phi_moins_1_sigma = phi_moins_1(sigma)

    def f_smoothed(x):
        '''
        x belongs to Rd
        '''

        nonlocal draw_to_do
        nonlocal g
        nonlocal draws

        if draw_to_do:
            draws = []
            for _ in range(n):
                draws.append(G(x))
            draw_to_do = False

        if tuple(x) not in g:
            experiment = []
            for draw in draws:
                x_with_noise = x+draw
                experiment.append(float(f(x_with_noise)))

            f_exp = choice_exp(experiment)
            g[tuple(x)] = l+(u-l)*phi_sigma((sigma*phi_moins_1_sigma((f_exp-l)/(u-l))-epsilon-securite) /
                                            sigma), f_exp, l+(u-l)*phi_sigma((sigma*phi_moins_1_sigma((f_exp-l)/(u-l))+epsilon+securite)/sigma)

        return g[tuple(x)]

    return f_smoothed


def smoothing_and_bounds(f, n, sigma, p, alpha, epsilon):
    '''
    Prend une fonction f de Rd dans R et retourne sa fonction lissée.
    Necessite de choisir :
    Le nombre d'itération du tirage aléatoire du bruit n,
    La variable aléatoire du bruit (ex: gaussienne centrée réduite) G,
    La méthode de choix du tirage retenu (ex médiane). Si on se limite à des quantils alors p.
    '''

    G = good_gaussian(sigma)
    draw_to_do = True
    h = {}
    draws = None
    ql = q_bot(p, n, alpha, epsilon, sigma)
    qp = q_p(p, n)
    qu = q_top(p, n, alpha, epsilon, sigma)

    '''
    All the calculations are made with the same sample.
    It notably allows to obtain the same result when the f_smoothed(x) is evaluated once more at a given x.
    '''

    def f_smoothed(x):
        '''
        x belongs to Rd
        '''

        nonlocal draw_to_do
        nonlocal h
        nonlocal draws

        if draw_to_do:
            draws = []
            for _ in range(n):
                draws.append(G(x))
            draw_to_do = False

        if tuple(x) not in h:
            experiment = []
            for draw in draws:
                x_with_noise = x+draw
                experiment.append(float(f(x_with_noise)))
            experiment.sort()

            h[tuple(x)] = experiment[ql], experiment[qp], experiment[qu]

        return h[tuple(x)]

    return f_smoothed


def q_bot(p, n, alpha, epsilon, sigma):
    p_bot = phi(sigma)(phi_moins_1(sigma)(p)-epsilon/sigma)
    ql = max(0, int(n - scipy.stats.binom.ppf(alpha, n, 1 - p_bot) - 2))
    return ql


def q_top(p, n, alpha, epsilon, sigma):
    p_top = phi(sigma)(phi_moins_1(sigma)(p)+epsilon/sigma)
    qu = min(n-1, int(scipy.stats.binom.ppf(alpha, n, p_top)))
    return qu


def graph_and_bounds(f, n, sigma, p, alpha, epsilon):

    f_smoothed = smoothing_and_bounds(f, n, sigma, p, alpha, epsilon)

    l_x = pl.linspace(-10, 10, 1000)

    l_f = [f([x]) for x in l_x]
    l_smoothed = [f_smoothed([x])[1] for x in l_x]
    l_inf = [f_smoothed([x])[0] for x in l_x]
    l_sup = [f_smoothed([x])[2] for x in l_x]

    pl.plot(l_x, l_f, label='f')
    pl.plot(l_x, l_smoothed, label='f_smoothed')
    pl.plot(l_x, l_inf, label='f_inf')
    pl.plot(l_x, l_sup, label='f_sup')

    pl.legend()

    pl.show()


# graph_and_bounds(pl.sin, 1000, 1, 0.5, 0.99, 0.1)


def graph_and_bounds_exp(f, n, sigma, u, l, alpha, epsilon):

    f_smoothed = smoothing_and_bounds_exp(f, n, sigma, u, l, epsilon, alpha)

    l_x = pl.linspace(-10, 10, 1000)

    l_f = [f([x]) for x in l_x]
    l_smoothed = [f_smoothed([x])[1] for x in l_x]
    l_inf = [f_smoothed([x])[0] for x in l_x]
    l_sup = [f_smoothed([x])[2] for x in l_x]

    pl.plot(l_x, l_f, label='f')
    pl.plot(l_x, l_smoothed, label='f_smoothed')
    pl.plot(l_x, l_inf, label='f_inf')
    pl.plot(l_x, l_sup, label='f_sup')

    pl.legend()

    pl.show()


# graph_and_bounds_exp(pl.sin, 1000, 1, -1, 1, 0.99, 0.1)


# test_smoothed = smoothing_and_bounds(test, 100, 1, 0.5, 0.9, 1)
# print(test_smoothed([17.76, 42.42, 1009.09, 66.26]))
