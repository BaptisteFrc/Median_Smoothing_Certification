import scipy.stats


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


def bruit(G):
    '''
    pour le moment ne fonctionne qu'avec les gaussiennes de scipy
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


lissee = lissage(lambda x: x[0]+x[1], 100,
                 scipy.stats.multivariate_normal(0, 1), 0.5)
print(lissee((10, 10)))
