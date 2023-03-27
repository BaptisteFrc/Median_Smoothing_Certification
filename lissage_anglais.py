from regression_model import test
import scipy.stats
import pylab as pl


def good_gaussian(sigma, mean=0):
    '''
    Ensures that the Gaussian is well-valued in Rd.
    '''
    def inner(x):
        d = len(x)
        return scipy.stats.multivariate_normal(mean*pl.ones(d), sigma*pl.identity(d)).rvs()
    return inner


def smoothing(f, n, G, p):
    '''
    Takes a function f from Rd to R and returns its smoothed function.
    Needs to choose:
    The number of iterations of the random noise draw n,
    The random variable of noise (e.g. standard centered Gaussian) G,
    The method of choosing the selected draw (e.g. median). If limited to quantiles, then p.
    '''

    draw_to_do = True
    h = {}
    draws = None
    qp = q_p(p, n)

    '''
    All calculations will be made from the same sample.
    This allows in particular to obtain the same result when recalculating f_smoothed at the same point.
    '''

    def f_smoothed(x):
        '''
        x is an element of Rd
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
            experience = []
            for draw in draws:
                x_noisy = x+draw
                experience.append(float(f(x_noisy)))
            experience.sort()

            h[tuple(x)] = experience[qp]

        return h[tuple(x)]

    return f_smoothed


def smoothing_esp(f, n, G):
    '''
    Takes a function f from Rd to R and returns its smoothed function.
    Requires to choose:
    The number of random noise draws n,
    The random noise variable (e.g. centered reduced Gaussian) G,
    The method for choosing the draw: here the expectation.
    '''

    draw_to_do = True
    g = {}
    draws = None

    def smoothed_f(x):
        '''
        x is an element of Rd
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
            experience = []
            for draw in draws:
                noisy_x = x+draw
                experience.append(float(f(noisy_x)))

            g[tuple(x)] = esp_choice(experience)

        return g[tuple(x)]

    return smoothed_f


def q_p(p, n):
    '''
    we do not take the average of two values. here we took the lower index.
    '''
    return min(n-1, max(0, int((n+1)*p)-1))


def esp_choice(experience):
    '''
    Returns the expectation of experience.
    '''
    res = 0
    for el in experience:
        res += el
    return res/len(experience)


def diff_curve(f, n, G, p):
    '''
    only works for d=1
    '''

    l_x = pl.linspace(-10, 10, 1000)

    smoothed_f = smoothing(f, n, G, p)
    esp_f = smoothing_esp(f, n, G)

    l_f = [f([x]) for x in l_x]
    l_smoothed = [smoothed_f([x]) for x in l_x]
    l_esp = [esp_f([x]) for x in l_x]

    pl.plot(l_x, l_f, label='f')
    pl.plot(l_x, l_smoothed, label='f_p')
    pl.plot(l_x, l_esp, label='f_esp')

    pl.legend()

    pl.show()


# diff_curve(pl.sin, 10, good_gaussian(2), 0.5)


def phi(sigma, moy=0):
    '''
    Returns the cdf of the centered Gaussian.
    '''
    def inner_phi(x):
        return scipy.stats.norm.cdf(x, moy, sigma)

    return inner_phi


def phi_minus_1(sigma, moy=0):
    '''
    Returns the inverse of the cdf of the centered Gaussian.
    '''
    def inner_phi_minus_1(p):
        return scipy.stats.norm.ppf(p, moy, sigma)

    return inner_phi_minus_1


def smoothing_and_bounds_esp(f, n, sigma, u, l, delta, alpha):
    '''
    To obtain the bounds of the paper, it is necessary to normalize f and therefore that it is bounded in [u,l].
    The formula only works with a centered Gaussian so no need for G but only for sigma.
    Needs to know the bound on the delta attacks (for now I put 0.1 at random for the 1D case).
    alpha is the confidence we want to have in the bound (0.999 for example).
    n is used to calculate smoothed_f and also
    '''


def smoothing_and_bounds_esp(f, n, sigma, u, l, delta, alpha):
    '''
    To have the bounds of the paper, we need to normalize f, and thus it should be bounded in [u, l].
    The formula only works with a centered Gaussian, so there is no need for G, only sigma.
    It is necessary to know the bound on the attacks delta (for now, I randomly put 0.1 for the 1D case).
    alpha is the confidence we want to have in the bound (0.999 for example).
    n is used to calculate f_smoothed and also for the quality of the bound because the larger n is, the more confident we are.
    The security expression follows from the weak law of large numbers.
    '''

    G = good_gaussian(sigma)

    draw_to_do = True
    g = {}
    draws = None
    security = (u-l)/(2*pl.sqrt(n*(1-alpha)))

    phi_sigma = phi(sigma)
    phi_minus_1_sigma = phi_minus_1(sigma)

    def f_smoothed(x):
        '''
        x is an element of Rd
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
            experience = []
            for draw in draws:
                x_noisy = x+draw
                experience.append(float(f(x_noisy)))

            f_esp = esp_choice(experience)
            g[tuple(x)] = l+(u-l)*phi_sigma((sigma*phi_minus_1_sigma((f_esp-l)/(u-l))-delta-security) /
                                            sigma), f_esp, l+(u-l)*phi_sigma((sigma*phi_minus_1_sigma((f_esp-l)/(u-l))+delta+security)/sigma)

        return g[tuple(x)]

    return f_smoothed


def smoothing_and_bounds(f, n, sigma, p, alpha, epsilon):
    '''
    Takes a function f from Rd to R and returns its smoothed function.
    Requires choosing:
    The number of iterations of random noise generation n,
    The random variable of the noise (e.g. centered reduced Gaussian) G,
    The method for selecting the drawn sample (e.g. median). If we limit ourselves to quantiles, then p.
    '''

    G = good_gaussian(sigma)
    to_do_sampling = True
    h = {}
    samples = None
    ql = q_bot(p, n, alpha, epsilon, sigma)
    qp = q_p(p, n)
    qu = q_top(p, n, alpha, epsilon, sigma)

    '''
    All calculations will be done from the same sample.
    This makes it possible in particular to obtain the same result when recalculating f_smoothed at the same point.
    '''

    def f_smoothed(x):
        '''
        x is an element of Rd
        '''

        nonlocal to_do_sampling
        nonlocal h
        nonlocal samples

        if to_do_sampling:
            samples = []
            for _ in range(n):
                samples.append(G(x))
            to_do_sampling = False

        if tuple(x) not in h:
            experience = []
            for sample in samples:
                x_noisy = x+sample
                experience.append(float(f(x_noisy)))
            experience.sort()

            h[tuple(x)] = experience[ql], experience[qp], experience[qu]

        return h[tuple(x)]

    return f_smoothed


def q_bot(p, n, alpha, epsilon, sigma):
    p_bot = phi(sigma)(phi_minus_1(sigma)(p)-epsilon/sigma)
    ql = max(0, int(n - scipy.stats.binom.ppf(alpha, n, 1 - p_bot) - 2))
    return ql


def q_top(p, n, alpha, epsilon, sigma):
    p_top = phi(sigma)(phi_minus_1(sigma)(p)+epsilon/sigma)
    qu = min(n-1, int(scipy.stats.binom.ppf(alpha, n, p_top)))
    return qu


def curves_and_bounds(f, n, sigma, p, alpha, epsilon):
    smoothed_f = smoothing_and_bounds(f, n, sigma, p, alpha, epsilon)

    l_x = pl.linspace(-10, 10, 1000)

    l_f = [f([x]) for x in l_x]
    l_smoothed = [smoothed_f([x])[1] for x in l_x]
    l_inf = [smoothed_f([x])[0] for x in l_x]
    l_sup = [smoothed_f([x])[2] for x in l_x]

    pl.plot(l_x, l_f, label='f')
    pl.plot(l_x, l_smoothed, label='smoothed_f')
    pl.plot(l_x, l_inf, label='f_inf')
    pl.plot(l_x, l_sup, label='f_sup')

    pl.legend()

    pl.show()

# curves_and_bounds(pl.sin, 1000, 1, 0.5, 0.99, 0.1)


def curves_and_bounds_esp(f, n, sigma, u, l, alpha, epsilon):
    smoothed_f = smoothing_and_bounds_esp(f, n, sigma, u, l, epsilon, alpha)

    l_x = pl.linspace(-10, 10, 1000)

    l_f = [f([x]) for x in l_x]
    l_smoothed = [smoothed_f([x])[1] for x in l_x]
    l_inf = [smoothed_f([x])[0] for x in l_x]
    l_sup = [smoothed_f([x])[2] for x in l_x]

    pl.plot(l_x, l_f, label='f')
    pl.plot(l_x, l_smoothed, label='smoothed_f')
    pl.plot(l_x, l_inf, label='f_inf')
    pl.plot(l_x, l_sup, label='f_sup')

    pl.legend()

    pl.show()

# curves_and_bounds_esp(pl.sin, 1000, 1, -1, 1, 0.99, 0.1)

# test_smoothed = smoothing_and_bounds(test, 100, 1, 0.5, 0.9, 1)
# print(test_smoothed([17.76, 42.42, 1009.09, 66.26]))