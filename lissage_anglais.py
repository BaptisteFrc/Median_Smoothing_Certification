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
    """Returns the smoothed function of the function in input f

    Args:
        f (function): Rd -> R
        n (int): number of iterations for the random draw for the noise
        G (function): random variable for the noise
        p (float): depends on the method of draw chosen, here quantiles. p is between 0 and 1

    Returns:
        function: f_smoothed
    """

    draw_to_do = True
    h = {}
    draws = None
    qp = q_p(p, n)

    '''
    All calculations will be made from the same sample.
    This allows in particular to obtain the same result when recalculating f_smoothed at the same point.
    '''

    def smoothed_f(x):
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
            sample = []
            for draw in draws:
                x_with_noise = x+draw
                sample.append(float(f(x_with_noise)))
            sample.sort()

            h[tuple(x)] = sample[qp]

        return h[tuple(x)]

    return smoothed_f


def smoothing_exp(f, n, G):
    """

    Args:
        f (function)): from Rd to R
        n (int): number of iterations for the random draw of the noise
        G (function): random variable of the noise (e.g. standard normal distribution)

    Returns:
        function: smoothed version of f
    """

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
            sample = []
            for draw in draws:
                x_with_noise = x+draw
                sample.append(float(f(x_with_noise)))

            g[tuple(x)] = exp(sample)

        return g[tuple(x)]

    return smoothed_f


def q_p(p, n):
    '''
    We do not take the average of two values. Here we choose to consider the lower index.
    '''
    return min(n-1, max(0, int((n+1)*p)-1))


def exp(sample):
    '''
    Returns the expected value of the experiment.
    '''
    res = 0
    for el in sample:
        res += el
    return res/len(sample)


def graph_diff(f, n, G, p):
    '''
    only works for d=1
    '''

    l_x = pl.linspace(2, 5, 1000)

    smoothed_f = smoothing(f, n, G, p)
    exp_f = smoothing_exp(f, n, G)

    l_f = [f([x]) for x in l_x]
    l_smoothed = [smoothed_f([x]) for x in l_x]
    l_exp = [exp_f([x]) for x in l_x]

    pl.plot(l_x, l_f, label='f')
    pl.plot(l_x, l_smoothed, label='f_p')
    pl.plot(l_x, l_exp, label='f_exp')

    pl.legend()

    pl.show()


# graph_diff(lambda x: abs(pl.sin(x)), 300, good_gaussian(0.5), 0.5)


def phi(sigma, mean=0):
    '''
    Returns the cdf of the centered Gaussian.
    '''
    def inner_phi(x):
        return scipy.stats.norm.cdf(x, mean, sigma)

    return inner_phi


def phi_minus_1(sigma, mean=0):
    '''
    Returns the inverse of the cdf of the centered Gaussian.
    '''
    def inner_phi_minus_1(p):
        return scipy.stats.norm.ppf(p, mean, sigma)

    return inner_phi_minus_1


def smoothing_and_bounds_exp(f, n, sigma, u, l, epsilon, alpha):
    """
    To have the bounds of the paper, we need to normalize f, and thus it should be bounded in [u, l].
    The formula only works with a centered Gaussian, so there is no need for G, only sigma.
    It is necessary to know the bound on the attacks epsilon (for now, I randomly put 0.1 for the 1D case).
    alpha is the confidence we want to have in the bound (0.999 for example).
    n is used to calculate f_smoothed and also for the quality of the bound because the larger n is, the more confident we are.
    The security expression follows from the weak law of large numbers.

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
            sample = []
            for draw in draws:
                x_with_noise = x+draw
                sample.append(float(f(x_with_noise)))

            f_exp = exp(sample)
            g[tuple(x)] = l+(u-l)*phi_sigma((sigma*phi_minus_1_sigma((f_exp-l)/(u-l))-epsilon-security) /
                                            sigma), f_exp, l+(u-l)*phi_sigma((sigma*phi_minus_1_sigma((f_exp-l)/(u-l))+epsilon+security)/sigma)

        return g[tuple(x)]

    return f_smoothed


def smoothing_and_bounds(f, n, sigma, p, alpha, epsilon):
    """Takes a function f and returns its smoothed function.

    Args:
        f (function): from Rd to R
        n (int): number of iterations of random noise generation
        sigma (float): standard deviation for her centered Gaussian distribution
        p (float): quantile, between 0 and 1
        alpha (float): confidence rate
        epsilon (float): bounds for the attack

    Returns:
        function: the smoothed version of the function f
    """

    G = good_gaussian(sigma)
    draw_to_do = True
    h = {}
    draws = None
    ql = q_lower(p, n, alpha, epsilon, sigma)
    qp = q_p(p, n)
    qu = q_upper(p, n, alpha, epsilon, sigma)

    '''
    All calculations will be done from the same sample.
    This makes it possible in particular to obtain the same result when recalculating f_smoothed at the same point.
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
            sample = []
            for draw in draws:
                x_with_noise = x+draw
                sample.append(float(f(x_with_noise)))
            sample.sort()

            h[tuple(x)] = sample[ql], sample[qp], sample[qu]

        return h[tuple(x)]

    return f_smoothed


def q_lower(p, n, alpha, epsilon, sigma):
    p_bot = phi(sigma)(phi_minus_1(sigma)(p)-epsilon/sigma)
    ql = max(0, int(n - scipy.stats.binom.ppf(alpha, n, 1 - p_bot) - 2))
    return ql


def q_upper(p, n, alpha, epsilon, sigma):
    p_top = phi(sigma)(phi_minus_1(sigma)(p)+epsilon/sigma)
    qu = min(n-1, int(scipy.stats.binom.ppf(alpha, n, p_top)))
    return qu


def graph_and_bounds(f, n, sigma, p, alpha, epsilon):
    smoothed_f = smoothing_and_bounds(f, n, sigma, p, alpha, epsilon)

    l_x = pl.linspace(2, 5, 1000)

    l_f = [f([x]) for x in l_x]
    l_smoothed = [smoothed_f([x])[1] for x in l_x]
    l_lower = [smoothed_f([x])[0] for x in l_x]
    l_upper = [smoothed_f([x])[2] for x in l_x]

    pl.plot(l_x, l_f, label='f')
    pl.plot(l_x, l_smoothed, label='smoothed_f')
    pl.plot(l_x, l_lower, label='f_l')
    pl.plot(l_x, l_upper, label='f_u')

    pl.legend()

    pl.show()


# graph_and_bounds(lambda x: abs(pl.sin(x)), 1000, 0.1, 0.5, 0.99, 0.1)


def graph_and_bounds_exp(f, n, sigma, u, l, alpha, epsilon):
    smoothed_f = smoothing_and_bounds_exp(f, n, sigma, u, l, epsilon, alpha)

    l_x = pl.linspace(-10, 10, 1000)

    l_f = [f([x]) for x in l_x]
    l_smoothed = [smoothed_f([x])[1] for x in l_x]
    l_lower = [smoothed_f([x])[0] for x in l_x]
    l_upper = [smoothed_f([x])[2] for x in l_x]

    pl.plot(l_x, l_f, label='f')
    pl.plot(l_x, l_smoothed, label='smoothed_f')
    pl.plot(l_x, l_lower, label='f_l')
    pl.plot(l_x, l_upper, label='f_u')

    pl.legend()

    pl.show()


# graph_and_bounds_exp(pl.sin, 1000, 1, -1, 1, 0.99, 0.1)

test_smoothed = smoothing_and_bounds(test, 100, 1, 0.5, 0.9, 1)
print(test_smoothed([17.76, 42.42, 1009.09, 66.26]),
      test([17.76, 42.42, 1009.09, 66.26]))
