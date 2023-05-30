'''
mean
commentaires
cohérence des formalisme
segmentation
pylab

sigma/epsilon
'''

from Smoothing_v3.utils import *
from types import FunctionType


def smoothing(f : FunctionType, n : int, G : FunctionType, p : float) ->FunctionType:
    """Returns the smoothed function of the function in input f, using the quantil method.

    Args:
        f (function): Rd -> R
        n (int): number of iterations for the random draw for the noise
        G (function): random variable for the noise
        p (float): depends on the method of draw chosen, here quantiles. p is between 0 and 1

    Returns:
        function: f_smoothed
    """

    qp = q_p(n, p)

    """
    supprimer si v3
    All calculations will be made from the same sample.
    This allows in particular to obtain the same result when recalculating f_smoothed at the same point.
    This is also the case for every function below.
    """

    def smoothed_f(x : list):

        sample = []

        for _ in range(n):
            x_with_noise = x+G(x)
            sample.append(float(f(x_with_noise)))
        sample.sort()

        return sample[qp]

    return smoothed_f


def smoothing_exp(f : FunctionType, n : int, G : FunctionType) ->FunctionType:
    """Returns the smoothed function of the function in input f, using the mean method.

    Args:
        f (function)): from Rd to R
        n (int): number of iterations for the random draw of the noise
        G (function): random variable of the noise (e.g. standard normal distribution)

    Returns:
        function: smoothed version of f
    """

    def smoothed_f(x : list):

        sample = []

        for _ in range(n):
            x_with_noise = x+G(x)
            sample.append(float(f(x_with_noise)))

        return exp(sample)

    return smoothed_f


def smoothing_and_bounds(f : FunctionType, n : int, sigma : float, p : float, alpha : float, epsilon : float) ->FunctionType:
    """
    Args:
        f (function): from Rd to R
        n (int): number of iterations of random noise generation
        sigma (float): standard deviation for her centered Gaussian distribution
        p (float): quantile, between 0 and 1
        alpha (float): confidence rate
        epsilon (float): bounds for the attack

    Returns:
        function: smoothed_f and the bounds
    """

    G = good_gaussian(sigma)
    ql = q_lower(n, p, alpha, epsilon, sigma)
    qp = q_p(n, p)
    qu = q_upper(n, p, alpha, epsilon, sigma)

    def f_smoothed(x : list):

        sample = []
        for _ in range(n):
            x_with_noise = x+G(x)
            sample.append(float(f(x_with_noise)))
        sample.sort()

        return sample[ql], sample[qp], sample[qu]

    return f_smoothed


def smoothing_and_bounds_exp(f : FunctionType, n : int, sigma : float, l : float, u : float, epsilon : float, alpha : float) ->FunctionType:
    """
    To have the bounds of the paper, we need f to be normalized, and thus it should be bounded in [u, l].
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
        l (float): minimum of f
        u (float): maximum of f
        epsilon (float): bound of the attack
        alpha (float): confidence rate of the bounds obtained for the output of the function

    Returns:
        function: smoothed_f and the bounds
    """

    G = good_gaussian(sigma)
    security = (u-l)/(2*np.sqrt(n*(1-alpha)))

    def f_smoothed(x : list):

        sample = []
        for _ in range(n):
            x_with_noise = x+G(x)
            sample.append(float(f(x_with_noise)))

        f_exp = exp(sample)

        return l+(u-l)*phi((sigma*phi_minus_1((f_exp-l)/(u-l), sigma)-epsilon-security)/sigma, sigma), f_exp, l+(u-l)*phi((sigma*phi_minus_1((f_exp-l)/(u-l), sigma)+epsilon+security)/sigma, sigma)

    return f_smoothed


def max_bound(f : FunctionType, n : int, sigma : float, p : float, alpha : float, epsilon : float, precision : float) ->FunctionType:
    """This time we bound the theoritical function but also the practical and computational smoothed_f.

    Args:
        f (function): from Rd to R
        n (int): number of iterations of random noise generation
        sigma (float): standard deviation for her centered Gaussian distribution
        p (float): quantile, between 0 and 1
        alpha (float): confidence rate
        epsilon (float): bounds for the attack
        precision (float) : how precise p_minus and p_plus should be

    Returns:
        function: f_smoothed and the 4 bounds.
    """

    G = good_gaussian(sigma)
    ql = q_lower(n, p, alpha, epsilon, sigma)
    qp = q_p(n, p)
    qu = q_upper(n, p, alpha, epsilon, sigma)
    qlmax = q_lower(n, p_minus(n, p, alpha, precision), alpha, epsilon, sigma)
    qumax = q_upper(n, p_plus(n, p, alpha, precision), alpha, epsilon, sigma)

    def f_smoothed(x : list):

        sample = []
        for _ in range(n):
            x_with_noise = x+G(x)
            sample.append(float(f(x_with_noise)))
        sample.sort()

        return sample[qlmax], sample[ql], sample[qp], sample[qu], sample[qumax]

    return f_smoothed


def max_bound_exp(f : FunctionType, n : int, sigma : float, l : float, u : float, epsilon : float, alpha : float) ->FunctionType:
    """
    Same for the mean method.

    Args:
        f (function): _description_
        n (int): number of iterations for the random draw of the noise
        sigma (float): standard deviation of the noise, has an impact on the quality of the bound, 
            the bigger the more trustworthy 
        l (float): minimum of f
        u (float): maximum of f
        epsilon (float): bound of the attack
        alpha (float): confidence rate of the bounds obtained for the output of the function

    Returns:
        function: f_smoothed and the 4 bounds.
    """

    G = good_gaussian(sigma)
    security = (u-l)/(2*np.sqrt(n*(1-alpha)))

    def f_smoothed(x : list):

        sample = []
        for _ in range(n):
            x_with_noise = x+G(x)
            sample.append(float(f(x_with_noise)))

        f_exp = exp(sample)
        f_l = l+(u-l)*phi((sigma*phi_minus_1((f_exp-l) /
                                             (u-l), sigma)-epsilon-security)/sigma, sigma)
        f_u = l+(u-l)*phi((sigma*phi_minus_1((f_exp-l) /
                                             (u-l), sigma)+epsilon+security)/sigma, sigma)

        return f_l-security, f_l, f_exp, f_u, f_u+security

    return f_smoothed
