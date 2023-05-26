from scipy.integrate import nquad
from Smoothing_v3.utils import norm_2
from numpy import array, exp, log

def u(x, E) :
    res=[]
    for i in range(len(x)) :
        res.append((x[i]-E[i][0])/(E[i][1]-E[i][0]))
    return res

def mesure(F) :
    return nquad(lambda *args : 1, F)[0]

def sensitivity_at_x(f, x, E, F, fmax, fmin) :
    def inner(*args) :
        y=[yi for yi in args]
        return abs(f(x)-f(y))/norm_2(array(u(x, E))-array(u(y, E)))
    return nquad(inner, F, opts={'points' : x})[0]/mesure(F)/(fmax-fmin)

print(sensitivity_at_x(lambda x : x[0], [0.5, 0.5], [[0,1], [0,1]], [[0,1], [0,1]], 1, 0))

def sensitivity(f, E, fmax, fmin) :
    def inner(*args) :
        x=[xi for xi in args]
        return log(sensitivity_at_x(f, x, E, E, fmax, fmin))
    return exp(nquad(inner, E)[0]/mesure(E))

def robustness(f, E, fmax, fmin) :
    return 1/sensitivity(f, E, fmax, fmin)

print(robustness(lambda x : x[0], [[0,1], [0,1]], 1, 0))