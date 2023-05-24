def normalize(f, u, l) :
    def inner(x) :
        return (f(x)-l)/(u-l)
    return inner


# def robustness_at_x(f, x, n) :
#     '''Pour que cela ait du sens il faut que f soit normalisée.'''
#     value=f(x)
#     for _ in range(n)
#     return 


