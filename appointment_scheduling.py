import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import optimize # finding roots
from scipy import special # n choose k function
import itertools # enumerating subsets

import matplotlib.pyplot as plt
plt.switch_backend('agg')
np.random.seed(42)

n = 20 # number of customers (physicians see about 20 patients per day)
mean_service_time = 5 # mean service times
stddev_service_time = mean_service_time / 2 # std. dev. of service time
p = 0.05 # cancellation probability
c_e = 5 # unit earliness cost
c_l = 10 # unit lateness cost
q = c_l / (c_e + c_l) # critical ratio
SQRT_2 = math.sqrt(2)
mu_s = mean_service_time * np.ones(n+1)
sigma_s = stddev_service_time * np.ones(n+1)
mu_s[0] = 0 # no service time at the depot
sigma_s[0] = 0 
d = np.zeros([n+1, n+1])
x_coord = np.random.uniform(low=0, high=10, size=n+1)
y_coord = np.random.uniform(low=0, high=10, size=n+1)

a = [] # list of appointment times
a.append(0) # appointment at depot is 0

def get_distance(i,j):
    return np.sqrt((x_coord[i] - x_coord[j]) ** 2 + (y_coord[i] - y_coord[j]) ** 2)

for i in range(n+1):
    for j in range(i,n+1):
        d[i,j] = get_distance(i,j)
        d[j,i] = d[i,j]
mu_t = np.array(d)
sigma_t = np.array(d) / 2

def G(mu, sigma, x):
    '''Calculates the CDF of a normal distribution with mean mu and
    standard deviation sigma evaluated at a point x'''
    return (1 / 2) * (1 + math.erf((x - mu)/(SQRT_2 * sigma)))

def H(delta, x):
    '''Cumulative distribution function for the normally-distributed
    random variable reflecting the elapsed time between arriving at
    the first stop and arriving at the last stop along the subroute
    delta, assuming service commences immediately upon arrival'''
    mu = 0
    sigma = 0
    for i in range(len(delta) - 1):
        current_stop = delta[i]
        next_stop = delta[i+1]
        mu += mu_s[current_stop]
        mu += mu_t[current_stop, next_stop]
        sigma += sigma_s[current_stop] ** 2
        sigma += sigma_t[current_stop, next_stop] ** 2
    sigma = math.sqrt(sigma)
    return G(mu, sigma, x)

def Q(gamma, i, j, x):
    return H(gamma[i:(j+1)], x)


def get_gamma_cdf(gamma, x):
    result = 0
    #print('Gamma', gamma)
    k = len(gamma) - 1
    # compute xi_i for i in 1 to k-1
    xi = [get_gamma_cdf(gamma[0:i+1], a[gamma[i]]) for i in range(1,k)]
    xi.insert(0,0)
    result += np.product([(1-xi[i]) for i in range(1,k)]) * Q(gamma, 0, k, x)
    #print(xi)
    for i in range(1, k):
        result += np.product([(1-xi[j]) for j in range(i+1, k)]) * xi[i] * Q(gamma, i, k, x - a[gamma[i]])
    return result

def get_subroutes(route, k):
    """Given a route and a number k, returns the set of routes that emerge from deleting k customers"""
    assert k <= len(route)-2, 'k is too large'
    original = route.copy()
    intermediates = original[1:-1]
    m = len(intermediates) 
    subroutes = [[original[0]] + list(subroute) + [original[-1]] for subroute in itertools.combinations(intermediates, m - k)]
    return subroutes

def get_CDF(j, x):
    route = [i for i in range(0,j+1)]
    result = 0
    for k in range(j):
        subroutes = get_subroutes(route, k)
        term = 0
        for gamma in subroutes:
            term += get_gamma_cdf(gamma, x)
        term = special.comb(j - 1, k) * term
        term = ( p ** k) * ((1 - p) ** (j - 1 - k)) * term
        term = (1 / len(subroutes)) * term
        result += term
    return result - q

def F1(x):
    return get_CDF(1,x)

def plot_CDF(j):
    Z = np.arange(-50,60,.1)
    fig = plt.figure()
    plt.scatter(Z, [get_CDF(j, z) for z in Z])
    plt.show()
    def my_F(x):
        return get_CDF(j,x)
    sol = optimize.root_scalar(my_F, bracket=[-50,50])
    print("Root at ", sol.root)
    if len(a) < j+1:
        a.append(sol.root)
print(a)
plot_CDF(1)
print(a)