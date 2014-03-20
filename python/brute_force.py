"""
Evaluates the integral by brute force by doing a Riemann
sum to "infinity."  This is obviously inefficient and cannot be
used for large nets but can be used to compare the accuracy of MCMC
and importance sampling
...

This is futile
"""

from testing_net import net
from orderings import *
from tools import prob_model_given_data, gen_data, gen_logn_fact
import numpy as np

def brute(net, data, T, deltat):
    logn_fact = gen_logn_fact(data)
    orders = gen_orderings(net, {'A': 'infected'
                             , 'B': 'normal', 'C': 'normal', 'D': 'normal'})

    orders = [orders[i] for i in xrange(len(orders)) if len(orders[i])==4]
    evals = []
    for order in orders:
        print order
        for n1 in np.arange(0, 2*T, deltat):
            for n2 in np.arange(n1+.0001, 2*T, deltat):
                print n2
                for n3 in np.arange(n2+.0001, 2*T, deltat):
                    times = dict(zip(order, (0, n1, n2,n3)))
                    evals.append(sum(prob_model_given_data(net, data, times, T, logn_fact)))
    return np.mean(evals)

if __name__ == "__main__":
    data = gen_data(10000, net, {'A': 'infected'
                             , 'B': 'normal', 'C': 'normal', 'D': 'normal'})
    res = brute(net, data, 10000, 5)
