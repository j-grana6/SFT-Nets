"""
The model with Munsky's parameters
"""

import numpy as np
from sft import *
from sft_net import *
from tools import *
from lhood_comps import *
import os
import copy

#  Create 4 nodes
A = SFT('A', ['normal', 'infected'], ['B', 'C'],
      {'B':np.array([[1, 0], [1, 1/10000.]]),
       'C': np.array([[1,0], [1,1/10000.]])},
       ['clean', 'malicious'], 'external')

B = SFT('B', ['normal', 'infected'], ['C', 'D'],
       {'C': np.array([[1,0], [1,.1]]),
        'D' : np.array([[1,0], [1,.1]])},
       ['clean', 'malicious'], 'internal')

C = SFT('C', ['normal', 'infected'], ['D'],
       {'D': np.array([[1,0], [1,.1]])},
       ['clean', 'malicious'], 'internal')

D = SFT('D', ['normal', 'infected'], [], {}, [], 'internal')

T=10000
nodes = [A, B, C, D]
net = SFTNet(nodes)
# Put the nodes together and create a net
data = gen_data(10000, net, ['infected', 'normal', 'normal', 'normal'])
# Generate artificial data
print data[-1]
# print the true infection times
guess_times = {'A': 0, 'B': 5000, 'C': 4000, 'D': 4500}
lp = -np.inf
# This loop picks starting values for the MCMC
# It loops though possible values of B and C by 250
# and sets D 'close' to the minimum of the two
logn_fact = gen_logn_fact(data)
for j in np.arange(1000,10000,1000):
    for k in np.arange(1000,10000, 1000):
        s0 =  {'A': 0, 'B': j, 'C': k-1, 'D': min(j,k) + 10}
        newprob = sum(prob_model_given_data(net, data, s0, 10000,
                                            logn_fact))
        if newprob > lp:
            lp = newprob
            guess_times = s0
    print j
print guess_times
monte_carlo_samples = 500
mcmc_samples = 500


prob_no_attacker = prob_model_no_attacker(net, data, T)
prob_true_value = prob_model_given_data(net, data, data[-1], T, logn_fact)

print 'Probability no attacker is', prob_no_attacker, '/n'
print 'Probability at true params is ', prob_true_value


smc = MC_int(net, data, monte_carlo_samples,
                            guess_times, T)
# do simple Monte Carlo
mcmc = MCMC_MH(net, data, mcmc_samples, guess_times, T,
                uniform=True)
# Do MCMC
#nonumcmc = MCMC_MH(net, data, 10000, guess_times, T,
                    #uniform=False)

simple_result = Results(smc, data[-1], prob_no_attacker,
                        prob_true_value, data)

mcmc_results = Results(mcmc, data[-1], prob_no_attacker,
                       prob_true_value, data, metropolis = True)



### Write Results.  Mix this in with the results class













