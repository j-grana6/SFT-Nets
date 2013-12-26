"""
The model with Munsky's parameters
"""

import numpy as np
from sft import *
from sft_net import *
from tools import *
from sft_mcmc import MCMC_SFT

#  Create 4 nodes
A = SFT('A', ['normal', 'infected'], ['B', 'C'],
      {'B':np.array([[1, 0], [1, 1/10000.]]),
       'C': np.array([[1,0], [1,1/10000.]])},
       ['clean', 'malicious'])

B = SFT('B', ['normal', 'infected'], ['C', 'D'],
       {'C': np.array([[1,0], [1,.1]]),
        'D' : np.array([[1,0], [1,.1]])},
       ['clean', 'malicious'])

C = SFT('C', ['normal', 'infected'], ['D'],
       {'D': np.array([[1,0], [1,.1]])},
       ['clean', 'malicious'])

D = SFT('D', ['normal', 'infected'], [], {}, [])


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
for j in np.arange(1,13000,500):
    for k in np.arange(1,13000, 500):
        s0 =  {'A': 0, 'B': j, 'C': k, 'D': j+5}
        newprob = prob_model_given_data(net, data[1], s0,
                                        data[2], data[3], 10000)
        if newprob > lp:
            lp = newprob
            guess_times = s0
        s0 = {'A': 0, 'B': j, 'C': k, 'D': k+5}
        newprob = prob_model_given_data(net, data[1], s0,
                                        data[2], data[3], 10000)
        if newprob > lp:
            lp = newprob
            guess_times = s0
guess_times['C'] +=1
res = MCMC_SFT(net, data, 10000, guess_times, 10000)

## Probability given no attacker is just the probability of the time
## time sequence of messages.

