import numpy as np
from sft import *
from sft_net import *
from tools import *
from sft_mcmc import MCMC_SFT

#  Create 4 nodes
A = SFT('A', ['normal', 'infected'], ['B', 'C'],
      {'B':np.array([[1, 0], [1, 0.]]),
       'C': np.array([[1,0], [1,0.]])},
       ['clean', 'malicious'])

B = SFT('B', ['normal', 'infected'], ['C', 'D'],
       {'C': np.array([[1,0], [1,0]]),
        'D' : np.array([[1,0], [1,0]])},
       ['clean', 'malicious'])

C = SFT('C', ['normal', 'infected'], ['D'],
       {'D': np.array([[1,0], [1,0]])},
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
# It loops though possible values of B and C by 500
# and sets D 'close' to the minimum of the two
for j in np.arange(1000,10000,1000):
    for k in np.arange(1000,10000, 1000):
        s0 =  {'A': 0, 'B': j, 'C': k-1, 'D': min(j,k) + 10}
        newprob = prob_model_given_data(net, data[1], s0,
                                        data[2], data[3], 10000)
        print newprob
        if newprob > lp:
            lp = newprob
            guess_times = s0
print guess_times
mcmc_steps1 = 1000
burn_in1 = 200
mcmc_steps2 = 5000
burn_in2 = 500
prob_no_attacker = prob_model_no_attacker(net, data, 10000)
print 'Probability no attacker is', prob_no_attacker, '/n'
res1 = MCMC_SFT(net, data, mcmc_steps1, guess_times, 10000)
res2 = MCMC_SFT_old(net, data, mcmc_steps2, guess_times, 10000)
prob_with_attacker1 = np.sum(res1[1][burn_in1 : ])/(mcmc_steps1 - burn_in1)
prob_with_attacker2 = np.sum(res2[1][burn_in2 : ])/(mcmc_steps2 - burn_in2)



