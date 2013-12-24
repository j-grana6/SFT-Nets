"""
The model with Munsky's parameters
"""

import numpy as np
from sft import *
from sft_net import *
from tools import *
from sft_mcmc import MCMC_SFT

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
data = gen_data(10000, net, ['infected', 'normal', 'normal', 'normal'])
print data[-1]
guess_times = {'A': 0, 'B': 5000, 'C': 4000, 'D': 4500}
res = MCMC_SFT(net, data, 10000, guess_times, 10000)




