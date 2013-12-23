"""
The model with Munsky's parameters
"""

import numpy as np
from sft import *
from sft_net import *
from tools import *

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
guess_times = {'A': 0, 'B': 3500, 'C': 3506, 'D': 3000}
prob_model = prob_model_given_data(net, data[1],
                                   guess_times, data[2], data[3], 10000)
