from sft import *
from sft_net import *

A = SFT('A', ['normal', 'infected'], ['B', 'C'],
      {'B':np.array([[1, 0], [1, 1./10000.]]),
       'C': np.array([[1, 0], [1, 1./10000.]])},
       ['clean', 'malicious'], 'external')

B = SFT('B', ['normal', 'infected'], ['C', 'D'],
       {'C': np.array([[1, 0], [1, .1]]),
        'D' : np.array([[1, 0], [1, .1]])},
       ['clean', 'malicious'], 'internal')

C = SFT('C', ['normal', 'infected'], ['D'],
       {'D': np.array([[1, 0], [1, .2]])},
       ['clean', 'malicious'], 'internal')

D = SFT('D', ['normal', 'infected'], [], {}, [], 'internal')


T=10000
nodes = [A, B, C, D]
net = SFTNet(nodes)

A_clean = SFT('A', ['normal', 'infected'], ['B', 'C'],
      {'B':np.array([[1, 0], [1,  0]]),
       'C': np.array([[1, 0], [1, 0]])},
       ['clean', 'malicious'], 'external')


nodes_clean = [A_clean, B, C, D]
net_clean = SFTNet(nodes_clean)

