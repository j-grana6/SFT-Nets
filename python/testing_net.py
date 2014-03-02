from sft import *
from sft_net import *

A = SFT('A', ['normal', 'infected'], ['B', 'C'],
      {'B':np.array([[1, 0], [1, 1/10000.]]),
       'C': np.array([[1, 0], [1,1/10000.]])},
       ['clean', 'malicious'], 'external')

B = SFT('B', ['normal', 'infected'], ['C', 'D'],
       {'C': np.array([[1, 0], [1, .1]]),
        'D' : np.array([[1, 0], [1, .1]])},
       ['clean', 'malicious'], 'internal')

C = SFT('C', ['normal', 'infected'], ['D'],
       {'D': np.array([[1, 0], [1,.1]])},
       ['clean', 'malicious'], 'internal')

D = SFT('D', ['normal', 'infected'], [], {}, [], 'internal')

T=10000
nodes = [A, B, C, D]
net = SFTNet(nodes)

A2 = SFT('A', ['normal', 'infected'], ['B', 'C'],
      {'B':np.array([[1, 0], [4, 1/10000.]]),
       'C': np.array([[1,0], [4,1/10000.]])},
       ['clean', 'malicious'], 'external')
nodes = [A2, B, C,D]
net2 = SFTNet(nodes)
