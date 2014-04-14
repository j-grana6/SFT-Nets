from sft import SFT as SFTnew
from sft_net import SFTNet as SFTNetnew
from sft_old import SFT as SFTold
from sft_net_old import SFTNet as SFTNetold
import numpy as np
from tools import prob_model_given_data, gen_data
from tools_old import prob_model_given_data as pmgd_old
from tools_old import gen_data as gd_old
#from direct_sample import Direct_Sample
#from direct_sample_old import Direct_Sample as DSold

Anew = SFTnew('A', ['normal', 'infected'], ['B', 'C'],
      {'B':np.array([[1, 0], [1, 1./10000.]]),
       'C': np.array([[1, 0], [1, 1./10000.]])},
       ['clean', 'malicious'], 'external')

Bnew = SFTnew('B', ['normal', 'infected'], ['C', 'D'],
       {'C': np.array([[1, 0], [1, .1]]),
        'D' : np.array([[1, 0], [1, .1]])},
       ['clean', 'malicious'], 'internal')

Cnew = SFTnew('C', ['normal', 'infected'], ['D'],
       {'D': np.array([[1, 0], [1, .2]])},
       ['clean', 'malicious'], 'internal')

Dnew = SFTnew('D', ['normal', 'infected'], [], {}, [], 'internal')


T=10000
nodes = [Anew, Bnew, Cnew, Dnew]
netnew = SFTNetnew(nodes)

Aold = SFTold('A', ['normal', 'infected'], ['B', 'C'],
      {'B':np.array([[1, 0], [1, 1./10000.]]),
       'C': np.array([[1, 0], [1, 1./10000.]])},
       ['clean', 'malicious'], 'external')

Bold = SFTold('B', ['normal', 'infected'], ['C', 'D'],
       {'C': np.array([[1, 0], [1, .1]]),
        'D' : np.array([[1, 0], [1, .1]])},
       ['clean', 'malicious'], 'internal')

Cold = SFTold('C', ['normal', 'infected'], ['D'],
       {'D': np.array([[1, 0], [1, .2]])},
       ['clean', 'malicious'], 'internal')

Dold = SFTold('D', ['normal', 'infected'], [], {}, [], 'internal')


T=10000
nodes = [Aold, Bold, Cold, Dold]
netold = SFTNetold(nodes)

s0 = {'A': 'infected', 'B': 'normal', 'C': 'normal', 'D': 'normal'}

# A_clean = SFT('A', ['normal', 'infected'], ['B', 'C'],
#       {'B':np.array([[1, 0], [1,  0]]),
#        'C': np.array([[1, 0], [1, 0]])},
#        ['clean', 'malicious'], 'external')


# nodes_clean = [A_clean, B, C, D]
# net_clean = SFTNet(nodes_clean)

net=0
