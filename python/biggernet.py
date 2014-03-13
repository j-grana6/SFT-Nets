from sft import *
from sft_net import *
#from roc import *
from uniform_approx import *
import numpy as np



A = SFT('A', ['normal' , 'infected'], ['B', 'F'], 
    {'B': np.array([[.5, 0], [.5,.005]]), 'F' : np.array([[.5, 0], [.5, .005]])},
    ['clean', 'malicious'], 'external')
# Node A sends messages to B and F

B = SFT('B', ['normal' , 'infected'], ['A', 'D', 'F'], 
    {'A': np.array([[.5, 0], [.5,.01]]), 'D': np.array([[.5,0], [.5, .01]]), 'F' : np.array([[.5, 0], [.5, .01]])},
    ['clean', 'malicious'], 'internal')
# B sends messages to A, D and F

C = SFT('C', ['normal' , 'infected'], ['A', 'B', 'F'], 
    {'A': np.array([[.5, 0], [.5,.01]]), 'B': np.array([[.5,0], [.5, .01]]), 'F' : np.array([[.5, 0], [.5, .01]])},
    ['clean', 'malicious'], 'internal')
# C sends messages to A, B and F

D = SFT('D', ['normal' , 'infected'], ['A', 'C', 'F'], 
    {'A': np.array([[.5, 0], [.5,.01]]), 'C': np.array([[.5,0], [.5, .001]]), 'F' : np.array([[.5, 0], [.5, .01]])},
    ['clean', 'malicious'], 'internal')
# D sends nodes to A, C and F

E = SFT('E', ['normal' , 'infected'], ['A', 'B', 'C', 'F'], 
    {'A': np.array([[.5, 0], [.5,.1]]), 'B': np.array([[.5,0], [.5, .1]]), 
     'C': np.array([[.5,0], [.5, .1]]), 'F' : np.array([[.5, 0], [.5, .1]])},
    ['clean', 'malicious'], 'internal')
# E (slowly) sends nodes to A, B, C and F

F = SFT('F', ['normal' , 'infected'], ['A', 'E'], 
    {'A': np.array([[3, 0], [3,.001]]), 'E' : np.array([[3, 0], [3, .001]])},
    ['clean', 'malicious'], 'external')
# F sends nodes to A and E

nodes= [A,B,C,D,E,F]
net = SFTNet(nodes)
