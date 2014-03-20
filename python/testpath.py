import copy as copy
def find_all_paths(graph, start,end, path=[], at_risk = []):
    path = path + [start]
    
    at_risk1 = copy.copy(at_risk)
    at_risk1.extend(graph[start])
    at_risk1 = list(set(at_risk1))
    if start==end:
        return [path]
    paths = []
    for node in at_risk1:
        if node not in path:
            newpaths = find_all_paths(graph, node, end,
                                      path=path
                                      , at_risk = at_risk1)
            for i  in range(len(newpaths)):
                paths.append(newpaths[i])
    return paths

g= {'A': ['B', 'C'],
    'B': [],
    'C': [],
    'D': []}
all = []

for nd in g.keys():
    all.extend(find_all_paths(g, 'A', nd))

from sft import *
from sft_net import *

A = SFT('A', ['normal' , 'infected'], ['B', 'F'], 
    {'B': np.array([[.5, 0], [.5,.0001]]), 'F' : np.array([[.5, 0], [.5, .0001]])},
    ['clean', 'malicious'], 'external')
# Node A sends messages to B and F

B = SFT('B', ['normal' , 'infected'], [ 'E'], 
    {'E' : np.array([[.5 , 0], [.5, .001]])},
    ['clean', 'malicious'], 'internal')
# B sends messages to A, D and F
C = SFT('C', ['normal' , 'infected'], ['E'],
    {'E': np.array([[.5, 0], [.5,.001]])},
    ['clean', 'malicious'], 'internal')
# C sends messages to A, B and F


D = SFT('D', ['normal' , 'infected'], ['E'], 
    {'E': np.array([[.5, 0], [.5,.001]])},
    ['clean', 'malicious'], 'internal')
# D sends nodes to A, C and F

E = SFT('E', ['normal' , 'infected'], ['A', 'B', 'C', 'D', 'F'], 
    {'A': np.array([[.5, 0], [.5,.001]]), 'B': np.array([[.5,0], [.5, .001]]), 
     'C': np.array([[.5,0], [.5, .001]]), 'D' : np.array([[.5, 0], [.5, .001]]), 'F' : np.array([[.5, 0], [.5, .001]])},
    ['clean', 'malicious'], 'internal')
# E (slowly) sends nodes to A, B, C and F

F = SFT('F', ['normal' , 'infected'], ['A1', 'E'], 
    {'A1': np.array([[.5, 0], [.5,.001]]), 'E' : np.array([[.5, 0], [.5, .001]])},
    ['clean', 'malicious'], 'external')
# F sends nodes to A and E

A1 = SFT('A1', ['normal' , 'infected'], ['B1', 'F1'], 
    {'B1': np.array([[.5, 0], [.5,.001]]), 'F1' : np.array([[.5, 0], [.5, .001]])},
    ['clean', 'malicious'], 'external')
# Node A sends messages to B and F

B1 = SFT('B1', ['normal' , 'infected'], [ 'E1'], 
    {'E1' : np.array([[.5, 0], [.5, .001]])},
    ['clean', 'malicious'], 'internal')
# B sends messages to A, D and F
C1 = SFT('C1', ['normal' , 'infected'], ['E1'],
    {'E1': np.array([[.5, 0], [.5,.001]])},
    ['clean', 'malicious'], 'internal')
# C sends messages to A, B and F


D1 = SFT('D1', ['normal' , 'infected'], ['E1'], 
    {'E1': np.array([[.5, 0], [.5,.001]])},
    ['clean', 'malicious'], 'internal')
# D sends nodes to A, C and F

E1 = SFT('E1', ['normal' , 'infected'], ['A1', 'B1', 'C1', 'D1', 'F1'], 
    {'A1': np.array([[.5, 0], [.5,.001]]), 'B1': np.array([[.5,0], [.5, .001]]), 
     'C1': np.array([[.5,0], [.5, .001]]), 'D1' : np.array([[.5, 0], [.5, .001]]), 'F1' : np.array([[.5, 0], [.5, .001]])},
    ['clean', 'malicious'], 'internal')
# E (slowly) sends nodes to A, B, C and F

F1 = SFT('F1', ['normal' , 'infected'], ['A1', 'E1'], 
    {'A1': np.array([[.5, 0], [.5,.001]]), 'E1' : np.array([[.5, 0], [.5, .001]])},
    ['clean', 'malicious'], 'external')
# F sends nodes to A and E

nodes = [A, B,C,D,E,F, A1, B1,C1,D1,E1,F1]
net = SFTNet(nodes)
net.node_names

g = {}
for key, val in net.node_dict.iteritems():
    g[key] = net.node_dict[key].sends_to

all2 = []
for nd in g.keys():
    all2.extend(find_all_paths(g, 'A', nd))
