import numpy as np
import random
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import networkx as nx
import multiprocessing as mp
import itertools
from sympy import binomial
import os
os.dont_write_bytecode = True

class DiamondNetMCTS(object):
    def __init__(self,pb,pc,pd,qab,qac,qbd,qcd,w,threshold,gamma,default_policy,c):
        self.pb = pb
        self.pc = pc
        self.pd = pd
        self.qab = qab
        self.qac = qac
        self.qbd = qbd
        self.qcd = qcd
        self.w = w
        self.threshold = threshold
        self.dp = default_policy
        self.gamma = gamma
        self.c = c
    def compute_prob(self,kab,kac,kbd,kcd):
        prob_ab = binomial(self.w,kab) * (self.qab**kab) * ((1-self.qab)**(self.w-kab))
        prob_ac = binomial(self.w,kac) * (self.qac**kac) * ((1-self.qac)**(self.w-kac))
        prob_bd = binomial(self.w,kbd) * (self.qbd**kbd) * ((1-self.qbd)**(self.w-kbd))
        prob_cd = binomial(self.w,kcd) * (self.qcd**kcd) * ((1-self.qcd)**(self.w-kcd))
        joint = prob_ab * prob_ac * prob_bd * prob_cd
        return joint
    def draw_s0(self):
        alarm = True
        while alarm == True:
            s0_ab = list(np.random.binomial(1, self.qab, size=self.w))
            s0_ac = list(np.random.binomial(1, self.qac, size=self.w))
            s0_bd = list(np.random.binomial(1, self.qbd, size=self.w))
            s0_cd = list(np.random.binomial(1, self.qcd, size=self.w))
            kab = np.sum(s0_ab)
            kac = np.sum(s0_ac)
            kbd = np.sum(s0_bd)
            kcd = np.sum(s0_cd)
            alarm = (self.compute_prob(kab,kac,kbd,kcd) < self.threshold)
        s0 = dict(ab = s0_ab, ac = s0_ac, bd = s0_bd, cd = s0_cd)
        return s0
    def draw_s_o_r(self,s0,actions,infected_nodes):
        newstate = {}
        newstate['ab'] = s0['ab'][1:]
        newstate['ac'] = s0['ac'][1:]
        newstate['bd'] = s0['bd'][1:]
        newstate['cd'] = s0['cd'][1:]
        bgm_ab = int(random.random() < self.qab)
        bgm_ac = int(random.random() < self.qac)
        bgm_bd = int(random.random() < self.qbd)
        bgm_cd = int(random.random() < self.qcd)
        newstate['ab'].append(bgm_ab + actions['ab'])
        newstate['ac'].append(bgm_ac + actions['ac'])
        newstate['bd'].append(bgm_bd + actions['bd'])
        newstate['cd'].append(bgm_cd + actions['cd'])
        kab = np.sum(newstate['ab'])
        kac = np.sum(newstate['ac'])
        kbd = np.sum(newstate['bd'])
        kcd = np.sum(newstate['cd'])
        if self.compute_prob(kab,kac,kbd,kcd) < self.threshold:
            newstate = 'Alarm'
        else:
            if infected_nodes == ['A']:
                assert actions['bd'] == actions['cd'] == 0
                if random.random() * actions['ab'] > 1 - self.pb:
                    infected_nodes.append('B')
                if random.random() * actions['ac'] > 1 - self.pc:
                    infected_nodes.append('C')
            elif infected_nodes == ['A','B']:
                assert actions['cd'] == 0
                if random.random() * actions['ac'] > 1 - self.pc:
                    infected_nodes.append('C')
                if random.random() * actions['bd'] > 1 - self.pd:
                    newstate = 'Infected'
            elif infected_nodes == ['A','C']:
                assert actions['bd'] == 0
                if random.random() * actions['ab'] > 1 - self.pb:
                    infected_nodes.insert(1,'B')
                if random.random() * actions['cd'] > 1 - self.pd:
                    newstate = 'Infected'
            elif infected_nodes == ['A','B','C']:
                if random.random() * actions['bd'] > 1 - self.pd:
                    newstate = 'Infected'
                if random.random() * actions['cd'] > 1 - self.pd:
                    newstate = 'Infected'
        return newstate,infected_nodes
    def rollout(self,s,h):
        if s == 'Alarm':
            return 0
        if s == 'Infected':
            return 1
        infected_nodes = deepcopy(h['infect'][-1])
        actions = dict(ab = 0, ac = 0, bd = 0, cd = 0)
        if 'B' not in infected_nodes:
            actions['ab'] = int(random.random() < self.dp['ab'])
        if 'C' not in infected_nodes:
            actions['ac'] = int(random.random() < self.dp['ac'])
        if 'B' in infected_nodes:
            actions['bd'] = int(random.random() < self.dp['bd'])
        if 'C' in infected_nodes:
            actions['cd'] = int(random.random() < self.dp['cd'])
        newstate,infc = self.draw_s_o_r(s,actions,infected_nodes)
        h['act'].append(actions)
        h['infect'].append(deepcopy(infc))
        assert len(h['act']) == len(h['infect'])
        return self.gamma * self.rollout(newstate,h)
    def simulate(self,s,history,T={},returnT=False):
        if s == 'Alarm':
            return 0
        if s == 'Infected':
            return 1
        if str(history) not in T:
            T[str(history)] = {}
            infected_nodes = deepcopy(history['infect'][-1])
            if infected_nodes == ['A']:
                T[str(history)]['[]'] = [0,0] # 1st - V; 2nd - N
                T[str(history)]['ab'] = [0,0]
                T[str(history)]['ac'] = [0,0]
                T[str(history)]['ab ac'] = [0,0]
            elif infected_nodes == ['A','B']:
                T[str(history)]['[]'] = [0,0] # 1st - V; 2nd - N
                T[str(history)]['ac'] = [0,0]
                T[str(history)]['bd'] = [0,0]
                T[str(history)]['ac bd'] = [0,0]
            elif infected_nodes == ['A','C']:
                T[str(history)]['[]'] = [0,0] # 1st - V; 2nd - N
                T[str(history)]['ab'] = [0,0]
                T[str(history)]['cd'] = [0,0]
                T[str(history)]['ab cd'] = [0,0]
            elif infected_nodes == ['A','B','C']:
                T[str(history)]['[]'] = [0,0] # 1st - V; 2nd - N
                T[str(history)]['bd'] = [0,0]
                T[str(history)]['cd'] = [0,0]
                T[str(history)]['bd cd'] = [0,0]
            else:
                raise ValueError
            T[str(history)]['num'] = 0 # total count
            if returnT: 
                return self.rollout(s,history),T
            else:
                return self.rollout(s,history)
        this_history = T[str(history)]
        vseq = []
        moves = []
        for (key,val) in this_history.items():
            if key != 'num':
                moves.append(key)
                if val[1] == 0:
                    v = np.inf
                else:
                    assert this_history['num'] > 0
                    v = val[0] + self.c * (np.log(this_history['num']) /float(val[1]))**0.5
                vseq.append(v)
        assert len(moves) == len(vseq)
        maxix = np.argmax(vseq)
        com = moves[maxix]
        actions = dict(ab = 0, ac = 0, bd = 0, cd = 0)
        if 'ab' in com:
            actions['ab'] = 1
        if 'ac' in com:
            actions['ac'] = 1
        if 'bd' in com:
            actions['bd'] = 1
        if 'cd' in com:
            actions['cd'] = 1
        newstate,infected_nodes = self.draw_s_o_r(s,actions,deepcopy(history['infect'][-1]))
        updated_history = deepcopy(history)
        updated_history['act'].append(actions)
        updated_history['infect'].append(deepcopy(infected_nodes))
        R = self.gamma * self.simulate(newstate,updated_history,T)
        # Update:
        T[str(history)][com][1] += 1
        T[str(history)][com][0] += (R - T[str(history)][com][0])/float(T[str(history)][com][1])
        T[str(history)]['num'] += 1
        if returnT:
            return R,T
        else:
            return R
    def search(self,num_samps = 50000):
        history0 = dict(act = [{'ab':0,'ac':0,'bd':0,'cd':0}], infect = [['A']])
        nosend = []
        to_b = []
        to_c = []
        to_bc = []
        for i in xrange(num_samps):
            print i
            s0 = self.draw_s0()
            r,T = self.simulate(s0,deepcopy(history0),returnT = True)
            t0 = T[str(history0)]
            nosend.append(t0['[]'][0])
            to_b.append(t0['ab'][0])
            to_c.append(t0['ac'][0])
            to_bc.append(t0['ab ac'][0])
        return T,nosend,to_b,to_c,to_bc



if __name__ == '__main__':
    default_policy = dict(ab = 0.25, ac = 0.25, bd = 0.15, cd = 0.15)
    test = DiamondNetMCTS(0.05,0.02,0.01,0.5,0.5,0.3,0.3,50,1e-6,1,default_policy,5)
    y = test.search()
    
    plt.plot(y[1],label = 'Send to Neither')
    plt.plot(y[2],label = 'Send to B')
    plt.plot(y[3],label = 'Send to C')
    plt.plot(y[4],label = 'Send to Both')
    plt.ylabel('Expected Utility')
    plt.xlabel('Simulation Number')
    plt.legend()
    plt.show()
