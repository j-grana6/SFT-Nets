import numpy as np
import random
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import networkx as nx
import multiprocessing as mp
import itertools
import os
os.dont_write_bytecode = True

class NNodesMCTS(object):
    def __init__(self,nodes,node_s0,params,default_policy,reward_function):
        self.nodes = nodes # nodes
        self.node_s0 = node_s0 # initial states of nodes
        self.params = params # parameters
        self.graph = self.create_network() # network
        self.dp = default_policy # attacker's default policy
        self.rf = reward_function # reward function maps state to reward
    def create_network(self,draw=False):
        """Returns directed network.
        """
        nodes,node_s0 = self.nodes,self.node_s0
        G = nx.DiGraph()
        G.add_nodes_from(node_s0.keys())
        edges = []
        for nd in nodes.values():
            for outnode in nd.sends_to:
                edges.append((nd.name,outnode))
        G.add_edges_from(edges)
        if draw == True:
            colors = []
            nds = []
            for (key,val) in node_s0.iteritems():
                nds.append(key)
                if val == 'Infected':
                    colors.append('r')
                elif val == 'Normal':
                    colors.append('g')
            assert len(nds) == len(colors)
            nx.draw(G, nodelist = nds, node_color = colors)
        return G
    def draw_s0(self,s0=None):
        """Returns s0. Draw initial states.
        """
        params,nodes = self.params,self.nodes
        alarm = True
        while alarm == True:
            s0 = {}
            for nd in nodes.values():
                if nd.endnode == False:
                    for out in nd.sends_to:
                        key = nd.name + out
                        s0[key] = list(np.random.binomial(1,params['q'][nd.name][out],size = params['w']))
            s,_ = self.rf(s0,nodes,params)
            alarm = (s=='Alarm')
        return s0
    def draw_s_o_r(self,current_state,actions):
        """Returns new state (alarm, infection, or neither). Draw state, observation and reward.
        """
        nodes,params = self.nodes,self.params
        newstate = {}
        for key in current_state.keys():
            newstate[key] = current_state[key][1:] 
        newinfect = []
        for nd in nodes.values():
            #if nd.state == 'Infected': 
             #   nd.infect_duration += 1
            if nd.endnode == False:
                for out in nd.sends_to:
                    outnode = nodes[out]
                    if nd.state == 'Infected':
                        action = actions[nd.name][out]
                        if random.random() * action > 1 - params['p'][out]:
                            newinfect.append(outnode)
                    else:
                        action = 0
                    backgroundmessage = int(random.random() < params['q'][nd.name][out])
                    newstate[nd.name + out].append(backgroundmessage + action)
        assert len(newinfect) == len(set(newinfect))
        for new in newinfect:
            new.state = 'Infected'
        newstate,reward = self.rf(newstate,nodes,params)
        return newstate,reward
    def rollout(self,s,reward,h):
        """Returns outcome (alarm or infected) and total reward.
        """
        nodes = self.nodes
        if s == 'Alarm':
            return reward
        if s == 'Infected':
            return reward
        actions = {}
        for nd in nodes.values():
            if nd.state == 'Infected':
                actions[nd.name] = {}
                for out in nd.sends_to:
                    actions[nd.name][out] = int(random.random() < self.dp[nd.name][out])
        newstate,reward = self.draw_s_o_r(s,actions)
        h.append(actions)
        return self.params['gamma'] * self.rollout(newstate,reward,h)
    def simulate(self,s,history,T = {},reward = None, returnT = False):
        """Returns the reward.
        """
        nodes,params = self.nodes,self.params
        if s == 'Alarm':
            return reward
        if s == 'Infected':
            return reward
        if str(history) not in T:
            T[str(history)] = {}
            act_com = []
            ndnames = []
            for nd in nodes.values():
                if nd.state == 'Infected':
                    ndnames.append(nd.name)
                    nout = len(nd.sends_to)
                    all_moves = [[0,1] for i in range(nout)]
                    combos = list(itertools.product(*all_moves))
                    act_com.append(combos)
            all_actions = list(itertools.product(*act_com))
            T[str(history)]['for_searching'] = {}
            for i in range(len(all_actions)):
                all_actions[i] = dict(zip(ndnames,all_actions))
                T[str(history)][str(all_actions[i])] = [0,0] # 1st - V; 2nd - N
                T[str(history)]['for_searching'][str(all_actions[i])] = all_actions[i]
            T[str(history)]['num'] = 0 # total count (# of times the history is visited)
            if returnT:
                return self.rollout(s,reward,history),T
            else:
                return self.rollout(s,reward,history)
        this_history = T[str(history)]
        values = []
        moves = []
        for (key,val) in this_history.items():
            if key != 'num'  and key != 'for_searching': 
                if val[1] == 0: # if this history-action pair not visited before
                    v = np.inf # set to be an infinite large number to ensure exploration over exploitation
                else:
                    v = val[0] + params['c']*(np.log(this_history['num'])/float(val[1]))**0.5
                values.append(v)
                moves.append(key)
        maxix = np.argmax(values)
        kk = moves[maxix]
        actions = this_history['for_searching'][kk]
        updated_history = copy(history)
        updated_history.append(actions)
        newstate,reward = self.draw_s_o_r(s,actions)
        R = params['gamma'] * self.simulate(newstate,updated_history,T,reward=reward)
        # Update:
        T[str(history)][str(actions)][1] += 1
        T[str(history)][str(actions)][0] += (R - T[str(history)][str(actions)][0]) \
                /float(T[str(history)][str(actions)][1])
        if returnT:
            return R,T
        else:
            return R
    def search(self,history0=[],s0=None):
        """Returns
        """
        nodes,params = self.nodes,self.params
        num_samps = params['N']
        T = {}
        util = {}
        act_com = []
        ndnames = []
        for nd in nodes.values():
            if nd.state == 'Infected':
                ndnames.append(nd.name)
                nout = len(nd.sends_to)
                all_moves = [[0,1] for i in range(nout)]
                combos = list(itertools.product(*all_moves))
                act_com.append(combos)
        all_actions = list(itertools.product(*act_com))
        for i in range(len(all_actions)):
            all_actions[i] = dict(zip(ndnames,all_actions))
        for act in all_actions:
            util[str(act)] = []
        for i in xrange(num_samps):
            print i
            s0 = self.draw_s0(s0)
            r,T = self.simulate(s0,list(history0),T,returnT = True)
            t0 = T[str(list(history0))]
            for act in all_actions:
                util[str(act)].append(t0[str(act)])
        return r,T,util


if __name__ == '__main__':

    class Node(object):
        def __init__(self,name,sends_to,endnode,state):
            self.name = name
            self.sends_to = sends_to
            self.endnode = endnode
            self.state = state
    
    A = Node('A',['B','C'],False,'Infected')
    B = Node('B',['D'],False,'Normal')
    C = Node('C',['D'],False,'Normal')
    D = Node('D',[],True,'Normal')
    nodes = {A.name : A, B.name : B, C.name : C, D.name : D}
    node_s0 = {A.name:A.state, B.name:B.state, C.name:C.state, D.name:D.state}

    params = dict(gamma = 1,
                  w = 80,
                  p = {'B':0.1, 'C':0.1, 'D':0.05},
                  q = dict(A = {'B':0.5, 'C':0.5},
                           B = {'D':0.2},
                           C = {'D':0.2}
                            ),
                  N = 5000
                  )

    default_policy = dict(A = {'B':0.3, 'C':0.3},
                          B = {'D':0.1},
                          C = {'D':0.05}
                          )
    
    from sympy import binomial

    def reward_function(newstate,nodes,params):
        w,q = params['w'],params['q']
        reward = None
        probs = []
        for nd in nodes.values():
            if nd.endnode == False:
                for out in nd.sends_to:
                    q1 = q[nd.name][out]
                    key = nd.name + out
                    seq = newstate[key]
                    k = np.sum(seq)
                    probs.append(binomial(w,k) * (q1**k) * ((1-q1)**(w-k)))
        lhood = np.prod(probs)
        if lhood < 1e-6:
            newstate = 'Alarm'
            reward = 0
        else:
            for nd in nodes.values():
                if nd.endnode == True and nd.state == 'Infected':
                    newstate = 'Infected'
                    reward = 1
                    break
        return newstate,reward

    diamondNet = NNodesMCTS(nodes,node_s0,params,default_policy,reward_function)
    sr = diamondNet.search()
