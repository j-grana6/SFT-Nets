import numpy as np
import random
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import networkx as nx
import multiprocessing as mp
from itertools import chain, combinations
import multiprocessing as mp
import os
os.dont_write_bytecode = True


def powerset(iterable):
    """Returns the powerset of the iterable list.
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class NNodeMCTS(object):
    def __init__(self,net,params,default_policy,reward_function,alarm_function):
        self.net = net # sft net
        self.w = params['w'] # observation window length
        self.threshold = params['threshold'] # alarm threshold
        self.gamma = params['gamma'] # discount rate
        self.c = params['c'] # MCTS regret parameter
        self.samplesize = params['samplesize']
        self.dp = default_policy # attacker's default policy
        self.rf = reward_function # reward function maps state to reward
        self.af = alarm_function # whether the alarm sounds
    def draw_s0(self,s0=None):
        """Returns s0. Draw initial states.
        """
        nodes = self.net.nodes.values()
        allrates = {}
        for sender in nodes:
            for r in sender.sends_to:
                k = sender.name + r
                allrates[k] = sender.backgroundrates[r]    
        if s0 is not None:
            assert not self.af(s0,self.threshold,allrates), "Alarms sounds on initial states!"
        else:
            alarm = True
            while alarm == True:
                s0 = {}
                for nd in nodes:
                    if len(nd.sends_to) > 0:
                        for c in nd.sends_to:
                            key = nd.name + c
                            s0[key] = list(np.random.binomial(1,nd.backgroundrates[c],
                                size = self.w))
                alarm = self.af(s0,self.threshold,allrates)
        return s0
    def draw_s_o_r(self,current_state,actions):
        """Returns new state (alarm, infection, or neither). Draw state, observation and reward.
        """
        nodes = self.net.nodes
        allrates = {}
        for sender in nodes.values():
            for r in sender.sends_to:
                k = sender.name + r
                allrates[k] = sender.backgroundrates[r]
        newstate = {}
        for key in current_state.keys():
            newstate[key] = current_state[key][1:] 
        newly_infected = [] 
        for nd in nodes.values():
            #if nd.state == 'Infected':
             #   nd.infect_duration += 1 #chkchk
            for c in nd.sends_to:
                cnd = nodes[c]
                k = nd.name + c
                if nd.state == 'Infected' and cnd.state == 'Normal':
                    #assert cnd.infect_duration == 0
                    act = actions[k]
                    if np.random.random() * act > 1 - cnd.infectionrate:
                        newly_infected.append(cnd)
                else:
                    act = 0
                backgroundmessage = int(np.random.random() < nd.backgroundrates[c])
                newstate[k].append(backgroundmessage + act)
        for nd in newly_infected:
            nd.state = 'Infected'
            #nd.infect_duration = 1
        alarm = self.af(newstate,self.threshold,allrates)
        outcome,reward = self.rf(self.net,alarm)
        if outcome is not None:
            newstate = outcome
        return newstate,reward
    def gen_action_space(self,init=False):
        """Returns action space. Dictionary - key: sender name + receiver name, 
        value: 0/1 (init == False) or None (init == True)
        """
        actions = {}
        for sender in self.net.nodes.values():
            if sender.state == 'Infected' and len(sender.sends_to) > 0:
                for r in sender.sends_to:
                    receiver = self.net.nodes[r]
                    if receiver.state == 'Normal':
                        k = sender.name + receiver.name
                        if init == False:
                            actions[k] = int(np.random.random() < self.dp[k])
                        else:
                            actions[k] = None
        return actions
    def gen_action_combinations(self,action_space):
        """Return all action combinations. 
        """
        all_actions = action_space.keys()
        PSet = powerset(all_actions)
        all_combinations = []
        for ps in PSet: 
            if ps == ():
                all_combinations.append(" ")
            else:
                g = ""
                for p in sorted(ps): #chkchk
                    g = g + p + "_"
                all_combinations.append(g)
        return tuple(sorted(all_combinations)) #chkchk
    def rollout(self,s,reward,h):
        """Returns outcome (alarm or infected) and total reward.
        """
        if s == 'Alarm':
            return reward
        if s == 'Succeed':
            return reward
        actions = self.gen_action_space()
        actions_copy = deepcopy(actions) # copy for safe
        infected_nodes = tuple(sorted([nd.name for nd in self.net.nodes.values() if nd.state == 'Infected']))
        infected_nodes_copy = deepcopy(infected_nodes) # copy for safe
        newstate,reward = self.draw_s_o_r(s,actions) # draw (s,o,r)
        h['act'].append(actions_copy)
        h['infectednodes'].append(infected_nodes_copy)
        return self.gamma * self.rollout(newstate,reward,h)
    def simulate(self,s,history,T = {},reward = None, returnT = False):
        """Returns the reward.
        """
        nodes = self.net.nodes.values()
        # Reset state
        for nd in nodes:
            if nd.name in history['infectednodes'][-1]:
                nd.state = 'Infected'
            else:
                nd.state = 'Normal'
                nd.duration = 0
        if s == 'Alarm':
            return reward
        if s == 'Succeed':
            return reward
        if str(history) not in T:
            T[str(history)] = {}
            action_space = self.gen_action_space(init=True)
            all_combinations = self.gen_action_combinations(action_space)
            """
            if history['infectednodes'] == [('A',), ('A',), ('A',)]:
                print 'doam'
                print self.net.nodes['B'].state
                print self.net.nodes['C'].state
                print action_space
                print all_combinations
            """
            for ac in all_combinations:
                T[str(history)][ac] = [0,0]
            T[str(history)]['num'] = 0 # total count (# of times the history is visited)
            if returnT:
                return self.rollout(s,reward,history),T
            else:
                return self.rollout(s,reward,history)
        this_history = T[str(history)]
        vseq = []
        moves = []
        for (key,val) in this_history.items():
            if key != 'num': 
                moves.append(key)
                """
                if val[1] == 0: # if this history-action pair not visited before
                    v = np.inf # set to be an infinite large number to ensure exploration over exploitation
                else:
                    assert this_history['num'] > 0
                    v = val[0] + self.c * (np.log(this_history['num'])/float(val[1]))**0.5
                vseq.append(v)
                """
                v = val[0] + self.c * (this_history['num'] / float(val[1] + 1))**0.5
                vseq.append(v)
        maxix = np.argmax(vseq)
        com = moves[maxix]
        actions = self.gen_action_space(init=True)
        all_combos = self.gen_action_combinations(actions)
        forcompare = deepcopy(this_history) # for check purpose L1
        del forcompare['num'] # for check purpose L2
        """
        if sorted(all_combos) != sorted(forcompare.keys()):
            print history['infectednodes']
            print history['act']
            print this_history
            print 'B',self.net.nodes['B'].state
            print 'C',self.net.nodes['C'].state
            print 'a',all_combos
            print 'b',forcompare.keys()
        """
        assert sorted(all_combos) == sorted(forcompare.keys()) # for check purpose L3
        for key in actions.keys():
            if key + "_" in com:
                actions[key] = 1
            else:
                actions[key] = 0
        updated_history = deepcopy(history)
        updated_history['act'].append(actions)
        infected_nodes = tuple(sorted([nd.name for nd in nodes if nd.state == 'Infected']))
        updated_history['infectednodes'].append(infected_nodes) #chkchk, which first?
        newstate,reward = self.draw_s_o_r(s,actions)
        R = self.gamma * self.simulate(newstate,updated_history,T,reward=reward)
        # Update:
        T[str(history)][com][1] += 1
        T[str(history)][com][0] += (R - T[str(history)][com][0]) /float(T[str(history)][com][1])
        T[str(history)]['num'] += 1
        if returnT:
            return R,T
        else:
            return R
    def search(self,history0=[],s0=None):
        """Returns payoff sequence for each action combination.
        """
        nodes = self.net.nodes.values()
        T = {}
        action_space0 = self.gen_action_space(init=True)
        all_combinations0 = self.gen_action_combinations(action_space0)
        infected_nodes0 = tuple(sorted([nd.name for nd in nodes if nd.state == 'Infected']))
        actions0 = deepcopy(action_space0)
        for k in actions0.keys():
            actions0[k] = 0
        history0 = dict(act = [deepcopy(actions0)],infectednodes = [infected_nodes0])
        empties = [[] for i in range(len(all_combinations0))]
        payoffs = dict(zip(all_combinations0,empties)) # initialize
        for ep in range(self.samplesize):
            print ep
            # Reset state to original
            for nd in nodes:
                nd.state = self.net.states0[nd.name]
                #if nd.state == 'Infected':
                #    nd.infect_duration = 1
                #else:
                #    nd.infect_duration = 0
            s0 = self.draw_s0(s0)
            r,T = self.simulate(s0,deepcopy(history0),T=T,returnT = True)
            t0 = T[str(history0)]
            for ac in all_combinations0:
                payoffs[ac].append(t0[ac][0])
        return payoffs
