import numpy as np
import os
os.dont_write_bytecode = True
import csv
from sympy import binomial

from sft4mcts import SFT4mcts, SFTnet4mcts
from GeneralizedDiscreteMCTS import *

def input_to_csv(filename,payoffs):
    with open(filename + '.csv','wb') as fp:
        csvout = csv.writer(fp)
        csvout.writerow(sorted(payoffs.keys()))
        for l in range(len(payoffs[' '])):
            the_row = []
            for k in sorted(payoffs.keys()):
                the_row.append(payoffs[k][l])
            csvout.writerow(the_row)

def test_2nodes():
    A = SFT4mcts('A','Infected',{'B':0.5},p=0.02,duration=1)
    B = SFT4mcts('B','Normal',{},p=0.01)
    simplenet = SFTnet4mcts('Simple 2-node Net',(A,B))
    default_policy = {'AB':0.5}
    def reward_function(net,alaram):
        nodeB = net.nodes['B']
        if alaram == True:
            return 'Alarm',0
        else:
            if nodeB.state == 'Infected':
                return 'Succeed',1
            else:
                return None,None
    def alarm_function(seq,threshold,backgroundrates):
        flowB = seq.values()[0]
        if np.sum(flowB) > threshold:
            return True
        else:
            return False
    params = dict(gamma=1,w=50,c=5,threshold=45,samplesize=5000)
    TwoNodeMCTS = NNodeMCTS(simplenet,params,default_policy,reward_function,alarm_function)
    payoffs = TwoNodeMCTS.search()
    input_to_csv('2nodeMCTSexamples',payoffs)

def test_diamond_net():
    A = SFT4mcts('A','Infected',{'B':0.5,'C':0.5},p=0.5,duration=1)
    B = SFT4mcts('B','Normal',{'D':0.3},p=0.3)
    C = SFT4mcts('C','Normal',{'D':0.3},p=0.03)
    D = SFT4mcts('D','Normal',{},p=0.05)
    diamondnet = SFTnet4mcts('Diamond Net', (A,B,C,D))
    default_policy = {'AB':0.5,'AC':0.5,'BD':0.5,'CD':0.5}
    def reward_function(net,alarm):
        nodeD = net.nodes['D']
        if alarm == True:
            return 'Alarm',0
        else:
            if nodeD.state == 'Infected':
                return 'Succeed',1
            else:
                return None,None
    def alarm_function(seq,threshold,backgroundrates):
        probs = []
        for s in seq.keys():
            k = np.sum(seq[s])
            w = len(seq[s])
            rate = backgroundrates[s]
            probs.append(binomial(w,k) * (rate**k) * ((1-rate)**(w-k)))
        joint = 1
        for p in probs:
            joint *= p
        if np.log(float(joint)) < threshold:
            return True
        else:
            return False
    params = dict(gamma=1,w=50,c=5,threshold=-25,samplesize=20000)
    DiamondMCTS = NNodeMCTS(diamondnet,params,default_policy,reward_function,alarm_function)
    payoffs = DiamondMCTS.search()
    input_to_csv('diamondMCTSexamples',payoffs)

def test_8node_net():
    A = SFT4mcts('A','Infected',{'B':0.5, 'C':0.5, 'D':0.5},p=0.5,duration=1)
    B = SFT4mcts('B','Normal',{'E':0.5},p=0.1)
    C = SFT4mcts('C','Normal',{'D':0.8,'E':0.8,'F':0.8,'H':0.8},p=0.1)
    D = SFT4mcts('D','Normal',{'F':0.5},p=0.1)
    E = SFT4mcts('E','Normal',{'G':0.5},p=0.1)
    F = SFT4mcts('F','Normal',{'H':0.5},p=0.1)
    G = SFT4mcts('G','Normal',{},p=0.1)
    H = SFT4mcts('H','Normal',{},p=0.1)
    eightnodenet = SFTnet4mcts('8-Node Net', (A,B,C,D,E,F,G,H))
    default_policy = {'AB':0.5,'AC':0.5,'AD':0.5,'BE':0.5,'CD':0.5,'CE':0.5,'CF':0.5,'CH':0.5,'DF':0.5,
            'EG':0.5,'FH':0.5}
    def reward_function(net,alarm):
        nodeG = net.nodes['G']
        nodeH = net.nodes['H']
        if alarm == True:
            return 'Alarm',0
        else:
            if nodeG.state == 'Infected' or nodeH.state == 'Infected':
                return 'Succeed',1
            else:
                return None, None
    def alarm_function(seq,threshold,backgroundrates):
        probs = []
        for s in seq.keys():
            k = np.sum(seq[s])
            w = len(seq[s])
            rate = backgroundrates[s]
            probs.append(binomial(w,k) * (rate**k) * ((1-rate)**(w-k)))
        joint = 1
        for p in probs:
            joint *= p
        if np.log(float(joint)) < threshold:
            return True
        else:
            return False
    params = dict(gamma=.95,w=50,c=5,threshold=-50,samplesize=15000)
    EightNodeMCTS = NNodeMCTS(eightnodenet,params,default_policy,reward_function,alarm_function)
    payoffs = EightNodeMCTS.search()
    input_to_csv('EightNodeMCTSexamples',payoffs)

if __name__ == '__main__':
    test_8node_net()
