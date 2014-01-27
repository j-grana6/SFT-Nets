"""
The model with Munsky's parameters
"""
from collections import defaultdict
import numpy as np
from sft import *
from sft_net import *
from tools import *
from lhood_comps import *
import os
import copy

#  Create 4 nodes
def go():
    A = SFT('A', ['normal', 'infected'], ['B', 'C'],
          {'B':np.array([[1, 0], [1, 1/10000.]]),
           'C': np.array([[1,0], [1,1/10000.]])},
           ['clean', 'malicious'], 'external')

    B = SFT('B', ['normal', 'infected'], ['C', 'D'],
           {'C': np.array([[1,0], [1,.1]]),
            'D' : np.array([[1,0], [1,.1]])},
           ['clean', 'malicious'], 'internal')

    C = SFT('C', ['normal', 'infected'], ['D'],
           {'D': np.array([[1,0], [1,.1]])},
           ['clean', 'malicious'], 'internal')

    D = SFT('D', ['normal', 'infected'], [], {}, [], 'internal')

    T=10000
    nodes = [A, B, C, D]
    net = SFTNet(nodes)
    # Put the nodes together and create a net
    data = gen_data(10000, net, ['infected', 'normal', 'normal', 'normal'])
    # Generate artificial data
    print data[-1]
    # print the true infection times
    guess_times = {'A': 0, 'B': 5000, 'C': 4000, 'D': 4500}
    lp = -np.inf
    # This loop picks starting values for the MCMC
    # It loops though possible values of B and C by 250
    # and sets D 'close' to the minimum of the two
    logn_fact = gen_logn_fact(data)
    for j in np.arange(1000,2*T,1000):
        for k in np.arange(1000,2*T, 1000):
            s0 =  {'A': 0, 'B': j, 'C': k-1, 'D': min(j,k) + 10}
            newprob = sum(prob_model_given_data(net, data, s0, 10000,
                                                logn_fact))
            if newprob > lp:
                lp = newprob
                guess_times = s0
        print j
    print guess_times
    #monte_carlo_samples = 30000
    mcmc_samples = 80000


    prob_no_attacker = prob_model_no_attacker(net, data, T)
    prob_true_value = prob_model_given_data(net, data, data[-1], T, logn_fact)

    #print 'Probability no attacker is', prob_no_attacker, '/n'
    #print 'Probability at true params is ', prob_true_value


    #smc = MC_int(net, data, monte_carlo_samples,
    #                            guess_times, T)
    # do simple Monte Carlo
    mcmc = MCMC_MH(net, data, mcmc_samples, guess_times, T,
                    uniform=True)
    # Do MCMC
    #nonumcmc = MCMC_MH(net, data, 10000, guess_times, T,
                        #uniform=False)

    # simple_result = Results(smc, data[-1], prob_no_attacker,
    #                        prob_true_value, data)

    mcmc_results = Results(mcmc, data[-1], prob_no_attacker,
                           prob_true_value, data, metropolis = True)
    return mcmc_results
    #idnum = np.str(np.random.random())[:15]
    #simple_result.write_results(idnum)
    #mcmc_results.write_results('MH' + idnum)
    ### Write Results.  Mix this in with the results class

sdsdsd  
if __name__ == '__main__':
    true_times = defaultdict(list)
    likelihoods = []
    no_attacker = []
    for i in range(50):
        a= go()
        likelihoods.append(a.calc_log_likelihood())
        no_attacker.append(a.p_no_attacker)
        for (key, value) in a.true_times.iteritems():
            true_times[key].append(value)


def plot_diffs():
    from matplotlib import pyplot as plt
    plt.rc('text', usetex=True)
    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    ax1.scatter(true_times['B'], np.asarray(likelihoods) - np.asarray(no_attacker))
    ax1.set_xlabel('Node B True Infection Time')
    ax1.set_ylabel('Likelihood Difference')
    ax1.hlines(y=0, xmin=0, xmax =10000, color='red')
    ax1.set_xlim([0,11000])

    ax2.scatter(true_times['C'], np.asarray(likelihoods) - np.asarray(no_attacker))
    ax2.set_xlabel('Node C True Infection Time')
    ax2.set_ylabel('Likelihood Difference')
    ax2.hlines(y=0, xmin=0, xmax= 10000, color='red')
    ax2.set_xlim([0,11000])
    
    ax3.scatter(true_times['D'], np.asarray(likelihoods) - np.asarray(no_attacker))
    ax3.set_xlabel('Node D True Infection Time')
    ax3.set_ylabel('Likelihood Difference')
    ax3.hlines(y=0, xmin=0, xmax=10000, color='red')
    ax3.set_xlim([0,11000])
    fig.suptitle(r'Metropolis Hastings over [0, \infty]')
    plt.tight_layout()




