"""
The model with Munsky's parameters
"""
from collections import defaultdict
from testing_net import *
import numpy as np
from tools import *
from lhood_comps import *
import os
import copy
import itertools

def go(net, T, t0, mcmc_samples):
    """
    Does an entire simulation.  I.e. generates data, picks starting params
    does the mcmc and returns a result

    Parameters
    ----------

    net : SFTNet
        An SFTNet

    T : int
        Length of observation Period

    t0 : Initial state of the net
        Dict.  Keys are node names, values are either 'clean' or 'infected'

    mcmc_samples : int
        How many mcmc samples to run

    """
    data = gen_data(T, net, t0)
    # Generate artificial data
    # lp = -np.inf
    # This loop picks starting values for the MCMC
    # It loops though possible values of B and C by 250
    # and sets D 'close' to the minimum of the two
    logn_fact = gen_logn_fact(data)
    orderings = gen_orderings(net, t0)
    orderings = [order for order in orderings if len(order) ==4]
    lp = -np.inf
    nodes_to_change = [nd for nd in net.node_names if t0[nd] == 'normal']
    nodes_no_change = [nd for nd in net.node_names if t0[nd] == 'infected']
    no_change_dict = dict(zip(nodes_no_change, [0]*len(nodes_no_change)))
    start_loop = itertools.permutations(range(1, 2*T, 1000), len(nodes_to_change))
    while True:
        try :
            totry = start_loop.next()
            totry = list(np.asarray(totry) + np.random.random(size=len(totry)))
            s0 =  dict(zip(nodes_to_change, totry))
            s0.update(no_change_dict)
            newprob = sum(prob_model_given_data(net, data, s0, 10000, logn_fact))
            if newprob > lp:
                lp = newprob
                guess_times = s0
        except StopIteration:
            break
    #monte_carlo_samples = 30000
    prob_no_attacker = prob_model_no_attacker(net, data, T)
    prob_true_value = prob_model_given_data(net, data, data[-1], T, logn_fact)
    mcmc = MCMC_MH(net, data, mcmc_samples, guess_times, T, orderings, nodes_no_change)

    mcmc_results = Results(mcmc, data[-1], prob_no_attacker,
                           prob_true_value, data, metropolis = True)
    return mcmc_results

if __name__ == '__main__':
    true_times = defaultdict(list)
    likelihoods = []
    no_attacker = []
    for i in range(1):
        a= go(net, 10000, { 'A' : 'infected', 'B': 'normal', 'C': 'normal', 'D': 'normal'}, 60000 )
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




