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

# This setup function is no longer necessary
# I didn't delete the module because we might want
# to pull the plotting stuff from the bottom

 
# def MH_to_infty(net, T, t0, mcmc_samples, data, print_jumps=False):
#     """
#     Does an entire simulation.  I.e. generates data, picks starting params
#     does the mcmc and returns a result

#     Parameters
#     ----------

#     net : SFTNet
#         An SFTNet

#     T : int
#         Length of observation Period

#     t0 : Initial state of the net
#         Dict.  Keys are node names, values are either 'clean' or 'infected'

#     mcmc_samples : int
#         How many mcmc samples to run

#     """
#     # Generate artificial data
#     # lp = -np.inf
#     # This loop picks starting values for the MCMC
#     # It loops though possible values of B and C by 250
#     # and sets D 'close' to the minimum of the two
#     # logn_fact = gen_logn_fact(data)
#     # orderings = gen_orderings(net, t0)
#     # orderings = [order for order in orderings if len(order) ==4]
    
    
#     # mcmc = MCMC_MH(net, data, mcmc_samples, guess_times, T, orderings, nodes_no_change, print_jumps=print_jumps)
    
#     # return mcmc_results

# if __name__ == '__main__':
#     true_times = defaultdict(list)
#     likelihoods = []
#     no_attacker = []
#     for i in range(1):
#         t0 = { 'A' : 'infected', 'B': 'normal', 'C': 'normal', 'D': 'normal'}
#         data = gen_data(T, net, t0)
#         a= MH_to_infty(net, 10000, t0, 60000, data )
#         likelihoods.append(a.calc_log_likelihood())
#         no_attacker.append(a.p_no_attacker)
#         for (key, value) in a.true_times.iteritems():
#             true_times[key].append(value)


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




