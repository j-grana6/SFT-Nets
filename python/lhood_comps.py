import numpy as np
from tools import prob_model_given_data,  \
      rhs_integral, gen_logn_fact, prob_model_no_attacker
import copy
import pandas as pandas
from results import Results
import random
from orderings import gen_orderings

def MCMC_MH(SFTNet, data, s0, N,  T, proposal_var=100, print_jumps=False):
    #  TODO Need to profile this
    #  TODO: Need to make this more general.  Not trivial
    #  TODO : Add sample from possible node orderings
    
    """
    Performs MCMC integration using Metropolis Hastings.  Returns
    the sampled times, their associated probabilities and the
    associated likelihood value.  This method corresponds to David's
    "half-way" approach in the 2nd version of the ASCII where we
    sample (accept / reject) according to P(z | attacker) and then
    take the average of P(data | z, attacker) of the accepted
    values.


    SFTNet : SFTNet instance
        The net to do MCMC over

    data : list
        The data as outputted by gen_data

    N : int
        The number of MCMC proposals

    s0 : dict
        State of the net at t=0

    T : int
        How long the process ran for.
    """
    logn_fact = gen_logn_fact(data)
    n = 1
    nodes_to_change = [nd for nd in SFTNet.node_names if s0[nd] == 'normal' ]
    nodes_no_change = [nd for nd in SFTNet.node_names if s0[nd] == 'infected']
    prob_no_attacker = prob_model_no_attacker(SFTNet, data, T)
    prob_true_value = prob_model_given_data(SFTNet, data, data[-1], T, logn_fact)

    prob_mod = lambda x : prob_model_given_data(SFTNet, data, x, T,
                                                logn_fact)
    guess_times = np.sort(np.random.random(size=len(nodes_to_change))*T)
    z0 = dict(zip(nodes_to_change, guess_times))
    for nd in nodes_no_change:
        z0[nd] = 0
    # lambda function that calls prob_model_given_data for
    # specified infection times
    p0 = prob_mod(z0)
    # Initiial probability
    # actual times
    time_samples = {node.name : [] for node in SFTNet.nodes}
    # container for samples
    probs = []
    # container for probabilities
    z1 = copy.deepcopy(z0)
    while n < N:
        #if np.random.random() < alpha:
        #    order = random.sample(orderings, 1)[0]
        for nd in nodes_to_change:
            z1[nd] = z0[nd] + np.random.normal() * proposal_var
        p1 = prob_mod(z1)
        if (p1[0] - p0[0] > np.log(np.random.random())):
            if print_jumps :
                print 'A Jump at, ', n, 'to ', z1, 'with prob', p1, '\n'
            p0 = p1
            z0 = copy.deepcopy(z1)
        for key, val in z0.iteritems():
            time_samples[key].append(val)
        probs.append(p0[:2])
        n += 1
    probs = np.asarray(probs)
    out_ar = np.hstack((np.asarray(time_samples.values()).T, probs))
    columns = copy.copy(time_samples.keys())
    columns.append('P(z | attacker)')
    columns.append('P(data | z, attacker)')
    out = pandas.DataFrame(out_ar, columns = columns)
    mcmc_results = Results(out, data[-1], prob_no_attacker,
                           prob_true_value, data, metropolis = True)
    return mcmc_results


# def MC_int(SFTNet, data, N, z0, T):
#     """
#     Performs Simple Monte Carlo integration over z^n.  This method
#     corresponds to David's point 9) in the ASCII


#     SFTNet : SFTNet instance
#         The net to do MCMC over

#     data : list
#         The data as outputted by gen_data

#     N : int
#         The number of MCMC proposals

#     z0 : dict
#         Initial guess for infection times.  Keys are node names
#         values are floats

#     T : int
#         How long the process ran for.
#     """
#     logn_fact = gen_logn_fact(data)
#     n = 1
#     # initiate step
#     prob_mod = lambda x : prob_model_given_data(SFTNet, data, x, T,
#                                                 logn_fact)
#     # lambda function that calls prob_model_given_data for
#     # specified infection times
#     p0 = prob_mod(z0)
#     # Initiial probability
#     time_samples = {node.name : [] for node in SFTNet.nodes}
#     # container for samples
#     probs = []
#     lower_bound =rhs_integral(SFTNet, data, T)
#     num_internal = len(SFTNet.internals)
#     V = T ** num_internal
#     while n < N:
#         z1 = {nd:  np.random.random() * T
#                   for nd in z0}
#         z1['A'] = 0
#         p1 = prob_mod(z1)
#         if min(z1.values()) >=  0 and max(z1.values()) <=T:
#             if p1[0] > - np.inf:
#                 p0 = p1
#                 z0 = z1
#                 for key, val in z0.iteritems():
#                     time_samples[key].append(val)
#                 probs.append(p0)
#                 n += 1
#                 if n/500*500 == n:
#                     print n
#     probs = np.asarray(probs)
#     out_ar = np.hstack((np.asarray(time_samples.values()).T, probs))
#     columns = copy.copy(time_samples.keys())
#     columns.append('P(z | attacker)')
#     columns.append('P(data | z, attacker)')
#     out = pandas.DataFrame(out_ar, columns = columns)
#     return out,  lower_bound, V
