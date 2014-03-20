import numpy as np
from tools import prob_model_given_data,  \
      rhs_integral, gen_logn_fact, prob_model_no_attacker
import copy
import pandas as pandas
from results import Results
import random
from orderings import gen_orderings
import operator
from tools import trunc_expon

def MCMC_sequence(SFTNet, data, s0, N,  T, proposal_var=100, print_jumps=False, alpha=1):
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
    numattackers = len(nodes_no_change)

    prob_mod = lambda x : prob_model_given_data(SFTNet, data, x, T,
                                                logn_fact)
    guess_times = np.sort(np.random.random(size=len(nodes_to_change))*T)
    z0 = dict(zip(nodes_to_change, guess_times))
    for nd in nodes_no_change:
        z0[nd] = 0
    order = sorted(z0.iterkeys(), key = lambda k: z0[k])
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
    orders = gen_orderings(SFTNet, s0)
    state0 = ['infected'] * numattackers + ['normal'] * len(nodes_to_change)
    while n < N:
        z1 = dict(zip(nodes_no_change, [0] * numattackers))
        last_infect = 0
        state = copy.copy(state0)
        if np.random.random() < alpha:
            new_order  = random.choice(orders)
            switch_order = True
        else :
            switch_order = False
            new_order = order
        for nd in new_order[numattackers:]:
            cross_s_ix = SFTNet.cross_S.index(state)
            nd_ix = SFTNet.node_names.index(nd)
            incoming_rate = np.sum(SFTNet.mal_trans_mats[cross_s_ix][:, nd_ix])
            last_infect = last_infect   + trunc_expon(incoming_rate, T-last_infect)
            z1[nd] = last_infect
            state[nd_ix] = 'infected'
        p1 = prob_mod(z1)
        if (p1[2]  -p0[2] > np.log(np.random.random())):
            if print_jumps :
                print 'A Jump at, ', n, 'to ', z1, 'with prob', p1, '\n'
            if switch_order:
                #print ' new order ', order, ' at ', n
                p0 = p1
                z0 = copy.deepcopy(z1)
                order = new_order
        for key, val in z0.iteritems():
            time_samples[key].append(val)
        for nd in nodes_to_change:
            if nd not in z0.keys():
                time_samples[nd].append(T)
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
