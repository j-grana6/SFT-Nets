import numpy as np
from tools import prob_model_given_data, convoluted_cdf_func, \
      convoluted_pdf_func, qij_over_qji, rhs_integral, gen_logn_fact
import copy
import pandas as pandas
from results import Results


def MCMC_MH(SFTNet, data, N, z0, T, uniform = True):
    #  TODO Need to profile this
    #  TODO: Need to make this more general.  Not trivial
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

    z0 : dict
        Initial guess for infection times.  Keys are node names
        values are floats

    T : int
        How long the process ran for.
    """
    logn_fact = gen_logn_fact(data)
    n = 1
    # initiate step
    prob_mod = lambda x : prob_model_given_data(SFTNet, data, x, T,
                                                logn_fact)
    # lambda function that calls prob_model_given_data for
    # specified infection times
    p0 = prob_mod(z0)
    # Initiial probability
    # actual times
    time_samples = {node.name : [] for node in SFTNet.nodes}
    # container for samples
    probs = []
    # container for probabilities
    lower_bound =rhs_integral(SFTNet, data, T)
    num_internal = len(SFTNet.internals)
    V = T ** num_internal
    if uniform :
        while n < N:
            z1 = {nd: z0[nd] + np.random.normal() * 100
                  for nd in z0}
            z1['A'] = 0
            p1 = prob_mod(z1)
            if min(z1.values()) >=  0 and max(z1.values()) <=T:
                if (p1[0] - p0[0] >
                    np.log(np.random.random())):
                    print 'A Jump at, ', n, 'to ', z1, 'with prob', p1, '\n'
                    p0 = p1
                    z0 = z1
                for key, val in z0.iteritems():
                    time_samples[key].append(val)
                probs.append(p0)
                n += 1
    else:
        con_cdf = convoluted_cdf_func(20000 **.5, 0, 50)
        while n < N:
            za = 0
            zb = z0['B'] + np.random.normal() *  100
            zc = min(z0['C'] + np.random.normal() *  100,
                     zb + np.random.random()* 50)
            zd = min(zb, zc) + np.random.random() * 50
            z1 = dict(zip(['A', 'B', 'C', 'D'], [za, zb,zc, zd]))
            p1 = prob_mod(z1)
            if min(z1.values()) >=  0 and max(z1.values()) < T:
                log_q_ratio = qij_over_qji(z0,z1, con_cdf, convoluted_pdf_func)
                if (p1[0] - p0[0]  + log_q_ratio >
                    np.log(np.random.random())):
                    print 'A Jump at, ', n, 'to ', z1, 'with prob', p1, '\n'
                    p0 = p1
                    z0 = z1
                for i in z0.keys():
                    time_samples[i].append(z0[i])
                probs.append(p0)
                n += 1
    probs = np.asarray(probs)
    out_ar = np.hstack((np.asarray(time_samples.values()).T, probs))
    columns = copy.copy(time_samples.keys())
    columns.append('P(z | attacker)')
    columns.append('P(data | z, attacker)')
    out = pandas.DataFrame(out_ar, columns = columns)
    return out, lower_bound, None


def MC_int(SFTNet, data, N, z0, T):
    """
    Performs Simple Monte Carlo integration over z^n.  This method
    corresponds to David's point 9) in the ASCII


    SFTNet : SFTNet instance
        The net to do MCMC over

    data : list
        The data as outputted by gen_data

    N : int
        The number of MCMC proposals

    z0 : dict
        Initial guess for infection times.  Keys are node names
        values are floats

    T : int
        How long the process ran for.
    """
    logn_fact = gen_logn_fact(data)
    n = 1
    # initiate step
    prob_mod = lambda x : prob_model_given_data(SFTNet, data, x, T,
                                                logn_fact)
    # lambda function that calls prob_model_given_data for
    # specified infection times
    p0 = prob_mod(z0)
    # Initiial probability
    time_samples = {node.name : [] for node in SFTNet.nodes}
    # container for samples
    probs = []
    lower_bound =rhs_integral(SFTNet, data, T)
    num_internal = len(SFTNet.internals)
    V = T ** num_internal
    while n < N:
        z1 = {nd:  np.random.random() * T
                  for nd in z0}
        z1['A'] = 0
        p1 = prob_mod(z1)
        if min(z1.values()) >=  0 and max(z1.values()) <=T:
            if p1[0] > - np.inf:
                p0 = p1
                z0 = z1
                for key, val in z0.iteritems():
                    time_samples[key].append(val)
                probs.append(p0)
                n += 1
                if n/500*500 == n:
                    print n
    probs = np.asarray(probs)
    out_ar = np.hstack((np.asarray(time_samples.values()).T, probs))
    columns = copy.copy(time_samples.keys())
    columns.append('P(z | attacker)')
    columns.append('P(data | z, attacker)')
    out = pandas.DataFrame(out_ar, columns = columns)
    return out,  lower_bound, V
