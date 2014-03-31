import numpy as np
from testing_net import net
import copy
from tools import prob_model_given_data, gen_logn_fact, prob_model_no_attacker
from itertools import chain



def direct_sample(SFTNet, data, num_samples, T, s0):
    net = copy.deepcopy(SFTNet)
    logn_fact = gen_logn_fact(data)
    n = 1
    nodes_to_change = [nd for nd in SFTNet.node_names if s0[nd] == 'normal' ]
    nodes_no_change = [nd for nd in SFTNet.node_names if s0[nd] == 'infected']
    prob_no_attacker = prob_model_no_attacker(SFTNet, data, T)
    prob_true_value = prob_model_given_data(SFTNet, data, data[-1], T, logn_fact)
    numattackers = len(nodes_no_change)
    prob_mod = lambda x : prob_model_given_data(SFTNet, data, x, T,
                                                logn_fact)
    probs = []
    while n < num_samples:
        t = 0
        state = [s0[nd] for nd in net.node_names]
        times = {nd: 0 for nd in nodes_no_change}
        # Corresponds to correct order
        while t<T :
            state_ix = net.cross_S.index(state)
            infected = [net.node_names[i] for i in range(len(net.nodes)) if state[i]=='infected']
            at_risk = set(chain(*[net.node_dict[nd].sends_to for nd in infected])) - set(infected)
            if len(at_risk) == 0:
                break
            at_risk_ix = [net.node_names.index(nd) for nd in at_risk]
            mt_rates = np.sum(net.mal_trans_mats[state_ix][:, at_risk_ix], axis=0)
            print at_risk, mt_rates, infected, n
            r_rate = np.sum(mt_rates)
            t += np.random.exponential(scale=1/r_rate)
            if t<T:
                next_infected = np.random.choice(list(at_risk), p = mt_rates/sum(mt_rates))
                times[next_infected] = t
                nd_ix = net.node_names.index(next_infected)
                state[nd_ix] = 'infected'
        print times, n
        probs.append(prob_mod(times)[1])
        n+=1
    e_probs = np.exp(probs)
    return np.log(np.mean(e_probs)), e_probs
