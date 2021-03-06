import numpy as np
#from testing_net import net
import copy
from tools import prob_model_given_data, gen_logn_fact, prob_model_no_attacker
from itertools import chain



def Direct_Sample(SFTNet, data, num_samples, T, s0):
    net = copy.deepcopy(SFTNet)
    logn_fact = gen_logn_fact(data)
    n = 1
    nodes_to_change = [nd for nd in net.node_names if s0[nd] == 'normal' ]
    nodes_no_change = [nd for nd in net.node_names if s0[nd] == 'infected']
    prob_no_attacker = prob_model_no_attacker(net, data, T)
    prob_true_value = prob_model_given_data(net, data, data[-1], T, logn_fact, s0)
    numattackers = len(nodes_no_change)
    prob_mod = lambda x : prob_model_given_data(net, data, x, T,
                                                logn_fact, s0)
    probs = []
    while n < num_samples:
        t = 0
        for nd in net.node_names:
            net.node_dict[nd].state = s0[nd]
        times = {nd: 0 for nd in nodes_no_change}
        # Corresponds to correct order
        while t<T :
            infected = [nd.name for nd in net.nodes if nd.state =='infected']
            at_risk = set(chain(*[net.node_dict[nd].sends_to for nd in infected])) - set(infected)
            if len(at_risk) == 0:
                break
            at_risk_ix = [net.node_names.index(nd) for nd in at_risk]
            mt_rates = np.sum(net.get_mal_trans()[:, at_risk_ix], axis=0)
            #print at_risk, mt_rates, infected, n
            r_rate = np.sum(mt_rates)
            t += np.random.exponential(scale=1/r_rate)
            if t<T:
                next_infected = np.random.choice(list(at_risk), p = mt_rates/sum(mt_rates))
                times[next_infected] = t
                net.node_dict[next_infected].state = 'infected'
        #print times, n
        probs.append(prob_mod(times)[1])
        n+=1
    e_probs = np.exp(probs)
    return np.log(np.mean(e_probs)), e_probs
