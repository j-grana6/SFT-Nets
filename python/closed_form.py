import numpy as np
from tools import prob_model_given_data,  \
      rhs_integral, gen_logn_fact, prob_model_no_attacker
import copy
import pandas as pandas
from results import Results
import random
from orderings import gen_orderings
from collections import defaultdict
import itertools

#@profile
def closed_form_calc(SFTNet, s0, data,T):
    """The usual inputs"""
    lnf = gen_logn_fact(data)
    orderings = gen_orderings(SFTNet, s0)
    attackers  = [nd for nd in SFTNet.node_names if s0[nd] == 'infected']
    probs = []
    z0 = dict(zip(attackers, [0]*len(attackers)))
    for order in orderings :
        print order
        num_infected = len(order)
        already_infected = copy.copy(attackers)
        possibles = []
        for nd in order[len(attackers):] :
            possible_time = []
            for o_node in already_infected:
                try :
                    possible_time.extend(
                    list(data[5][o_node +'-'+ nd]))
                except KeyError:
                    print o_node + ' does not send to ' + nd
            possibles.append(possible_time)
            already_infected.append(nd)
        order_cart = itertools.product(*possibles)
        i = 0 
        while True:
            i +=1
            try :
                times = order_cart.next()
                if sorted(times) == list(times):
                    infect_z = dict(zip(order[len(attackers):], times))
                    new_z = dict(z0.items() + infect_z.items())
                    newprob = prob_model_given_data(SFTNet,\
                            data, new_z, T, lnf)
                    probs.append(newprob)
            except StopIteration:
                print order, 'number of samples', i
                break
    return probs

if __name__ == '__main__':
    from testing_net import net
    from tools import gen_data
    np.random.seed(6)
    s0 = {'A' : 'infected', 'B': 'normal', 'C' : 'normal', 'D': 'normal'}
    T = 10000
    data = gen_data(T, net, s0)
    exact = closed_form_calc(net, s0, data, T)
    exact = np.asarray(exact)
    exact = np.exp(exact)
    ps = exact[:, 0] + exact[:, 1]
    lval = np.log(np.sum(ps))