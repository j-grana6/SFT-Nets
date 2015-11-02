### THIS SCRIPT GENERATES DATA
### such that the attacker does not
### send messages to already infected nodes

### It is different from our original ROC formulation
### However, the prob_model function does not account for this.  In other words
### I have not yet coded a way to compute the likelihood if the attacker stops
### sending messages

import numpy as np
import copy
import operator
from scipy.stats import poisson, norm
import pandas as pd


def gen_data(T, SFTNet, t0):
    """

    Parameters
    ----------

    T : int
        Length of time

    SFTNet : SFTNet instance:
        An SFTNet

    s0 : list
        A list of strings that sets an initial state.  The
        order of the list should correspond to the order of
        SFTNet.nodes

    """
    ##Make s0
    s0 = []
    for nd in SFTNet.node_names:
        s0.append(t0[nd])
    n_nodes = len(SFTNet.nodes)
    # Number of nodes
    reaction_times = []
    # empty list to collect reaction times
    reaction_sender = []
    # empty list to collect reaction sender
    reaction_receiver = []
    # collect the receiving nodes
    msgs_sent = []
    # Record the messages sent
    n_reactions = 0
    # initialize number of reactions
    t = 0
    # initialize start time
    infect_times = dict(zip([x.name for x in SFTNet.nodes], [T]*n_nodes))
    for key, val in t0.iteritems():
        if val == 'infected':
            infect_times[key] = 0
    state = s0
    for s in range(len(state)):
        SFTNet.nodes[s].state = state[s]
    state_change = 0
    t_rates = SFTNet.get_all_trans()
    while t < T:
        if t == 0 or state_change == 1:
            # If we are just starting, get the correct
            # transmission rates.  If there is a state
            # change, get the new transmission rates.
            # If the state of the system didn't change.
            # use the same transmission rate and skip
            # this block.
            # Get the index of the state of the net
            unadj_t_rates = SFTNet.get_all_trans()
            mal_rates = SFTNet.get_mal_trans()
            c_rates = unadj_t_rates - mal_rates
            ### Below we are going to account for
            ### attackers not sending messages to already
            ### infected nodes.
            I_mat = np.identity(c_rates.shape[0])
            for nd_ix in range(len(SFTNet.nodes)):
                if state[nd_ix]=='infected':
                    I_mat[nd_ix, nd_ix] =0
            t_rates = c_rates + np.dot(mal_rates, I_mat)
            # The transmission matrix that corresponds
            # to that state.
            r_rate = np.sum(t_rates)
            # Reaction rate
        t_temp = t + np.random.exponential(scale=1 / r_rate)
        if t_temp > T:
            break
        draw = r_rate*np.random.random()
        # Random number to determine sender and receiver
        reaction_ix = np.argmin(np.cumsum(t_rates) < draw)
        # argmin returns the index of the **first** element the
        # cum sum that is less than draw.  Therefore, the random
        # number was between ix-1 and ix of the cumsum, which is the
        # area of the distribution associated with a draw of reaction
        # given by reaction_ix
        sender_ix, receiver_ix = int(reaction_ix)/int(n_nodes),\
            reaction_ix % n_nodes
        # This is the location in the matrix of the reaction
        # The first term is the sending node index, the second
        # term is the receiving node index.
        sender, receiver = SFTNet.nodes[sender_ix], SFTNet.nodes[receiver_ix]
        # Get the actual sender and receiver nodes
        sndr_state_ix = sender.states.index(state[sender_ix])
        # What state is the sender in (from which sending distribution
        # do we draw?)
        reaction_sender.append(sender.name)
        reaction_receiver.append(receiver.name)
        t = t_temp
        if receiver.state =='infected':
            msg = 'clean'
        else:
            msg_distribution = np.cumsum(sender.rates[receiver.name][sndr_state_ix])
            msg_ix = np.argmin(msg_distribution <
                            np.random.random() * msg_distribution[-1])
            # Determine the index of the message to send
            # Note that this data generating algorithm calls 2 random numbers.
            # One to determine the sender-receiver and the other to determine
            # the message to send.  Theoretically, these steps can be combined
            # and we can use only 1 random number.
            msg = sender.messages[msg_ix]
            # The message string
        msgs_sent.append(msg)
        # Record the transmission
        receiver.react(msg, sender.name)
        reaction_times.append(t)
            # Draw the next time and append.
            # Marginally faster than t - log(random.random())/r_rate
        n_reactions += 1
        if state == SFTNet.get_state():
        # If the message is not malicious or the node was already
        # infected, this will hold.
            state_change = 0
        else:
            # The only time this happens is if a node gets infected
            state_change = 1
            infect_times[receiver.name] = t
            # Dictionary
            state = SFTNet.get_state()
        # Update state
    transmissions = {}
    for node in SFTNet.nodes:
        for o_node in node.sends_to:
            key = node.name+'-'+str(o_node)
            from_node = np.asarray(reaction_sender) == node.name
            to_o_node = np.asarray(reaction_receiver) == o_node
            times = from_node * to_o_node
            transmissions[key] = np.asarray(reaction_times)[times]

    return (np.asarray(msgs_sent), np.asarray(reaction_times),
            np.asarray(reaction_sender),
            np.asarray(reaction_receiver), n_reactions, transmissions,
            infect_times)



def prob_model_given_data(SFTNet, data, infect_times, T, logn_fact, s0):  ## TODO: Profile this
    """
    Returns a tuple whose first element is P(z | attacker) and the
    second element is P(data | z, attacker).  Note the function is misnamed.

    Parameters
    ----------

    SFTNet : SFTNet instance
        An SFTNet

    data : list
        Output of gendata

    infect_times : dict
        Dictionary of infection times

    T : float
        Total running time

    Notes
    -----

    This function cannot take the same infection time for multiple nodes
    as a starting parameters.  Need to look into why.

    """
    eps = 0
    transmissions = data[-2]
    # First order the infections
    attackers = [nd for nd in s0.keys() if s0[nd] =='infected']
    sorted_infect = sorted(infect_times.iteritems(),
                           key=operator.itemgetter(1))
    if min(infect_times.values()) < 0:
        return [-np.inf, - np.inf]
    not_infected = [nd for nd in SFTNet.node_names \
                    if nd not in infect_times.keys()]
    # Creates a list of tuples and then sorts by the value
    # Assuming the first node in SFTNet.nodes is infected.
    # This can be generalized to any initial condition state
    for nd in SFTNet.nodes:
        if nd.name in attackers:
            nd.state='infected'
        else:
            nd.state='normal'
    prob_sequence = 0
    prob_exact_times = 0
    time_minus_1 = 0
    for node, time in sorted_infect[len(attackers):]:
        # TODO IMPORTANT.  Generalize to be able to specify intial
        # infected node.  
        ### Here we need to control for the fact that we only
        ### care about the ordering if the nodes that do
        ### get infected since we are taking as *given* some
        ### nodes do not get infected when summing over DAGS
        ### The p exact time is ok
        infect_ix = SFTNet.node_names.index(node)
        # The index of the node that gets infected
        # The state of the net when the node gets infected
        mal_trans = SFTNet.get_mal_trans()
        # The transmission matrix of malicious messages given config
        prob_node = np.sum(mal_trans[:, infect_ix])
        # The (relative) probability that the node
        # is infected by any other node
        prob_total = np.sum(mal_trans[:,
                np.asarray(SFTNet.get_state()) == 'normal'])
        # Only the nodes that are not already infected can become
        # infected.
        prob_sequence += np.log(prob_node + eps) - \
          np.log(prob_total + eps)
        # Update the probability of the sequence order
        deltat = time - time_minus_1 + eps
        # The time between infections
        prob_exact_times += np.log(prob_total + eps) - \
            deltat * prob_total
        # Update the probability of the specific times.  We use deltat
        # because of the memoryless property of the process.
        SFTNet.node_dict[node].state = 'infected'
        time_minus_1 = time
        ## The above loop combines the first 2 functions of Munsky's
        ## matlab code.

    deltat = T - time_minus_1
    noinfect_prob = 0
    for node in not_infected:
        infect_ix = SFTNet.node_names.index(node)
        # The index of the node that gets infected
        # The state of the net when the node gets infected
        mal_trans = SFTNet.get_mal_trans() 
        # The transmission matrix of malicious messages given config
        prob_node = np.sum(mal_trans[:, infect_ix])
        # This is the sum of all of the rate constants connected to the node
        noinfect_prob += - prob_node * deltat

    prob_no_infect_data = 0
    for node in not_infected:
        # For each node.  node is the node name, not the instance
        _node_inst = SFTNet.node_dict[node]
        # We need the node instance here.  This should be added as a method
        # of SFTNet instance
        norm_ix = _node_inst.states.index('normal')
        # Index of normal state
        for o_node in _node_inst.sends_to :
            rate =  np.sum(_node_inst.rates[o_node][norm_ix, :])
            num_msgs = len(transmissions[node+'-'+o_node])
            prob_msgs = (num_msgs *
                    np.log(rate * T) -
                    logn_fact[num_msgs] -
                    rate * T)
            prob_no_infect_data += prob_msgs

    prob_data = prob_no_infect_data
    for node, time in sorted_infect:
        # For each node.  node is the node name, not the instance
        _node_inst = SFTNet.node_dict[node]
        # We need the node instance here.  This should be added as a method
        # of SFTNet instance
        norm_ix = _node_inst.states.index('normal')
        # Index of normal state
        infect_ix = _node_inst.states.index('infected')
        # Index of infected state
        for o_node in _node_inst.sends_to:
        # If two nodes are connected
            if time == 0:
                # Because of problems with 0 in logs
                # we use this for nodes that are initially
                # infected
                num_msgs = len(transmissions[node+'-'+o_node])
                prob_msgs = (num_msgs *
                    np.log(
                    np.sum(_node_inst.rates[o_node][infect_ix, :]) * T) -
                    logn_fact[num_msgs] -
                    np.sum(_node_inst.rates[o_node][infect_ix, :]) * T)
                prob_data += prob_msgs
            else:
                num_before =  np.searchsorted(
                    transmissions[node+'-'+o_node],time)
                # Number of reactions before
                num_after = len(transmissions[node+'-'+o_node]) - num_before
                # Number of reactions after infection
                if num_before == 0 :
                    prob_before = - np.sum(_node_inst.rates[o_node][norm_ix, :]) * \
                                    min(T, time)
                else:
                    prob_before = (num_before *
                            np.log(eps +
                            np.sum(_node_inst.rates[o_node][norm_ix, :]) *
                            min(T, time)) -
                            logn_fact[num_before] -
                            np.sum(_node_inst.rates[o_node][norm_ix, :]) *
                            min(T, time))
                # prob before is the probability of node sending num_before
                # messages to o_node before it gets infected.  This is a bit
                # different from Munsky's in 3 ways.  The first is the min
                # function.  This allows us to compute the probability of the
                # model even when our 'guess' times are above the simulation time.
                # For example, we can compute the probability of the model
                # that says node 2 gets infected at time 10001 when we only
                # sample up to time 1000.  The second difference is the the
                # 10**-10 term.  This is simply because I assume the simulation
                # starts at time 0, not time 1.  Finally, I use numpy to compute
                # the factorial instead of his function.  We will see if this is
                # a significant bottle neck later.
                if num_after == 0:
                    prob_after = - np.sum(_node_inst.rates[o_node][infect_ix, :]) * \
                            (T-time)
                else:
                    prob_after = ( num_after *
                            np.log(eps +
                            np.sum(_node_inst.rates[o_node][infect_ix, :]) *
                            (T- time)) -
                            logn_fact[num_after] -
                            np.sum(_node_inst.rates[o_node][infect_ix, :]) *
                            (T-time))
                prob_data += prob_before + prob_after


    return [prob_sequence + prob_exact_times + noinfect_prob,  prob_data, prob_sequence + noinfect_prob, prob_sequence]

def prob_model_no_attacker(SFTnet, data, T):
    """
    Calculates the probability of the data when no node is
    initially infected.  It is just the probability of each
    sequence of observations.
    """
    total_prob = 0
    for node in SFTnet.nodes:
        # For each node
        for rec in node.sends_to:
        # For each possible receiver
            normal_ix = node.states.index('normal')
            clean_ix = node.messages.index('clean')
            rate = node.rates[rec][normal_ix, clean_ix]
            num_sent = np.sum((data[2] == node.name) * (data[3] == rec))
            logprob = -rate * T + num_sent * (np.log(rate * T))  \
                - np.sum(np.log(np.arange(1, num_sent + 1, 1)))
            total_prob += logprob
    return total_prob

def get_alarm_time(SFTnet, data, T, window, lp_limit):
    """
    This is hackish and can definitely be improved
    """
    alarm_sounds =False
    d = [0,0]
    leading_ix = np.argmin(np.asarray(data[1]<=window))
    trailing_ix = 0
    d.append(data[2][trailing_ix:leading_ix])
    d.append(data[3][trailing_ix:leading_ix])
    p_window = prob_model_no_attacker(SFTnet, d, window)
    if p_window < lp_limit:
        return 0
    leading_ix = np.argmax(np.asarray(data[1])>= window)
    while not alarm_sounds:
        d= [0,0] #This is a placeholder so I can easily use
                 # p_model_no_attacker function
        trailing_ix = np.argmax(data[1] > data[1][leading_ix]-window)
        #print leading_ix, trailing_ix
        d.append(data[2][trailing_ix:leading_ix])
        d.append(data[3][trailing_ix:leading_ix])
        p_window = prob_model_no_attacker(SFTnet, d, window)
        #print p_window
        if p_window < lp_limit:
            alarm_sounds = True
        else:
            leading_ix +=1
        if leading_ix >= len(data[1]):
            alarm_sounds=True
            return T
    return data[1][leading_ix]
        
    

## I'm pretty sure we can delete this stuff, we found a better way
## to construct a proposal distribution
# def qij_over_qji(zi, zj, con_cdf, con_pdf):
#     def qij(zi, zj, con_cdf=con_cdf, con_pdf=con_pdf):
#         """
#         Returns the probability of proposing zj given we are in
#         zi.  This is used because we are using a non-uniform
#         MCMC sampler.  This is very non-general.  It only applies
#         to the example in munsky's original code.

#         Notes
#         -----

#         The convolution of a normal random variable with mean 0 and
#         standard deviation sigma and a continuous uniform random
#         variable on (a,b) is given by

#         (norm.cdf((x-a)/sigma) - norm.cdf((x-b)/sigma))/b-a
#         """
#         pc = norm.pdf(zj['C'] - zi['C'], 0, 100) * \
#           (1 - con_cdf(zi['C'] - zi['B'])) + \
#           con_pdf(zj['C'] - zi['B']) * \
#           con_cdf(zi['C'] - zi['B'])

#         prob_zjc_less_than_zjb = (1 - con_cdf(zi['C'] - zi['B'])) * \
#             norm.cdf(zi['B'] - zi['C'], 0 , 20000**.5)

#         pd = prob_zjc_less_than_zjb * max(0, 0 < zj['D'] - zj['C'] < 50) + \
#             (1 - prob_zjc_less_than_zjb) * max(0, 0 < zj['D'] - zj['B'] < 50)
#         return pc * pd
#     return np.log(qij(zj, zi)) - np.log(qij(zi,zj))



# def convoluted_cdf_func(sigma, a,b, grain=.01):
#     """
#     Returns a function that Generates the cdf of the convolution of a
#     random variable with mean 0 and standard deviation sigma and
#     a uniform random variable on the interval (a,b).
#         """
#     x = np.arange(-4 *sigma + a, 4 * sigma + b, grain)
#     #support to consider
#     con_pdf = (norm.cdf((x - a) / sigma) -
#                norm.cdf((x - b) / sigma)) / (b - a)
#     # the convoluted pdf
#     con_cdf = np.cumsum(con_pdf)/np.sum(con_pdf)
#     # approximate cdf
#     def get_prob(t, support=x, cdf=con_cdf):
#         if t > max(support):
#             return 1
#         elif t < min(support):
#             return 0
#         else:
#             idx = np.argmin(t > support)
#             return cdf[idx]
#     return get_prob

# def convoluted_pdf_func(x, sigma=100, a=0, b=50):
#     return (norm.cdf((x-a)/sigma) - norm.cdf((x-b)/sigma))/ (b - a)


# def rhs_integral(SFTNet, data, T):
#     """
#     Returns \int_T^\infty P(d|z, attacker)P(z|attacker)
#     *Not* the log.
#     """
#     data_greater_t = {node.name: T+1 for node in SFTNet.nodes}
#     data_greater_t['A'] = 0 # Keep initially infected
#     p_data = prob_model_given_data(SFTNet, data, data_greater_t,T,
#                                    gen_logn_fact(data))[1]
#     prob_z = 0
#     for node in SFTNet.nodes:
#         for rec in node.sends_to:
#             if node.location =='external':
#                 infect_ix = node.states.index('infected')
#                 mal_ix = node.messages.index('malicious')
#                 rate = node.rates[rec][infect_ix, mal_ix]
#                 prob_z += -rate * T
#     return np.exp(p_data + prob_z)

def gen_logn_fact(data):
    return np.hstack((np.zeros(1), np.cumsum(np.log(np.arange(1, len(data[0])+2,1)))))


def gen_trans_frame(net):
    import pandas as pandas
    ix1 = []
    for node in net.node_names:
        ix1.extend([node]*len(net.node_names))
    ix2 = net.node_names * len(net.node_names)
    ix1 = np.asarray(ix1)
    ix2 = np.asarray(ix2)
    ix = pandas.MultiIndex.from_arrays([ix1, ix2],names= ['sender', 'receiver'])
    df = pandas.DataFrame(np.zeros(shape=(len(ix1), 3)),
                          columns = ['normal-clean rate', 'infected-clean rate', 'infected-malicious rate'],index=ix)
    for i in range(len(net.nodes)):
        nd = net.nodes[i]
        normstate_ix = nd.states.index('normal')
        infstate_ix = nd.states.index('infected')
        try :
            cleanmsg_ix = nd.messages.index('clean')
            malmsg_ix = nd.messages.index('malicious')
        except Exception:
            print 'Doesn\'t send'
        for o_node in nd.sends_to:
            df.set_value((nd.name, o_node), 'normal-clean rate', nd.rates[o_node][normstate_ix, cleanmsg_ix])
            df.set_value((nd.name, o_node), 'infected-clean rate', nd.rates[o_node][infstate_ix, cleanmsg_ix])
            df.set_value((nd.name, o_node), 'infected-malicious rate', nd.rates[o_node][infstate_ix, malmsg_ix])
    return df

# def trunc_expon(rate, truncation):
#     """
#     rate :
#         the rate

#     truncation :
#         truncation time
#     """
#     rate = float(rate)
    
#     R = np.random.random()*(1-np.exp(-truncation*rate))
#     return -np.log(1-R)*1./rate



# if __name__ =='__main__':
    # import numpy as np
    # #from direct_sample import Direct_Sample
    # from sft import SFT
    # from sft_net import SFTNet
    # import networkx as nx
    # from IPython.display import HTML
    # from tools import gen_trans_frame
    # from roc import get_roc_coords
    # A = SFT('A', ['normal' , 'infected'], ['B', 'F'], 
    # {'B': np.array([[.5, 0], [.5,.001]]), 'F' : np.array([[.5, 0], [.5, .001]])},
    # ['clean', 'malicious'], 'external')
    # # Node A sends messages to B and F

    # B = SFT('B', ['normal' , 'infected'], [ 'E'], 
    #     {'E' : np.array([[2, 0], [2, .01]])},
    #     ['clean', 'malicious'], 'internal')
    # # B sends messages to A, D and F
    # C = SFT('C', ['normal' , 'infected'], ['E'],
    #     {'E': np.array([[.25, 0], [.25,.01]])},
    #     ['clean', 'malicious'], 'internal')
    # # C sends messages to A, B and F


    # D = SFT('D', ['normal' , 'infected'], ['E'], 
    #     {'E': np.array([[.8, 0], [.8,.01]])},
    #     ['clean', 'malicious'], 'internal')
    # # D sends nodes to A, C and F

    # E = SFT('E', ['normal' , 'infected'], ['A', 'B', 'C', 'D', 'F'], 
    #     {'A': np.array([[3, 0], [3,.001]]), 'B': np.array([[5,0], [5, .001]]), 
    #      'C': np.array([[.5,0], [.5, .001]]), 'D' : np.array([[4, 0], [4, .001]]), 'F' : np.array([[.5, 0], [.5, .001]])},
    #     ['clean', 'malicious'], 'internal')
    # # E (slowly) sends nodes to A, B, C and F

    # F = SFT('F', ['normal' , 'infected'], ['A', 'E'], 
    #     {'A': np.array([[10, 0], [10,.01]]), 'E' : np.array([[.9, 0], [.9, .01]])},
    #     ['clean', 'malicious'], 'external')
    # # F sends nodes to A and E

    # nodes_2 = [A, B,C,D,E,F]
    # net_2 = SFTNet(nodes_2)
    # s0_2 = {'A': 'infected', 'B': 'normal', 'C': 'normal', 'D': 'normal', 'E': 'normal', 'F': 'normal'}
    # gen_data(15000, net_2, s0_2)
