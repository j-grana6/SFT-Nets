import numpy as np
import copy
import operator


def gen_data(T, SFTNet, s0):
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
    infect_times['A'] = 0
    state = s0
    for s in range(len(state)):
        SFTNet.nodes[s].state = state[s]
    state_change = 0
    while t < T:
        if t == 0 or state_change == 1:
            # If we are just starting, get the correct
            # transmission rates.  If there is a state
            # change, get the new transmission rates.
            # If the state of the system didn't change.
            # use the same transmission rate and skip
            # this block.
            state_ix = SFTNet.cross_S.index(state)
            # Get the index of the state of the net
            t_rates = SFTNet.transmission_mats[state_ix]
            # The transmission matrix that corresponds
            # to that state.
        r_rate = np.sum(t_rates)
        # Reaction rate
        t += np.random.exponential(scale=1 / r_rate)
        if t > T:
            break
        reaction_times.append(t)
        # Draw the next time and append.
        # Marginally faster than t - log(random.random())/r_rate
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
        reaction_sender.append(sender.name)
        reaction_receiver.append(receiver.name)
        msgs_sent.append(msg)
        # Record the transmission
        receiver.react(msg, sender.name)
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
        n_reactions += 1
    return (np.asarray(msgs_sent), np.asarray(reaction_times),
            np.asarray(reaction_sender),
            np.asarray(reaction_receiver), n_reactions, infect_times)



def prob_model_given_data(SFTNet, msg_times, infect_times, senders,
                          receivers, T):
    """
    Parameters
    ----------

    net : SFTNet instance
        An SFTNet

    msg_times : list
        A list of times of transmissions (in order)

    infect_times : dict
        Dictionary of infection times

    senders : list
        List of senders of messages.  Order is given by msg_times.

    receivers : list
        List of receivers of messages.  Order is given by msg times.

    T : float
        Total running time

    """

    # First order the infections
    sorted_infect = sorted(infect_times.iteritems(),
                           key=operator.itemgetter(1))

    # Creates a list of tuples and the sorts by the value
    state = ['infected'] + (len(SFTNet.nodes) - 1) * ['normal']
    # Assuming the first node in SFTNet.nodes is infected.
    # This can be generalized to any initial condition state
    prob_sequence = 0
    prob_exact_times = 0
    time_minus_1 = 0
    for node, time in sorted_infect[1:]:
        infect_ix = SFTNet.node_names.index(node)
        # The index of the node that gets infected
        cross_S_ix = SFTNet.cross_S.index(state)
        # The state of the net when the node gets infected
        mal_trans = SFTNet.mal_trans_mats[cross_S_ix]
        # The transmission matrix of malicious messages given config
        prob_node = np.sum(mal_trans[:, infect_ix])
        # The (relative) probability that the node
        # is infected by any other node
        prob_total = np.sum(mal_trans[:, np.asarray(state) == 'normal'])
        # Only the nodes that are not already infected can become
        # infected.
        prob_sequence += np.log(prob_node + 10 ** -10) - \
          np.log(prob_total)
        # Update the probability of the sequence order
        deltat = time - time_minus_1 + 10 ** -10
        # The time between infections
        prob_exact_times += np.log(prob_total) - \
        np.log(deltat * prob_total)
        # Update the probability of the specific times.  We use deltat
        # because of the memoryless property of the process.
        state[infect_ix] = 'infected'
        time_minus_1 = time
        ## The above loop combines the first 2 functions of Munsky's
        ## matlab code.
    prob_data = 0
    for node, time in sorted_infect:
        # For each node.  Node is the node name, not the instance
        _node_ix = SFTNet.node_names.index(node)
        _node_inst = SFTNet.nodes[_node_ix]
        # We need the node instance here.  This should be added as a method
        # of SFTNet instance
        norm_ix = _node_inst.states.index('normal')
        # Index of normal state
        infect_ix = _node_inst.states.index('infected')
        # Index of infected state
        for o_node in _node_inst.sends_to:
            # If two nodes are connected
            num_before = np.sum(((senders == node) *
                                (receivers == o_node)
                                )[msg_times <= time])
            # Number of reactions before
            num_after = np.sum(((senders== node) *
                                (receivers == o_node)
                                )[msg_times > time])
            # Number of reactions after infection
            prob_before = ( num_before *
                        np.log(10**-10 +
                        np.sum(_node_inst.rates[o_node][norm_ix, :]) *
                        min(time + 10 ** -10, T)) -
                        np.sum(np.log(10 ** -10 + np.arange(1, num_before+1))) -
                        np.sum(_node_inst.rates[o_node][norm_ix, :]) *
                        min(time, T))
            # prob before is the probability of node sending num_before
            # messages to o_node before it gets infected.  This is a bit
            # different from Munsky's in 3 ways.  The first is the min
            # function.  This allows us to compute the probability of the model
            # even when our 'guess' times are above the simulation time.
            # For example, we can compute the probability of the model
            # that says node 2 gets infected at time 10001 when we only
            # sample up to time 1000.  The second difference is the the
            # 10**-10 term.  This is simply because I assume the simulation
            # starts at time 0, not time 1.  Finally, I use numpy to compute
            # the factorial instead of his function.  We will see if this is
            # a significant bottle neck later.
            prob_after = ( num_after *
                        np.log(10 ** -10 +
                        np.sum(_node_inst.rates[o_node][infect_ix, :]) *
                        max(T- time, 10**-10)) -
                        np.sum(np.log(10 ** -10 + np.arange(1, num_after + 1))) -
                        np.sum(_node_inst.rates[o_node][infect_ix, :]) *
                        max(T-time, 10**-10))
            prob_data += prob_before + prob_after
    return prob_sequence + prob_exact_times + prob_data










