import numpy as np


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
    infect_times = dict(zip([x.name for x in SFTNet.nodes], [0]*n_nodes))
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
            print t
            # The only time this happens is if a node gets infected
            state_change = 1
            infect_times[receiver.name] = t
            state = SFTNet.get_state()
        n_reactions += 1
    return (msgs_sent, reaction_times, reaction_sender,
            reaction_receiver, n_reactions, infect_times)

