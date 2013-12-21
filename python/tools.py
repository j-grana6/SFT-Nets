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
    reaction_type = []
    # empty list to collect type of reaction
    n_reactions = 0
    # initialize number of reactions
    t = 0
    # initialize start time
    state=s0
    while t < T:
        if t==0 or state_change==1:
            # If we are just starting, get the correct
            # transmission rates.  If there is a state
            # change, get the new transmission rates.
            # If the state of the system didn't change.
            # use the same transmission rate and skip
            # this block.
            state_ix = SFTNet.cross_S.index[state]
            # Get the index of the state
            t_rates = SFTNet.transmission_mats[state_ix]
            # The transmission matrix that corresponds
            # to that state.
        r_rate = np.sum(t_rates)
        # Reaction rate
        t += np.random.exponential(scale=1/w0)
        # Draw the next time
        # Marginally faster than t - log(random.random())/r_rate
        draw = r_rate*np.random.random()
        # Random number to determine which reaction
        reaction_ix = np.argmin(np.cumsum(t_rates)< draw)
        # argmin returns the index of the **first** element the
        #  cum sum that is less than draw.  Therefore, the random
        # number was between ix-1 and ix of the cumsum, which is the
        # area of the distribution associated with a draw of reaction
        # given by reaction_ix
        sender_ix, receiver_ix =  int(reaction_ix)/int(n_nodes),\
                                            reaction_ix % n_nodes
        # This is the location in the matrix of the reaction
        # The first term is the sending node index, the second
        # term is the receiving node index.


