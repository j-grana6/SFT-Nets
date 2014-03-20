
"""
This script contains a function to uniformly sample the SFT net. The results 
will be used to compare to that from MH, Imp Samp, etc.

The pseudo-code is as follows.

- Create a set of V of all possible node orderings.
- For each v in V
  - For sample in 1: num_samps
    - Draw an infection time for each z in v
    - Compute and record P(data | z, net infected) X P(z | net infected)
  - Add up all of the samples in the previous step and multiply by the
    normalizing constant
- Sum up all of the averages for each v
"""

import numpy as np
import copy
from math import factorial
from orderings import *
from sft import *
from sft_net import *
from tools import *

def uniform_samp(SFTnet,s0,samp_size,T, data):
    """
    Returns a likelihood. Uniformly sample the SFT net and calculate the 
    likelihood.

    SFTnet: SFTnet instance
	The net that we would like to sample.

    s0: dict
	The net state at t = 0.
	
    samp_size: int
	The number of samples that we want to draw.
	
    T: int
	observation length.
    """
    numattackers = 0
    for stat in s0.values():
        if stat == 'infected':
            numattackers += 1
    logn_fact = gen_logn_fact(data)
    # Local assignments.
    nodes = SFTnet.nodes
    node_names = SFTnet.node_names
    # Create a set V of all possible node orderings.
    V = gen_orderings(SFTnet, s0)
    # Initialize the list to store the averages for each ordering, v
    samp_col = []
    averages = []
    ncs = []
    # For every possible ordering
    for v in V:
        # Initialize the sample
        samples = []
        # If any node is infected, corresponding to attacker existence case
        if len(v) != 0:
            # Check every node. If it is initially infected, append it to 
            # the list. In usual cases, only the first node is infected at
            # t = 0, so we can simply assign 0 to the first node. But we 
            # code it in a general way for a possible later extension that
            # multiple nodes can be infected at 0.
            # Initialize the vector of infection times
            # Zip into a dict
            zvec0 = dict(zip(v, [0]*len(v)))
            # Number of nodes to be sampled. Initialize it as the length of 
            # the ordering, but initially infected node(s) will be removed 
            # because their infection times are deterministically 0, there
            # is no need to be sampled
            to_be_samp = len(v)
            # Loop through each node.
            for n in v:
                if s0[n].lower() == 'infected':
                    zvec0[n] = 0
                    to_be_samp -= 1		
                # Sample an user-specified number of times
        for _ in range(samp_size):
            # If more nodes will be infected later on
            if to_be_samp > 0:
                # The way we do this is as follows. (1) The first node's 
                # infection time must be 0; (2) Assume there are r other
                # nodes than the first node in the ordering, then we
                # generate a list of size r, each element of which is
                # uniformly drawn from [1,T]; (3) Sort the list ascend and
                # assign each infection time to the nodes accordingly.
                # Initialize
                zvec = copy.copy(zvec0)
                # Sample infection times for initially normal nodes
                ran = np.random.randint(1, T, size = to_be_samp)
                # Sort infection times
                ran.sort()
                # Assign
                for n in v:
                    if s0[n].lower() == 'normal':
                        ix = v.index(n) - (len(v) - to_be_samp)
                        zvec[n] = ran[ix]
                    # If there are no other nodes than the initial node are infected 
            else:
                # Initialize
                zvec = copy.copy(zvec0)
                # Calculate P(data | z, attacker) and P(z | attacker)
                # given data.
            probs = prob_model_given_data(SFTnet,data,zvec,T,logn_fact)
            pz_a = probs[0]; pd_za = probs[1]
            # Combine them to get P(data | attacker)
            pd_a = pd_za + pz_a
            # Add the probability to samples
            samples.append(pd_a)
            # If no node is infected at all time, corresponding to no attacker case.
            # Number of infected nodes in the ordering.
        m = len(v) - numattackers
        #Normalizing constant.
        nc = T ** m / factorial(m) * 1/.54450
        # Average up the samples for this v.
        # Need to take out -inf's
        samples = np.asarray(samples)
        ix =  samples == -np.inf
        samples1 = samples[~ix]
        av = np.sum(np.exp(samples)) * nc / float(samp_size)
        ncs.append(nc)
        # Add this average number to "averages".
        averages.append(av)
        samp_col.append(samples)
    # Sum up all of the averages for each v
    lhood = np.log(np.sum(averages))
    return lhood, samp_col, ncs, samples, averages

def plot_uniform_convergence(unires):
    unif_samps = np.exp(np.asarray(unires[1]))
    (num_configs, samps_per) = unif_samps.shape
    ncs = np.asarray(unires[2]).reshape(num_configs, 1)
    unif_samps = unif_samps * ncs
    unif_samps = np.cumsum(unif_samps, axis=1)
    unif_samps = unif_samps / np.arange(1, samps_per +1, 1)
    uni_ma = np.log(np.sum(unif_samps, axis=0))
    return uni_ma

if __name__ =='main':
    # Test the uniform_samp()
    from testing_net import *

    # Initial net state
    state0 = dict(A = 'infected',
			  B = 'normal',
			  C = 'normal',
			  D = 'normal'
			 )
    data = gen_data(T,net,state0)
    lhood = uniform_samp(net, state0, 1000, T, data)
    print lhood
