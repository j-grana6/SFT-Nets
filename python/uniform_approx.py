"""
This script contains a function to uniformly sample the SFT net. The results 
will be used to compare to that from MH, Imp Samp, etc.

The algorithm is as follows.

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
import matplotlib.pyplot as plt
import copy

from orderings import *
from sft import *
from sft_net import *

def uniform_samp(SFTnet,s0,samp_size,T):
	"""
	Returns a probability. Uniformly sample the SFT net and calculate the 
	likelihood.

	SFTnet: SFTnet instance
		The net that we would like to sample.

	s0: list
		The net state at t = 0.
	
	samp_size: int
		The number of samples that we want to draw.
	
	T: int
		Observation length.
	"""

	# Local assignments.
	nodes = SFTnet.nodes
	node_names = SFTnet.node_names

	# Create a dictionary where keys are node names and values are node states
	# at t = 0. Such a dict is required to generate all node orderings.

	# First, assert the list of initial states, s0, is complete, i.e. has the 
	# length of the number of nodes. Otherwise, the zip() function will 
	# truncate the zipped list as the shortest argument sequence.
	assert len(node_names) == len(s0)
	# Second, zip node_names and s0 to a dictionary.
	dict_s0 = dict(zip(node_names,s0))

	# Create a set V of all possible node orderings.
	V = gen_orderings(SFTnet, dict_s0)

	# Initialize the sample
	sample = []

	# For every possible ordering
	for v in V:
		# If any node is infected, corresponding to attacker existence case
		if len(v) != 0:
			# Check every node. If it is initially infected, append it to 
			# the list. In usual cases, only the first node is infected at
			# t = 0, so we can simply assign 0 to the first node. But we 
			# code it in a general way for a possible later extension that
			# multiple nodes can be infected at 0.

			# Initialize the vector of infection times
			times = [-1] * len(v)
			# Zip into a dict
			zvec0 = dict(zip(node_names, times))
			# Number of nodes to be sampled. Initialize it as the length of 
			# the ordering, but initially infected node(s) will be removed 
			# because their infection times are deterministically 0, there
			# is no need to be sampled
			to_be_samp = len(v)
			# Loop through each node.
			for n in v:
				if dict_s0[n].lower() == 'infected':
					zvec0[n] = 0
					to_be_samp -= 1
			

			# Sample an user-specified number of times
			for _ in range(samp_size):
				# The way we do this is as follows. (1) The first node's 
				# infection time must be 0; (2) Assume there are r other nodes
				# than the first node in the ordering, then we generate a list
				# of size r, each element of which is uniformly drawn from 
				# [1,T]; (3) Sort the list ascend and assign each infection 
				# time to the nodes accordingly.

				# Initialize
				zvec = copy.copy(zvec0)
				# Sample infection times for initially normal nodes
				ran = np.random.randint(1, T, size = to_be_samp)
				# Sort infection times
				ran.sort()
				# Assign
				for n in v:
					if dict_s0[n].lower() == 'normal':
						ix = v.index(n)
						zvec[n] = ran[ix]
