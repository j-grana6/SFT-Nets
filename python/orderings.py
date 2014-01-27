import numpy as np
import copy
from itertools import permutations

def validate(order, sends_fr, s0):
	"""
	Ruturn a True or False to indicate whether this order is valid.

	order: tuple
		The order to be tested its validity

	sends_fr: dict
		Lists of tuples of size 2, which indicate out links among nodes in
		the order

	s0: dict
		The state of the SFT net at 0. Key is the node name, value is the state.
	"""

	# Initialize the returned value.
	valid = True

	# Infected nodes stay infected. So any node cannot appear in the order twice.
	if len(order) != len(set(order)):
		valid = False
	# No duplicates in the order
	else:
		# The first node in the order must be a spontaneously infected node at
		# t = 0, otherwise the infection cannot be spread.

		# If no node is infected at all (no attacker case)
		if order == ():
			# This case is only possible to happen when all nodes are normal 
			# at t = 0
			normal = ['normal'] * len(s0.keys())
			if s0.values() == normal:
				valid = True
			else: 
				valid = False
		# At least one node is infected (attacker existence case)
		else:
			first = order[0]
			if s0[first].lower() == 'normal':
				valid = False
			# The first node is infected.
			elif s0[first].lower() == 'infected':
				# Set up a list to store already infected nodes.
				already = [first]

				# Loop through every other node in the order.
				for i in range(1,len(order)):
					# Get the node name (destination node).
					inode = order[i]
					# Find all sources that can infect this destination node.
					sources = sends_fr[inode]
					# Initialize the indicator: whether the ordering is valid up
					# to this node.
					ind = False
					# See whether any source node has been infected.
					for s in range(len(sources)):
						if sources[s] in already:
							already.append(inode)
							ind = True
							break
					# No source nodes have been infected
					if ind == False:
						valid = False
						break

			# Unknown state
			else:
				raise ValueError('Unknow state: {}!').format(s0[first])

	return valid


def gen_orderings(SFTnet,s0):
	"""
	Returns a list of all allowable node infection orderings.

	SFTnet: SFTnet instance
		The net where all allowable node infection orderings are computed

	s0: dict
		The state of the SFT net at 0. Key is the node name, value is the state.
	"""

	# Local assignments
	nodes = SFTnet.nodes
	names = SFTnet.node_names

	# Assert the nodes are unique
	assert len(names) == len(set(names))

	# Initialize the returned list.
	orderings0 = []

	# Loop through s = 0:N, where N is the number of nodes, and extend the list
	# all permutations of cardinality s. Note that the list after this loop is
	# not our returned list yet.
	n_nodes = len(nodes)
	for s in range(n_nodes + 1):
		orderings0.extend(list(permutations(names,s)))

	orderings = copy.copy(orderings0)

	# Loop through the collection of the orderings.
	for o in range(len(orderings0)):
		# Get the specific order
		order = orderings0[o]
		# Initialize the dictionary that stores all source nodes of the nodes
		# in the order
		sends_fr = {}

		# Loop through all nodes in the order and find all source nodes of each.
		for n in order:
			# Intialize the list of source nodes of this node
			sends_fr[n] = []
			# Loop through all nodes
			for nn in nodes:
				if n in nn.sends_to:
					sends_fr[n].append(nn.name)

		# Call the "validate" function to validate this ordering
		valid = validate(order,sends_fr,s0)

		# If the flag is positive, keep the ordering; otherwise, remove it.
		if valid == False:
			orderings.remove(order)


	return orderings


# Test the gen_orderings function.
if __name__ == '__main__':
	from sft import *
	from sft_net import *

	# Create 4 nodes
	A = SFT('A', ['normal', 'infected'], ['B', 'C'],
		  {'B':np.array([[1, 0], [1, 1/10000.]]),
		   'C': np.array([[1,0], [1,1/10000.]])},
		   ['clean', 'malicious'], 'external')

	B = SFT('B', ['normal', 'infected'], ['C', 'D'],
		   {'C': np.array([[1,0], [1,.1]]),
			'D' : np.array([[1,0], [1,.1]])},
		   ['clean', 'malicious'], 'internal')

	C = SFT('C', ['normal', 'infected'], ['D'],
		   {'D': np.array([[1,0], [1,.1]])},
		   ['clean', 'malicious'], 'internal')

	D = SFT('D', ['normal', 'infected'], [], {}, [], 'internal')

	# Create an SFT net
	nodes = [A, B, C, D]
	net = SFTNet(nodes)
	state0 = dict(A = 'infected',
				  B = 'normal',
				  C = 'normal',
				  D = 'normal'
				 )

	allowable_orderings = gen_orderings(net,state0)
	print allowable_orderings
