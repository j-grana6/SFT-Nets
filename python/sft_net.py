"""
Constructs an SFT net from SFT's
"""
import itertools
import numpy as np


class SFTNet(object):
    """
    Creates an SFT-Net from SFT objects.

    The basic functionality of the SFTNet is to aggregate the information
    in each SFT instance and provide basic functions that facilitate Munsky's
    version of the Gillespie algorithm.  It removes all of the boiler plate
    so minimal computation is needed while doing the MCMC.

    Parameters
    ----------

    nodes : list
         A list containing SFT instances to be included in the Net.

    utility : list
        A list of utility functions


    Attributes
    ----------

    cross_S : list
        The Cartesian product of all states of the node.  Each
        element of cross_S represents a possible state of
        \sigma.

    transmission_mats : list
        A list whose order is determined by cross_S.  The kth element
        of transmission_mats is a matrix whose i,j entry represents
        the rate at which i sends messages to j given that the state
        of the system is given by cross_S[k].




    Methods
    -------

    _gen_cross_S :
        Generates a list of all possible states.

    _gen_transmission :
        Generates the transmission matrix for every possible
        state of the net.




    """

    def __init__(self, nodes):
        self.nodes = nodes
        self._gen_cross_s()
        self._get_transmission()

    def _gen_cross_s(self):
        """
        Generates the Cartesian product of S_v.  Creates
        attribute self.cross_S.  Each element represents
        a (possibly probability 0) state  of the net.
        """
        S_collection = []
        for node in self.nodes:
            S_collection.append(node.states)
            # Create a list of lists.  Each element
            # is a list of states for a node
        self.cross_S = list(itertools.product(*S_collection))
        # Gives the Cartesian product of the sets of states of
        # each node.  This will determine the transmission rates.

    def _gen_transmission(self):
        num_nodes = len(self.nodes)  # How many SFTs
        transmission_mats = []
        # A container for all transmission matrices.  The order is determined
        # by the order of self.cross_S
        for config in self.cross_S:
            # Step through states of the net.
            transmission = np.zeros((num_nodes, num_nodes))
            for nodenum in xrange(len(self.nodes)):
                # It is convenient to keep the index
                this_node = self.nodes[nodenum]
                # The node whose sending rate we are filling in
                state = config[nodenum]
                # The state of the node
                state_ix = this_node.states.index(state)
                # The index of the state of the node
                for o_node_num in xrange(len(self.nodes)):
                    # Step through other nodes
                    if self.nodes[o_node_num].name in this_node.sends_to:
                        # If this_node sends  messages to another_node
                        o_node_name = self.nodes[o_node_num].name
                        transmission[nodenum, o_node_num] = \
                            this_node.rates[o_node_name][state_ix]
                        # Fill in the transmission rate matrix.  The i,j
                        # element of the matrix is the rate at which i sends
                        # messages to j, given i is in the state determined
                        # by config.
            transmission_mats.append(transmission)
        self.transmission_mats = transmission_mats






