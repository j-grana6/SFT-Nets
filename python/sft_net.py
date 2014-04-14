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

    mal_trans_mats : list
        A list whose order is determined by cross_S.  The kth element
        is a matrix whose i,j, entry represents the rate at which i sends
        *malicious* messages to j given that the state of the system is
        given by cross_S[k].


    Methods
    -------

    _gen_cross_S :
        Generates a list of all possible states.

    _gen_transmission :
        Generates the transmission matrix for every possible
        state of the net.

    get_state :
        returns the state of the nodes.  Order is given by node order




    """

    def __init__(self, nodes):
        self.nodes = nodes
        # self._gen_cross_s()
        # self._gen_transmission()
        # self._gen_mal_transmission()
        self.node_names = [x.name for x in nodes]
        self.node_dict = dict([(node.name, node) for node in self.nodes])
        self.internals = [ node for node in self.nodes
                           if node.location == 'internal']
        self.externals = [ node for node in self.nodes
                           if node.location=='external']
        self._gen_trans_vect()


    # def _gen_cross_s(self):
    #     """
    #     Generates the Cartesian product of S_v.  Creates
    #     attribute self.cross_S.  Each element represents
    #     a (possibly probability 0) state  of the net.
    #     """
    #     S_collection = []
    #     for node in self.nodes:
    #         S_collection.append(node.states)
    #         # Create a list of lists.  Each element
    #         # is a list of states for a node
    #     cross_S = list(itertools.product(*S_collection))
    #     self.cross_S = [list(x) for x in cross_S]
    #     # Gives the Cartesian product of the sets of states of
    #     # each node.  This will determine the transmission rates


    # def _gen_transmission(self):
    #     num_nodes = len(self.nodes)  # How many SFTs
    #     transmission_mats = []
    #     # A container for all transmission matrices.
    #     # The order is determined by the order of self.cross_S
    #     for config in self.cross_S:
    #         # Step through states of the net.
    #         transmission = np.zeros((num_nodes, num_nodes))
    #         for nodenum in xrange(len(self.nodes)):
    #             # It is convenient to keep the index
    #             this_node = self.nodes[nodenum]
    #             # The node whose sending rate we are filling in
    #             state = config[nodenum]
    #             # The state of the node
    #             state_ix = this_node.states.index(state)
    #             # The index of the state of the node
    #             for o_node_num in xrange(len(self.nodes)):
    #                 # Step through other nodes
    #                 if self.nodes[o_node_num].name in this_node.sends_to:
    #                     # If this_node sends  messages to another_node
    #                     o_node_name = self.nodes[o_node_num].name
    #                     transmission[nodenum, o_node_num] = \
    #                         np.sum(this_node.rates[o_node_name][state_ix, :])
    #                     # Fill in the transmission rate matrix.  The i,j
    #                     # element of the matrix is the rate at which i sends
    #                     # messages *of any type* to j, given i is in the state
    #                     # determined by config.
    #         transmission_mats.append(transmission)
    #     self.transmission_mats = transmission_mats

    def get_state(self):
        return [x.state for x in self.nodes]

    def _gen_trans_vect(self):
        """
        Used to create arrays of rates for given states, then we can
        use them to build the entire matrix
        """
        clean_rates = {}
        mal_rates = {}
        inf_rates = {}
        for nd in self.nodes:
            clean_r = []
            mal_r = []
            inf_r = []
            for nd2 in self.node_names:
                if nd2 in nd.sends_to:
                    clean_r.append(nd.rates[nd2][0][0])
                    mal_r.append(nd.rates[nd2][1][1])
                    inf_r.append(np.sum(nd.rates[nd2][1,:]))
                else:
                    clean_r.append(0)
                    mal_r.append(0)
                    inf_r.append(0)
            clean_rates[nd.name] =clean_r
            mal_rates[nd.name] = mal_r
            inf_rates[nd.name] = inf_r
            
        self.clean_trans = clean_rates
        self.mal_trans = mal_rates
        self.inf_trans = inf_rates
                    
    # def _gen_mal_transmission(self):
    #     """
    #     Returns a list of transmission probabilities of malicious
    #     messages given the state of the network.  The order
    #     is given by cross_S in SFTNet.

    #     Parameters
    #     ----------

    #     SFTNet : SFTNet instance
    #         The net

    #     """
    #     num_nodes = len(self.nodes)  # How many SFTs
    #     transmission_mats = []
    #     # A container for all transmission matrices.  The order is determined
    #     # by the order of self.cross_S
    #     for config in self.cross_S:
    #         # Step through states of the net.
    #         transmission = np.zeros((num_nodes, num_nodes))
    #         for nodenum in xrange(len(self.nodes)):
    #             # It is convenient to keep the index
    #             this_node = self.nodes[nodenum]
    #             # The node whose sending rate we are filling in
    #             infct_ix = this_node.states.index('infected')
    #             # The index of 'infected' state.
    #             state = config[nodenum]
    #             # The state of the node, not the Network
    #             # I will use 'config' as the state of a Network
    #             # and 'state' as the state of the node.
    #             state_ix = this_node.states.index(state)
    #             # The index of the state of the node
    #             if state == 'infected':
    #                 for o_node_num in xrange(len(self.nodes)):
    #                     # Step through other nodes
    #                     if self.nodes[o_node_num].name in this_node.sends_to:
    #                         # If this_node sends  messages to another_node
    #                         o_node_name = self.nodes[o_node_num].name
    #                         transmission[nodenum, o_node_num] = \
    #                             (this_node.rates[o_node_name]
    #                              [state_ix, infct_ix])
    #                         # Fill in the transmission rate matrix.  The i,j
    #                         # element of the matrix is the rate at which i
    #                         # sends * infected * messages  to j, given i is
    #                         # in the state determined by config.
    #         transmission_mats.append(transmission)
    #     self.mal_trans_mats = transmission_mats


        
    def get_mal_trans(self):
        mal_trans = []
        non_infect = list(np.zeros(len(self.nodes)))
        i=0
        for nd in self.nodes:
            if nd.state =='normal':
                mal_trans.append(non_infect)
            else:
                mal_trans.append(self.mal_trans[nd.name])
        return np.asarray(mal_trans)

    def get_all_trans(self):
        all_trans = []
        for nd in self.nodes:
            if nd.state =='normal':
                all_trans.append(self.clean_trans[nd.name])
            else:
                all_trans.append(self.inf_trans[nd.name])
        return np.asarray(all_trans)
