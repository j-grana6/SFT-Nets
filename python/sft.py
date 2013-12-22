"""
An OO approach to SFT's
"""


class SFT(object):
    """
    An SFT node.

    Parameters
    ----------

    name : str
        The name of the node.  ex. 'attacker'.

    states : list of str
        A list containing strings where each string represents a possible
        state.  ex. ['normal', 'infected']

    sends_to  : list of str
        A list whose elements are names of SFTs that receive messages from
        the SFT instance.


    messages : list
        A list of possible messages.  e.x. ['clean', 'malicious']

    rates : dict
        A dictionary with keys being the elements of sends_to.  The entry is a
        pxq array where p = len(states) and q = len(messages).  The order is
        determined by the order of sends_to and messages.
        e.x. for some node, if sends_to is ['A', 'B'], states is
        ['normal', 'infected'] and messages are ['clean', 'malicious'] then
        rates can be
        >>> {'A': [[1, 0],[1, .00001]] , 'B': [[1,0], [1, .1]]}
        which means that the SFT sends clean messages to 'A' at a rate of 1
        when it is in the normal state and sends message no malicious messages.
        When the SFT is in an infected state, it sends clean messages to 'A'
        at a rate of 1 and malicious messages at a rate of .00001.


    Notes
    -----

    - Right now, there is no spontaneous state change.  However, when it comes
      time to include it, we can consider it as a node having a transmission
      with itself.

    - As is, there are also no messages emitted up receiving message
      and the only possible state change is a change to infected upon
      receiving an infected message.

    - To generalize each node should take a pi-sp and pi-st input

    Attributes
    ----------

    state : string, float, arr


    Methods
    -------

    react :
        Given a message and a sender, the node reacts by changing
        state and/or sending out a message.
    """


    def __init__(self, name, states, sends_to, rates, messages):
        self.name = name
        self.states = states
        self.sends_to = sends_to
        self.rates = rates
        self.messages = messages
        self.state = None

    def react(self, message, source):
        if message == 'malicious':
            self.state = 'infected'
    #  This is obviously not general but is the case for the
    #  cyber project.  Ideally, it would be a draw from
    #  pi-sp given message, source and state.


