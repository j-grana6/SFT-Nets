mport numpy as np
import random
import copy
# Parameters
p = .01 # Infection probability
q =.5  # Background probability
w = 50 # Window
k =45 # Threshold
gamma = 1

class TwoNodeMCTS(object):
    def __init__(self, p=p, q=q, w=w, k=k,gamma=gamma, default_policy =.5, c=5):
        self.p = p # Infection probability
        self.q =q  # Background probability
        self.w = w # Window
        self.k = k # Threshold
        self.gamma = gamma # Discount
        self.dp = default_policy # Rollout policy of sending a message
        self.c = c
    
    def draw_s_o_r(self, current_state, action):
        """
        Returns the next state, observation and reward for 
        a given current action and reward.
        """
        newstate = current_state[1:]
        backgroundmessage = int( random.random() < self.q )
        newstate.append(backgroundmessage + action)
        if np.sum(newstate) > self.k:
            newstate='Alarm'
        elif random.random() * action > 1-self.p:
            newstate = 'Infected'
        return newstate

    def draw_s0(self):
        s0  = list(np.random.binomial(1, self.q, size=self.w))
        while np.sum(s0) > self.k:
            s0 = list(np.random.binomial(1, self.q, size=self.w))
        return s0
    
    def search(self, num_samps=5000):
        T= {}
        noactu = []
        actu = []
        for i in xrange(num_samps):
            s= self.draw_s0()
            r, T = self.simulate(s, list(np.int_(np.zeros(self.w))),T, returnT=True)
            t0 =  T[str(list(np.int_(np.zeros(self.w))))]
            noactu.append(t0[0][0])
            actu.append(t0[1][0])
        return r, T, noactu, actu
    
    def rollout(self, s, h):
        if s == 'Alarm':
            return 0
        if s =='Infected':
            return 1
        action = int(np.random.random() < self.dp)
        new_state = self.draw_s_o_r(s, action)
        h.append(action)
        return self.gamma * self.rollout(new_state, h)
        
        
    
    def simulate(self, s, history, T = {}, returnT=False):
        """
        T is a dictionary keyed by histories. 
        values are 3 element lists.  The first is the Q
        value and the second is the count. The third keeps
        track of the total.
        
        Only terminal rewards.
        
        Might want to convert the history to asci as it should be
        less in memory...maybe?
        """
        if s == 'Alarm': 
            return 0
        if s =='Infected':
            return 1
        if str(history) not in T:
            T[str(history)] = [[0,0], [0,0], 0] # Also stores state count
            if returnT:
                return self.rollout(s, history), T
            else:
                return self.rollout(s, history)
        this_history = T[str(history)]
        action = np.argmax([this_history[0][0] 
                            + self.c*(this_history[2] / float(this_history[0][1] + 1))**.5, 
                            this_history[1][0] 
                            + self.c*(this_history[2] / float(this_history[1][1]+ 1))**.5])
        updated_history = copy.copy(history)
        updated_history.append(action)
        newstate = self.draw_s_o_r(s, action)
        R = self.gamma * self.simulate(newstate, updated_history, T)
        T[str(history)][action][1]  += 1
        T[str(history)][action][0] += (R - T[str(history)][action][0])/float(T[str(history)][action][1])
        T[str(history)][2] += 1
        if returnT:
            return R, T
        else:
            return R

        
        
