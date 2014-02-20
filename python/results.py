"""
A script to handle results, make plots, etc
"""
import numpy as np
from matplotlib import pyplot as plt
import os
import copy as copy
import pandas as pandas

class Results(object):
    """
    The results class

    Parameters
    ----------

    mc : tuple
        The output of a MCMC.  Should contain data, likelihood
        and the lower bound term.

    true_times : dict
        The true infection times

    lp_no_attacker : float
        log probability with no attacker

    lp_true_vals :
        log probability at true infection times


    metropolis : bool
        If True, result is from Metropolis Hastings, otherwise
        it is simple integral approximation


    """

    def __init__(self, mc, true_times, lp_no_attacker, lp_true_vals,
                  data, metropolis=False):
        self.res = mc
        self.log_pdata = self.res['P(data | z, attacker)']
        self.log_pz = self.res['P(z | attacker)']
        self.true_times = true_times
        self.metropolis = metropolis
        self.p_no_attacker = lp_no_attacker
        self.p_true_vals = lp_true_vals
        self.data = data

    def calc_log_likelihood(self, burnin=0):
        """
        Calculates the log likelihood
        """
        probs = self.log_pdata[burnin:]
        log_likelihood = np.log(np.mean(np.exp(probs)))
        return log_likelihood

    def plot_convergence(self):
        if  self.metropolis:
            running_llhood = np.log(np.cumsum(np.exp(self.log_pdata))\
                            / np.arange(len(self.log_pdata)))

        # else:
        #     log_probs = self.log_pz + self.log_pdata
        #     probs = np.exp(log_probs)
        #     ra_probs = np.cumsum(probs)/np.arange(len(probs))
        #     scaled_by_v = ra_probs
        #     running_llhood = np.log(scaled_by_v + self.lower_bound)
        return running_llhood

    def write_results(self, f_pre):
        """
        Writes the results to a directory "results"
        """
        if not os.path.exists("results"):
            os.makedirs("results")
        data = copy.deepcopy(self.res)
        data['running'] = self.plot_convergence()
        data_path = os.path.join('results', f_pre + '_raw.csv')
        data.to_csv(data_path, index=False, sep=',')
        fig, ax = plt.subplots()
        ax.plot(self.plot_convergence())
        ax.set_ylabel('Integral Approximation')
        ax.set_xlabel('Monte Carlo Iteration')
        ax.set_title('Convergence of (MC)MC')
        fig_path = os.path.join('results', f_pre+'_fig.png')
        fig.savefig(fig_path)
        obs_path = os.path.join('results', f_pre + '_obs.csv')
        cols = ['msg_type', 'time', 'sender', 'receiver']
        obsframe = pandas.DataFrame({cols[i]: self.data[i]
                                   for i in range(4)})
        obsframe.to_csv(obs_path, index=False, sep= ',')
        info_path = os.path.join('results', f_pre + '_info.txt')
        with open(info_path , 'wb') as f:
            integralval = self.calc_log_likelihood()
            f.write('The integral Value is .....' + str(integralval) +'\n')
            f.write('Likelihood of no attacker is .....' +
                    str(self.p_no_attacker) + '\n')
            f.write('The true infection times are ..... \n')
            for key in self.true_times.keys():
                f.write(str(key) + ":" + str(self.true_times[key]) + ",")
            f.write('\n Was this done using MH?.....' + str(self.metropolis))

