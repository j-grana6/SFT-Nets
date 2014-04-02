from biggernet import net as bignet
from testing_net import net as briansnet
from uniform_approx import uniform_samp
from lhood_comps import MCMC_MH
from tools import gen_data, prob_model_no_attacker 
import numpy as np
from mcmc_over_dags import MCMC_sequence
from direct_sample import Direct_Sample

def get_roc_coords(seed, num_pos, num_neg, i_net,s0,
                   method = 'Direct_Sample', T=10000, uni_samp_size = 2000, mcmc_steps =5000, directsamps=1000,
                   burnin_rate = .25, printsteps=False):
    """
    num_pos : int
        Number pf infected nets in the sample

    num_neg :
        Number of clean nets to generate

    i_net : SFTNet
        The net instance with an attacker


    s0 : dict
       Initial state of the net when there is an attacker
    
    T : int
        Observation Window

    uni_samp_size : int
        The sample size for each infection ordering in uniform sampling

    """
    np.random.seed(seed)
    infected_lhoods = []
    # Will store the lhood w attacker and lhood difference
    clean_lhoods = []
    for i in range(num_pos):
        if printsteps:
            print 'i= ', i
        data = gen_data(T, i_net, s0)
        if method == 'uniform' :
            res = uniform_samp(i_net, s0, uni_samp_size, T, data)[0]
        elif method == 'mcmc' :
            mh_res = MCMC_sequence(i_net, data, s0, mcmc_steps, T , print_jumps=False)
            res = mh_res.calc_log_likelihood(burnin=int(burnin_rate * mcmc_steps))
        else:
            res = Direct_Sample(i_net, data, directsamps, T, s0)[0]
        p_no_attacker = prob_model_no_attacker(i_net, data, T)
        # infected_lhoods.append((uni_res[0], p_no_attacker))
        infected_lhoods.append((res, p_no_attacker))
    for j in range(num_neg):
        if printsteps:
            print 'j =', j
        data = gen_data(T, i_net, dict(zip(i_net.node_names, ['normal'] * len(i_net.nodes))))
        if method == 'uniform':
            res = uniform_samp(i_net, s0, uni_samp_size, T, data)[0]
        elif method=='mcmc' :
            mh_res = MCMC_sequence(i_net, data, s0, mcmc_steps, T, print_jumps=False)
            res = mh_res.calc_log_likelihood(burnin=int(burnin_rate * mcmc_steps))
        else:
            res = Direct_Sample(i_net, data, directsamps, T, s0)[0]
        p_no_attacker = prob_model_no_attacker(i_net, data, T)
        
        clean_lhoods.append((res, p_no_attacker))
    return infected_lhoods, clean_lhoods
    
def handle_parallel_res(res, numcores=4):
    pos = []
    neg = []
    for i in range(numcores):
        pos.append(res[i][0])
        neg.append(res[i][1])
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    pos = pos.reshape(-1, 2)
    neg = neg.reshape(-1, 2)
    return pos, neg

def plot_our_roc(infect_res, clean_res, lhood_ratio_step):
    """
    infect_res : array
        array with columns P(data | attacker) and P(data | no attacker) when there
        is an attacker

    clean res : array
        array with columns P(data | attacker) and P(data | no attacker) when there
        is not an attacker

    step : float
        Threshhold step
    """
    roc_pts = []
    infect_res = np.asarray(infect_res)
    infect_lhood_diff  = infect_res[:, 0] - infect_res[:, 1]
    clean_res = np.asarray(clean_res)
    clean_lhood_diff = clean_res[:, 0] - clean_res[:, 1]
    threshmin = min( min(clean_lhood_diff), min(infect_lhood_diff))
    threshmax = max( max(clean_lhood_diff), max(infect_lhood_diff))
    for step in np.arange(threshmin, threshmax, lhood_ratio_step):
        tps = np.sum(infect_lhood_diff > step)
        tps_rate = float(tps) / float( len(infect_lhood_diff) )
        fps = np.sum(clean_lhood_diff > step)
        fps_rate = float(fps) / float( len ( clean_lhood_diff))
        roc_pts.append((fps_rate, tps_rate, step))
    return roc_pts


def plot_anomaly_roc(infect_res, clean_res, lhood_step):
    """
    See above
    """
    lhood_infect = np.asarray(infect_res)[:,1]
    lhood_clean = np.asarray(clean_res)[:, 1]
    roc_pts = []
    threshmin = min( min(lhood_infect), min(lhood_clean))
    threshmax = max( max(lhood_infect), max(lhood_clean))
    for step in np.arange(threshmin, threshmax, lhood_step):
        tps = np.sum(lhood_infect < step)
        tps_rate = float(tps) / float( len(lhood_infect) )
        fps = np.sum(lhood_clean < step)
        fps_rate = float(fps) / float( len(lhood_clean))
        roc_pts.append((fps_rate, tps_rate, step))
    return roc_pts

def clean_res(results, percore,numcores=4): 
    pos = []
    neg = []
    for i in range(numcores):
        pos.append(res[i][0])
        neg.append(res[i][1])
    pos = np.asarray(pos)
    pos = pos.reshape(percore * 4,2)
    neg = np.asarray(neg)
    neg = neg.reshape(percore * 4,2)
    return pos, neg
    
if __name__ == '__main__':
    percore =1
    s0 = {'A' : 'infected', 'B': 'normal', 'C': 'normal', 'D': 'normal'} 
    def f(seed):
        return get_roc_coords(seed, percore, percore, briansnet, s0=s0, T=10000, printsteps=True)

    from multiprocessing import Pool
    P = Pool(4)
    res = P.map(f, [1,2,3,4])
    pos = []
    neg = []
    for i in range(4):
        pos.append(res[i][0])
        neg.append(res[i][1])
    pos = np.asarray(pos)
    pos = pos.reshape(-1,2)
    neg = np.asarray(neg)
    neg = neg.reshape(-1,2)


    our_res = np.asarray(plot_our_roc(pos, neg, .5))
    anom_res = np.asarray(plot_anomaly_roc(pos, neg,.5))
    label = str(np.random.random())[2:5]
    np.savetxt('./'+str(label) + 'our_roc.csv', our_res, delimiter = ',')
    np.savetxt('./' + str(label) + 'anom_roc.csv', anom_res, delimiter =',')

