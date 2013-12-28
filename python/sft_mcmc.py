import numpy as np
from tools import prob_model_given_data, convoluted_cdf_func, \
      convoluted_pdf_func, qij_over_qji

def MCMC_SFT(SFTNet, data, N, z0, T):
    """
    SFTNet : SFTNet instance
        The net to do MCMC over

    data : list
        The data as outputted by gen_data

    N : int
        The number of MCMC proposals

    z0 : dict
        Initial guess for infection times.  Keys are node names
        values are floats

    T : int
        How long the process ran for.
    """
    n = 1
    # initiate step
    prob_mod = lambda x : prob_model_given_data(SFTNet, data[1], x,
                                                data[2], data[3], T)
    # lambda function that calls prob_model_given_data for
    # specified infection times
    p0 = prob_mod(z0)
    # Initiial probability
    t0 =np.asarray(z0.values())
    za, zc, zb, zd = list(t0)
    # actual times
    time_samples = []
    # container for samples
    probs = []
    # container for probabilities
    con_cdf = convoluted_cdf_func(20000 **.5, 0, 50)
    while n < N:
        za = 0
        zb = z0['B'] + np.random.normal() *  100
        zc = min(z0['C'] + np.random.normal() *  100,
                 zb + np.random.random()* 50)
        zd = min(zb, zc) + np.random.random() * 50
        z1 = dict(zip(['A', 'B', 'C', 'D'], [za, zb,zc, zd]))
        p1 = prob_mod(z1)
        if min(z1.values()) >=  0:
            log_q_ratio = qij_over_qji(z0,z1, con_cdf, convoluted_pdf_func)
            if (p1 - p0   + log_q_ratio >
                np.log(np.random.random())):
                print 'A Jump at, ', n, 'to ', z1, 'with prob', p1, '\n'
                t0 = z1.values()
                p0 = p1
                z0 = z1
        time_samples.append(t0)
        probs.append(p0)
        n += 1
    return time_samples, probs, z1.keys()
    # For some reason the dictionary was switching order
