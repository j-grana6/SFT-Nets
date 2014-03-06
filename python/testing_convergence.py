import numpy as np
import matplotlib.pyplot as plt

from testing_net import *
from mh_and_uniform import go
from uniform_approx import *
from lhood_comps import MCMC_MH
from sft import *
from sft_net import *

# parameters ranges
ratesA = [1/10000.,2/10000.]#np.arange(1/10000., 2.5/10000., 0.5/10000.)
ratesB = [0.1,0.2]#np.arange(0.05, 0.2, 0.05)
ratesC = [0.1,0.2]#np.arange(0.05,0.2,0.05)
tmax = [10000,12500,15000]#np.arange(10000,16000,2000)
draws = 10

print "The case of attacker existence."
for rA in ratesA:
    for rB in ratesB:
        for rC in ratesC:
            for T in tmax:
                print "Infection rate of A:", rA
                print "Infection rate of B:", rB
                print "Infection rate of C:", rC
                unis = []
                mhs = []
                diff1 = []
                print "Observation window:", T
                for d in range(draws):
                    print 'Draw:', d + 1
                    A = SFT('A', ['normal', 'infected'], ['B', 'C'],
                          {'B':np.array([[1, 0], [1, rA]]),
                           'C': np.array([[1, 0], [1, rA]])},
                           ['clean', 'malicious'], 'external')

                    B = SFT('B', ['normal', 'infected'], ['C', 'D'],
                           {'C': np.array([[1, 0], [1, rB]]),
                            'D' : np.array([[1, 0], [1, rB]])},
                           ['clean', 'malicious'], 'internal')

                    C = SFT('C', ['normal', 'infected'], ['D'],
                           {'D': np.array([[1, 0], [1,rC]])},
                           ['clean', 'malicious'], 'internal')

                    D = SFT('D', ['normal', 'infected'], [], {}, [], 'internal')

                    nodes = [A, B, C, D]
                    net = SFTNet(nodes)
                    
                    s0 = { 'A' : 'infected', 'B': 'normal', 'C': 'normal', \
                            'D': 'normal'}

                    data = gen_data(T,net,s0)
                    mh_res = MCMC_MH(net,data,s0,500000,T,print_jumps=False)
                    uni_res = uniform_samp(net,s0,10000,T,data)
                    mhs.append(mh_res)
                    unis.append(uni_res)

                    uni_lhood = uni_res[0]
                    mh_lhood = mh_res.calc_log_likelihood(burnin=0)

                    diff1.append(uni_lhood - mh_lhood)

                    burn = 50000
                    mh_conv = mh_res.plot_convergence()
                    mh_w_burnin = np.log(np.cumsum(np.exp(mh_res.log_pdata[burn:]\
                            ))/np.arange(len(mh_res.log_pdata[burn:])))
                    unif_samps = np.exp(np.asarray(uni_res[1]))
                    (num_configs, samps_per) = unif_samps.shape
                    ncs = np.asarray(uni_res[2]).reshape(num_configs, 1)
                    unif_samps = unif_samps * ncs
                    unif_samps = np.cumsum(unif_samps, axis=1)
                    unif_samps = unif_samps / np.arange(1, samps_per +1, 1)
                    uni_ma = np.log(np.sum(unif_samps, axis=0))
                    
                    randstr = str(np.random.random())[:5]
                    x_axis_unif = np.arange(num_configs, num_configs *
                                            (samps_per +1), num_configs)
                    fig,(ax1,ax2) = plt.subplots(2,1)
                    ax1.plot(np.arange(len(mh_conv)),mh_conv,label = 'MH')
                    ax1.plot(np.arange(len(mh_w_burnin)), mh_w_burnin, label = \
                            'with burnin')
                    ax1.legend(bbox_to_anchor = (1,.75), prop = {'size': 10})
                    ax2.plot(x_axis_unif, uni_ma, label = 'Uniform')
                    ax2.legend(bbox_to_anchor = (1,.75), prop = {'size': 10})
                    fig.set_size_inches(15,12)
                    fig.suptitle('rA:'+str(rA)+' rB:'+str(rB)+' rC:'+\
                            str(rC)+' T:'+str(T)+' Draw:'+str(d+1)+' Att')
                    fig.tight_layout()
                    fig.savefig('images/' + randstr + 'Convs.png')


                
                randstr = str(np.random.random())[:5]
                hist, bins = np.histogram(diff1,bins=12)
                fig, ax = plt.subplots(1,1)
                center = (bins[:-1] + bins[1:])/2.
                ax.bar(center,hist,align='center',width=5)
                ax.set_xticks(xrange(-30,40,10))
                ax.set_yticks(xrange(0,6,1))
                ax.set_title('Histogram of Estimation Difference, Uni - MH')
                fig.suptitle('rA:'+str(rA)+' rB:'+str(rB)+' rC:'+\
                        str(rC)+' T:'+str(T)+' Att')
                fig.tight_layout()
                fig.savefig('images/' + randstr + 'HistUniVsMH.png')
                    


