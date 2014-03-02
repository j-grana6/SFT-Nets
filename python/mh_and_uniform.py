from testing_net import *
from uniform_approx import *
#from mh_0_infty import *
from lhood_comps import MCMC_MH

def go(SFTNet, T, s0, uniform_sample_size, Mh_steps):
    data = gen_data(T, SFTNet, s0)
    mh_res = MCMC_MH(SFTNet, data, s0, Mh_steps, T, print_jumps=True)
    uni_res = uniform_samp(SFTNet, s0, uniform_sample_size, T, data)
    return uni_res, mh_res, data

if __name__ == '__main__':
    reps = 1
    t0 = { 'A' : 'infected', 'B': 'normal', 'C': 'normal', 'D': 'normal'}
    mh_t = []
    mh_res = []
    uni = []
    uni_times = []
    diffs=[]
    times = []
    truep = []
    for i in range(reps):
        res = go(net, 10000, t0, 10000, 1000000)
        mh_res.append(res[1])
        uni.append(res[0])
        uni_times.append(res[0][0])
        mh_time = res[1].calc_log_likelihood(burnin=0)
        mh_t.append(mh_time)
        diffs.append(res[0][0] - mh_time)
        times.append(res[2][-1])
        truep.append(res[1].p_true_vals)
        print mh_time -res[0][0]
        print '================'
        print res[1].calc_log_likelihood(burnin=50000) - res[0][0]
        print '================'
        print res[1].calc_log_likelihood(burnin=100000) - res[0][0]

    from matplotlib import pyplot as plt
    b_times = [time['B'] for time in times]
    c_times = [time['C'] for time in times]
    d_times = [time['D'] for time in times]
    p_no_attacker = [res.p_no_attacker for res in mh_res]
    num = str(np.random.random())[:5]
    fig, ax = plt.subplots()
    ax.hist(diffs, bins = np.arange(-30, 10,30))
    ax.set_title('Histogram of Estimation Difference, Uni - MH')
    fig.tight_layout()
    fig.savefig(num +'hist_of_diffs.png')

    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    fig.suptitle('Difference in estimation against node infection time')
    ax1.scatter(b_times, diffs)
    ax1.set_ylabel('Difference in Estimation')
    ax1.set_xlabel('B\'s infection time')


    ax2.scatter(c_times, diffs)
    ax2.set_ylabel('Difference in Estimation')
    ax2.set_xlabel('C\'s infection time')

    ax3.scatter(d_times, diffs)
    ax3.set_ylabel('Difference in Estimation')
    ax3.set_xlabel('D\'s infection time')
    fig.tight_layout()
    fig.savefig(num + 'diff_vs_times.png')

    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    ax1.scatter(p_no_attacker, diffs)
    ax1.set_ylabel('Difference in Estimation')
    ax1.set_xlabel('Probability no Attacker')
    ax1.set_title(' Estimation Difference against P(d|no attacker)')

    ax2.scatter(p_no_attacker, uni_times)
    ax2.set_ylabel('P(data|attacker')
    ax2.set_xlabel('P(data | no attacker)')
    ax2.plot(np.arange(-200, 0,1), np.arange(-200,0,1))
    ax2.set_title('Uniform vs no attacker')

    ax3.scatter(p_no_attacker, mh_t)
    ax3.set_ylabel('P(data|attacker')
    ax3.set_xlabel('P(data | no attacker)')
    ax3.plot(np.arange(-200, 0,1), np.arange(-200,0,1))
    ax3.set_title('MH vs no attacker')
    fig.tight_layout()

    fig.savefig(num+'against_no_attacker.png')

    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import rc
    rc('mathtext', default='regular')
    pdf_doc = PdfPages('convergence.pdf')
    burn =50000
    #plt.tight_layout()
    for r in range(reps):
        mh_conv = mh_res[r].plot_convergence()
        mh_w_burnin = np.log(np.cumsum(np.exp(mh_res[r].log_pdata[burn:]))/np.arange(len(mh_res[r].log_pdata[burn:])))
        unif_samps = np.exp(np.asarray(uni[r][1]))
        (num_configs, samps_per) = unif_samps.shape
        ncs = np.asarray(uni[r][2]).reshape(num_configs, 1)
        unif_samps = unif_samps * ncs
        unif_samps = np.cumsum(unif_samps, axis=1)
        unif_samps = unif_samps / np.arange(1, samps_per +1, 1)
        uni_ma = np.log(np.sum(unif_samps, axis=0))
        tt= times[r]
        true_txt = 'True infection times are ' +'A:' +str(tt['A']) +\
          ', B: ' + str(tt['B']) + ', C: ' +str(tt['C']) + \
          ', D: ' + str(tt['D'])
        x_axis_unif = np.arange(num_configs, num_configs *
                                (samps_per +1), num_configs)
        fig, (ax12, ax11) = plt.subplots(2,1)
        p1 = ax11.plot(np.arange(len(mh_conv)) , mh_conv, label ='MH')
        ax11.set_xlabel('Metropolis Samples')
        ax11.set_ylabel('Running Likelihood')
        p2 = ax11.plot(np.arange(len(mh_w_burnin)), mh_w_burnin, label = 'w/ burn')
        ax11.set_position([.05,.05,.7, .35])
        ax22 = ax11.twiny()
        p3 = ax22.plot(x_axis_unif, uni_ma, label ='Uni', color='red')
        ax22.set_xlabel('Uniform Sample')
        ax22.set_position([.05,.05,.7,.35])
        ax11.legend(p1+ p2+ p3, ['MH', '50kburnin', 'uni'], loc='center left', bbox_to_anchor = (1, .75))
        fig.suptitle(true_txt)
        n1 = ax12.plot(mh_res[r].res['B'], label='B')
        n2 = ax12.plot(mh_res[r].res['C'], label='C')
        n3 = ax12.plot(mh_res[r].res['D'], label='D')
        ax12.legend(n1+n2+n3, ['B', 'C', 'D'], loc = 'upper center', fancybox=True, ncol=3, bbox_to_anchor=(.5, 1.15))
        ax12.set_position([.1, .5, .35, .35])
        for tick in ax12.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        axfin = fig.add_axes([.5, .5, .45, .35])
        f1 = axfin.plot(mh_res[r].res['P(data | z, attacker)'], label = 'P(data)')
        f2 = axfin.plot(mh_res[r].res['P(z | attacker)'], label = 'P(z)')
        axfin.legend(f1+f2, ['P(data)', 'P(z)'], loc='lower center', prop={'size':10}, bbox_to_anchor = (1, .75))
        pdf_doc.savefig(fig)
        plt.close(fig)
    pdf_doc.close()


    

    
