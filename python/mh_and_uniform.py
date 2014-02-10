from testing_net import *
from uniform_approx import *
from mh_0_infty import *

def go(SFTNet, T, s0, uniform_sample_size, Mh_steps):
    data = gen_data(T, SFTNet, s0)
    uni_res = uniform_samp(SFTNet, s0, uniform_sample_size, T, data)
    mh_res = MH_to_infty(SFTNet, T, s0, Mh_steps, data)
    return uni_res, mh_res, data

if __name__ == '__main__':
    t0 = { 'A' : 'infected', 'B': 'normal', 'C': 'normal', 'D': 'normal'}
    mh_t = []
    mh_res = []
    uni = []
    diffs=[]
    times = []
    truep = []
    for i in range(30):
        res = go(net, 10000, t0, 20000, 1000000)
        mh_res.append(res[1])
        uni.append(res[0])
        mh_time = res[1].calc_log_likelihood(burnin=100000)
        mh_t.append(mh_time)
        diffs.append(res[0] - mh_time)
        times.append(res[2][-1])
        truep.append(res[1].p_true_vals)
        print mh_time -res[0]

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

ax2.scatter(p_no_attacker, uni)
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
